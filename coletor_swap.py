#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coletor da série DIÁRIA de swap DI x Pré no vértice de 360 dias corridos.

Fonte: arquivo "Taxas de Mercado para Swaps" da B3, acessado pela biblioteca `pyettj`
(curva 'PRE' = DIxPRE). Para cada dia útil, extrai a taxa no vértice de 360 dias
corridos (exata, se existir; senão, interpolação linear entre vértices vizinhos).

Saída: swap_di_pre_360_diario.csv  (colunas: Data, Taxa, Metodo)
       Taxa em % a.a. (ex.: 14.65). É o arquivo lido por script_painel.py.

Características:
  - PARALELO: baixa várias datas ao mesmo tempo (o gargalo é a rede, não a CPU);
  - retomável: se o CSV de saída já existe, só baixa as datas que faltam;
  - salva o progresso a cada N datas (um erro/queda de rede não perde o trabalho);
  - feriados/dias sem publicação são ignorados; falhas reais vão para um arquivo de log;
  - checagens finais de sanidade (cobertura, saltos diários atípicos).

Uso:
    pip install pyettj pandas
    python coletor_swap.py                                  # 2011-01-03 até hoje, 8 threads
    python coletor_swap.py --workers 12                     # mais requisições simultâneas
    python coletor_swap.py --modo processos --workers 6     # usa vários núcleos (processos)
    python coletor_swap.py --inicio 2011-01-03 --fim 2026-10-09
    python coletor_swap.py --proxy http://usuario:senha@proxy:porta   # rede corporativa
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

try:
    import pyettj as ettj
    from pyettj import HolidayError, NoDataError, PyETTJError
except ImportError:  # pragma: no cover
    sys.exit("Biblioteca 'pyettj' não encontrada. Instale com: pip install pyettj")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("coletor_swap")

# Silencia avisos do próprio pyettj (ex.: "curva com poucos vértices"). Feito globalmente
# porque warnings.catch_warnings() NÃO é seguro entre threads.
warnings.filterwarnings("ignore", message=r"\[pyettj\]")

CURVA = "PRE"                       # DIxPRE no arquivo TaxaSwap da B3
COLUNAS = ["Data", "Taxa", "Metodo"]


# =============================================================================
# EXTRAÇÃO DO VÉRTICE
# =============================================================================
def taxa_no_vertice(curva_df: pd.DataFrame, alvo_dc: int, tolerancia_dc: int):
    """
    Devolve (taxa em % a.a., método) no vértice `alvo_dc` (dias corridos), ou None.

    - Se existe vértice exatamente em alvo_dc: usa-o ('exato').
    - Senão, interpola linearmente entre o vértice anterior e o posterior, desde que
      ambos estejam a no máximo `tolerancia_dc` dias do alvo ('interpolado').
    """
    g = (curva_df[["dias_corridos", "taxa"]].dropna()
         .drop_duplicates("dias_corridos").sort_values("dias_corridos"))
    exato = g[g["dias_corridos"] == alvo_dc]
    if not exato.empty:
        return float(exato["taxa"].iloc[0]) * 100.0, "exato"

    inf = g[g["dias_corridos"] < alvo_dc].tail(1)
    sup = g[g["dias_corridos"] > alvo_dc].head(1)
    if inf.empty or sup.empty:
        return None
    x0, y0 = int(inf["dias_corridos"].iloc[0]), float(inf["taxa"].iloc[0])
    x1, y1 = int(sup["dias_corridos"].iloc[0]), float(sup["taxa"].iloc[0])
    if (alvo_dc - x0) > tolerancia_dc or (x1 - alvo_dc) > tolerancia_dc:
        return None
    y = y0 + (y1 - y0) * (alvo_dc - x0) / (x1 - x0)
    return y * 100.0, "interpolado"


def coletar_dia(data: pd.Timestamp, vertice: int, tolerancia: int, proxies, timeout: int):
    """
    Baixa a curva de um dia e extrai o vértice. Função de nível de módulo (picklável),
    segura para threads e processos.
    Retorna (data, status, payload):
        (d, "ok", (taxa, metodo)) | (d, "sem_dados", None) | (d, "erro", mensagem)

    cache=False: evita o cache em disco do pyettj (que grava TODAS as curvas de cada dia)
    e também as mensagens impressas por ele, que não são seguras em execução paralela.
    O "cache" aqui é o próprio CSV de saída (retomada).
    """
    data_str = data.strftime("%d/%m/%Y")
    try:
        df = ettj.get_ettj(data_str, curva=CURVA, proxies=proxies, timeout=timeout, cache=False)
    except (HolidayError, NoDataError):
        return data, "sem_dados", None                  # feriado / sem publicação
    except Exception as exc:                            # PyETTJError, rede, parsing etc.
        return data, "erro", f"{type(exc).__name__}: {exc}"

    res = taxa_no_vertice(df, vertice, tolerancia)
    if res is None:
        return data, "erro", f"vértice {vertice}d indisponível (tolerância {tolerancia}d)"
    return data, "ok", res


# =============================================================================
# PERSISTÊNCIA
# =============================================================================
def carregar_existente(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUNAS)
    df = pd.read_csv(path, dtype={"Metodo": str})
    df["Data"] = pd.to_datetime(df["Data"], format="%Y-%m-%d", errors="coerce")
    return df.dropna(subset=["Data", "Taxa"])[COLUNAS]


def salvar(df: pd.DataFrame, path: Path) -> None:
    out = (df.drop_duplicates("Data", keep="last").sort_values("Data").copy())
    out["Data"] = pd.to_datetime(out["Data"]).dt.strftime("%Y-%m-%d")
    out["Taxa"] = out["Taxa"].round(6)
    out.to_csv(path, index=False)


# =============================================================================
# CHECAGENS DE SANIDADE
# =============================================================================
def relatorio_final(df: pd.DataFrame, inicio: pd.Timestamp, fim: pd.Timestamp) -> None:
    df = df.copy()
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df["Taxa"] = pd.to_numeric(df["Taxa"], errors="coerce")
    df = df.dropna(subset=["Data", "Taxa"])

    s = df.drop_duplicates("Data", keep="last").set_index("Data")["Taxa"].sort_index()
    if s.empty:
        log.error("Nenhuma observação coletada.")
        return

    esperados = len(pd.bdate_range(inicio, fim))
    log.info("Cobertura: %d obs. (%s a %s) para ~%d dias úteis no intervalo (%.0f%%; a "
             "diferença inclui feriados).", len(s), s.index.min().date(), s.index.max().date(),
             esperados, 100 * len(s) / max(esperados, 1))

    gaps = s.index.to_series().diff().dt.days.dropna()
    longos = gaps[gaps > 5]
    if not longos.empty:
        log.warning("%d lacuna(s) > 5 dias corridos; maiores: %s", len(longos),
                    ", ".join(f"{i.date()} ({int(v)}d)" for i, v in longos.nlargest(5).items()))

    saltos = s.diff().abs().dropna()
    atipicos = saltos[saltos > 1.0]                    # > 100 bps em 1 dia é suspeito
    if not atipicos.empty:
        log.warning("%d variação(ões) diária(s) > 1,0 p.p. — confira se são reais "
                    "(ex.: %s)", len(atipicos),
                    ", ".join(f"{i.date()} ({v:.2f})" for i, v in atipicos.nlargest(5).items()))

    metodos = df["Metodo"].value_counts().to_dict()
    log.info("Método de extração: %s", metodos)


# =============================================================================
# MAIN
# =============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Coleta diária do swap DI x Pré (vértice em dias corridos).")
    ap.add_argument("--inicio", default="2011-01-03", help="AAAA-MM-DD (padrão: 2011-01-03)")
    ap.add_argument("--fim", default=None, help="AAAA-MM-DD (padrão: hoje)")
    ap.add_argument("--vertice", type=int, default=360, help="vértice em dias corridos (padrão: 360)")
    ap.add_argument("--tolerancia", type=int, default=45,
                    help="distância máxima (dias) aos vértices vizinhos na interpolação")
    ap.add_argument("--saida", default="swap_di_pre_360_diario.csv")
    ap.add_argument("--falhas", default="swap_di_pre_360_falhas.txt")
    ap.add_argument("--proxy", default=None, help="ex.: http://usuario:senha@proxy:porta")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--modo", choices=["threads", "processos"], default="threads",
                    help="threads (padrão, ideal p/ rede) ou processos (usa mais núcleos)")
    ap.add_argument("--workers", type=int, default=None,
                    help="execuções simultâneas (padrão: 8 em threads; nº de núcleos em processos)")
    ap.add_argument("--pausa", type=float, default=0.1,
                    help="pausa (s) por worker entre requisições, para não sobrecarregar a B3")
    ap.add_argument("--salvar-a-cada", type=int, default=100)
    args = ap.parse_args(argv)

    n_cpu = os.cpu_count() or 4
    workers = args.workers or (8 if args.modo == "threads" else n_cpu)
    Executor = ThreadPoolExecutor if args.modo == "threads" else ProcessPoolExecutor

    inicio = pd.Timestamp(args.inicio)
    fim = pd.Timestamp(args.fim) if args.fim else pd.Timestamp.today().normalize()
    saida, falhas_path = Path(args.saida), Path(args.falhas)
    proxies = {"http": args.proxy, "https": args.proxy} if args.proxy else None

    existente = carregar_existente(saida)
    ja = set(existente["Data"])
    pendentes = [d for d in pd.bdate_range(inicio, fim) if d not in ja]
    log.info("Intervalo %s a %s | já coletadas: %d | a coletar: %d dias úteis | modo=%s, "
             "workers=%d (CPU lógicas: %d)", inicio.date(), fim.date(), len(ja), len(pendentes),
             args.modo, workers, n_cpu)

    estado = {"novos": [], "erros": [], "sem_dados": 0, "existente": existente}
    t0 = time.time()

    def tarefa(d):
        out = coletar_dia(d, args.vertice, args.tolerancia, proxies, args.timeout)
        time.sleep(args.pausa)
        return out

    def checkpoint():
        if estado["novos"]:
            estado["existente"] = pd.concat([estado["existente"], pd.DataFrame(estado["novos"])],
                                            ignore_index=True)
            estado["novos"] = []
            salvar(estado["existente"], saida)

    def processar(lista, n_workers, rotulo):
        if not lista:
            return
        # Em processos, `tarefa` (closure) não é picklável: usamos a função de módulo direto.
        ex = Executor(max_workers=n_workers)
        try:
            if args.modo == "threads":
                futs = [ex.submit(tarefa, d) for d in lista]
            else:
                futs = [ex.submit(coletar_dia, d, args.vertice, args.tolerancia, proxies,
                                  args.timeout) for d in lista]
            for i, fut in enumerate(as_completed(futs), 1):
                d, status, payload = fut.result()
                if status == "ok":
                    estado["novos"].append({"Data": d, "Taxa": payload[0], "Metodo": payload[1]})
                elif status == "sem_dados":
                    estado["sem_dados"] += 1
                else:
                    estado["erros"].append((d, payload))
                if i % args.salvar_a_cada == 0:
                    checkpoint()
                    dt = time.time() - t0
                    eta = dt / i * (len(lista) - i)
                    log.info("[%s] %d/%d | ok acumulado: %d | sem dados: %d | erros: %d | "
                             "%.1f datas/s | decorrido %.0fs | ETA ~%.0f min", rotulo, i,
                             len(lista), len(estado["existente"]), estado["sem_dados"],
                             len(estado["erros"]), i / max(dt, 1e-9), dt, eta / 60)
        except KeyboardInterrupt:
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            ex.shutdown(wait=True)

    try:
        processar(pendentes, workers, "coleta")
        # Segunda passada, mais gentil (poucos workers), só nas datas que falharam
        if estado["erros"]:
            repetir = [d for d, _ in estado["erros"]]
            log.info("Reprocessando %d data(s) com erro (2 workers)...", len(repetir))
            estado["erros"] = []
            processar(repetir, min(2, workers), "retry")
    except KeyboardInterrupt:
        log.warning("Interrompido pelo usuário; salvando progresso.")
    finally:
        checkpoint()

    existente = estado["existente"]
    erros = estado["erros"]
    if erros:
        falhas_path.write_text("\n".join(f"{d.date()}\t{m}" for d, m in erros), encoding="utf-8")
        tipos = Counter(m[:80] for _, m in erros).most_common(3)
        log.warning("%d data(s) falharam (detalhes em %s). Principais causas: %s",
                    len(erros), falhas_path, tipos)

    relatorio_final(existente, inicio, fim)
    if existente.empty:
        return 1
    log.info("Arquivo gerado: %s | tempo total: %.1f min", saida.resolve(), (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())