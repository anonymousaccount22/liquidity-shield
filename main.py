#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - Orquestrador da pipeline (execução sequencial, robusta e reprodutível)

Fluxo: Ativo Fixo via Balanço Patrimonial (Opção A, sem DFC)

Ordem estrita:
  1. script_painel.py     (painel; 1ª passagem: sem Tobin_Q, ou com o Q da rodada anterior)
  2. calcular_q.py        (Tobin's Q: valor de mercado + dívida / ativo total -> painel_final_q.csv)
  3. script_painel.py#2   (2ª passagem: incorpora o Q na grade e cria Tobin_Q_L1 -> painel_final.csv)
  4. calcular_q.py#2      (recalcula sobre o painel final -> painel_final_q.csv com Tobin_Q_L1)
  5. ortogonalizacao.py   (isolamento do choque monetário de alta frequência, HFI)
  6. capex_bp.py          (Ativo Fixo via BP, log_Ativo_Fixo e integração do painel)
  7. projecoes_locais.py  (estimação das projeções locais e IRFs de Delta_log_Ativo_Fixo)

Por que script_painel.py e calcular_q.py rodam duas vezes:
  calcular_q.py precisa de Ativo_Total e Empréstimos (vindos do script_painel.py) e o
  script_painel.py precisa do Tobin_Q para criar Tobin_Q_L1. Como Ativo_Total/Empréstimos não
  dependem do Q, uma volta completa fecha o ciclo (ponto fixo).
  Pré-requisito: calcular_q.py deve descartar as colunas de Q já existentes ao ler o painel.

Rótulos:
  Cada etapa tem um rótulo único (ex.: "script_painel.py#2"); o arquivo executado é a parte
  antes do "#". Os dicionários ENTRADAS/SAIDAS aceitam o rótulo (prioridade) ou o nome do script.

Uso:
  python main.py                        -> roda tudo
  python main.py capex_bp.py            -> roda a partir dessa etapa (as anteriores ficam PULADAS)
  python main.py "script_painel.py#2"   -> retoma a partir da 2ª passagem (aspas por causa do #)

Comportamento:
  - Cada sub-script roda em um processo próprio (cwd = raiz do projeto), com a saída
    (stdout + stderr) exibida em tempo real.
  - Antes de cada etapa, confere se as entradas obrigatórias existem; depois, se as saídas
    esperadas foram (re)geradas nesta execução (evita seguir adiante com arquivo obsoleto).
  - Se um script falha, o traceback é exibido e a pipeline é interrompida imediatamente.
  - Ao final (com sucesso ou falha) grava `pipeline_execution.log` com data/hora, status
    e tempo de cada etapa, ambiente e SHA-256 dos scripts e dos CSVs da raiz.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
LOG_FILE = RAIZ / "pipeline_execution.log"

PIPELINE = [
    "script_painel.py",
    "calcular_q.py",
    "script_painel.py#2",      # fecha o ciclo: lê painel_final_q.csv e cria Tobin_Q_L1
    "calcular_q.py#2",         # regrava painel_final_q.csv a partir do painel já com Q_L1
    "ortogonalizacao.py",
    "capex_bp.py",
    "projecoes_locais.py",
]

# Arquivos que cada etapa PRECISA encontrar antes de rodar (falha clara e imediata se faltar).
# Chave = rótulo da etapa (prioridade) ou nome do script.
ENTRADAS = {
    "script_painel.py":    (),
    "calcular_q.py":       ("painel_final.csv", "tickers.csv", "precos_trimestrais.csv"),
    "script_painel.py#2":  ("painel_final_q.csv",),
    "ortogonalizacao.py":  ("painel_final_q.csv", "vix_trimestral.csv", "risco_fiscal_trimestral.csv"),
    "capex_bp.py":         ("painel_final.csv", "painel_regressao_q_ortog.csv", "dados_cvm"),
    "projecoes_locais.py": ("painel_regressao_q_bp.csv",),
}

# Arquivos que cada etapa deve ter (re)gerado ao terminar com sucesso
SAIDAS = {
    "script_painel.py":    ("painel_final.csv",),
    "calcular_q.py":       ("painel_final_q.csv",),
    "ortogonalizacao.py":  ("painel_regressao_q_ortog.csv",),
    "capex_bp.py":         ("painel_regressao_q_bp.csv", "cobertura_ativo_fixo_ano.csv"),
    # projecoes_locais.py grava o consolidado abaixo (não existe "resultados_lp_...csv")
    "projecoes_locais.py": ("irf_consolidado_Delta_log_Ativo_Fixo_ctrlL1.csv",),
}

# Pares (arquivo_derivado, arquivo_fonte): aviso se o derivado for mais antigo que a fonte
DEPENDENCIAS_TEMPORAIS = {
    "capex_bp.py": (("painel_regressao_q_ortog.csv", "painel_final.csv"),),
}

PACOTES = ("numpy", "pandas", "requests", "statsmodels", "linearmodels", "matplotlib")
TAIL_LINHAS = 40   # últimas linhas da saída do script guardadas para diagnóstico
TOL_MTIME = 2.0    # tolerância (s) na comparação de datas de modificação


class PipelineError(RuntimeError):
    """Falha na execução de uma etapa da pipeline."""


@dataclass
class Etapa:
    rotulo: str                          # único, ex.: "script_painel.py#2"
    status: str = "NAO EXECUTADO"       # OK | FALHOU | PULADO | NAO EXECUTADO
    duracao: float | None = None
    returncode: int | None = None
    tail: str = ""
    avisos: list[str] = field(default_factory=list)

    @property
    def script(self) -> str:             # nome do arquivo a executar (sem o "#N")
        return self.rotulo.split("#")[0]


def consulta(tabela: dict, etapa: Etapa) -> tuple:
    """Busca pelo rótulo (prioridade) e, na falta dele, pelo nome do script."""
    return tabela.get(etapa.rotulo, tabela.get(etapa.script, ()))


# =============================================================================
# UTILITÁRIOS
# =============================================================================
def fmt_tempo(seg: float) -> str:
    m, s = divmod(seg, 60)
    h, m = divmod(int(m), 60)
    return f"{h}h{m:02d}m{s:05.2f}s" if h else (f"{m}m{s:05.2f}s" if m else f"{s:.2f}s")


def sha256_arquivo(path: Path, bloco: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(bloco), b""):
            h.update(chunk)
    return h.hexdigest()


def versoes_pacotes() -> dict[str, str]:
    out = {}
    for nome in PACOTES:
        try:
            out[nome] = metadata.version(nome)
        except metadata.PackageNotFoundError:
            out[nome] = "não instalado"
    return out


# =============================================================================
# VERIFICAÇÕES DE ENTRADA / SAÍDA
# =============================================================================
def verificar_entradas(etapa: Etapa) -> None:
    ausentes = [n for n in consulta(ENTRADAS, etapa) if not (RAIZ / n).exists()]
    if ausentes:
        raise PipelineError(
            f"{etapa.rotulo}: entradas obrigatórias ausentes na raiz do projeto: {ausentes}. "
            f"Rode as etapas anteriores (ou confira qual script gera esses arquivos).")

    for derivado, fonte in DEPENDENCIAS_TEMPORAIS.get(etapa.script, ()):
        pd_, pf_ = RAIZ / derivado, RAIZ / fonte
        if pd_.exists() and pf_.exists() and pd_.stat().st_mtime + TOL_MTIME < pf_.stat().st_mtime:
            msg = (f"AVISO: {derivado} é mais antigo que {fonte}; pode estar obsoleto "
                   f"(nenhuma etapa desta pipeline o regenera?).")
            etapa.avisos.append(msg)
            print(f"\n[{msg}]", flush=True)


def verificar_saidas(etapa: Etapa, inicio_ts: float) -> None:
    ausentes, obsoletas = [], []
    for nome in consulta(SAIDAS, etapa):
        p = RAIZ / nome
        if not p.exists():
            ausentes.append(nome)
        elif p.stat().st_mtime < inicio_ts - TOL_MTIME:
            obsoletas.append(nome)
    if ausentes or obsoletas:
        raise PipelineError(
            f"{etapa.rotulo} terminou com código 0, mas saídas esperadas "
            f"ausentes={ausentes} / não atualizadas nesta execução={obsoletas}.")


# =============================================================================
# EXECUÇÃO DE UMA ETAPA
# =============================================================================
def executar(etapa: Etapa) -> None:
    """Roda o script em subprocesso, transmitindo a saída em tempo real."""
    caminho = RAIZ / etapa.script
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    verificar_entradas(etapa)

    print(f"\n{'=' * 70}\n>> {etapa.rotulo}\n{'=' * 70}", flush=True)
    env = {**os.environ,
           "PYTHONUNBUFFERED": "1",       # saída em tempo real
           "PYTHONIOENCODING": "utf-8",   # evita erro de encoding no Windows
           "PYTHONHASHSEED": "0"}         # ordem de hash determinística
    tail: deque[str] = deque(maxlen=TAIL_LINHAS)

    inicio_ts = time.time()
    inicio = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-u", str(caminho)],
        cwd=RAIZ,                          # caminhos relativos dos scripts funcionam
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,          # traceback do script aparece na mesma saída
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    try:
        assert proc.stdout is not None
        for linha in proc.stdout:
            print(linha, end="", flush=True)
            tail.append(linha)
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        raise
    finally:
        etapa.duracao = time.perf_counter() - inicio
        etapa.returncode = proc.returncode
        etapa.tail = "".join(tail)

    if proc.returncode != 0:
        raise PipelineError(f"{etapa.rotulo} terminou com código {proc.returncode}")
    verificar_saidas(etapa, inicio_ts)


# =============================================================================
# LOG DE REPRODUTIBILIDADE
# =============================================================================
def escrever_log(etapas: list[Etapa], inicio: datetime, total: float, ok: bool) -> None:
    L: list[str] = []
    L.append("PIPELINE EXECUTION LOG")
    L.append(f"Data/hora de início : {inicio.isoformat(timespec='seconds')}")
    L.append(f"Data/hora de término: {datetime.now().isoformat(timespec='seconds')}")
    L.append(f"Resultado geral     : {'SUCESSO' if ok else 'FALHA'}")
    L.append(f"Tempo total         : {fmt_tempo(total)}")
    L.append("")
    L.append("[ETAPAS]")
    for i, e in enumerate(etapas, 1):
        dur = fmt_tempo(e.duracao) if e.duracao is not None else "-"
        L.append(f"{i}. {e.rotulo:<22} status={e.status:<14} tempo={dur:<12} returncode={e.returncode}")
    avisos = [(e.rotulo, a) for e in etapas for a in e.avisos]
    if avisos:
        L.append("")
        L.append("[AVISOS]")
        for rotulo, a in avisos:
            L.append(f"{rotulo}: {a}")
    falhou = next((e for e in etapas if e.status == "FALHOU"), None)
    if falhou and falhou.tail:
        L.append("")
        L.append(f"[ULTIMAS LINHAS DE SAIDA - {falhou.rotulo}]")
        L.append(falhou.tail.rstrip())
    L.append("")
    L.append("[AMBIENTE]")
    L.append(f"python   : {sys.version.split()[0]} ({platform.python_implementation()})")
    L.append(f"sistema  : {platform.platform()}")
    for nome, ver in versoes_pacotes().items():
        L.append(f"{nome:<13}: {ver}")
    L.append("")
    L.append("[HASH SHA-256 DOS SCRIPTS]")
    for s in dict.fromkeys(e.script for e in etapas):     # sem repetir script_painel/calcular_q
        p = RAIZ / s
        L.append(f"{sha256_arquivo(p) if p.exists() else 'ARQUIVO AUSENTE'}  {s}")
    L.append("")
    L.append("[HASH SHA-256 DOS CSVs NA RAIZ]")
    csvs = sorted(RAIZ.glob("*.csv"))
    if not csvs:
        L.append("(nenhum CSV encontrado na raiz)")
    for p in csvs:
        L.append(f"{sha256_arquivo(p)}  {p.name}  ({p.stat().st_size} bytes)")
    LOG_FILE.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nLog de execução salvo em: {LOG_FILE.name}")


# =============================================================================
# MAIN
# =============================================================================
def main(a_partir_de: str | None = None) -> int:
    etapas = [Etapa(r) for r in PIPELINE]
    inicio_dt = datetime.now()
    t0 = time.perf_counter()
    ok = False

    # --- retomada a partir de uma etapa (aceita o rótulo; o nome sem "#N" vale para a 1ª ocorrência) ---
    idx_ini = 0
    if a_partir_de:
        if a_partir_de not in PIPELINE:
            print(f"[ERRO] Etapa inválida: {a_partir_de}. Opções: {PIPELINE}")
            return 2
        idx_ini = PIPELINE.index(a_partir_de)
        for e in etapas[:idx_ini]:
            e.status = "PULADO"

    try:
        for etapa in etapas[idx_ini:]:
            try:
                executar(etapa)
                etapa.status = "OK"
                print(f"\n[OK] {etapa.rotulo} concluído em {fmt_tempo(etapa.duracao or 0)}", flush=True)
            except KeyboardInterrupt:
                etapa.status = "FALHOU"
                print(f"\n[INTERROMPIDO] {etapa.rotulo} cancelado pelo usuário.")
                raise
            except Exception:
                etapa.status = "FALHOU"
                print(f"\n[ERRO] Falha em {etapa.rotulo} "
                      f"(após {fmt_tempo(etapa.duracao or 0)}). Pipeline interrompido.\n")
                print(traceback.format_exc())
                if etapa.tail:
                    print(f"--- últimas {TAIL_LINHAS} linhas da saída de {etapa.rotulo} ---")
                    print(etapa.tail.rstrip())
                break
        else:
            ok = True
    except KeyboardInterrupt:
        pass
    finally:
        total = time.perf_counter() - t0
        print(f"\n{'=' * 70}")
        for e in etapas:
            dur = fmt_tempo(e.duracao) if e.duracao is not None else "-"
            print(f"  {e.rotulo:<22} {e.status:<14} {dur}")
        print(f"{'=' * 70}\n{'Pipeline completo' if ok else 'Pipeline NÃO concluído'} "
              f"- tempo total: {fmt_tempo(total)}")
        escrever_log(etapas, inicio_dt, total, ok)

    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Orquestrador da pipeline firma-trimestre / LP.")
    ap.add_argument("a_partir_de", nargs="?", default=None,
                    help=f"opcional: etapa inicial ({', '.join(PIPELINE)})")
    sys.exit(main(ap.parse_args().a_partir_de))