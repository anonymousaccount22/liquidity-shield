#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolida o painel firma-trimestre para regressões de Projeções Locais.

Pipeline:
  1. Choques de política monetária (alta frequência) a partir do Swap DI-Pré 360d
  2. Índice de Incerteza (EPU) mensal -> trimestral (média)
  3. Dados contábeis da CVM (BPA_con, BPP_con, DRE_con) -> variáveis do modelo
     (inclui Ativo_Fixo = Imobilizado + Intangivel, via Balanço Patrimonial)
  3b. Grade trimestral contínua por empresa (linhas NaN nos trimestres ausentes)
  3c. Merge opcional do Tobin_Q (calcular_q.py) NA GRADE, antes dos lags
  4. Merge final por Trimestre e exportação para painel_final.csv

Dependência circular com calcular_q.py (e como ela é resolvida):
  - calcular_q.py precisa de Ativo_Total, Emprestimos_CP e Emprestimos_LP (vindos deste script);
  - este script precisa do Tobin_Q para criar Tobin_Q_L1.
  Como Ativo_Total/Empréstimos NÃO dependem do Q, basta uma iteração (ponto fixo):
    Passo 1: python script_painel.py   -> painel_final.csv (sem Tobin_Q; aviso informativo)
    Passo 2: python calcular_q.py      -> painel_final_q.csv (Tobin_Q winsorizado)
    Passo 3: python script_painel.py   -> lê painel_final_q.csv, cria Tobin_Q_L1 e regrava
                                          painel_final.csv (agora com Tobin_Q e Tobin_Q_L1)
  Rodadas seguintes são idempotentes (o conjunto de linhas não depende do Q, logo a
  winsorização cross-section é a mesma). calcular_q.py descarta as colunas de Q
  pré-existentes ao ler o painel (ver patch no final da resposta).

Convenções de timing (anti-vazamento):
  - `t` é o ordinal inteiro do trimestre (PeriodIndex.astype(int)); trimestres consecutivos
    têm t consecutivo.
  - Todo .shift() é feito por empresa, com o painel ORDENADO por [CNPJ_CIA, t] e com a
    grade trimestral COMPLETA (verificado por assert), logo shift(1) == trimestre t-1.
  - Variáveis sem sufixo que entram como controles (Cash_Ratio, Tamanho) JÁ são
    informação de t-1. Alavancagem é contemporânea (t); sua versão defasada é Alavancagem_L1.
  - Ativo_Fixo, log_Ativo_Fixo e Ativo_Total são contemporâneos (t); suas versões
    defasadas são Ativo_Fixo_L1, log_Ativo_Fixo_L1 e Ativo_Total_L1.
  - Tobin_Q é contemporâneo (t); a versão defasada é Tobin_Q_L1.

Dependências: pandas, requests
"""
from __future__ import annotations

import logging
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
BASE_DIR = Path(".")
COPOM_CSV = BASE_DIR / "datas_copom.csv"          # coluna 'Data' com reuniões do COPOM
EPU_CSV = BASE_DIR / "epu_brasil.csv"                    # EPU mensal
CVM_ROOT = BASE_DIR / "dados_cvm"                 # subpastas por ano (itr_cia_aberta_YYYY, ...)
CAD_CVM_CSV = BASE_DIR / "cad_cia_aberta.csv"     # OPCIONAL: cadastro CVM (traz SETOR_ATIV)
OUTPUT_CSV = BASE_DIR / "painel_final.csv"

# Saída do calcular_q.py (OPCIONAL): chaves (CNPJ_CIA, Trimestre='2019Q3') + Tobin_Q.
# Se existir, o Tobin_Q é incorporado à grade ANTES dos lags (gera Tobin_Q_L1).
TOBIN_Q_CSV = BASE_DIR / "painel_final_q.csv"
TOBIN_Q_COLS = ["Tobin_Q"]          # colunas de Q a trazer e defasar (contemporâneas, em t)

# Janela esperada do painel (grade trimestral contínua)
Q_INI = pd.Period("2011Q1", freq="Q")
Q_FIM = pd.Period("2026Q2", freq="Q")

# False (padrão): as linhas inseridas para preencher lacunas servem para garantir shifts
#   corretos e são removidas na exportação (dropna das variáveis críticas).
# True : mantém as linhas-lacuna (NaN nas variáveis contábeis) no CSV; a coluna `obs_real`
#   indica se o trimestre foi realmente observado. Só Choque_HFI e EPU são exigidos.
MANTER_LINHAS_GAP = False


# Fonte da série DIÁRIA de DI-Pré ~360d usada nos choques HFI:
#   "csv" -> lê RATES_CSV (coluna de data + coluna de taxa em % a.a.)   [padrão]
#   "sgs" -> baixa do BCB usando SGS_CODE (a série PRECISA ser diária)
# ATENÇÃO: os SGS 3802 e 7821 são MENSAIS (1 obs./mês) e não servem para choques T -> T+1.
HFI_SOURCE = "csv"
SGS_CODE: int | None = None
RATES_CSV = BASE_DIR / "swap_di_pre_360_diario.csv"
RATES_VALUE_COL: str | None = None   # nome da coluna de taxa (None = 1ª coluna numérica)
BCB_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json"

# Nome da coluna do índice no epu_brasil.csv (None = detecta a 1ª coluna numérica que não seja data)
EPU_VALUE_COL: str | None = None

# Padrões de arquivos CVM aceitos (busca case-insensitive no nome do arquivo)
CVM_FILE_TAGS = ("BPA_con", "BPP_con", "DRE_con")

# Contas contábeis -> nomes das colunas no formato largo
# (plano de contas padrão CVM: 1.02.01 = Realizável a LP, 1.02.02 = Investimentos,
#  1.02.03 = Imobilizado, 1.02.04 = Intangível)
CONTAS = {
    "1": "Ativo_Total",
    "1.01": "Ativo_Circulante",
    "1.01.01": "Caixa_Equiv",
    "1.01.02": "Aplic_Financeiras",
    "1.02.03": "Imobilizado",
    "1.02.04": "Intangivel",
    "2.01.04": "Emprestimos_CP",
    "2.02.01": "Emprestimos_LP",
    "3.01": "Receita_Liquida",
    "3.05": "Lucro_Operacional",
}
CONTAS_FLUXO = ["Receita_Liquida", "Lucro_Operacional"]   # DRE: vem acumulada no ano

# Fallback por descrição exata (sem acento, maiúscula) -> código canônico CVM.
# Só é usado para empresas cujo plano de contas não traz Imobilizado/Intangível nos códigos padrão.
CONTAS_DESC_FALLBACK = {"IMOBILIZADO": "1.02.03", "INTANGIVEL": "1.02.04"}

CVM_USECOLS = {
    "CNPJ_CIA", "DENOM_CIA", "DT_REFER", "DT_INI_EXERC", "CD_CONTA", "DS_CONTA",
    "VL_CONTA", "ESCALA_MOEDA", "ORDEM_EXERC", "VERSAO", "SETOR_ATIV",
}

# Utilities (setores). Confira a lista que o log vai imprimir e ajuste se precisar.
SETOR_UTIL_REGEX = re.compile(
    r"ENERGIA ELETRICA|SANEAMENTO|AGUA E ESGOTO|GAS CANALIZADO|DISTRIBUICAO DE GAS",
    re.IGNORECASE,
)
# Contas que, se ausentes, são tratadas como zero (empresa sem dívida / sem aplicações)
# (aplicado apenas em trimestres realmente observados, nunca em linhas-lacuna).
# Intangivel NÃO está aqui: seu zero só vale se o Imobilizado estiver presente
# (tratado em build_model_variables).
CONTAS_FILL_ZERO = ["Emprestimos_CP", "Emprestimos_LP", "Aplic_Financeiras"]


# Variáveis críticas: linhas com NaN em qualquer uma delas são removidas no final
VARS_CRITICAS = ["Cash_Ratio", "Alavancagem", "Choque_HFI", "EPU"]

# Heurísticas para identificar empresas financeiras
SETOR_FIN_REGEX = re.compile(
    r"INTERMEDIA|BANCO|SEGUR|FINANC|PREVIDEN|CAPITALIZ|CREDITO|SECURITIZ|CORRETORA",
    re.IGNORECASE,
)
# Grupo de NÃO-captura (?:...) -> elimina o UserWarning do pandas ("match groups")
NOME_FIN_REGEX = re.compile(
    r"\b(?:BANCO|BCO|SEGURADORA|SEGUROS|PREVIDENCIA|CAPITALIZACAO|FINANCEIRA|"
    r"SECURITIZADORA|CORRETORA|BRADESCO|ITAU|SANTANDER|BTG PACTUAL)\b"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("painel")


# =============================================================================
# UTILITÁRIOS
# =============================================================================
def read_csv_flex(path: Path, **kwargs) -> pd.DataFrame:
    """
    Lê CSV tentando encodings comuns e escolhendo o separador (; , tab) pela 1ª linha.
    (O sniffer automático do pandas erra em arquivos de coluna única.)
    """
    last_exc: Exception | None = None
    for enc in ("utf-8-sig", "latin1"):
        try:
            with open(path, encoding=enc) as fh:
                header = fh.readline()
            sep = max((";", ",", "\t"), key=header.count)
            if header.count(sep) == 0:
                sep = ","                       # coluna única: qualquer separador serve
            return pd.read_csv(path, sep=sep, encoding=enc, **kwargs)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_exc = exc
    raise ValueError(f"Não foi possível ler {path}: {last_exc}")


def to_period_q(dates: pd.Series | pd.DatetimeIndex) -> pd.PeriodIndex | pd.Series:
    """Converte datas para período trimestral (Q-DEC)."""
    if isinstance(dates, pd.Series):
        return dates.dt.to_period("Q")
    return dates.to_period("Q")


def strip_accents_upper(s: pd.Series) -> pd.Series:
    """Normaliza texto: sem acentos e em maiúsculas (para regex robusto)."""
    return (
        s.fillna("")
        .astype(str)
        .map(lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii"))
        .str.upper()
    )


# =============================================================================
# PARTE 1 — CHOQUES HFI (COPOM x SWAP DI-PRÉ 360)
# =============================================================================
def parse_dates_flex(s: pd.Series) -> pd.Series:
    """
    Converte texto em datas SEM usar dayfirst (evita o UserWarning do pandas).
    Tenta formatos explícitos em ordem: ISO (YYYY-MM-DD), BR (DD/MM/YYYY), YYYY-MM.
    """
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s.str.slice(0, 10), format="%Y-%m-%d", errors="coerce")
    out = out.fillna(pd.to_datetime(s, format="%d/%m/%Y", errors="coerce"))
    out = out.fillna(pd.to_datetime(s, format="%Y-%m", errors="coerce"))
    return out


def load_copom_dates(path: Path) -> pd.DatetimeIndex:
    """Lê as datas das reuniões do COPOM (coluna 'Data')."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    df = read_csv_flex(path)
    df.columns = [c.strip() for c in df.columns]
    if "Data" not in df.columns:
        raise KeyError(f"Coluna 'Data' ausente em {path}. Colunas: {list(df.columns)}")
    dates = parse_dates_flex(df["Data"]).dropna()
    if dates.empty:
        raise ValueError("Nenhuma data válida de COPOM foi encontrada.")
    return pd.DatetimeIndex(sorted(dates.unique()))


def fetch_sgs_series(
    code: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout: int = 60,
    retries: int = 3,
) -> pd.Series:
    """
    Baixa uma série do SGS em janelas de 9 anos (a API exige filtro de datas e limita
    consultas por período a 10 anos). Janela sem dados (HTTP 404/lista vazia) é tratada
    como vazia, não como falha. O recorte final [start, end] é refeito no Pandas.
    """
    url = BCB_URL.format(code=code)   # .../dados?formato=json
    headers = {"Accept": "application/json", "User-Agent": "painel-hfi/1.0"}

    frames: list[pd.DataFrame] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + pd.DateOffset(years=9), end)
        params = {"dataInicial": chunk_start.strftime("%d/%m/%Y"),
                  "dataFinal": chunk_end.strftime("%d/%m/%Y")}
        payload = None
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=timeout)
                if resp.status_code == 404:      # sem dados nesse período
                    payload = []
                    break
                resp.raise_for_status()
                payload = resp.json()
                if not isinstance(payload, list):
                    raise ValueError(f"Resposta inesperada da API: {payload}")
                break
            except (requests.RequestException, ValueError) as exc:
                log.warning("BCB: tentativa %d/%d falhou (%s a %s): %s", attempt, retries,
                            params["dataInicial"], params["dataFinal"], exc)
                if attempt < retries:
                    time.sleep(2 ** attempt)
        if payload is None:
            raise RuntimeError(f"Falha ao obter SGS {code} ({params}) após {retries} tentativas.")
        log.info("SGS %s: %s a %s -> %d obs.", code, params["dataInicial"],
                 params["dataFinal"], len(payload))
        if payload:
            frames.append(pd.DataFrame(payload))
        chunk_start = chunk_end + pd.Timedelta(days=1)

    if not frames:
        raise RuntimeError(f"SGS {code} não retornou nenhuma observação entre "
                           f"{start.date()} e {end.date()} (série descontinuada?).")

    raw = pd.concat(frames, ignore_index=True)
    raw["data"] = pd.to_datetime(raw["data"], format="%d/%m/%Y")
    raw["valor"] = pd.to_numeric(raw["valor"].astype(str).str.replace(",", ".", regex=False),
                                 errors="coerce")
    serie = (raw.dropna(subset=["valor"]).drop_duplicates("data")
                .set_index("data")["valor"].sort_index().loc[start:end])
    return serie


def load_rates_csv(path: Path) -> pd.Series:
    """
    Lê a série diária de DI-Pré 360d de um CSV local (ex.: B3/ANBIMA/Bloomberg).
    Precisa de uma coluna de data (Data/Date) e uma coluna numérica (taxa, % a.a.).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de juros diários não encontrado: {path}. Forneça um CSV com colunas "
            f"'Data' e 'Taxa' (DI-Pré 360d, diário) ou defina HFI_SOURCE='sgs' com um "
            f"SGS_CODE diário."
        )
    df = read_csv_flex(path)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    dcol = lower.get("data", lower.get("date"))
    if dcol is None:
        raise KeyError(f"Coluna 'Data'/'Date' ausente em {path}: {list(df.columns)}")
    if RATES_VALUE_COL is not None:
        if RATES_VALUE_COL not in df.columns:
            raise KeyError(f"RATES_VALUE_COL='{RATES_VALUE_COL}' não existe em {path}: "
                           f"{list(df.columns)}")
        vcol = RATES_VALUE_COL
    else:
        vcol = next((c for c in df.columns if c != dcol and pd.to_numeric(
            df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce").notna().mean() > 0.8),
            None)
    if vcol is None:
        raise KeyError(f"Nenhuma coluna numérica de taxa encontrada em {path}.")
    log.info("Juros diários: usando coluna '%s' de %s.", vcol, path.name)
    serie = pd.Series(
        pd.to_numeric(df[vcol].astype(str).str.replace(",", ".", regex=False), errors="coerce").values,
        index=parse_dates_flex(df[dcol]),
    )
    return serie[serie.index.notna()].dropna().groupby(level=0).last().sort_index()


def validate_daily_series(rates: pd.Series, copom_dates: pd.DatetimeIndex) -> None:
    """Falha cedo, com mensagem clara, se a série não for diária ou não cobrir o COPOM."""
    gap = rates.index.to_series().diff().dt.days.dropna().median()
    if gap > 4:
        raise ValueError(
            f"A série de juros NÃO é diária (mediana de {gap:.0f} dias entre observações). "
            f"Choques T -> T+1 exigem cotações em dias úteis consecutivos. Use outra série "
            f"SGS diária ou HFI_SOURCE='csv' com um arquivo diário."
        )
    cobertas = ((copom_dates >= rates.index.min()) & (copom_dates <= rates.index.max())).mean()
    if cobertas < 0.8:
        raise ValueError(
            f"A série ({rates.index.min().date()} a {rates.index.max().date()}) cobre só "
            f"{cobertas:.0%} das datas de COPOM ({copom_dates.min().date()} a "
            f"{copom_dates.max().date()})."
        )


def compute_hfi_shocks(copom_dates: pd.DatetimeIndex, rates: pd.Series) -> pd.Series:
    """
    Choque(T) = Taxa(T+1) - Taxa(T), onde:
      - Taxa(T)   = última cotação disponível em ou antes de T;
      - Taxa(T+1) = primeira cotação disponível estritamente depois de T (próx. dia útil).
    O COPOM divulga a decisão após o fechamento, logo a reação aparece em T+1.
    Resultado em pontos percentuais (mesma unidade da série).
    """
    idx = rates.index
    shocks: dict[pd.Timestamp, float] = {}
    for t in copom_dates:
        pos = idx.searchsorted(t, side="right")  # 1º índice com data > t
        if pos == 0 or pos >= len(idx):
            log.warning("COPOM %s fora da cobertura da série; ignorada.", t.date())
            continue
        d0, d1 = idx[pos - 1], idx[pos]
        if (t - d0).days > 4 or (d1 - t).days > 5:
            log.warning("COPOM %s: lacuna na série (T=%s, T+1=%s); ignorada.",
                        t.date(), d0.date(), d1.date())
            continue
        shocks[t] = rates.iloc[pos] - rates.iloc[pos - 1]

    out = pd.Series(shocks, name="Choque_HFI", dtype="float64").sort_index()
    if out.empty:
        raise RuntimeError("Nenhum choque pôde ser calculado.")
    return out


def build_hfi_quarterly(copom_csv: Path) -> pd.DataFrame:
    """Orquestra a Parte 1 e devolve DataFrame com colunas [Trimestre, Choque_HFI]."""
    copom = load_copom_dates(copom_csv)

    if HFI_SOURCE == "csv":
        rates = load_rates_csv(RATES_CSV)
    elif HFI_SOURCE == "sgs":
        if SGS_CODE is None:
            raise ValueError("HFI_SOURCE='sgs' exige SGS_CODE (série diária).")
        rates = fetch_sgs_series(SGS_CODE,
                                 start=copom.min() - pd.Timedelta(days=10),
                                 end=copom.max() + pd.Timedelta(days=10))
    else:
        raise ValueError(f"HFI_SOURCE inválido: {HFI_SOURCE!r} (use 'csv' ou 'sgs').")

    log.info("Juros: %d obs. (%s a %s)", len(rates), rates.index.min().date(),
             rates.index.max().date())
    validate_daily_series(rates, copom)

    shocks = compute_hfi_shocks(copom, rates)

    # Equivalente a resample('Q').sum(), porém imune à troca 'Q' -> 'QE' do pandas >= 2.2
    trimestral = shocks.groupby(to_period_q(shocks.index)).sum()
    trimestral.index.name = "Trimestre"
    return trimestral.rename("Choque_HFI").reset_index()


# =============================================================================
# PARTE 2 — EPU
# =============================================================================
def build_epu_quarterly(epu_csv: Path, value_col: str | None = EPU_VALUE_COL) -> pd.DataFrame:
    """Lê o EPU mensal e devolve a média trimestral: colunas [Trimestre, EPU]."""
    if not epu_csv.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {epu_csv}")
    df = read_csv_flex(epu_csv)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    # --- identifica a data mensal ---
    date_cols: list[str] = []
    if "data" in lower or "date" in lower:
        col = lower.get("data", lower.get("date"))
        dates = parse_dates_flex(df[col])
        date_cols = [col]
    else:
        ycol = next((lower[k] for k in ("year", "ano") if k in lower), None)
        mcol = next((lower[k] for k in ("month", "mes", "mês") if k in lower), None)
        if ycol is None or mcol is None:
            raise KeyError(f"Não achei coluna de data (Data/Date ou Year+Month) em {epu_csv}: "
                           f"{list(df.columns)}")
        dates = pd.to_datetime(
            {"year": pd.to_numeric(df[ycol], errors="coerce"),
             "month": pd.to_numeric(df[mcol], errors="coerce"), "day": 1},
            errors="coerce",
        )
        date_cols = [ycol, mcol]

    # --- identifica a coluna do índice ---
    if value_col is None:
        candidates = [c for c in df.columns if c not in date_cols]
        value_col = next(
            (c for c in candidates
             if pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False),
                              errors="coerce").notna().mean() > 0.8),
            None,
        )
        if value_col is None:
            raise KeyError("Não foi possível detectar a coluna do EPU; defina EPU_VALUE_COL.")
    log.info("EPU: usando a coluna '%s'.", value_col)

    values = pd.to_numeric(df[value_col].astype(str).str.replace(",", ".", regex=False),
                           errors="coerce")
    serie = pd.Series(values.values, index=dates).dropna()
    serie = serie[serie.index.notna()].sort_index()
    if serie.empty:
        raise ValueError("Série EPU vazia após limpeza.")

    # Equivalente a resample('Q').mean()
    trimestral = serie.groupby(to_period_q(serie.index)).mean()
    trimestral.index.name = "Trimestre"
    return trimestral.rename("EPU").reset_index()


# =============================================================================
# PARTE 3 — CVM
# =============================================================================
def find_cvm_files(root: Path, tags: tuple[str, ...] = CVM_FILE_TAGS) -> list[Path]:
    """Busca recursiva por CSVs consolidados (BPA_con, BPP_con, DRE_con)."""
    if not root.exists():
        raise FileNotFoundError(f"Pasta raiz da CVM não encontrada: {root}")
    tags_l = tuple(t.lower() for t in tags)
    files = sorted(
        p for p in root.rglob("*.csv")
        if any(t in p.name.lower() for t in tags_l)
    )
    log.info("CVM: %d arquivos consolidados encontrados em %s", len(files), root)
    return files


def read_cvm_file(path: Path) -> pd.DataFrame | None:
    """
    Lê um CSV da CVM (latin1, ';'), mantém só as contas de interesse e a última
    versão do exercício. Retorna None em caso de erro (o loop principal segue).
    """
    try:
        df = pd.read_csv(
            path, sep=";", encoding="latin1", dtype=str,
            usecols=lambda c: c in CVM_USECOLS,
        )
    except Exception as exc:  # arquivo corrompido, encoding, etc.
        log.error("Falha ao ler %s: %s", path.name, exc)
        return None

    obrig = {"CNPJ_CIA", "DT_REFER", "CD_CONTA", "VL_CONTA"}
    if not obrig.issubset(df.columns):
        log.warning("%s sem colunas obrigatórias %s; ignorado.", path.name, obrig - set(df.columns))
        return None

    df["CD_CONTA"] = df["CD_CONTA"].str.strip()

    # Fallback por descrição para Imobilizado/Intangível: se a conta não vier no código
    # padrão, mas existir um grupo 1.02.xx com descrição exata, é remapeada para o código
    # canônico. `_canon` = 1 (lida pelo código) tem prioridade sobre 0 (resgatada por descrição).
    df["_canon"] = 1
    if "DS_CONTA" in df.columns:
        desc = strip_accents_upper(df["DS_CONTA"]).str.strip()
        alvo = desc.map(CONTAS_DESC_FALLBACK)
        fb = (alvo.notna()
              & ~df["CD_CONTA"].isin(CONTAS)
              & df["CD_CONTA"].str.startswith("1.02")
              & (df["CD_CONTA"].str.count(r"\.") <= 2))      # só nível de grupo, não subcontas
        df.loc[fb, "CD_CONTA"] = alvo[fb]
        df.loc[fb, "_canon"] = 0

    df = df[df["CD_CONTA"].isin(CONTAS)].copy()
    if df.empty:
        return None  # ex.: DRE_con não contém as contas de balanço pedidas

    # Mantém só o exercício corrente ('ÚLTIMO'), descartando a coluna comparativa ('PENÚLTIMO')
    if "ORDEM_EXERC" in df.columns:
        ordem = strip_accents_upper(df["ORDEM_EXERC"])
        df = df[ordem.str.contains("ULTIMO", regex=False, na=False)]

    # DRE: mantém só as linhas acumuladas no ano (início em 1º de janeiro).
    # O trimestre isolado é recuperado depois por diferença (YTD_t - YTD_t-1).
    if "DT_INI_EXERC" in df.columns:
        ini = pd.to_datetime(df["DT_INI_EXERC"], errors="coerce")
        eh_fluxo = df["CD_CONTA"].isin([c for c, n in CONTAS.items() if n in CONTAS_FLUXO])
        ytd = (ini.dt.month == 1) & (ini.dt.day == 1)
        df = df[~eh_fluxo | ytd]

    # Valor numérico e escala (MIL / UNIDADE)
    df["VL_CONTA"] = pd.to_numeric(df["VL_CONTA"].str.replace(",", ".", regex=False),
                                   errors="coerce")
    if "ESCALA_MOEDA" in df.columns:
        escala = strip_accents_upper(df["ESCALA_MOEDA"]).map({"MIL": 1_000.0}).fillna(1.0)
        df["VL_CONTA"] = df["VL_CONTA"] * escala

    df["VERSAO"] = pd.to_numeric(df.get("VERSAO", 0), errors="coerce").fillna(0)
    df["DT_REFER"] = pd.to_datetime(df["DT_REFER"], errors="coerce")
    df = df.dropna(subset=["VL_CONTA", "DT_REFER"])
    df["Trimestre"] = to_period_q(df["DT_REFER"])
    return df


def flag_financials(long: pd.DataFrame) -> set[str]:
    """
    Identifica CNPJs financeiros, em ordem de preferência:
      1) coluna SETOR_ATIV, se vier nos arquivos;
      2) cadastro opcional cad_cia_aberta.csv (SETOR_ATIV);
      3) heurística por razão social (DENOM_CIA).
    O prefixo do CNPJ NÃO é usado: não distingue setor de forma confiável.
    """
    cnpjs_fin: set[str] = set()

    setor = None
    if "SETOR_ATIV" in long.columns and long["SETOR_ATIV"].notna().any():
        setor = long[["CNPJ_CIA", "SETOR_ATIV"]].drop_duplicates("CNPJ_CIA")
    elif CAD_CVM_CSV.exists():
        try:
            cad = pd.read_csv(CAD_CVM_CSV, sep=";", encoding="latin1", dtype=str,
                              usecols=lambda c: c in {"CNPJ_CIA", "SETOR_ATIV"})
            setor = cad.drop_duplicates("CNPJ_CIA")
            log.info("Setor lido do cadastro %s.", CAD_CVM_CSV.name)
        except Exception as exc:
            log.warning("Falha ao ler cadastro %s: %s", CAD_CVM_CSV.name, exc)

    if setor is not None:
        txt = strip_accents_upper(setor["SETOR_ATIV"])
        cnpjs_fin |= set(setor.loc[txt.str.contains(SETOR_FIN_REGEX), "CNPJ_CIA"])

    if "DENOM_CIA" in long.columns:
        nomes = long[["CNPJ_CIA", "DENOM_CIA"]].drop_duplicates("CNPJ_CIA")
        txt = strip_accents_upper(nomes["DENOM_CIA"])
        # NOME_FIN_REGEX usa grupo de não-captura -> sem UserWarning do pandas
        cnpjs_fin |= set(nomes.loc[txt.str.contains(NOME_FIN_REGEX), "CNPJ_CIA"])

    if setor is None:
        log.warning("SETOR_ATIV indisponível: filtro financeiro baseado apenas no nome da "
                    "empresa (aproximado). Considere fornecer cad_cia_aberta.csv.")
    return cnpjs_fin


def flag_estatais_utilities() -> set[str]:
    """CNPJs de estatais (CONTROLE_ACIONARIO) e utilities (SETOR_ATIV) pelo cadastro da CVM."""
    if not CAD_CVM_CSV.exists():
        log.warning("Cadastro %s não encontrado: estatais/utilities NÃO foram filtradas.",
                    CAD_CVM_CSV.name)
        return set()
    try:
        cad = pd.read_csv(CAD_CVM_CSV, sep=";", encoding="latin1", dtype=str,
                          usecols=lambda c: c in {"CNPJ_CIA", "SETOR_ATIV", "CONTROLE_ACIONARIO"})
    except Exception as exc:
        log.warning("Falha ao ler cadastro %s: %s", CAD_CVM_CSV.name, exc)
        return set()

    cad = cad.drop_duplicates("CNPJ_CIA")
    vazio = pd.Series("", index=cad.index)
    ctrl = strip_accents_upper(cad["CONTROLE_ACIONARIO"]) if "CONTROLE_ACIONARIO" in cad else vazio
    setor = strip_accents_upper(cad["SETOR_ATIV"]) if "SETOR_ATIV" in cad else vazio

    m_est = ctrl.str.contains("ESTATAL", regex=False)
    m_util = setor.str.contains(SETOR_UTIL_REGEX)
    log.info("Cadastro: %d estatais, %d utilities.", m_est.sum(), m_util.sum())
    if "SETOR_ATIV" in cad:
        log.info("Setores tratados como utilities: %s",
                 sorted(cad.loc[m_util, "SETOR_ATIV"].dropna().unique()))
    return set(cad.loc[m_est | m_util, "CNPJ_CIA"])


def ytd_to_quarterly(wide: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Converte contas de fluxo acumuladas no ano (YTD) em valores trimestrais:
    Q1 = YTD; Q2..Q4 = YTD(t) - YTD(t-1) no mesmo ano. Se o trimestre anterior
    faltar, o resultado é NaN (em vez de um valor errado).
    """
    w = wide.reset_index()
    w["Ano"] = w["Trimestre"].dt.year
    w["Q"] = w["Trimestre"].dt.quarter
    for c in cols:
        prev = w[["CNPJ_CIA", "Ano", "Q", c]].copy()
        prev["Q"] += 1
        prev = prev.rename(columns={c: f"{c}_prev"})
        w = w.merge(prev, on=["CNPJ_CIA", "Ano", "Q"], how="left")
        w[c] = w[c].where(w["Q"] == 1, w[c] - w[f"{c}_prev"])
        w = w.drop(columns=f"{c}_prev")
    return w.drop(columns=["Ano", "Q"]).set_index(["CNPJ_CIA", "Trimestre"])


def build_cvm_wide(root: Path) -> pd.DataFrame:
    """Empilha os CSVs, filtra contas/setor e pivoteia: índice (CNPJ_CIA, Trimestre)."""
    frames = [df for f in find_cvm_files(root) if (df := read_cvm_file(f)) is not None]
    if not frames:
        raise RuntimeError("Nenhum dado utilizável foi lido dos arquivos da CVM.")
    long = pd.concat(frames, ignore_index=True)
    log.info("CVM empilhada: %d linhas, %d empresas.", len(long), long["CNPJ_CIA"].nunique())

    # --- remove financeiras, estatais e utilities (filtros institucionais) ---
    fin = flag_financials(long)
    exc = flag_estatais_utilities()
    excluir = fin | exc
    long = long[~long["CNPJ_CIA"].isin(excluir)]
    log.info("Removidas: %d financeiras + %d estatais/utilities (união: %d). Restam %d empresas.",
             len(fin), len(exc), len(excluir), long["CNPJ_CIA"].nunique())

    # --- duplicatas: fica a versão mais recente; conta lida pelo código (_canon=1) vence
    #     a resgatada por descrição (_canon=0) ---
    keys = ["CNPJ_CIA", "Trimestre", "CD_CONTA"]
    long = (long.sort_values(keys + ["_canon", "VERSAO", "DT_REFER"])
                .drop_duplicates(keys, keep="last"))

    # --- pivot: uma coluna por conta ---
    wide = (long.set_index(keys)["VL_CONTA"].unstack("CD_CONTA")
                .reindex(columns=list(CONTAS))
                .rename(columns=CONTAS))

    setor = None
    if "SETOR_ATIV" in long.columns:
        setor = (long.groupby("CNPJ_CIA")["SETOR_ATIV"]
                 .agg(lambda s: s.dropna().iloc[-1] if s.notna().any() else pd.NA)
                 .rename("SETOR_ATIV").reset_index())
    if CAD_CVM_CSV.exists():
        cadastro = pd.read_csv(CAD_CVM_CSV, sep=";", encoding="latin1", dtype=str,
                               usecols=lambda c: c in {"CNPJ_CIA", "SETOR_ATIV"})
        cadastro = cadastro.dropna(subset=["SETOR_ATIV"]).drop_duplicates("CNPJ_CIA", keep="last")
        if setor is None:
            setor = cadastro
        else:
            setor = setor.merge(cadastro, on="CNPJ_CIA", how="outer", suffixes=("", "_cad"))
            setor["SETOR_ATIV"] = setor["SETOR_ATIV"].fillna(setor["SETOR_ATIV_cad"])
            setor = setor[["CNPJ_CIA", "SETOR_ATIV"]]
    if setor is not None:
        wide = (wide.reset_index().merge(setor, on="CNPJ_CIA", how="left")
                .set_index(["CNPJ_CIA", "Trimestre"]))

    # --- DRE: acumulado no ano -> trimestre isolado ---
    return ytd_to_quarterly(wide, CONTAS_FLUXO)


# =============================================================================
# PARTE 3b — GRADE TRIMESTRAL CONTÍNUA
# =============================================================================
def complete_calendar(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Garante equidistância temporal: para cada empresa, cria uma linha por trimestre entre
    a 1ª e a última observação (dentro do intervalo da própria empresa). Trimestres
    ausentes entram com NaN nas contas contábeis e `obs_real = False`.
    Retorna DataFrame plano, ordenado por [CNPJ_CIA, t], com as colunas `t` (ordinal
    inteiro) e `Trimestre` (Period[Q-DEC]).
    """
    w = wide.reset_index()
    w["t"] = w["Trimestre"].map(lambda q: q.ordinal).astype("int64")
    w["obs_real"] = True
    w = w.sort_values(["CNPJ_CIA", "t"])

    rng = w.groupby("CNPJ_CIA")["t"].agg(["min", "max"])
    grid = pd.DataFrame(
        [(c, t) for c, (a, b) in zip(rng.index, rng.to_numpy()) for t in range(int(a), int(b) + 1)],
        columns=["CNPJ_CIA", "t"],
    )
    out = grid.merge(w.drop(columns="Trimestre"), on=["CNPJ_CIA", "t"], how="left")
    out["obs_real"] = out["obs_real"].fillna(False).astype(bool)
    out["Trimestre"] = out["t"].map(lambda o: pd.Period(ordinal=int(o), freq="Q"))

    # SETOR_ATIV é atributo da firma: propaga para as linhas inseridas
    if "SETOR_ATIV" in out.columns:
        out["SETOR_ATIV"] = out.groupby("CNPJ_CIA")["SETOR_ATIV"].transform(
            lambda s: s.ffill().bfill())

    out = out.sort_values(["CNPJ_CIA", "t"]).reset_index(drop=True)

    n_ins = int((~out["obs_real"]).sum())
    firmas_gap = out.loc[~out["obs_real"], "CNPJ_CIA"].nunique()
    log.info("Grade trimestral: %d linhas observadas + %d linhas-lacuna inseridas "
             "(%d empresas com buracos internos).", int(out["obs_real"].sum()), n_ins, firmas_gap)
    return out


def verificar_calendario_global(df: pd.DataFrame) -> None:
    """Confere se cada trimestre de Q_INI a Q_FIM tem ao menos uma empresa observada."""
    esperado = range(Q_INI.ordinal, Q_FIM.ordinal + 1)
    presentes = set(df.loc[df["obs_real"], "t"].unique())
    faltam = [str(pd.Period(ordinal=o, freq="Q")) for o in esperado if o not in presentes]
    fora = sorted(str(pd.Period(ordinal=int(o), freq="Q")) for o in presentes
                  if o < Q_INI.ordinal or o > Q_FIM.ordinal)
    if faltam:
        log.warning("Trimestres da janela %s-%s SEM nenhuma empresa observada: %s",
                    Q_INI, Q_FIM, faltam)
    else:
        log.info("Calendário global OK: todos os trimestres de %s a %s têm observações.",
                 Q_INI, Q_FIM)
    if fora:
        log.info("Trimestres fora da janela (usados só como defasagem/serão cortados): %s",
                 fora[:6] + (["..."] if len(fora) > 6 else []))


# =============================================================================
# PARTE 3c — TOBIN'S Q (saída do calcular_q.py) INCORPORADO À GRADE
# =============================================================================
def merge_tobin_q(grade: pd.DataFrame, path: Path = TOBIN_Q_CSV,
                  cols: list[str] = TOBIN_Q_COLS) -> pd.DataFrame:
    """
    Traz as colunas de Q (contemporâneas, em t) do painel_final_q.csv para a grade, por
    (CNPJ_CIA, Trimestre). Deve rodar DEPOIS de complete_calendar() e ANTES de
    build_model_variables(), para que o Tobin_Q_L1 seja criado com lag_seguro().

    - Arquivo ausente (1ª passagem): segue sem Q e informa como gerar (ver docstring do módulo).
    - Linhas-lacuna ficam com Q = NaN (não há Q para trimestres não observados).
    - A ordem [CNPJ_CIA, t] é restaurada após o merge (pré-requisito do shift).
    - Colunas de Q que já estejam na grade são descartadas antes (evita sufixos _x/_y).
    """
    if not path.exists():
        log.info("Tobin_Q: %s não encontrado (1ª passagem?). Seguindo sem Tobin_Q_L1. "
                 "Rode calcular_q.py e depois este script novamente.", path.name)
        return grade

    try:
        q = read_csv_flex(path, dtype={"CNPJ_CIA": str},
                          usecols=lambda c: c in {"CNPJ_CIA", "Trimestre", *cols})
    except Exception as exc:
        log.warning("Falha ao ler %s (%s); seguindo sem Tobin_Q.", path.name, exc)
        return grade

    faltam = [c for c in ("CNPJ_CIA", "Trimestre", *cols) if c not in q.columns]
    if faltam:
        log.warning("%s sem as colunas %s; seguindo sem Tobin_Q.", path.name, faltam)
        return grade

    q["Trimestre"] = q["Trimestre"].astype(str).str.strip()
    q = q.drop_duplicates(["CNPJ_CIA", "Trimestre"], keep="last")

    out = grade.drop(columns=[c for c in cols if c in grade.columns])
    chave = out["Trimestre"].astype(str)                       # Period -> '2019Q3'
    out = (out.assign(_tri=chave)
              .merge(q.rename(columns={"Trimestre": "_tri"}), on=["CNPJ_CIA", "_tri"], how="left")
              .drop(columns="_tri")
              .sort_values(["CNPJ_CIA", "t"]).reset_index(drop=True))

    for c in cols:
        log.info("Tobin_Q: coluna '%s' incorporada (%.0f%% das linhas observadas com valor).",
                 c, 100 * out.loc[out["obs_real"], c].notna().mean())
    return out


# =============================================================================
# LAGS SEGUROS E VARIÁVEIS DO MODELO
# =============================================================================
def assert_grade_continua(df: pd.DataFrame) -> None:
    """
    Pré-condição de qualquer .shift(): painel ordenado por [CNPJ_CIA, t], sem duplicatas
    e com trimestres estritamente consecutivos dentro de cada empresa.
    """
    ordenado = df.sort_values(["CNPJ_CIA", "t"])
    if not df.index.equals(ordenado.index):
        raise AssertionError("Painel não está ordenado por [CNPJ_CIA, t] antes do shift.")
    if df.duplicated(["CNPJ_CIA", "t"]).any():
        raise AssertionError("Há pares (CNPJ_CIA, t) duplicados.")
    passos = df.groupby("CNPJ_CIA")["t"].diff().dropna()
    if not (passos == 1).all():
        raise AssertionError(f"Grade trimestral com lacunas: {int((passos != 1).sum())} saltos != 1.")


def lag_seguro(df: pd.DataFrame, cols: list[str], k: int = 1) -> pd.DataFrame:
    """Cria {col}_L{k} = valor da MESMA empresa k trimestres antes (exige grade contínua)."""
    assert_grade_continua(df)
    g = df.groupby("CNPJ_CIA")
    for c in cols:
        df[f"{c}_L{k}"] = g[c].shift(k)
    return df


def verificar_lags(df: pd.DataFrame, cols: list[str], k: int = 1) -> None:
    """
    Teste independente anti-vazamento: refaz a defasagem por MERGE em t+k e compara
    com o resultado do shift. Qualquer divergência interrompe o pipeline.
    """
    for c in cols:
        base = df[["CNPJ_CIA", "t", c]].copy()
        base["t"] += k
        via_merge = df[["CNPJ_CIA", "t"]].merge(base, on=["CNPJ_CIA", "t"], how="left")[c]
        if not np.array_equal(via_merge.to_numpy(float), df[f"{c}_L{k}"].to_numpy(float),
                              equal_nan=True):
            raise AssertionError(f"Defasagem inconsistente em {c}_L{k}: possível vazamento.")
    log.info("Verificação de lags OK (shift == merge em t-%d) para: %s", k, cols)


def log_participacao_aplicacoes(df: pd.DataFrame) -> None:
    """
    Documenta a relevância da tesouraria de alto rendimento: participação das
    Aplicações Financeiras (1.01.02) no Caixa Total (Caixa e Equivalentes + Aplicações).
    Considera só trimestres realmente observados e Caixa Total > 0.
    """
    d = df[df["obs_real"] & (df["Caixa"] > 0)]
    if d.empty:
        log.warning("Sem observações para calcular a participação das aplicações financeiras.")
        return
    share = d["Aplic_Financeiras"] / d["Caixa"]
    log.info("Aplicações Financeiras / Caixa Total: média = %.1f%% | mediana = %.1f%% | "
             "agregado (Σ aplic / Σ caixa) = %.1f%% | %.1f%% das obs. têm aplicações > 0 "
             "(N = %d firma-trimestres; ausência da conta é tratada como zero).",
             100 * share.mean(), 100 * share.median(),
             100 * d["Aplic_Financeiras"].sum() / d["Caixa"].sum(),
             100 * (d["Aplic_Financeiras"] > 0).mean(), len(d))


def log_validacao_ativo_fixo(df: pd.DataFrame) -> None:
    """Validação estatística simples de Ativo_Fixo / Ativo_Total (trimestres observados)."""
    d = df[df["obs_real"]] if "obs_real" in df.columns else df
    r = (d["Ativo_Fixo"] / d["Ativo_Total"]).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        log.warning("Ativo_Fixo / Ativo_Total: sem observações válidas.")
        return
    print("\n=== Validação: Ativo_Fixo / Ativo_Total ===")
    print(f"N             : {len(r):,}")
    print(f"Média         : {r.mean():.4f}")
    print(f"Mediana       : {r.median():.4f}")
    print(f"Desvio-padrão : {r.std():.4f}")
    print(f"Fora de [0,1] : {int((~r.between(0, 1)).sum())} obs.\n")


def build_model_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula Ativo_Fixo, log_Ativo_Fixo, Delta_Capex, Delta_log_Ativo_Fixo, Cash_Ratio,
    Tamanho e Alavancagem(_L1).

    Definições (todas usam apenas informação até t-1, exceto Alavancagem, Ativo_Fixo e
    log_Ativo_Fixo em t):
      Caixa_t       = Caixa e Equivalentes (1.01.01) + Aplicações Financeiras de CP (1.01.02)
      Ativo_Fixo    = Imobilizado (1.02.03) + Intangível (1.02.04)  (contemporâneo, em t)
                      Intangível ausente = 0 SOMENTE se o Imobilizado estiver presente;
                      sem Imobilizado, Ativo_Fixo = NaN.
                      Ativo_Fixo <= 0 é convertido em NaN (log indefinido); isso vale também
                      para Delta_Capex e para Ativo_Fixo_L1.
      log_Ativo_Fixo= ln(Ativo_Fixo_t)                          (contemporâneo, em t)
      Cash_Ratio    = Caixa_{t-1} / Ativo_Total_{t-1}
                      (liquidez corporativa já defasada: numerador E denominador em t-1,
                       logo não captura a reação contemporânea do caixa ao choque em t)
      Tamanho       = ln(Ativo_Total_{t-1})
      Alavancagem   = (Empréstimos CP + LP)_t / Ativo_Total_t   (contemporânea)
      Alavancagem_L1= Alavancagem_{t-1}                         (a que vai ao modelo)
      Delta_Capex   = (Ativo_Fixo_t - Ativo_Fixo_{t-1}) / Ativo_Total_{t-1}
      Delta_log_Ativo_Fixo = log_Ativo_Fixo_t - log_Ativo_Fixo_{t-1}

    Defasagens limpas (t-1) criadas: Ativo_Fixo_L1, log_Ativo_Fixo_L1, Ativo_Total_L1,
    Alavancagem_L1 e, para cada coluna de TOBIN_Q_COLS presente na grade (ver
    merge_tobin_q), Tobin_Q_L1. (Cash_Ratio já É a informação de t-1.)

    Pré-requisito: `df` vem de complete_calendar() (e, opcionalmente, merge_tobin_q()),
    ordenado e com grade contínua, de modo que groupby().shift(1) é exatamente o
    trimestre t-1 da mesma empresa.
    """
    df = df.sort_values(["CNPJ_CIA", "t"]).reset_index(drop=True)
    assert_grade_continua(df)

    # Ausência de conta = zero, SOMENTE em trimestres observados (linhas-lacuna ficam NaN)
    for c in CONTAS_FILL_ZERO:
        df[c] = df[c].where(df[c].notna() | ~df["obs_real"], 0.0)

    # Intangível ausente = 0 apenas se o Imobilizado estiver presente (trimestre observado)
    df["Intangivel"] = df["Intangivel"].where(
        df["Intangivel"].notna() | ~(df["obs_real"] & df["Imobilizado"].notna()), 0.0)

    df["Caixa"] = df["Caixa_Equiv"] + df["Aplic_Financeiras"]
    df["Ativo_Total"] = df["Ativo_Total"].where(df["Ativo_Total"] > 0)
    df["Ativo_Fixo"] = df["Imobilizado"] + df["Intangivel"]   # NaN se Imobilizado ausente

    # Ativo_Fixo <= 0 -> NaN (log indefinido); NaN permanece NaN (comparação com NaN é False)
    df["Ativo_Fixo"] = df["Ativo_Fixo"].where(~(df["Ativo_Fixo"] <= 0))
    df["log_Ativo_Fixo"] = np.log(df["Ativo_Fixo"])

    log_participacao_aplicacoes(df)

    # Versões contemporâneas (auxiliares; NÃO são exportadas)
    df["Cash_Ratio_t"] = df["Caixa"] / df["Ativo_Total"]
    df["Tamanho_t"] = np.log(df["Ativo_Total"])
    df["Alavancagem"] = (df["Emprestimos_CP"] + df["Emprestimos_LP"]) / df["Ativo_Total"]

    cols_lag = ["Ativo_Fixo", "log_Ativo_Fixo", "Ativo_Total", "Cash_Ratio_t", "Tamanho_t",
                "Alavancagem"]
    q_presentes = [c for c in TOBIN_Q_COLS if c in df.columns]
    cols_lag += q_presentes            # Tobin_Q vem de merge_tobin_q() (calcular_q.py)
    if q_presentes:
        log.info("Tobin_Q presente na grade: criando %s.", [f"{c}_L1" for c in q_presentes])
    df = lag_seguro(df, cols_lag, k=1)
    verificar_lags(df, cols_lag, k=1)

    df["Cash_Ratio"] = df["Cash_Ratio_t_L1"]          # (Caixa+Aplic)_{t-1} / Ativo_{t-1}
    df["Tamanho"] = df["Tamanho_t_L1"]                # ln(Ativo_{t-1})
    df["Delta_Capex"] = (df["Ativo_Fixo"] - df["Ativo_Fixo_L1"]) / df["Ativo_Total_L1"]
    df["Delta_log_Ativo_Fixo"] = df["log_Ativo_Fixo"] - df["log_Ativo_Fixo_L1"]
    df["Margem_Operacional"] = df["Lucro_Operacional"] / df["Receita_Liquida"].where(df["Receita_Liquida"] > 0)
    df["Receita_Ativo"] = df["Receita_Liquida"] / df["Ativo_Total_L1"]

    df = df.drop(columns=["Cash_Ratio_t", "Cash_Ratio_t_L1", "Tamanho_t", "Tamanho_t_L1"])

    num = [c for c in df.select_dtypes("number").columns if c != "t"]
    df[num] = df[num].replace([float("inf"), float("-inf")], pd.NA).astype("float64")
    return df


# =============================================================================
# PARTE 4 — MERGE FINAL
# =============================================================================
def merge_panel(cvm: pd.DataFrame, hfi: pd.DataFrame, epu: pd.DataFrame) -> pd.DataFrame:
    """Merge por Trimestre (Period[Q-DEC]), corte na janela Q_INI-Q_FIM e limpeza de nulos."""
    painel = (cvm.merge(hfi, on="Trimestre", how="left")
                 .merge(epu, on="Trimestre", how="left"))

    # Corte na janela só DEPOIS de calcular as defasagens (2011Q1 usa 2010Q4 como t-1)
    n_pre = len(painel)
    painel = painel[(painel["t"] >= Q_INI.ordinal) & (painel["t"] <= Q_FIM.ordinal)]
    log.info("Janela %s a %s: %d -> %d linhas.", Q_INI, Q_FIM, n_pre, len(painel))

    crit = VARS_CRITICAS if not MANTER_LINHAS_GAP else ["Choque_HFI", "EPU"]
    n0 = len(painel)
    painel = painel.dropna(subset=crit)
    log.info("Painel: %d -> %d linhas após remover nulos críticos (%s).", n0, len(painel), crit)
    return painel.sort_values(["CNPJ_CIA", "t"]).reset_index(drop=True)


# =============================================================================
# MAIN
# =============================================================================
def main() -> int:
    try:
        hfi = build_hfi_quarterly(COPOM_CSV)
        epu = build_epu_quarterly(EPU_CSV)
        wide = build_cvm_wide(CVM_ROOT)
        grade = complete_calendar(wide)
        verificar_calendario_global(grade)
        grade = merge_tobin_q(grade)             # Q entra na grade ANTES dos lags
        cvm = build_model_variables(grade)
        painel = merge_panel(cvm, hfi, epu)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, AssertionError) as exc:
        log.error("Pipeline interrompido: %s", exc)
        return 1

    cols = (["CNPJ_CIA", "Trimestre", "t"] + VARS_CRITICAS +
            [c for c in painel.columns
             if c not in {"CNPJ_CIA", "Trimestre", "t", *VARS_CRITICAS}])
    painel = painel[cols].copy()
    log_validacao_ativo_fixo(painel)
    painel["Trimestre"] = painel["Trimestre"].astype(str)   # ex.: '2019Q3'
    painel.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    log.info("Painel exportado: %s (%d obs., %d empresas, %s a %s)",
             OUTPUT_CSV, len(painel), painel["CNPJ_CIA"].nunique(),
             painel["Trimestre"].min(), painel["Trimestre"].max())
    return 0


if __name__ == "__main__":
    sys.exit(main())