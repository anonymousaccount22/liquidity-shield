#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capex_bp.py — Estoque de capital via Balanço Patrimonial (Opção A) + CFO da DFC.

Pipeline:
  1. Lê painel_final.csv (gerado por script_painel.py), ordena por [CNPJ_CIA, t],
     valida Ativo_Fixo (<= 0 ou nulo -> NaN) e calcula log_Ativo_Fixo.
  2. Diagnóstico: Delta_Ativo_Fixo_Ativo = (AF_t - AF_{t-1}) / Ativo_Total_{t-1},
     com estatísticas e cobertura por ano -> cobertura_ativo_fixo_ano.csv
  3. CFO (DFC, conta 6.01): lê DFC_MD_con / DFC_MI_con em dados_cvm, converte o acumulado
     no ano (YTD) em trimestral e calcula Fluxo_Caixa_Op_Ativo = CFO_t / Ativo_Total_{t-1}.
  4. Merge de log_Ativo_Fixo e Fluxo_Caixa_Op_Ativo com painel_regressao_q_ortog.csv
     -> painel_regressao_q_bp.csv

Anti-vazamento / defasagem:
  O painel_final.csv pode ter LACUNAS (script_painel.py remove linhas com NaN críticos),
  então .shift(1) NÃO garante t-1. A defasagem do painel é feita por MERGE em t+1, que
  devolve NaN se o t-1 não existir. O denominador do CFO usa Ativo_Total_L1 do
  script_painel.py (calculado na grade contínua) quando disponível.

Dependências: pandas, numpy
"""
from __future__ import annotations

import logging
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
BASE_DIR = Path(".")
PAINEL_CSV = BASE_DIR / "painel_final.csv"
ORTOG_CSV = BASE_DIR / "painel_regressao_q_ortog.csv"
COBERTURA_CSV = BASE_DIR / "cobertura_ativo_fixo_ano.csv"
OUTPUT_CSV = BASE_DIR / "painel_regressao_q_bp.csv"
CVM_ROOT = BASE_DIR / "dados_cvm"

COLS_OBRIGATORIAS = ["CNPJ_CIA", "t", "Ativo_Fixo", "Ativo_Total"]

# DFC: método direto (MD) e indireto (MI), consolidadas
DFC_TAGS = ("dfc_md_con", "dfc_mi_con")
# 6.01 = "Caixa Líquido das Atividades Operacionais" (CFO total).
# ATENÇÃO: 6.01.01 ("Caixa Gerado nas Operações") NÃO é o CFO: é antes da variação de capital
# de giro e dos juros/IR. Por isso a descrição de fallback é a do total (6.01), não a do 6.01.01.
CODIGO_CFO = "6.01"
DESC_CFO = {
    "CAIXA LIQUIDO DAS ATIVIDADES OPERACIONAIS",
    "CAIXA LIQUIDO ATIVIDADES OPERACIONAIS",
    "CAIXA LIQUIDO GERADO PELAS ATIVIDADES OPERACIONAIS",
}
DFC_USECOLS = {"CNPJ_CIA", "DT_REFER", "DT_INI_EXERC", "CD_CONTA", "DS_CONTA",
               "VL_CONTA", "ESCALA_MOEDA", "ORDEM_EXERC", "VERSAO"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("capex_bp")


# =============================================================================
# UTILITÁRIOS
# =============================================================================
def read_csv_flex(path: Path, **kwargs) -> pd.DataFrame:
    """Lê CSV tentando encodings comuns e escolhendo o separador (; , tab) pela 1ª linha."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    last_exc: Exception | None = None
    for enc in ("utf-8-sig", "latin1"):
        try:
            with open(path, encoding=enc) as fh:
                header = fh.readline()
            sep = max((";", ",", "\t"), key=header.count)
            if header.count(sep) == 0:
                sep = ","
            return pd.read_csv(path, sep=sep, encoding=enc, **kwargs)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_exc = exc
    raise ValueError(f"Não foi possível ler {path}: {last_exc}")


def strip_accents_upper(s: pd.Series) -> pd.Series:
    """Normaliza texto: sem acentos e em maiúsculas."""
    return (
        s.fillna("")
        .astype(str)
        .map(lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii"))
        .str.upper()
    )


def ano_de_t(t: pd.Series) -> pd.Series:
    """Ano a partir do ordinal trimestral (Period 'Q': ordinal 0 = 1970Q1)."""
    return (1970 + t // 4).astype("int64")


# =============================================================================
# PASSO 1 — LIMPEZA E LOG DO ATIVO FIXO
# =============================================================================
def carregar_e_limpar(path: Path) -> pd.DataFrame:
    df = read_csv_flex(path, dtype={"CNPJ_CIA": str})
    df.columns = [str(c).strip() for c in df.columns]
    faltam = [c for c in COLS_OBRIGATORIAS if c not in df.columns]
    if faltam:
        raise KeyError(f"Colunas ausentes em {path.name}: {faltam}. "
                       f"Rode script_painel.py atualizado (gera Ativo_Fixo).")

    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df.dropna(subset=["CNPJ_CIA", "t"]).copy()
    df["t"] = df["t"].astype("int64")

    # Ordenação estrita antes de qualquer cálculo temporal
    df = df.sort_values(["CNPJ_CIA", "t"]).reset_index(drop=True)
    if df.duplicated(["CNPJ_CIA", "t"]).any():
        raise AssertionError("Há pares (CNPJ_CIA, t) duplicados no painel.")

    for c in ("Ativo_Fixo", "Ativo_Total"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        n_inv = int((df[c] <= 0).sum())
        if n_inv:
            log.warning("%s: %d valores <= 0 convertidos em NaN.", c, n_inv)
        df[c] = df[c].where(df[c] > 0)          # nulo/<=0 -> NaN (evita log de não positivos)

    # Ativo_Total_L1 calculado na grade contínua pelo script_painel.py (mais completo que um
    # lag feito sobre as linhas que sobraram). Guardado à parte; as colunas _L1 de mesmo nome
    # são removidas para não colidirem (sufixos _x/_y) no merge de defasagem abaixo.
    if "Ativo_Total_L1" in df.columns:
        at = pd.to_numeric(df["Ativo_Total_L1"], errors="coerce")
        df["Ativo_Total_L1_grade"] = at.where(at > 0)
    df = df.drop(columns=[c for c in ("Ativo_Fixo_L1", "Ativo_Total_L1") if c in df.columns])

    df["log_Ativo_Fixo"] = np.log(df["Ativo_Fixo"])
    return df


# =============================================================================
# PASSO 2 — INVESTIMENTO LÍQUIDO DESCRITIVO
# =============================================================================
def adicionar_delta_ativo_fixo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Delta_Ativo_Fixo_Ativo = (AF_t - AF_{t-1}) / Ativo_Total_{t-1}.
    Defasagem por merge em t+1: só existe se o trimestre t-1 está de fato no painel.
    """
    lag = df[["CNPJ_CIA", "t", "Ativo_Fixo", "Ativo_Total"]].copy()
    lag["t"] += 1
    lag = lag.rename(columns={"Ativo_Fixo": "Ativo_Fixo_L1", "Ativo_Total": "Ativo_Total_L1"})
    out = df.merge(lag, on=["CNPJ_CIA", "t"], how="left", validate="one_to_one")
    out["Delta_Ativo_Fixo_Ativo"] = (out["Ativo_Fixo"] - out["Ativo_Fixo_L1"]) / out["Ativo_Total_L1"]
    out["Delta_Ativo_Fixo_Ativo"] = out["Delta_Ativo_Fixo_Ativo"].replace([np.inf, -np.inf], np.nan)
    return out


# =============================================================================
# PASSO 3 — CFO (DFC) -> Fluxo_Caixa_Op_Ativo
# =============================================================================
def find_dfc_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Pasta raiz da CVM não encontrada: {root}")
    files = sorted(p for p in root.rglob("*.csv")
                   if any(t in p.name.lower() for t in DFC_TAGS))
    log.info("DFC: %d arquivos (MD/MI consolidados) encontrados em %s", len(files), root)
    return files


def read_dfc_file(path: Path) -> pd.DataFrame | None:
    """Lê um CSV de DFC e devolve só o CFO (6.01) acumulado no ano, em reais."""
    try:
        df = pd.read_csv(path, sep=";", encoding="latin1", dtype=str,
                         usecols=lambda c: c in DFC_USECOLS)
    except Exception as exc:
        log.error("Falha ao ler %s: %s", path.name, exc)
        return None
    obrig = {"CNPJ_CIA", "DT_REFER", "CD_CONTA", "VL_CONTA"}
    if not obrig.issubset(df.columns):
        log.warning("%s sem colunas obrigatórias %s; ignorado.", path.name, obrig - set(df.columns))
        return None

    cd = df["CD_CONTA"].str.strip()
    por_codigo = cd == CODIGO_CFO
    por_desc = pd.Series(False, index=df.index)
    if "DS_CONTA" in df.columns:
        desc = strip_accents_upper(df["DS_CONTA"]).str.strip()
        # só nível de grupo (ex.: '6.01'), nunca subcontas como 6.01.01
        por_desc = desc.isin(DESC_CFO) & cd.str.startswith("6") & (cd.str.count(r"\.") <= 1)
    df = df[por_codigo | por_desc].copy()
    if df.empty:
        return None
    df["_canon"] = np.where(cd.loc[df.index] == CODIGO_CFO, 1, 0)

    if "ORDEM_EXERC" in df.columns:
        df = df[strip_accents_upper(df["ORDEM_EXERC"]).str.contains("ULTIMO", na=False)]
    if "DT_INI_EXERC" in df.columns:               # só acumulado desde 1º de janeiro
        ini = pd.to_datetime(df["DT_INI_EXERC"], errors="coerce")
        df = df[(ini.dt.month == 1) & (ini.dt.day == 1)]

    df["CFO_ytd"] = pd.to_numeric(df["VL_CONTA"].str.replace(",", ".", regex=False), errors="coerce")
    if "ESCALA_MOEDA" in df.columns:
        escala = strip_accents_upper(df["ESCALA_MOEDA"]).map({"MIL": 1_000.0}).fillna(1.0)
        df["CFO_ytd"] = df["CFO_ytd"] * escala
    df["VERSAO"] = pd.to_numeric(df.get("VERSAO", 0), errors="coerce").fillna(0)
    df["DT_REFER"] = pd.to_datetime(df["DT_REFER"], errors="coerce")
    df = df.dropna(subset=["CFO_ytd", "DT_REFER"])
    df["t"] = df["DT_REFER"].dt.to_period("Q").map(lambda q: q.ordinal).astype("int64")
    return df[["CNPJ_CIA", "t", "CFO_ytd", "_canon", "VERSAO", "DT_REFER"]]


def construir_cfo(root: Path) -> pd.DataFrame:
    """Devolve [CNPJ_CIA, t, CFO] trimestral isolado (Q1 = YTD; Qn = YTD_t - YTD_{t-1})."""
    frames = [d for f in find_dfc_files(root) if (d := read_dfc_file(f)) is not None]
    if not frames:
        raise RuntimeError(f"Nenhum CFO (conta {CODIGO_CFO}) foi lido dos arquivos DFC em {root}.")
    cfo = pd.concat(frames, ignore_index=True)
    cfo = (cfo.sort_values(["CNPJ_CIA", "t", "_canon", "VERSAO", "DT_REFER"])
              .drop_duplicates(["CNPJ_CIA", "t"], keep="last"))

    w = cfo[["CNPJ_CIA", "t", "CFO_ytd"]].copy()
    w["Ano"] = ano_de_t(w["t"])
    w["Q"] = (w["t"] % 4 + 1).astype("int64")
    prev = w[["CNPJ_CIA", "Ano", "Q", "CFO_ytd"]].copy()
    prev["Q"] += 1
    prev = prev.rename(columns={"CFO_ytd": "CFO_prev"})
    w = w.merge(prev, on=["CNPJ_CIA", "Ano", "Q"], how="left")
    # Trimestre anterior ausente -> NaN (nunca um valor errado)
    w["CFO"] = np.where(w["Q"] == 1, w["CFO_ytd"], w["CFO_ytd"] - w["CFO_prev"])
    log.info("CFO: %d firma-trimestres, %d empresas.", len(w), w["CNPJ_CIA"].nunique())
    return w[["CNPJ_CIA", "t", "CFO"]]


def adicionar_fluxo_caixa(df: pd.DataFrame, cfo: pd.DataFrame) -> pd.DataFrame:
    """Fluxo_Caixa_Op_Ativo = CFO_t / Ativo_Total_{t-1}."""
    out = df.merge(cfo, on=["CNPJ_CIA", "t"], how="left", validate="one_to_one")
    den = out["Ativo_Total_L1"]
    if "Ativo_Total_L1_grade" in out.columns:
        den = out["Ativo_Total_L1_grade"].combine_first(den)
    out["Fluxo_Caixa_Op_Ativo"] = (out["CFO"] / den.where(den > 0)).replace([np.inf, -np.inf], np.nan)
    n = int(out["Fluxo_Caixa_Op_Ativo"].notna().sum())
    log.info("Fluxo_Caixa_Op_Ativo: %d de %d linhas válidas (%.1f%%); %d empresas.",
             n, len(out), 100 * n / max(len(out), 1),
             out.loc[out["Fluxo_Caixa_Op_Ativo"].notna(), "CNPJ_CIA"].nunique())
    return out


def tabela_cobertura(df: pd.DataFrame) -> pd.DataFrame:
    """Estatísticas descritivas de Delta_Ativo_Fixo_Ativo e cobertura, por ano."""
    d = df.copy()
    d["Ano"] = ano_de_t(d["t"])

    def resumo(g: pd.DataFrame) -> pd.Series:
        x = g["Delta_Ativo_Fixo_Ativo"].dropna()
        n = len(g)
        return pd.Series({
            "N_obs_painel": n,
            "N_empresas": g["CNPJ_CIA"].nunique(),
            "N_valid_log_AF": int(g["log_Ativo_Fixo"].notna().sum()),
            "Cobertura_log_AF_pct": 100 * g["log_Ativo_Fixo"].notna().mean() if n else np.nan,
            "N_valid_Delta": len(x),
            "Cobertura_Delta_pct": 100 * len(x) / n if n else np.nan,
            "N_valid_FCO": int(g["Fluxo_Caixa_Op_Ativo"].notna().sum()),
            "Cobertura_FCO_pct": 100 * g["Fluxo_Caixa_Op_Ativo"].notna().mean() if n else np.nan,
            "Delta_media": x.mean() if len(x) else np.nan,
            "Delta_mediana": x.median() if len(x) else np.nan,
            "Delta_p1": x.quantile(0.01) if len(x) else np.nan,
            "Delta_p99": x.quantile(0.99) if len(x) else np.nan,
        })

    tab = d.groupby("Ano").apply(resumo, include_groups=False).reset_index()
    total = resumo(d)
    total["Ano"] = "Total"
    tab = pd.concat([tab.astype({"Ano": object}), total.to_frame().T], ignore_index=True)
    return tab


# =============================================================================
# PASSO 4 — MERGE COM A BASE ORTOGONALIZADA
# =============================================================================
def merge_com_ortog(df_bp: pd.DataFrame, ortog_path: Path) -> pd.DataFrame:
    ort = read_csv_flex(ortog_path, dtype={"CNPJ_CIA": str})
    ort.columns = [str(c).strip() for c in ort.columns]
    if "CNPJ_CIA" not in ort.columns:
        raise KeyError(f"'CNPJ_CIA' ausente em {ortog_path.name}.")

    # Chave temporal: t (ordinal) se existir; senão Trimestre ('2019Q3') convertido para t
    if "t" in ort.columns:
        ort["t"] = pd.to_numeric(ort["t"], errors="coerce").astype("Int64")
    elif "Trimestre" in ort.columns:
        ort["t"] = pd.PeriodIndex(ort["Trimestre"].astype(str), freq="Q").asi8
    else:
        raise KeyError(f"{ortog_path.name} precisa ter 't' ou 'Trimestre' para o merge.")
    ort["t"] = ort["t"].astype("int64")

    if ort.duplicated(["CNPJ_CIA", "t"]).any():
        raise AssertionError(f"Chaves (CNPJ_CIA, t) duplicadas em {ortog_path.name}.")

    # Evita sufixos _x/_y: substitui versões antigas das colunas que estamos trazendo
    novas = ["log_Ativo_Fixo", "Fluxo_Caixa_Op_Ativo"]
    ort = ort.drop(columns=[c for c in novas if c in ort.columns])

    chaves = df_bp[["CNPJ_CIA", "t"] + novas]
    final = ort.merge(chaves, on=["CNPJ_CIA", "t"], how="left", validate="one_to_one")
    if len(final) != len(ort):
        raise AssertionError("O merge alterou o número de linhas da base de regressão.")
    log.info("Merge: %d linhas da base ortogonalizada; %d com log_Ativo_Fixo; %d com "
             "Fluxo_Caixa_Op_Ativo.", len(final), int(final["log_Ativo_Fixo"].notna().sum()),
             int(final["Fluxo_Caixa_Op_Ativo"].notna().sum()))
    return final


# =============================================================================
# MAIN
# =============================================================================
def main() -> int:
    try:
        df = carregar_e_limpar(PAINEL_CSV)
        df = adicionar_delta_ativo_fixo(df)
        cfo = construir_cfo(CVM_ROOT)
        df = adicionar_fluxo_caixa(df, cfo)

        cob = tabela_cobertura(df)
        cob.to_csv(COBERTURA_CSV, index=False, encoding="utf-8")
        log.info("Cobertura por ano exportada: %s", COBERTURA_CSV)
        with pd.option_context("display.width", 220, "display.max_columns", None,
                               "display.float_format", "{:.4f}".format):
            print("\n=== Delta_Ativo_Fixo_Ativo, Fluxo_Caixa_Op_Ativo e cobertura por ano ===")
            print(cob.to_string(index=False))

        final = merge_com_ortog(df, ORTOG_CSV)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, AssertionError) as exc:
        log.error("Pipeline interrompido: %s", exc)
        return 1

    final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    validas = final["log_Ativo_Fixo"].notna()
    fco = final["Fluxo_Caixa_Op_Ativo"].notna()
    print("\n=== Resultado final ===")
    print(f"Arquivo: {OUTPUT_CSV}")
    print(f"Observações totais na base            : {len(final):,}")
    print(f"Obs. válidas de log_Ativo_Fixo        : {int(validas.sum()):,}")
    print(f"Empresas cobertas (log_AF válido)     : {final.loc[validas, 'CNPJ_CIA'].nunique():,}")
    print(f"Obs. válidas de Fluxo_Caixa_Op_Ativo  : {int(fco.sum()):,}")
    print(f"Empresas cobertas (FCO válido)        : {final.loc[fco, 'CNPJ_CIA'].nunique():,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())