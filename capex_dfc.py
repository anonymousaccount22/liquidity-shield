"""
capex_dfc.py
==========
Constrói Capex trimestral (e Fluxo de Caixa Operacional) a partir da DFC da CVM
(ITR = Q1-Q3, DFP = Q4), junta ao painel de regressão e salva:

  dfc_602_601_cache.pkl        cache da leitura bruta (6.01 e 6.02.x)
  capex_dfc_trimestral.csv     intermediário: capex YTD -> fluxo trimestral (por firma-trimestre)
  cfo_dfc_trimestral.csv       intermediário: CFO YTD -> fluxo trimestral
  cobertura_capex_ano.csv      tabela de cobertura por ano
  painel_regressao_q_dfc.csv   painel final (sem winsorização; com log_Capex)

Convenções
----------
* Capex_DFC          = fluxo trimestral de capex (R$, positivo = investimento). Negativos -> NaN.
* Capex_DFC_Ativo    = Capex_DFC / Ativo_Total_L1, SEM winsorização no arquivo salvo.
* log_Capex          = log(Capex_DFC) (NaN quando Capex_DFC <= 0 ou ausente).
* Winsorização 1%/99% de Capex_DFC_Ativo é usada SOMENTE nas estatísticas descritivas
  impressas aqui. A winsorização para a regressão fica no script de regressão.
"""
import re, io, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

PASTA_CVM = Path("dados_cvm")
CACHE = Path("dfc_602_601_cache.pkl")
PAINEL_IN = "painel_regressao_q.csv"
PAINEL_OUT = "painel_regressao_q_dfc.csv"
COLS = ["CNPJ_CIA", "DT_REFER", "CD_CONTA", "DS_CONTA", "VL_CONTA", "ESCALA_MOEDA", "ORDEM_EXERC"]
alvo = re.compile(r"(itr|dfp)_cia_aberta_DFC_(MD|MI)_(con|ind)_\d{4}\.csv$", re.I)

DIAGNOSTICO = True                       # False desliga os blocos de diagnóstico no fim
CNPJS_DIAG = ["33.035.536/0001-00", "51.466.860/0001-56"]


def log(msg=""):
    print(msg, flush=True)


def le_csv(buf, nome):
    df = pd.read_csv(buf, sep=";", encoding="latin1", usecols=lambda c: c in COLS, dtype=str)
    # Mantém CFO total (6.01), contas de investimento (6.02.x) e apenas o exercício atual.
    df = df[df["CD_CONTA"].eq("6.01") | df["CD_CONTA"].str.startswith("6.02.", na=False)]
    df = df[df["ORDEM_EXERC"].str.upper().str.startswith(("ÚLT", "ULT"), na=False)]
    df = df.drop(columns="ORDEM_EXERC")
    df["tipo"] = "con" if "_con_" in nome.lower() else "ind"
    df["fonte"] = "dfp" if "dfp" in nome.lower() else "itr"
    df["metodo"] = "mi" if "_mi_" in nome.lower() else "md"
    return df


def escala(s):
    return s.str.upper().map({"MIL": 1_000, "UNIDADE": 1}).fillna(1)


def desacumula(df, ytd_col, zero_como_ausente=False):
    """
    Converte valores YTD (acumulados no ano fiscal) em fluxos trimestrais discretos.

    Entrada: colunas CNPJ_CIA, DT_REFER (datetime), tipo, fonte, <ytd_col>.
    Regras:
      * Ano fiscal por firma = mês de fechamento mais frequente nas DFP.
        k = posição do mês dentro do ano fiscal (3 = 1º tri, 6 = 2º, 9 = 3º, 12 = 4º).
      * Escopo (consolidado > individual) escolhido por firma-ANO FISCAL, para não trocar de
        escopo no meio do ano fiscal (isso geraria diferenças espúrias entre escopos).
      * Duplicata de trimestre: DFP prevalece sobre ITR.
      * fluxo = YTD            se k == 3 (início do ano fiscal)
        fluxo = YTD - YTD(t-1) caso contrário; se t-1 não existe -> NaN (nunca vira o YTD cheio).
      * zero_como_ausente: YTD exatamente 0 fora do 1º tri fiscal é tratado como dado ausente.
    """
    df = df.reset_index(drop=True).copy()
    df["ano"] = df["DT_REFER"].dt.year
    df["q"] = df["DT_REFER"].dt.quarter
    mes = df["DT_REFER"].dt.month

    mes_fim = (df[df["fonte"] == "dfp"].assign(m=mes)
               .groupby("CNPJ_CIA")["m"].agg(lambda s: s.mode().iloc[0]))
    df["fim_fiscal"] = df["CNPJ_CIA"].map(mes_fim).fillna(12).astype(int)
    df["fy"] = np.where(mes <= df["fim_fiscal"], df["ano"], df["ano"] + 1)
    df["k"] = ((mes - df["fim_fiscal"] - 1) % 12) + 1

    df["pref_escopo"] = (df["tipo"] == "con").astype(int)
    df = df[df["pref_escopo"] == df.groupby(["CNPJ_CIA", "fy"])["pref_escopo"].transform("max")]

    df["pref_fonte"] = (df["fonte"] == "dfp").astype(int)
    df = (df.sort_values(["CNPJ_CIA", "DT_REFER", "pref_fonte"])
            .drop_duplicates(["CNPJ_CIA", "ano", "q"], keep="last"))

    if zero_como_ausente:
        df.loc[(df[ytd_col] == 0) & (df["k"] != 3), ytd_col] = np.nan

    df["qi"] = df["ano"] * 4 + df["q"]
    ant = df[["CNPJ_CIA", "qi", ytd_col]].copy()
    ant["qi"] += 1
    df = df.merge(ant.rename(columns={ytd_col: "ytd_ant"}), on=["CNPJ_CIA", "qi"], how="left")
    df["fluxo"] = np.where(df["k"] == 3, df[ytd_col], df[ytd_col] - df["ytd_ant"])
    df["Trimestre"] = df["ano"].astype(str) + "Q" + df["q"].astype(str)
    return df.rename(columns={ytd_col: "ytd"})


# ======================================================================
# 1. Leitura (com cache e progresso)
# ======================================================================
if CACHE.exists():
    log(f"Lendo cache {CACHE}")
    dfc = pd.read_pickle(CACHE)
else:
    arquivos = []
    for f in PASTA_CVM.rglob("*"):
        if f.suffix.lower() == ".zip":
            with zipfile.ZipFile(f) as z:
                arquivos += [(f, n) for n in z.namelist() if alvo.search(n)]
        elif f.suffix.lower() == ".csv" and alvo.search(f.name):
            arquivos.append((f, None))
    if not arquivos:
        raise SystemExit("Nenhum arquivo DFC encontrado. Confira PASTA_CVM e os nomes dos arquivos.")
    log(f"{len(arquivos)} arquivos DFC a ler")
    partes = []
    for i, (f, n) in enumerate(arquivos, 1):
        if n is None:
            partes.append(le_csv(f, f.name))
        else:
            with zipfile.ZipFile(f) as z:
                partes.append(le_csv(io.BytesIO(z.read(n)), n))
        log(f"  [{i}/{len(arquivos)}] {n or f.name}")
    dfc = pd.concat(partes, ignore_index=True)
    dfc.to_pickle(CACHE)
    log(f"Cache salvo em {CACHE}")
log(f"Linhas 6.01/6.02.x lidas: {len(dfc):,}")

# ======================================================================
# 2. CFO (conta 6.01, reportada em YTD) -> fluxo trimestral
# ======================================================================
log("\n[CFO] construindo fluxo de caixa operacional trimestral...")
cfo = dfc[dfc["CD_CONTA"].eq("6.01")].copy()
cfo["DT_REFER"] = pd.to_datetime(cfo["DT_REFER"], errors="coerce")
cfo["valor"] = pd.to_numeric(cfo["VL_CONTA"], errors="coerce") * escala(cfo["ESCALA_MOEDA"])
cfo = (cfo.groupby(["CNPJ_CIA", "DT_REFER", "tipo", "fonte", "metodo"], as_index=False)["valor"]
          .sum(min_count=1))
# método indireto (mi) preferido quando o mesmo relatório traz os dois
cfo["pref_metodo"] = (cfo["metodo"] == "mi").astype(int)
cfo = cfo[cfo["pref_metodo"] == cfo.groupby(["CNPJ_CIA", "DT_REFER", "tipo", "fonte"])["pref_metodo"]
          .transform("max")]
cfo = desacumula(cfo, "valor")
cfo = cfo.rename(columns={"fluxo": "Fluxo_Caixa_Operacional"})
cfo_out = cfo[["CNPJ_CIA", "Trimestre", "tipo", "fonte", "k", "ytd", "ytd_ant", "Fluxo_Caixa_Operacional"]]
cfo_out.to_csv("cfo_dfc_trimestral.csv", index=False)
log(f"[CFO] {cfo['Fluxo_Caixa_Operacional'].notna().sum():,} firma-trimestres com valor "
    f"-> cfo_dfc_trimestral.csv")
cfo = cfo[["CNPJ_CIA", "Trimestre", "Fluxo_Caixa_Operacional"]]

# ======================================================================
# 3. Linhas de capex (6.02.x)
# ======================================================================
d = dfc.copy()
ds = d["DS_CONTA"].str.lower()
inclui = ds.str.contains(r"imobilizado|intang[ií]vel|ativo fixo|ativo biol[oó]gico",
                         regex=True, na=False) \
       & ds.str.contains(r"aquisi|adi[cç]|compra|acr[eé]scimo|pagamento|aplica",
                         regex=True, na=False)
exclui = ds.str.contains(r"venda|aliena|baixa|receb|dividend|redu|caixa l[ií]quido|devolu|resgate",
                         regex=True, na=False)
d = d[inclui & ~exclui].copy()
d["chave"] = d["CNPJ_CIA"] + "|" + d["DT_REFER"] + "|" + d["tipo"] + "|" + d["fonte"]

# mantém a conta mais alta que casa o regex; descarta descendentes (evita dupla contagem)
match_idx = pd.MultiIndex.from_frame(d[["chave", "CD_CONTA"]])
tem_anc = np.zeros(len(d), dtype=bool)
cur = d["CD_CONTA"]
for _ in range(5):
    cur = cur.str.rpartition(".")[0]
    ok = (cur != "").to_numpy()
    if not ok.any():
        break
    idx = pd.MultiIndex.from_arrays([d["chave"].to_numpy()[ok], cur.to_numpy()[ok]])
    tem_anc[ok] |= idx.isin(match_idx)
d = d[~tem_anc].copy()
d["pref_metodo"] = (d["metodo"] == "mi").astype(int)
chaves_metodo = ["CNPJ_CIA", "DT_REFER", "tipo", "fonte"]
d = d[d["pref_metodo"] == d.groupby(chaves_metodo)["pref_metodo"].transform("max")]

if DIAGNOSTICO:
    log("\n[DIAG] linhas de capex selecionadas, CNPJ 33.035.536/0001-00, 2012:")
    x = d[(d["CNPJ_CIA"] == "33.035.536/0001-00") & (d["DT_REFER"].str[:4] == "2012")]
    log(x[["DT_REFER", "tipo", "fonte", "CD_CONTA", "DS_CONTA", "VL_CONTA"]]
        .sort_values(["DT_REFER", "CD_CONTA"]).to_string(index=False))

# ======================================================================
# 4. Escala, agregação e desacumulação YTD -> trimestre
# ======================================================================
d["valor"] = pd.to_numeric(d["VL_CONTA"], errors="coerce") * escala(d["ESCALA_MOEDA"])

# capex = saída de caixa (negativa na DFC) -> sinal invertido
g = (d.groupby(["CNPJ_CIA", "DT_REFER", "tipo", "fonte"])["valor"].sum(min_count=1).mul(-1)
       .rename("capex_ytd").reset_index())
g["DT_REFER"] = pd.to_datetime(g["DT_REFER"])

# YTD exatamente 0 fora do 1º trimestre fiscal = dado ausente, não "zero investimento"
g = desacumula(g, "capex_ytd", zero_como_ausente=True)
g = g.rename(columns={"ytd": "capex_ytd", "fluxo": "Capex_DFC_bruto"})
log(f"\n[CAPEX] {g['Capex_DFC_bruto'].notna().sum():,} firma-trimestres com fluxo trimestral")

# ----------------------------------------------------------------------
# Capex trimestral negativo -> NaN (tratamento documentado)
#   * O filtro de linhas acima já EXCLUI vendas/baixas/alienações; logo, um fluxo trimestral
#     negativo NÃO é desinvestimento capturado, e sim, na prática, reapresentação (restatement)
#     entre ITR/DFP ou mudança de classificação de contas entre períodos.
#   * A máscara é aplicada observação a observação: só o trimestre negativo vira NaN. Os demais
#     trimestres da mesma firma permanecem intactos (a desacumulação é feita sobre o YTD,
#     antes da máscara, então um NaN aqui não contamina o trimestre seguinte).
#   * Capex_DFC_neg guarda a flag para auditoria; Capex_DFC_bruto guarda o valor original.
# ----------------------------------------------------------------------
g["Capex_DFC_neg"] = g["Capex_DFC_bruto"] < 0
g["Capex_DFC"] = g["Capex_DFC_bruto"].mask(g["Capex_DFC_neg"])

n_val = g["Capex_DFC_bruto"].notna().sum()
n_neg = int(g["Capex_DFC_neg"].sum())
firmas_neg = g.loc[g["Capex_DFC_neg"], "CNPJ_CIA"].unique()
validos_firmas_neg = g[g["CNPJ_CIA"].isin(firmas_neg)]["Capex_DFC"].notna().sum()
log(f"[CAPEX] negativos -> NaN: {n_neg:,} ({n_neg / n_val:.2%} das obs. com valor), "
    f"em {len(firmas_neg):,} firmas")
log(f"[CAPEX] essas firmas mantêm {validos_firmas_neg:,} trimestres válidos (não foram removidas)")
assert g["Capex_DFC"].notna().sum() == n_val - n_neg, "mascaramento afetou mais que os negativos"

g_out = g[["CNPJ_CIA", "Trimestre", "tipo", "fonte", "k", "capex_ytd", "ytd_ant",
           "Capex_DFC_bruto", "Capex_DFC", "Capex_DFC_neg"]]
g_out.to_csv("capex_dfc_trimestral.csv", index=False)
log("[CAPEX] intermediário salvo em capex_dfc_trimestral.csv")

# ======================================================================
# 5. Junção ao painel de regressão
# ======================================================================
p = pd.read_csv(PAINEL_IN)
n_painel = len(p)
log(f"\nPainel lido: {n_painel:,} linhas, {p['CNPJ_CIA'].nunique():,} empresas")

p = p.merge(g[["CNPJ_CIA", "Trimestre", "Capex_DFC", "Capex_DFC_neg"]],
            on=["CNPJ_CIA", "Trimestre"], how="left")
p["Capex_DFC_neg"] = p["Capex_DFC_neg"].fillna(False).astype(bool)
p["Tamanho"] = np.log(p["Ativo_Total_L1"].where(p["Ativo_Total_L1"] > 0))

cfo_l1 = cfo.copy()
cfo_l1["Trimestre"] = (pd.PeriodIndex(cfo_l1["Trimestre"], freq="Q") + 1).astype(str)
p = p.merge(cfo_l1, on=["CNPJ_CIA", "Trimestre"], how="left")
assert len(p) == n_painel, "a junção duplicou linhas do painel (chave CNPJ_CIA+Trimestre não é única?)"

ativo = p["Ativo_Total_L1"].where(p["Ativo_Total_L1"] > 0)
p["Fluxo_Caixa_Op_Ativo"] = p["Fluxo_Caixa_Operacional"] / ativo
p["Capex_DFC_Ativo"] = p["Capex_DFC"] / ativo          # SEM winsorização

# log do nível de capex para as projeções locais (log só existe para capex > 0)
n_zero = int((p["Capex_DFC"] == 0).sum())
p["log_Capex"] = np.log(p["Capex_DFC"].where(p["Capex_DFC"] > 0))
log(f"log_Capex: {p['log_Capex'].notna().sum():,} obs. válidas "
    f"({n_zero:,} com Capex_DFC == 0 ficam NaN no log)")

# ======================================================================
# 6. Estatísticas descritivas (winsorização 1%/99% só aqui)
# ======================================================================
raw = p["Capex_DFC_Ativo"]
lo, hi = raw.quantile([.01, .99])
wins = raw.clip(lo, hi)
log(f"\nWinsorização apenas para descritivas: p1={lo:.5f}  p99={hi:.5f}  (coluna salva NÃO é winsorizada)")
log(f"Cobertura (linhas com Capex_DFC_Ativo): {raw.notna().mean():.1%}")
log(f"Mediana de Capex_DFC/Ativo (esperado ~0,005 a 0,03 por trimestre): {wins.median():.4f}")
if "Delta_Capex" in p.columns:
    corr = pd.concat([wins, p["Delta_Capex"]], axis=1).corr().iloc[0, 1]
    log(f"Correlação (winsorizada) com Delta_Capex: {corr:.3f}")
log("\nDescritivas de Capex_DFC_Ativo (winsorizada 1%/99%):")
log(wins.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).to_string())

# ======================================================================
# 7. Cobertura por ano (pós-junção)
# ======================================================================
p["Ano"] = p["Trimestre"].str[:4].astype(int)


def tabela_cobertura(df):
    base = df.groupby("Ano").agg(Obs_painel=("CNPJ_CIA", "size"),
                                 Empresas_painel=("CNPJ_CIA", "nunique"))
    com_capex = (df[df["Capex_DFC_Ativo"].notna()].groupby("Ano")
                   .agg(Obs_capex=("CNPJ_CIA", "size"), Empresas_capex=("CNPJ_CIA", "nunique")))
    com_log = df[df["log_Capex"].notna()].groupby("Ano").size().rename("Obs_log_capex")
    negs = df[df["Capex_DFC_neg"]].groupby("Ano").size().rename("Obs_neg_NaN")
    tab = base.join([com_capex, com_log, negs]).fillna(0).astype(int)
    tab["Cob_obs_%"] = 100 * tab["Obs_capex"] / tab["Obs_painel"]
    tab["Cob_empresas_%"] = 100 * tab["Empresas_capex"] / tab["Empresas_painel"]
    tot = pd.DataFrame({
        "Obs_painel": [len(df)],
        "Empresas_painel": [df["CNPJ_CIA"].nunique()],
        "Obs_capex": [df["Capex_DFC_Ativo"].notna().sum()],
        "Empresas_capex": [df.loc[df["Capex_DFC_Ativo"].notna(), "CNPJ_CIA"].nunique()],
        "Obs_log_capex": [df["log_Capex"].notna().sum()],
        "Obs_neg_NaN": [int(df["Capex_DFC_neg"].sum())],
    }, index=["Total"])
    tot["Cob_obs_%"] = 100 * tot["Obs_capex"] / tot["Obs_painel"]
    tot["Cob_empresas_%"] = 100 * tot["Empresas_capex"] / tot["Empresas_painel"]
    return pd.concat([tab, tot])


cob = tabela_cobertura(p)
log("\nCobertura de Capex_DFC por ano (após junção com o painel):")
log(cob.to_string(float_format=lambda v: f"{v:,.1f}"))
cob.to_csv("cobertura_capex_ano.csv")
log("Tabela salva em cobertura_capex_ano.csv")

# ======================================================================
# 8. Salva painel final
# ======================================================================
p = p.drop(columns="Ano")
p.to_csv(PAINEL_OUT, index=False)
log(f"\nPainel final salvo em {PAINEL_OUT}: {len(p):,} linhas, {p.shape[1]} colunas "
    f"(inclui Capex_DFC, Capex_DFC_Ativo, log_Capex, Capex_DFC_neg)")

# ======================================================================
# 9. Diagnóstico dos negativos (DIAGNOSTICO = False para desligar)
# ======================================================================
if DIAGNOSTICO:
    log("\n[DIAG] Share de negativos entre obs. com valor, por trimestre-do-ano:")
    tmp = p.assign(qtr=p["Trimestre"].str[-2:])
    log(tmp[tmp["Capex_DFC"].notna() | tmp["Capex_DFC_neg"]]
        .groupby("qtr")["Capex_DFC_neg"].mean().to_string())

    log("\n[DIAG] Top 10 firmas com mais negativos:")
    log(p.loc[p["Capex_DFC_neg"], "CNPJ_CIA"].value_counts().head(10).to_string())

    for c in CNPJS_DIAG:
        log(f"\n[DIAG] {c}")
        x = g[g["CNPJ_CIA"] == c][["Trimestre", "tipo", "fonte", "capex_ytd", "Capex_DFC_bruto", "Capex_DFC"]]
        log(x.sort_values("Trimestre").head(24).to_string(index=False))

    q4 = g[(g["q"] == 4) & (g["Capex_DFC_bruto"] < 0) & (g["k"] == 12)]
    log("\n[DIAG] Q4 (fim do ano fiscal) com fluxo negativo - possível YTD da DFP inconsistente com ITR:")
    log(q4[["CNPJ_CIA", "ano", "tipo", "capex_ytd", "ytd_ant", "Capex_DFC_bruto"]]
        .head(25).to_string(index=False))

    neg_firma = g.groupby("CNPJ_CIA")["Capex_DFC_bruto"].apply(
        lambda s: (s < 0).sum() / max(s.notna().sum(), 1))
    log(f"\n[DIAG] {(neg_firma > 0.15).sum()} firmas com >15% de trimestres negativos")
    log(neg_firma.sort_values(ascending=False).head(10).round(2).to_string())