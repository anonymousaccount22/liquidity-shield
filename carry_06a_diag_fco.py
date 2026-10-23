"""
carry_06a_diag_fco.py -- quantas firmas perdem CFO por ano fiscal != ano civil?
So le arquivos. Nao altera nada. Importa funcoes do capex_bp.py (main protegido).
"""
import pandas as pd
import capex_bp as cb

ARQ = "painel_regressao_q_bp.csv"

# 1) linhas de 6.01 nos DFC brutos, com a data de inicio do exercicio
parts = []
for f in cb.find_dfc_files(cb.CVM_ROOT):
    try:
        d = pd.read_csv(f, sep=";", encoding="latin1", dtype=str,
                        usecols=lambda c: c in cb.DFC_USECOLS)
    except Exception:
        continue
    if not {"CNPJ_CIA", "CD_CONTA", "DT_INI_EXERC"}.issubset(d.columns):
        continue
    d = d[d["CD_CONTA"].str.strip() == cb.CODIGO_CFO]
    if "ORDEM_EXERC" in d.columns:
        d = d[cb.strip_accents_upper(d["ORDEM_EXERC"]).str.contains("ULTIMO", na=False)]
    ini = pd.to_datetime(d["DT_INI_EXERC"], errors="coerce")
    parts.append(pd.DataFrame({"CNPJ_CIA": d["CNPJ_CIA"], "jan1": (ini.dt.month == 1) & (ini.dt.day == 1)}))
raw = pd.concat(parts)
g = raw.groupby("CNPJ_CIA")["jan1"].agg(["sum", "size"])
g["so_nao_jan1"] = g["sum"] == 0
print(f"Firmas com 6.01 nos DFC: {len(g)} | sem NENHUMA linha iniciada em 1/jan: {int(g['so_nao_jan1'].sum())}")

# 2) impacto no painel de regressao
p = pd.read_csv(ARQ, dtype={"CNPJ_CIA": str})
af = p["log_Ativo_Fixo"].notna()
fco = p["Fluxo_Caixa_Op_Ativo"].notna()
print(f"Obs. com log_Ativo_Fixo: {int(af.sum())} | com FCO tambem: {int((af & fco).sum())} "
      f"| perdidas por falta de FCO: {int((af & ~fco).sum())} ({(af & ~fco).sum()/af.sum():.1%})")
sem = p[af].groupby("CNPJ_CIA")["Fluxo_Caixa_Op_Ativo"].apply(lambda s: s.notna().sum() == 0)
nm = set(sem[sem].index)
print(f"Firmas com log_Ativo_Fixo mas NENHUM FCO: {len(nm)} | dessas, com fiscal != civil: "
      f"{len(nm & set(g.index[g['so_nao_jan1']]))}")
cob = p[af].groupby("CNPJ_CIA")["Fluxo_Caixa_Op_Ativo"].apply(lambda s: s.notna().mean())
print("Cobertura de FCO por firma (quantis):\n", cob.quantile([.05, .1, .25, .5]).round(2).to_string())