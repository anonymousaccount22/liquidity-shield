"""
carry_00_contagem_N.py -- reconcilia N de firmas/obs/trimestres por amostra, spec e horizonte.
NAO estima regressao, NAO altera arquivos. Replica o preparo de projecoes_locais.py.
"""
import numpy as np
import pandas as pd

ARQ = "painel_regressao_q_bp.csv"
HS_CONTAGEM = [-4, 0, 3, 5, 7]
WINSOR_P = (0.01, 0.99)


def zscore(s):
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def winsor(s, lo=WINSOR_P[0], hi=WINSOR_P[1]):
    s = pd.Series(s)
    return s if s.notna().sum() == 0 else s.clip(s.quantile(lo), s.quantile(hi))


p = pd.read_csv(ARQ, dtype={"CNPJ_CIA": str})
p["t"] = pd.PeriodIndex(p["Trimestre"], freq="Q").asi8.astype("int64")
p = p.sort_values(["CNPJ_CIA", "t"]).reset_index(drop=True)
p["log_Ativo_Fixo"] = p["log_Ativo_Fixo"].where(np.isfinite(p["log_Ativo_Fixo"]))

print(f"Painel bruto: {len(p)} linhas | {p['CNPJ_CIA'].nunique()} CNPJs | "
      f"t de {pd.Period(ordinal=int(p['t'].min()), freq='Q')} a {pd.Period(ordinal=int(p['t'].max()), freq='Q')}")
print(f"Shock_t nao nulo em {p.groupby('t')['Shock_t'].first().notna().sum()} trimestres")

for v in ["Cash_Ratio", "Alavancagem", "Tobin_Q", "Tamanho", "Fluxo_Caixa_Op_Ativo"]:
    p[v] = winsor(p[v])
p = p.drop(columns=[c for c in ("Alavancagem_L1", "Tobin_Q_L1", "Fluxo_Caixa_Op_Ativo_L1") if c in p.columns])
for v in ["Alavancagem", "Tobin_Q", "Fluxo_Caixa_Op_Ativo"]:
    lag = p[["CNPJ_CIA", "t", v]].copy()
    lag["t"] += 1
    p = p.merge(lag.rename(columns={v: v + "_L1"}), on=["CNPJ_CIA", "t"], how="left")

tri = p.groupby("t")["Shock_t"].first().astype(float).rename("Shock")   # pp, sem reescala
p = p.merge(tri, left_on="t", right_index=True, how="left")
tri_c = tri.reindex(np.arange(tri.index.min(), tri.index.max() + 1))
p = p.merge(tri_c.shift(1).rename("Shock_L1"), left_on="t", right_index=True, how="left")

for v in ["Cash_Ratio", "Alavancagem_L1", "Tobin_Q_L1", "Tamanho", "Fluxo_Caixa_Op_Ativo_L1"]:
    p[v + "_z"] = p.groupby("t")[v].transform(zscore)
p["Ch_Caixa"] = p["Shock"] * p["Cash_Ratio_z"]
p["Ch_Alav"] = p["Shock"] * p["Alavancagem_L1_z"]
p["Ch_Q"] = p["Shock"] * p["Tobin_Q_L1_z"]
p["Ch_Tam"] = p["Shock"] * p["Tamanho_z"]
p["Ch_FCO"] = p["Shock"] * p["Fluxo_Caixa_Op_Ativo_L1_z"]
p["in_restrita"] = p["Tobin_Q_L1_z"].notna()

T_ALL = np.arange(p["t"].min(), p["t"].max() + 1)
piv = p.pivot(index="CNPJ_CIA", columns="t", values="log_Ativo_Fixo").reindex(columns=T_ALL)


def serie_y(h):
    return (piv.shift(-h, axis=1) - piv.shift(1, axis=1)).stack().rename("y").reset_index()


def xs(com_q, spec):
    ctrl = ["Cash_Ratio_z", "Alavancagem_L1_z"] + (["Tobin_Q_L1_z"] if com_q else []) + \
           ["Tamanho_z", "Fluxo_Caixa_Op_Ativo_L1_z"]
    ints = ["Ch_Caixa", "Ch_Alav", "Ch_Tam", "Ch_FCO"] + (["Ch_Q"] if com_q else [])
    if spec == "ctrl_int":
        return ["Shock"] + ints + ["Shock_L1"] + ctrl
    return ints + ctrl            # setor_t (sem Shock_L1)


AMOSTRAS = {"Total": (p, False), "Restrita": (p[p["in_restrita"]], True),
            "Restrita_semQ": (p[p["in_restrita"]], False)}
linhas = []
for am, (dados, com_q) in AMOSTRAS.items():
    for spec in ("ctrl_int", "setor_t"):
        x = xs(com_q, spec)
        for h in HS_CONTAGEM:
            d = dados.merge(serie_y(h), on=["CNPJ_CIA", "t"], how="inner")
            d = d.dropna(subset=["y"] + x + ["SETOR_ATIV"])
            linhas.append(dict(amostra=am, spec=spec, h=h, obs=len(d),
                               firmas=d["CNPJ_CIA"].nunique(), trimestres=d["t"].nunique(),
                               firmas_no_painel_da_amostra=dados["CNPJ_CIA"].nunique()))
out = pd.DataFrame(linhas)
print(out.to_string(index=False))
out.to_csv("carry_00_contagem_N.csv", index=False)