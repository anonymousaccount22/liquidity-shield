"""
carry_03_composicao_caixa.py -- separa Aplic_Financeiras e Caixa_Equiv.
Reaproveita placebos.py. Nao altera nada. Uso: python carry_03_composicao_caixa.py
"""
import numpy as np
import pandas as pd

import placebos as pb
from placebos import build, fit_full, p, G, HS, AMOSTRAS_TESTE
from eventos import comb
from carry_02_sem2021 import regime_alto, Q2021

def zscore(s):
    sd = s.std()
    return (s - s.mean()) / sd if np.isfinite(sd) and sd > 0 else pd.Series(np.nan, index=s.index)

# --- razoes defasadas em t-1 (mesma convencao do Cash_Ratio) ---
q = p[["CNPJ_CIA", "t", "Caixa_Equiv", "Aplic_Financeiras", "Ativo_Total"]].copy()
q["Aplic_Financeiras"] = q["Aplic_Financeiras"].fillna(0.0)
q["r_eq"] = q["Caixa_Equiv"] / q["Ativo_Total"]
q["r_ap"] = q["Aplic_Financeiras"] / q["Ativo_Total"]
lag = q[["CNPJ_CIA", "t", "r_eq", "r_ap"]].copy()
lag["t"] += 1                                       # valor de t-1 vira linha t
lag = lag.rename(columns={"r_eq": "Equiv_L1", "r_ap": "Aplic_L1"})
lag["Equiv_L1"] = lag["Equiv_L1"].clip(lag["Equiv_L1"].quantile(.01), lag["Equiv_L1"].quantile(.99))
lag["Aplic_L1"] = lag["Aplic_L1"].clip(lag["Aplic_L1"].quantile(.01), lag["Aplic_L1"].quantile(.99))
lag["Equiv_z"] = lag.groupby("t")["Equiv_L1"].transform(zscore)
lag["Aplic_z"] = lag.groupby("t")["Aplic_L1"].transform(zscore)
LAG = lag[["CNPJ_CIA", "t", "Equiv_L1", "Aplic_L1", "Equiv_z", "Aplic_z"]]

print(f"Fração de obs. com Aplic_Financeiras = 0 (em t-1): {(LAG['Aplic_L1'] <= 0).mean():.1%}")
print("Correlação Equiv_z x Aplic_z:", round(LAG[["Equiv_z", "Aplic_z"]].corr().iloc[0, 1], 3))


def estima(am, h, excluir=frozenset(), regime=None):
    d, x = build(am, h, excl_q=sorted(excluir))
    if d is None:
        return None
    d = d.merge(LAG, on=["CNPJ_CIA", "t"], how="left").dropna(subset=["Equiv_z", "Aplic_z"]).copy()
    d["Ch_Equiv"] = d["Shock"] * d["Equiv_z"]
    d["Ch_Aplic"] = d["Shock"] * d["Aplic_z"]
    # troca Cash_Ratio_z/Ch_Caixa pelas duas partes; demais controles iguais
    x2 = [v for v in x if v not in ("Cash_Ratio_z", "Ch_Caixa")] + \
         ["Equiv_z", "Aplic_z", "Ch_Equiv", "Ch_Aplic"]
    if regime is not None:                         # regime: 'alto' ou 'baixo'
        W, _ = regime_alto(set(excluir))
        d["Alto"] = W[d["t"].to_numpy() - G[0]]
        d = d.dropna(subset=["Alto"])
        d = d[d["Alto"] == (1.0 if regime == "alto" else 0.0)]
    r = fit_full(d, x2, h)
    return dict(amostra=am, h=h, regime=regime or "todos",
                excluidos="sem 2021" if excluir else "nenhum",
                b_equiv=r.params["Ch_Equiv"], p_equiv=r.pvalues["Ch_Equiv"],
                b_aplic=r.params["Ch_Aplic"], p_aplic=r.pvalues["Ch_Aplic"],
                n=int(r.nobs), firmas=d["CNPJ_CIA"].nunique(), T=d["t"].nunique())


if __name__ == "__main__":
    pb.checar_baseline()
    rows = []
    for am in AMOSTRAS_TESTE:
        for h in HS:
            for reg in (None, "baixo", "alto"):
                for ex in (frozenset(), frozenset(Q2021)):
                    r = estima(am, h, ex, reg)
                    if r:
                        rows.append(r)
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out.to_csv("carry_03_composicao_caixa.csv", index=False)