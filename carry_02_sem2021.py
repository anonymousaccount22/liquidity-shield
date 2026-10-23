"""
carry_02_sem2021.py -- a diferenca de gradiente entre regimes de taxa real
sobrevive sem 2021? Reaproveita placebos.py/eventos.py. Nao altera nada.
Requer carry_juros_trimestral.csv. Uso: python carry_02_sem2021.py
"""
import numpy as np
import pandas as pd

import placebos as pb
from placebos import build, fit_full, G, V, HS, AMOSTRAS_TESTE, ZMAP
from eventos import comb

JUROS = "carry_juros_trimestral.csv"
Q2021 = set(range(pd.Period("2021Q1", freq="Q").ordinal,
                  pd.Period("2021Q4", freq="Q").ordinal + 1))
Q1315 = set(range(pd.Period("2013Q2", freq="Q").ordinal,
                  pd.Period("2015Q2", freq="Q").ordinal + 1))

j = pd.read_csv(JUROS)
o = [pd.Period(s, freq="Q").ordinal for s in j["trimestre"]]
RR = pd.Series(j["real_L1"].to_numpy(float), index=o).reindex(G).to_numpy(float)
ok = V[np.isfinite(RR[V])]


def regime_alto(excluir):
    """Dummy de regime alto, com a mediana recalculada SEM os trimestres excluidos."""
    uso = [i for i in ok if int(G[i]) not in excluir]
    med = np.median(RR[uso])
    return np.where(np.isfinite(RR), (RR >= med).astype(float), np.nan), med


def diferenca(excluir, rotulo):
    W, med = regime_alto(excluir)
    rows = []
    for am in AMOSTRAS_TESTE:
        for h in HS:
            d, x = build(am, h, excl_q=sorted(excluir))
            if d is None:
                continue
            d["Alto"] = W[d["t"].to_numpy() - G[0]]
            d = d.dropna(subset=["Alto"]).copy()
            d["Shock_A"] = d["Shock"] * d["Alto"]
            extra = ["Alto", "Shock_A"]
            for v in x:
                if v in ZMAP:
                    d[v + "_A"] = d[v] * d["Alto"]
                    extra.append(v + "_A")
            r = fit_full(d, x, h, extra=extra)
            eb, sb, pa = comb(r, "Ch_Caixa", "Ch_Caixa_A")
            rows.append(dict(exclusao=rotulo, mediana_taxa_real=round(med, 2),
                             amostra=am, h=h,
                             beta_baixo=r.params["Ch_Caixa"], p_baixo=r.pvalues["Ch_Caixa"],
                             beta_alto=eb, p_alto=pa,
                             dif_alto_menos_baixo=r.params["Ch_Caixa_A"],
                             p_dif=r.pvalues["Ch_Caixa_A"],
                             n=int(r.nobs), firmas=d["CNPJ_CIA"].nunique(),
                             T=d["t"].nunique(), trim_alto=int(d.loc[d["Alto"] == 1, "t"].nunique())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pb.checar_baseline()
    out = pd.concat([diferenca(set(), "nenhuma (referencia)"),
                     diferenca(Q2021, "sem 2021"),
                     diferenca(Q1315, "sem 2013-15"),
                     diferenca(Q2021 | Q1315, "sem 2021 e 2013-15")], ignore_index=True)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out.to_csv("carry_02_sem2021.csv", index=False)