"""
carry_01_regimes.py -- gradiente de caixa por regime de taxa real defasada.
Reaproveita placebos.py e eventos.py (mesma pasta). Nao altera nada existente.
Requer carry_juros_trimestral.csv (rode carry_00b_juros.py antes).
Uso: python carry_01_regimes.py
"""
import math
import time

import numpy as np
import pandas as pd

import placebos as pb
from placebos import build, fit_full, G, V, HS, AMOSTRAS_TESTE, rot
from eventos import ajusta, comb

JUROS = "carry_juros_trimestral.csv"


def ords(ini, fim):
    return set(range(pd.Period(ini, freq="Q").ordinal, pd.Period(fim, freq="Q").ordinal + 1))


Q1315 = ords("2013Q2", "2015Q2")      # mesma janela principal do ciclos.py
Q2021 = ords("2021Q1", "2021Q4")

# ---------- taxa real defasada na grade G ----------
j = pd.read_csv(JUROS)
o = [pd.Period(s, freq="Q").ordinal for s in j["trimestre"]]
RR = pd.Series(j["real_L1"].to_numpy(float), index=o).reindex(G).to_numpy(float)
ok = V[np.isfinite(RR[V])]                       # trimestres com choque E taxa real
print(f"Trimestres com choque valido: {len(V)} | com taxa real tambem: {len(ok)}")

med = np.median(RR[ok])
t1, t2 = np.quantile(RR[ok], [1 / 3, 2 / 3])
nan = ~np.isfinite(RR)
W_ALTO = np.where(nan, np.nan, (RR >= med).astype(float))                 # mediana
W_MED = np.where(nan, np.nan, ((RR >= t1) & (RR < t2)).astype(float))     # tercis
W_ALTO3 = np.where(nan, np.nan, (RR >= t2).astype(float))
print(f"Taxa real defasada: mediana = {med:.2f} | tercis = {t1:.2f} / {t2:.2f}")

# ---------- 0. diagnostico: o regime e so um rotulo dos ciclos? ----------
rows = []
for nome, W in {"alto (>= mediana)": W_ALTO, "baixo (< mediana)": 1 - W_ALTO,
                "tercil 1 (baixo)": np.where(nan, np.nan, (RR < t1).astype(float)),
                "tercil 2": W_MED, "tercil 3 (alto)": W_ALTO3}.items():
    qs = [int(G[i]) for i in ok if W[i] == 1]
    rows.append(dict(regime=nome, n_trim=len(qs),
                     n_em_2013_15=len(set(qs) & Q1315), n_em_2021=len(set(qs) & Q2021),
                     taxa_real_media=float(np.mean([RR[q - G[0]] for q in qs])),
                     trimestres=" ".join(rot(q) for q in qs)))
diag = pd.DataFrame(rows)
print("\n=== Composicao dos regimes (checar sobreposicao com os ciclos) ===")
print(diag.drop(columns="trimestres").to_string(index=False, float_format=lambda v: f"{v:.2f}"))
diag.to_csv("carry_01_diagnostico.csv", index=False)


# ---------- 1. gradiente por regime (um modelo, dummy de regime) ----------
def split():
    rows, a = [], "Ch_Caixa"
    for am in AMOSTRAS_TESTE:
        for h in HS:
            # (a) mediana: base = taxa real baixa
            out = ajusta(am, h, [("P1", W_ALTO)])
            if out is not None:
                r, d = out
                eb, sb, pb_ = comb(r, a, "Ch_Caixa_P1")
                rows.append(dict(corte="mediana", amostra=am, h=h,
                                 beta_baixo=r.params[a], p_baixo=r.pvalues[a],
                                 beta_alto=eb, p_alto=pb_,
                                 dif_alto_menos_baixo=r.params["Ch_Caixa_P1"],
                                 p_dif=r.pvalues["Ch_Caixa_P1"],
                                 n=int(r.nobs), firmas=d["CNPJ_CIA"].nunique(),
                                 T=d["t"].nunique(), obs_regime_alto=int(d["P1"].sum())))
            # (b) tercis: base = tercil baixo
            out = ajusta(am, h, [("PM", W_MED), ("PA", W_ALTO3)])
            if out is not None:
                r, d = out
                bm, _, pm = comb(r, a, "Ch_Caixa_PM")
                ba, _, pa = comb(r, a, "Ch_Caixa_PA")
                b0 = r.params[a]
                rows.append(dict(corte="tercis", amostra=am, h=h,
                                 beta_baixo=b0, p_baixo=r.pvalues[a],
                                 beta_medio=bm, p_medio=pm, beta_alto=ba, p_alto=pa,
                                 dif_alto_menos_baixo=r.params["Ch_Caixa_PA"],
                                 p_dif=r.pvalues["Ch_Caixa_PA"],
                                 monotonico_decrescente=bool(b0 > bm > ba),
                                 monotonico_crescente=bool(b0 < bm < ba),
                                 n=int(r.nobs), firmas=d["CNPJ_CIA"].nunique(), T=d["t"].nunique()))
    df = pd.DataFrame(rows)
    print("\n=== Gradiente por regime de taxa real defasada ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


# ---------- 2. dentro do regime: quanto dependem de 2013-15 e 2021? ----------
def subamostra():
    todos = [int(g) for g in G]
    regimes = {"alto": {int(G[i]) for i in ok if W_ALTO[i] == 1},
               "baixo": {int(G[i]) for i in ok if W_ALTO[i] == 0}}
    cfg = {"todos": set(), "sem 2013-15": Q1315, "sem 2021": Q2021, "sem ambos": Q1315 | Q2021}
    rows = []
    for rn, keep in regimes.items():
        for cn, ex in cfg.items():
            excl = [q for q in todos if (q not in keep) or (q in ex)]
            for am in AMOSTRAS_TESTE:
                for h in HS:
                    d, x = build(am, h, excl_q=excl)
                    if d is None:
                        rows.append(dict(regime=rn, excluidos=cn, amostra=am, h=h, obs="poucas obs"))
                        continue
                    r = fit_full(d, x, h)
                    rows.append(dict(regime=rn, excluidos=cn, amostra=am, h=h,
                                     coef=r.params["Ch_Caixa"], se=r.std_errors["Ch_Caixa"],
                                     p=r.pvalues["Ch_Caixa"], n=int(r.nobs),
                                     firmas=d["CNPJ_CIA"].nunique(), T=d["t"].nunique()))
    df = pd.DataFrame(rows)
    print("\n=== Ch_Caixa dentro de cada regime, excluindo ciclos ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


if __name__ == "__main__":
    pb.checar_baseline()
    for nome, f in {"split": split, "subamostra": subamostra}.items():
        t0 = time.time()
        f().to_csv(f"carry_01_{nome}.csv", index=False)
        print(f"[{nome}] {time.time() - t0:.0f}s -> carry_01_{nome}.csv")