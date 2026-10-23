"""
ciclos.py -- heterogeneidade de sinal de Ch_Caixa entre ciclos de aperto

Requer placebos.py e eventos.py na MESMA pasta (reaproveita build, fit_full, ajusta, comb).
Uso:
    python ciclos.py              # igualdade + exclusoes
    python ciclos.py igualdade
    python ciclos.py exclusao

[igualdade] UM modelo com Ch_Caixa (base = trimestres fora das duas janelas), Ch_Caixa_P1
(dummy da janela 2013-15) e Ch_Caixa_P2 (dummy de 2021), mais os mesmos blocos para as demais
interacoes e Shock_P1/Shock_P2. Testa H0: Ch_Caixa_P1 = Ch_Caixa_P2 (Driscoll-Kraay do fit_full).
Janela principal = 2013Q2-2015Q2 (a que voce ja usou); as outras sao sensibilidade.

[exclusao] Ch_Caixa (ctrl_int) na amostra cheia, sem 2013-15, sem 2021, e sem os dois.
Atencao: a exclusao e pelo trimestre-base t (como o EXCLUIR_ANOS original). O desfecho de
firmas observadas antes da janela ainda pode cair dentro dela em h>0.
"""
import math
import sys
import time

import numpy as np
import pandas as pd

import placebos as pb
from placebos import build, fit_full, G, HS, AMOSTRAS_TESTE
from eventos import ajusta, comb, W2021, Q2021

JANELAS = {
    "2013Q2-2015Q2 (principal)": ("2013Q2", "2015Q2"),
    "2013Q2-2014Q4 (sens.)": ("2013Q2", "2014Q4"),
    "2013Q1-2015Q4 (sens.)": ("2013Q1", "2015Q4"),
}


def trimestres(ini, fim):
    return set(range(pd.Period(ini, freq="Q").ordinal, pd.Period(fim, freq="Q").ordinal + 1))


def pval(z):
    return math.erfc(abs(z) / math.sqrt(2))


# ---------------- 1. teste de igualdade entre ciclos ----------------
def teste_igualdade():
    rows = []
    a, b1, b2 = "Ch_Caixa", "Ch_Caixa_P1", "Ch_Caixa_P2"
    for nome, (ini, fim) in JANELAS.items():
        W1 = np.isin(G, list(trimestres(ini, fim))).astype(float)
        for am in AMOSTRAS_TESTE:
            for h in HS:
                out = ajusta(am, h, [("P1", W1), ("P2", W2021)])
                if out is None:
                    continue
                r, d = out
                e1, s1, p1 = comb(r, a, b1)     # beta no ciclo 2013-15
                e2, s2, p2 = comb(r, a, b2)     # beta em 2021
                dif = r.params[b1] - r.params[b2]
                var = r.cov.loc[b1, b1] + r.cov.loc[b2, b2] - 2 * r.cov.loc[b1, b2]
                rows.append(dict(
                    janela=nome, amostra=am, h=h,
                    beta_calmo=r.params[a], p_calmo=r.pvalues[a],
                    beta_ciclo=e1, p_ciclo=p1,
                    beta_2021=e2, p_2021=p2,
                    dif_ciclo_menos_2021=dif, se_dif=math.sqrt(var), p_igualdade=pval(dif / math.sqrt(var)),
                    n_obs_ciclo=int(d["P1"].sum()), n_obs_2021=int(d["P2"].sum())))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


# ---------------- 2. exclusao de ciclos na amostra cheia ----------------
def teste_exclusao():
    q1315 = trimestres(*JANELAS["2013Q2-2015Q2 (principal)"])
    configs = {"cheia": (), "sem 2013-15": tuple(sorted(q1315)),
               "sem 2021": tuple(sorted(Q2021)),
               "sem 2013-15 e 2021": tuple(sorted(q1315 | Q2021))}
    rows = []
    for am in AMOSTRAS_TESTE:
        for h in HS:
            for nome, ex in configs.items():
                d, x = build(am, h, excl_q=ex)
                if d is None:
                    continue
                r = fit_full(d, x, h)
                rows.append(dict(amostra=am, h=h, excluidos=nome, coef=r.params["Ch_Caixa"],
                                 se=r.std_errors["Ch_Caixa"], p=r.pvalues["Ch_Caixa"],
                                 n=int(r.nobs), T=d["t"].nunique()))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


TESTES = {"igualdade": teste_igualdade, "exclusao": teste_exclusao}

if __name__ == "__main__":
    pb.checar_baseline()
    for nome in (sys.argv[1:] or list(TESTES)):
        t1 = time.time()
        print(f"\n=== {nome} ===")
        df = TESTES[nome]()
        df.to_csv(f"ciclos_{nome}.csv", index=False)
        print(f"[{nome}] {time.time() - t1:.0f}s -> ciclos_{nome}.csv")