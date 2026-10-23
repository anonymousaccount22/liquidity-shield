"""
setor_t.py -- Ch_Caixa com efeito fixo de firma + setor x trimestre

Requer placebos.py e eventos.py na MESMA pasta.
Uso:
    python setor_t.py               # exclusao + igualdade
    python setor_t.py exclusao
    python setor_t.py igualdade

Com FE setor x trimestre, Shock, Shock_L1 e qualquer dummy de periodo sao absorvidos:
so sobram as interacoes (Ch_*) e os controles z. Ch_Caixa passa a ser identificado
SOMENTE pela comparacao de firmas dentro do mesmo setor e trimestre.

[exclusao]  Ch_Caixa sob setor_t na amostra cheia, sem 2021, sem 2013-15, sem 2020+2021, sem 2013-15 e 2021.
[igualdade] Ch_Caixa, Ch_Caixa_P1 (2013Q2-2015Q2) e Ch_Caixa_P2 (2021) no mesmo modelo setor_t;
            testa P1 = P2.
Referencia de reproducao (seu output original, setor_t, Restrita, sem 2020/21):
            h5 = 0.0334 (p=0.13), h7 = 0.0228 (p=0.34).
"""
import math
import sys
import time
import warnings

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

import placebos as pb
from placebos import build, cov_kwargs, monta_specs, AMOSTRAS, G, HS, AMOSTRAS_TESTE
from eventos import comb, Q2021
from ciclos import trimestres, JANELAS, pval

Q1315 = trimestres(*JANELAS["2013Q2-2015Q2 (principal)"])
REF = {("Restrita", 5): 0.0334, ("Restrita", 7): 0.0228}


def x_setor(am):
    x = list(monta_specs(AMOSTRAS[am]["com_q"])["setor_t"]["x"])
    # Shock e Shock_L1 sao colineares com FE setor x trimestre; remove por seguranca
    return [v for v in x if v not in ("Shock", "Shock_L1")]


def fit_setor(d, x, h, extra=()):
    d = d.copy()
    d["setor_t"] = d["SETOR_ATIV"].astype(str) + "|" + d["t"].astype(str)
    dd = d.set_index(["CNPJ_CIA", "t"])
    other = pd.DataFrame({"setor_t": pd.factorize(dd["setor_t"])[0]}, index=dd.index)
    cols = list(x) + list(extra)
    faltam = [c for c in cols if c not in dd.columns]
    if faltam:
        raise KeyError(f"colunas ausentes em build(): {faltam}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PanelOLS(dd["y"], dd[cols], entity_effects=True, other_effects=other).fit(
            **cov_kwargs(h, d["t"].nunique()))


def checar_ref():
    print("[setor_t] x usado (Restrita):", x_setor("Restrita"))
    for (am, h), esp in REF.items():
        d, _ = build(am, h, excl_years=(2020, 2021))
        r = fit_setor(d, x_setor(am), h)
        cf = r.params["Ch_Caixa"]
        print(f"[baseline setor_t] {am} h={h} sem 2020/21: {cf:.4f} esperado={esp:.4f} "
              f"{'OK' if abs(cf - esp) < 1e-3 else '<<< NAO REPRODUZ (confira x e FE)'}")


def teste_exclusao():
    configs = {"cheia": {}, "sem 2021": dict(excl_q=tuple(sorted(Q2021))),
               "sem 2013-15": dict(excl_q=tuple(sorted(Q1315))),
               "sem 2020+2021": dict(excl_years=(2020, 2021)),
               "sem 2013-15 e 2021": dict(excl_q=tuple(sorted(Q1315 | Q2021)))}
    rows = []
    for am in AMOSTRAS_TESTE:
        for h in HS:
            for nome, kw in configs.items():
                d, _ = build(am, h, **kw)
                if d is None:
                    continue
                r = fit_setor(d, x_setor(am), h)
                rows.append(dict(amostra=am, h=h, excluidos=nome, coef=r.params["Ch_Caixa"],
                                 se=r.std_errors["Ch_Caixa"], p=r.pvalues["Ch_Caixa"],
                                 n=int(r.nobs), T=d["t"].nunique()))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


def teste_igualdade():
    W1 = np.isin(G, list(Q1315)).astype(float)
    W2 = np.isin(G, list(Q2021)).astype(float)
    a, b1, b2 = "Ch_Caixa", "Ch_Caixa_P1", "Ch_Caixa_P2"
    rows = []
    for am in AMOSTRAS_TESTE:
        for h in HS:
            d, _ = build(am, h)
            ti = d["t"].to_numpy() - pb.G0
            d["P1"], d["P2"] = W1[ti], W2[ti]
            x = x_setor(am)
            extra = []
            for v in x:
                if v in pb.ZMAP:                      # so as interacoes Ch_*
                    d[v + "_P1"], d[v + "_P2"] = d[v] * d["P1"], d[v] * d["P2"]
                    extra += [v + "_P1", v + "_P2"]
            r = fit_setor(d, x, h, extra=extra)
            e1, _, p1 = comb(r, a, b1)
            e2, _, p2 = comb(r, a, b2)
            dif = r.params[b1] - r.params[b2]
            var = r.cov.loc[b1, b1] + r.cov.loc[b2, b2] - 2 * r.cov.loc[b1, b2]
            rows.append(dict(amostra=am, h=h, beta_calmo=r.params[a], p_calmo=r.pvalues[a],
                             beta_ciclo1315=e1, p_ciclo=p1, beta_2021=e2, p_2021=p2,
                             dif=dif, se_dif=math.sqrt(var), p_igualdade=pval(dif / math.sqrt(var))))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


TESTES = {"exclusao": teste_exclusao, "igualdade": teste_igualdade}

if __name__ == "__main__":
    pb.checar_baseline()
    checar_ref()
    for nome in (sys.argv[1:] or list(TESTES)):
        t1 = time.time()
        print(f"\n=== {nome} ===")
        df = TESTES[nome]()
        df.to_csv(f"setor_t_{nome}.csv", index=False)
        print(f"[{nome}] {time.time() - t1:.0f}s -> setor_t_{nome}.csv")