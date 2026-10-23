"""
carry_04_setorial.py -- a inversao de 2013-15 se concentra em alguns setores?
Reaproveita placebos.py/eventos.py. Nao altera nada.
Uso: python carry_04_setorial.py
"""
import numpy as np
import pandas as pd

import placebos as pb
from placebos import build, fit_full, G, HS, AMOSTRAS_TESTE, ZMAP
from eventos import comb, W2021

# Preencha DEPOIS de ver a lista de setores impressa, e ANTES de ver os resultados.
# Trechos de texto (minusculas) que identificam setores expostos. Ex.: ["constru", "petr", "miner"]
EXPOSTOS_SUBSTR = []
MIN_FIRMAS_SETOR = 8     # so exclui setores com pelo menos isso de firmas


def ords(ini, fim):
    return set(range(pd.Period(ini, freq="Q").ordinal, pd.Period(fim, freq="Q").ordinal + 1))


Q1315 = ords("2013Q2", "2015Q2")
W1 = np.isin(G, list(Q1315)).astype(float)


def ajusta_filtro(am, h, excluir=()):
    d, x = build(am, h)
    if d is None:
        return None
    if len(excluir):
        d = d[~d["SETOR_ATIV"].isin(list(excluir))]
    d = d.copy()
    ti = d["t"].to_numpy() - G[0]
    d["P1"], d["P2"] = W1[ti], W2021[ti]
    if d["P1"].sum() == 0:
        return None
    extra = ["P1", "P2"]
    for nm in ("P1", "P2"):
        for v in x:
            if v in ZMAP:
                d[f"{v}_{nm}"] = d[v] * d[nm]
                extra.append(f"{v}_{nm}")
        d[f"Shock_{nm}"] = d["Shock"] * d[nm]
        extra.append(f"Shock_{nm}")
    try:
        r = fit_full(d, x, h, extra=extra)
    except Exception as e:
        print(f"  [falhou] {am} h={h} excl={list(excluir)[:3]}...: {e}")
        return None
    eb, sb, pv = comb(r, "Ch_Caixa", "Ch_Caixa_P1")
    return dict(beta_calmo=r.params["Ch_Caixa"], beta_2013_15=eb, se=sb, p_2013_15=pv,
                n=int(r.nobs), firmas=d["CNPJ_CIA"].nunique(),
                n_obs_2013_15=int(d["P1"].sum()))


if __name__ == "__main__":
    pb.checar_baseline()

    # --- lista de setores (para voce escolher o grupo exposto) ---
    d0, _ = build("Total", 3)
    tab = (d0.groupby("SETOR_ATIV")
              .agg(firmas=("CNPJ_CIA", "nunique"), obs=("t", "size"),
                   obs_2013_15=("t", lambda s: int(s.isin(Q1315).sum())))
              .sort_values("firmas", ascending=False))
    print("\n=== Setores (amostra Total, h=3) ===")
    print(tab.to_string())
    tab.to_csv("carry_04_setores.csv")
    setores = [s for s in tab.index if tab.loc[s, "firmas"] >= MIN_FIRMAS_SETOR]

    rows = []
    for am in AMOSTRAS_TESTE:
        for h in HS:
            base = ajusta_filtro(am, h)
            if base:
                rows.append(dict(amostra=am, h=h, excluido="TODOS (referencia)", **base))
            for s in setores:
                r = ajusta_filtro(am, h, excluir=[s])
                if r:
                    rows.append(dict(amostra=am, h=h, excluido=s, **r))
            if EXPOSTOS_SUBSTR:
                grupo = [s for s in tab.index
                         if any(k in str(s).lower() for k in EXPOSTOS_SUBSTR)]
                r = ajusta_filtro(am, h, excluir=grupo)
                if r:
                    rows.append(dict(amostra=am, h=h,
                                     excluido="GRUPO EXPOSTO: " + "; ".join(map(str, grupo)), **r))
    out = pd.DataFrame(rows)
    # quanto muda o gradiente de 2013-15 ao excluir cada setor
    ref = out[out["excluido"] == "TODOS (referencia)"].set_index(["amostra", "h"])["beta_2013_15"]
    out["delta_vs_ref"] = out.apply(lambda r: r["beta_2013_15"] - ref[(r["amostra"], r["h"])], axis=1)
    print("\n=== Gradiente de 2013-15 excluindo setores ===")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out.to_csv("carry_04_setorial.csv", index=False)