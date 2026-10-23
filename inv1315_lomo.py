"""
inv1315_lomo.py
Leave-one-meeting-out do gradiente de caixa na janela 2013Q2-2015Q2.

Para cada reuniao do COPOM dentro da janela, remove o choque dela do trimestre
correspondente (Shock_novo = Shock - choque_reuniao * escala), reconstroi
Shock_L1 e reestima o gradiente da janela (Ch_Caixa + Ch_Caixa_P1), mantendo o
controle de 2021 (P2), exatamente como em eventos.ajusta.

APROXIMACAO: a ortogonalizacao contra VIX e risco fiscal NAO e refeita. Na janela,
a diferenca entre Shock_t e Choque_HFI e de no maximo ~0,05.

Requer, na raiz: placebos.py, eventos.py e os CSVs gerados por
inv1315_choques_janela.py (inv1315_choques_reunioes.csv,
inv1315_choques_trimestres.csv). NAO modifica nenhum arquivo existente.

Uso:  python inv1315_lomo.py
Saida: inv1315_lomo.csv
"""
import math
import numpy as np
import pandas as pd

import placebos as pb
from placebos import build, fit_full, G, G0, S0, ZMAP, rot
from eventos import W2021, comb

AMOSTRAS_TESTE = ["Restrita", "Total"]
HORIZONTES = [3, 5, 7]
ORD = lambda s: pd.Period(s, freq="Q").ordinal
JAN = list(range(ORD("2013Q2"), ORD("2015Q2") + 1))
W1 = np.isin(G, JAN).astype(float)


def ajusta_S(am, h, Ws, S):
    """Copia de eventos.ajusta, mas com vetor de choque S (e Shock_L1 = lag de S)."""
    SL1 = np.r_[np.nan, S[:-1]]
    d, x = build(am, h, S=S, SL1=SL1)
    if d is None:
        return None
    ti = d["t"].to_numpy() - G0
    for nm, W in Ws:
        d[nm] = W[ti]
    d = d.dropna(subset=[nm for nm, _ in Ws]).copy()
    if any(d[nm].abs().sum() == 0 for nm, _ in Ws):
        return None

    def monta(com_shock):
        ex = []
        for nm, _ in Ws:
            ex.append(nm)
            for v in x:
                if v in ZMAP:
                    d[f"{v}_{nm}"] = d[v] * d[nm]
                    ex.append(f"{v}_{nm}")
            if com_shock and d.loc[d[nm] != 0, "t"].nunique() >= 2:
                d[f"Shock_{nm}"] = d["Shock"] * d[nm]
                ex.append(f"Shock_{nm}")
        return ex

    for com in (True, False):
        try:
            return fit_full(d, x, h, extra=monta(com)), d
        except ValueError:
            continue
    return None


def estima(am, h, S):
    out = ajusta_S(am, h, [("P1", W1), ("P2", W2021)], S)
    if out is None:
        return None
    r, _ = out
    return comb(r, "Ch_Caixa", "Ch_Caixa_P1")


def escala_choque():
    """Slope de S0 (usado nas PLs) sobre Shock_t do painel, para checar a unidade."""
    tri = pd.read_csv("inv1315_choques_trimestres.csv")
    o = tri["Trimestre"].map(ORD).to_numpy() - G0
    ok = (o >= 0) & (o < len(G))
    o, st = o[ok], tri["Shock_t"].to_numpy()[ok]
    s0 = S0[o]
    m = np.isfinite(s0) & np.isfinite(st)
    slope = np.polyfit(st[m], s0[m], 1)[0]
    print(f"[escala] S0 ~ {slope:.4f} * Shock_t | max|S0 - Shock_t| = "
          f"{np.max(np.abs(s0[m] - st[m])):.6f} em {m.sum()} trimestres")
    return slope


def main():
    escala = escala_choque()
    reu = pd.read_csv("inv1315_choques_reunioes.csv", parse_dates=["reuniao"])
    reu["ord"] = reu["Trimestre"].map(ORD)
    reu = reu[reu["ord"].isin(JAN)].reset_index(drop=True)

    linhas = []
    for am in AMOSTRAS_TESTE:
        for h in HORIZONTES:
            ref = estima(am, h, S0)
            if ref is None:
                print(f"[pula ref] {am} h={h}")
                continue
            b_ref, se_ref, p_ref = ref
            linhas.append(dict(amostra=am, h=h, reuniao_removida="nenhuma", trimestre="",
                               choque_reuniao=0.0, shock_antes=np.nan, shock_depois=np.nan,
                               beta=b_ref, se=se_ref, p=p_ref, delta_vs_ref=0.0))
            for _, m in reu.iterrows():
                S = S0.copy()
                i = int(m["ord"]) - G0
                antes = S[i]
                S[i] = antes - m["choque"] * escala
                res = estima(am, h, S)
                if res is None:
                    print(f"[pula] {am} h={h} {m['reuniao'].date()}")
                    continue
                b, se, p = res
                linhas.append(dict(amostra=am, h=h,
                                   reuniao_removida=str(m["reuniao"].date()),
                                   trimestre=m["Trimestre"], choque_reuniao=m["choque"],
                                   shock_antes=antes, shock_depois=S[i],
                                   beta=b, se=se, p=p, delta_vs_ref=b - b_ref))
            print(f"ok {am} h={h}")

    df = pd.DataFrame(linhas)
    df.to_csv("inv1315_lomo.csv", index=False)

    print("\nBASELINE (esperado ~ -0.0926 Restrita, ~ -0.2399 Total em h=3):")
    print(df[(df.reuniao_removida == "nenhuma") & (df.h == 3)][["amostra", "beta"]]
          .to_string(index=False))

    fmt = lambda v: f"{v:.4f}"
    for (am, h), g in df.groupby(["amostra", "h"]):
        print(f"\n=== {am}, h={h} (ordenado por delta_vs_ref) ===")
        print(g.sort_values("delta_vs_ref")
                [["reuniao_removida", "trimestre", "choque_reuniao", "shock_antes",
                  "shock_depois", "beta", "se", "p", "delta_vs_ref"]]
                .to_string(index=False, float_format=fmt))

    print("\n=== Resumo ===")
    r = []
    for (am, h), g in df[df.reuniao_removida != "nenhuma"].groupby(["amostra", "h"]):
        ref = df[(df.amostra == am) & (df.h == h) & (df.reuniao_removida == "nenhuma")].iloc[0]
        r.append(dict(amostra=am, h=h, beta_ref=ref.beta, beta_min=g.beta.min(),
                      beta_max=g.beta.max(), n_p_acima_10pct=int((g.p > 0.10).sum()),
                      n_sinal_oposto=int((np.sign(g.beta) != np.sign(ref.beta)).sum()),
                      n_reunioes=len(g)))
    print(pd.DataFrame(r).to_string(index=False, float_format=fmt))
    print("\nSalvo: inv1315_lomo.csv")


if __name__ == "__main__":
    main()