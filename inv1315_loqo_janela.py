"""
inv1315_loqo_janela.py
Leave-one-quarter-out RESTRITO a janela 2013Q2-2015Q2.
Para cada trimestre q da janela: P1 = janela sem q, P3 = apenas q.
P3 absorve o gradiente proprio de q, entao o beta da janela passa a ser
identificado pelos outros 8 trimestres. Se um trimestre carrega a inversao,
o beta cai (em modulo) quando ele sai.

Uso:  python inv1315_loqo_janela.py
Saidas: inv1315_loqo_janela.csv, inv1315_loqo_janela.png
Depende de: placebos.py (G), eventos.py (ajusta/comb/W2021).
NAO modifica nenhum arquivo existente.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from placebos import G
from eventos import W2021, ajusta, comb

AMOSTRAS_TESTE = ["Restrita", "Total"]
HORIZONTES = [3, 5, 7]
ORD = lambda s: pd.Period(s, freq="Q").ordinal
JANELA = list(range(ORD("2013Q2"), ORD("2015Q2") + 1))
OUT_CSV = "inv1315_loqo_janela.csv"
OUT_PNG = "inv1315_loqo_janela.png"


def estima(am, h, ws, nomes_soma):
    try:
        out = ajusta(am, h, ws)
    except Exception as e:
        return None, f"erro: {e}"
    if out is None:
        return None, "ajusta devolveu None"
    r, _ = out
    try:
        return comb(r, *nomes_soma), ""
    except Exception as e:
        return None, f"erro comb: {e}"


def rodar():
    linhas = []
    w_full = np.isin(G, JANELA).astype(float)
    for am in AMOSTRAS_TESTE:
        for h in HORIZONTES:
            # referencia: janela completa
            res, msg = estima(am, h, [("P1", w_full), ("P2", W2021)],
                              ("Ch_Caixa", "Ch_Caixa_P1"))
            if res is None:
                print(f"[pula ref] {am} h={h}: {msg}")
                continue
            b_ref, se_ref, p_ref = res
            linhas.append(dict(amostra=am, h=h, sai="nenhum", beta=b_ref,
                               se=se_ref, p=p_ref, delta_vs_ref=0.0))
            for q in JANELA:
                w1 = np.isin(G, [x for x in JANELA if x != q]).astype(float)
                w3 = np.isin(G, [q]).astype(float)
                res, msg = estima(am, h,
                                  [("P1", w1), ("P2", W2021), ("P3", w3)],
                                  ("Ch_Caixa", "Ch_Caixa_P1"))
                if res is None:
                    print(f"[pula] {am} h={h} sai {pd.Period(ordinal=q, freq='Q')}: {msg}")
                    continue
                b, se, p = res
                linhas.append(dict(amostra=am, h=h,
                                   sai=str(pd.Period(ordinal=q, freq="Q")),
                                   beta=b, se=se, p=p, delta_vs_ref=b - b_ref))
            print(f"ok {am} h={h}")
    df = pd.DataFrame(linhas)
    df.to_csv(OUT_CSV, index=False)
    return df


def grafico(df):
    fig, axes = plt.subplots(len(AMOSTRAS_TESTE), len(HORIZONTES),
                             figsize=(4.4 * len(HORIZONTES), 3.4 * len(AMOSTRAS_TESTE)),
                             squeeze=False)
    for i, am in enumerate(AMOSTRAS_TESTE):
        for j, h in enumerate(HORIZONTES):
            ax = axes[i][j]
            g = df[(df.amostra == am) & (df.h == h)]
            ref = g[g.sai == "nenhum"]
            g = g[g.sai != "nenhum"]
            if g.empty:
                continue
            x = np.arange(len(g))
            ax.errorbar(x, g.beta, yerr=1.96 * g.se, fmt="o", color="0.3", capsize=2)
            if not ref.empty:
                ax.axhline(float(ref.beta.iloc[0]), color="crimson", lw=1.5)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(g.sai, rotation=60, fontsize=7)
            ax.set_title(f"{am}, h={h}")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df = rodar()
    ref3 = df[(df.sai == "nenhum") & (df.h == 3)][["amostra", "beta"]]
    print("\nBASELINE (esperado ~ -0.0926 Restrita, ~ -0.2399 Total):")
    print(ref3.to_string(index=False))
    print(df.to_string(index=False))
    grafico(df)