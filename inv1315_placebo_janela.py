"""
inv1315_placebo_janela.py
Placebo temporal: desliza uma janela de 9 trimestres (mesmo tamanho de
2013Q2-2015Q2) por todo o painel e estima, em cada posicao, o gradiente de
caixa dentro da janela (Ch_Caixa + Ch_Caixa_P1), mantendo o controle de 2021.
Pergunta: o beta de 2013-15 e raro na distribuicao de janelas placebo?

Uso:  python inv1315_placebo_janela.py
Saidas: inv1315_placebo_janela.csv, inv1315_placebo_janela.png
Depende de: placebos.py (build/G), eventos.py (ajusta/comb/W2021).
NAO modifica nenhum arquivo existente.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from placebos import G
from eventos import W2021, Q2021, ajusta, comb

AMOSTRAS_TESTE = ["Restrita", "Total"]
HORIZONTES = [3, 5, 7]
TAM_JANELA = 9  # 2013Q2..2015Q2 = 9 trimestres
ORD = lambda s: pd.Period(s, freq="Q").ordinal
REAL = set(range(ORD("2013Q2"), ORD("2015Q2") + 1))
OUT_CSV = "inv1315_placebo_janela.csv"
OUT_PNG = "inv1315_placebo_janela.png"


def beta_janela(am, h, trimestres):
    """Gradiente de caixa dentro da janela (soma Ch_Caixa + Ch_Caixa_P1)."""
    w1 = np.isin(G, list(trimestres)).astype(float)
    try:
        out = ajusta(am, h, [("P1", w1), ("P2", W2021)])
    except Exception as e:  # janela sem variacao suficiente, colinearidade etc.
        return None, f"erro: {e}"
    if out is None:
        return None, "ajusta devolveu None"
    r, _ = out
    try:
        b, se, p = comb(r, "Ch_Caixa", "Ch_Caixa_P1")
    except Exception as e:
        return None, f"erro comb: {e}"
    return (b, se, p), ""


def rodar():
    ords = np.unique(np.asarray(G))
    q2021 = set(Q2021)
    linhas = []
    for am in AMOSTRAS_TESTE:
        for h in HORIZONTES:
            for ini in range(int(ords.min()), int(ords.max()) - TAM_JANELA + 2):
                jan = set(range(ini, ini + TAM_JANELA))
                if jan & q2021:      # colinear com o controle de 2021
                    continue
                res, msg = beta_janela(am, h, jan)
                if res is None:
                    print(f"[pula] {am} h={h} {pd.Period(ini, 'Q')}: {msg}")
                    continue
                b, se, p = res
                linhas.append(dict(
                    amostra=am, h=h,
                    ini=str(pd.Period(ordinal=ini, freq="Q")),
                    fim=str(pd.Period(ordinal=ini + TAM_JANELA - 1, freq="Q")),
                    ini_ord=ini, beta=b, se=se, p=p,
                    sobrepoe_real=bool(jan & REAL),
                    e_real=(jan == REAL),
                ))
            print(f"ok {am} h={h}")
    df = pd.DataFrame(linhas)
    df.to_csv(OUT_CSV, index=False)
    return df


def resumo(df):
    """Posicao do beta real na distribuicao das janelas placebo."""
    linhas = []
    for (am, h), g in df.groupby(["amostra", "h"]):
        real = g[g.e_real]
        if real.empty:
            continue
        b0 = float(real.beta.iloc[0])
        plac = g[~g.sobrepoe_real]  # placebo limpo: sem sobreposicao com a janela real
        linhas.append(dict(
            amostra=am, h=h, beta_real=b0, n_placebo=len(plac),
            pct_menor_ou_igual=float((plac.beta <= b0).mean()),
            pct_abs_maior_ou_igual=float((plac.beta.abs() >= abs(b0)).mean()),
            media_placebo=float(plac.beta.mean()),
            dp_placebo=float(plac.beta.std()),
            min_placebo=float(plac.beta.min()),
            max_placebo=float(plac.beta.max()),
        ))
    rs = pd.DataFrame(linhas)
    rs.to_csv("inv1315_placebo_janela_resumo.csv", index=False)
    print(rs.to_string(index=False))
    return rs


def grafico(df):
    fig, axes = plt.subplots(len(AMOSTRAS_TESTE), len(HORIZONTES),
                             figsize=(4.2 * len(HORIZONTES), 3.4 * len(AMOSTRAS_TESTE)),
                             squeeze=False)
    for i, am in enumerate(AMOSTRAS_TESTE):
        for j, h in enumerate(HORIZONTES):
            ax = axes[i][j]
            g = df[(df.amostra == am) & (df.h == h)]
            plac = g[~g.sobrepoe_real]
            real = g[g.e_real]
            ax.hist(plac.beta, bins=15, color="0.7")
            if not real.empty:
                ax.axvline(float(real.beta.iloc[0]), color="crimson", lw=2)
            ax.axvline(0, color="k", lw=0.6)
            ax.set_title(f"{am}, h={h}")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df = rodar()
    # Checagem de baseline (briefing): Restrita h=3 ~ -0.0926 ; Total h=3 ~ -0.2399
    base = df[df.e_real & (df.h == 3)][["amostra", "beta"]]
    print("\nBASELINE (esperado ~ -0.0926 Restrita, ~ -0.2399 Total):")
    print(base.to_string(index=False))
    resumo(df)
    grafico(df)