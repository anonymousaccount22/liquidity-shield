"""
eventos.py -- valor da liquidez TRIMESTRE A TRIMESTRE (sem janelas longas)

Requer placebos.py na MESMA pasta (importa as funcoes; nao roda os testes de la).
Uso:
    python eventos.py              # trimestre + epu
    python eventos.py trimestre
    python eventos.py epu

[trimestre] Para cada trimestre-base q estima beta_q(h): quanto firmas com mais caixa
(Cash_Ratio_z) ampliaram o ativo fixo entre t-1 e t+h, num UNICO modelo com efeito fixo de
firma e de trimestre, controles comuns (Alavancagem, Tamanho, FCO, Q) e Cash_Ratio_z x D_q
para cada trimestre. beta_q NAO depende do tamanho do choque no trimestre (que pode ser ~0).
Tambem testa: (i) se beta_q varia mais que o ruido (estatistica Q) e (ii) quanto do beta_q de
cada evento sobra depois de descontar a reta beta_q = a + b*Choque_q (resid_choque).
SE: cluster por firma (indicativo; nao captura dependencia entre firmas no mesmo trimestre).

[epu] Interage o bloco do choque com trimestres de EPU alto / EPU continuo (+ controle de 2021).
"""
import math
import sys
import time
import warnings

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy import stats

import placebos as pb
from placebos import build, fit_full, p, G, G0, S0, V, ZMAP, HS, AMOSTRAS_TESTE, rot

# Lista fixada ANTES de ver os resultados (um trimestre por evento).
EVENTOS_Q = {
    "2011Q3": "Crise da divida europeia / rebaixamento EUA",
    "2013Q2": "Manifestacoes de junho + taper tantrum",
    "2014Q1": "Inicio da Lava Jato",
    "2014Q4": "Eleicao presidencial 2014",
    "2015Q3": "Rebaixamento do Brasil (S&P)",
    "2016Q2": "Impeachment: admissibilidade no Senado",
    "2016Q3": "Impeachment: cassacao",
    "2017Q2": "JBS / Joesley",
    "2018Q2": "Greve dos caminhoneiros",
    "2018Q4": "Eleicao presidencial 2018",
    "2020Q1": "Pandemia: inicio",
    "2020Q2": "Pandemia: pico de incerteza",
    "2021Q4": "Maior choque de aperto (referencia)",
    "2022Q4": "Eleicao presidencial 2022",
}
Q2021 = set(range(pd.Period("2021Q1", freq="Q").ordinal, pd.Period("2021Q4", freq="Q").ordinal + 1))
W2021 = np.isin(G, list(Q2021)).astype(float)


# ====================== TRIMESTRE A TRIMESTRE ======================
def slopes(am, h):
    d, x = build(am, h)
    if d is None:
        return None
    ctrl = [v for v in x if v.endswith("_z") and v != "Cash_Ratio_z"]
    qs = sorted(d["t"].unique())
    cz, tt = d["Cash_Ratio_z"].to_numpy(), d["t"].to_numpy()
    idx = pd.MultiIndex.from_arrays([d["CNPJ_CIA"].to_numpy(), tt])
    X = pd.DataFrame({f"cq_{q}": cz * (tt == q) for q in qs}, index=idx)
    for c in ctrl:
        X[c] = d[c].to_numpy()
    y = pd.Series(d["y"].to_numpy(), index=idx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
            cov_type="clustered", cluster_entity=True)
    nq = d["t"].value_counts()
    rows = []
    for q in qs:
        nm = f"cq_{q}"
        rows.append(dict(amostra=am, h=h, t=int(q), trimestre=rot(q), choque=float(S0[q - G0]),
                         beta=r.params[nm], se=r.std_errors[nm], p=r.pvalues[nm], n_q=int(nq[q])))
    return pd.DataFrame(rows)


def resumo_e_eventos(df):
    res, ev = [], []
    for (am, h), g in df.groupby(["amostra", "h"]):
        g = g.reset_index(drop=True)
        b, se, s = g["beta"].to_numpy(), g["se"].to_numpy(), g["choque"].to_numpy()
        w = 1 / se ** 2
        bbar = np.sum(w * b) / np.sum(w)
        Q = np.sum(((b - bbar) / se) ** 2)
        inc, a0 = np.polyfit(s, b, 1)
        resid = b - (a0 + inc * s)
        sd_res = resid.std(ddof=2)
        res.append(dict(amostra=am, h=h, n_trim=len(g), beta_pond=bbar, dp_entre_trim=b.std(ddof=1),
                        se_medio=se.mean(), Q=Q, p_Q=stats.chi2.sf(Q, len(g) - 1),
                        corr_beta_choque=np.corrcoef(b, s)[0, 1], inclinacao_vs_choque=inc))
        for tri, nome in EVENTOS_Q.items():
            m = g.index[g["trimestre"] == tri]
            if len(m) == 0:
                continue
            i = m[0]
            outros = np.delete(b, i)
            ev.append(dict(amostra=am, h=h, trimestre=tri, evento=nome, choque=s[i], beta=b[i], se=se[i],
                           p=g.loc[i, "p"], pct_vs_outros=float(np.mean(outros < b[i])),
                           z_vs_outros=(b[i] - outros.mean()) / outros.std(ddof=1),
                           resid_choque=resid[i], z_resid=resid[i] / sd_res, n_q=g.loc[i, "n_q"]))
    return pd.DataFrame(res), pd.DataFrame(ev)


def teste_trimestre():
    partes = []
    for am in AMOSTRAS_TESTE:
        for h in HS:
            s = slopes(am, h)
            if s is not None:
                partes.append(s)
    det = pd.concat(partes, ignore_index=True)
    det.to_csv("eventos_trimestre_detalhe.csv", index=False)
    res, ev = resumo_e_eventos(det)
    fmt = lambda v: f"{v:.4f}"
    print("\n--- O valor do caixa varia entre trimestres? (Q grande / p_Q pequeno = varia alem do ruido) ---")
    print(res.to_string(index=False, float_format=fmt))
    res.to_csv("eventos_trimestre_resumo.csv", index=False)
    print("\n--- Trimestres de evento (z_vs_outros: vs demais trimestres; z_resid: apos descontar o choque) ---")
    print(ev.to_string(index=False, float_format=fmt))
    return ev


# ====================== EPU (trimestres, nao anos) ======================
def ajusta(am, h, Ws):
    """Ws = lista de (nome, vetor W na grade G). Interage o bloco do choque com cada W."""
    d, x = build(am, h)
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


def comb(r, a, b):
    est = r.params[a] + r.params[b]
    se = math.sqrt(r.cov.loc[a, a] + r.cov.loc[b, b] + 2 * r.cov.loc[a, b])
    return est, se, math.erfc(abs(est / se) / math.sqrt(2))


def acha_epu():
    cands = [c for c in p.columns if "epu" in c.lower()]
    if not cands:
        return None
    col = "EPU" if "EPU" in cands else cands[0]
    return col, p.groupby("t")[col].mean().reindex(G).to_numpy(float)


def teste_epu():
    achou = acha_epu()
    if achou is None:
        print("Nao achei coluna EPU em painel_regressao_q_bp.csv. Colunas:", list(p.columns)[:60])
        return pd.DataFrame()
    col, E = achou
    ok = V[np.isfinite(E[V])]
    Ez = np.where(np.isfinite(E), (E - E[ok].mean()) / E[ok].std(), np.nan)
    corte = np.quantile(E[ok], 0.75)
    Wd = np.where(np.isfinite(E), (E >= corte).astype(float), np.nan)
    tab = pd.DataFrame({"trimestre": [rot(G[i]) for i in ok], "EPU": E[ok],
                        "EPU_alto": Wd[ok], "choque": S0[ok]})
    tab.to_csv("eventos_epu_trimestres.csv", index=False)
    print(f"[EPU] coluna '{col}' | corr(EPU, choque) = {np.corrcoef(E[ok], S0[ok])[0, 1]:.3f}")
    print("[EPU] trimestres com EPU alto (top quartil):",
          ", ".join(tab.loc[tab["EPU_alto"] == 1, "trimestre"]))
    rows, a, b = [], "Ch_Caixa", "Ch_Caixa_P1"
    for nome, W in {"EPU alto (top quartil)": Wd, "EPU continuo (por 1 dp)": Ez}.items():
        for ctrl in (False, True):
            Ws = [("P1", W)] + ([("P2", W2021)] if ctrl else [])
            for am in AMOSTRAS_TESTE:
                for h in HS:
                    out = ajusta(am, h, Ws)
                    if out is None:
                        continue
                    r, d = out
                    est, se, pv = comb(r, a, b)
                    row = dict(regime=nome, amostra=am, h=h, controle_2021=ctrl,
                               beta_base=r.params[a], p_base=r.pvalues[a],
                               diferenca=r.params[b], se_dif=r.std_errors[b], p_dif=r.pvalues[b],
                               beta_regime=est, p_regime=pv)
                    if ctrl:
                        row.update(dif_2021=r.params["Ch_Caixa_P2"], p_dif_2021=r.pvalues["Ch_Caixa_P2"])
                    rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


TESTES = {"trimestre": teste_trimestre, "epu": teste_epu}

if __name__ == "__main__":
    pb.checar_baseline()
    for nome in (sys.argv[1:] or list(TESTES)):
        t1 = time.time()
        print(f"\n=== {nome} ===")
        df = TESTES[nome]()
        df.to_csv(f"eventos_{nome}.csv", index=False)
        print(f"[{nome}] {time.time() - t1:.0f}s -> eventos_{nome}.csv")