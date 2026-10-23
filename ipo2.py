"""
ipo2.py -- o efeito de IPO / caixa recem-captado e real ou e perda de amostra?

Requer ipo.py, placebos.py, eventos.py e ciclos.py na MESMA pasta.
Rode UM teste por vez, nesta ordem:
    python ipo2.py pre      # pre-tendencia: Ch_Caixa sobre a MEDIA do desfecho em h=-4,-3,-2
    python ipo2.py inter    # gradiente de caixa das entrantes vs incumbentes (todos os periodos e em 2021)
    python ipo2.py aleat    # exclusao ALEATORIA de firmas (mesmo n por setor) vs exclusao das entrantes
    python ipo2.py boot     # bootstrap de blocos de trimestres na amostra SEM as entrantes
    python ipo2.py          # todos

"Entrantes" = firmas cujo primeiro trimestre no painel cai em 2019Q1-2021Q4 (mesma proxy do ipo.py).

[pre]   y_pre = media de y(h=-4), y(h=-3), y(h=-2) (cada y = log AF(t+h) - log AF(t-1)). Um unico
        coeficiente de Ch_Caixa, com DK de 5 lags. Como y(h<0) e MENOS o crescimento passado,
        coeficiente negativo = firmas com mais caixa ja cresciam mais antes do choque.
[inter] Modelo A: Ch_Caixa, Ch_Caixa x Entrante (+ Shock x Entrante e Cash_z x Entrante como controles).
        Modelo B: o mesmo com os regimes 2013-15 e 2021 de ciclos.py e termos extras para entrantes em 2021.
[aleat] Sorteia, N_ALEAT vezes, firmas NAO entrantes na mesma quantidade por setor das entrantes,
        exclui e reestima Ch_Caixa. p = P(nulo <= coeficiente sem entrantes). Cuidado: as entrantes
        tem poucas observacoes por firma, entao o sorteio remove mais observacoes que o filtro real.
[boot]  Bootstrap de blocos (L = h+1) na amostra sem entrantes: Ch_Caixa, beta_calmo, beta_2021 e diferenca.
"""
import math
import sys
import time

import numpy as np
import pandas as pd

import placebos as pb
from placebos import (G0, S0, SL10, ZMAP, HS, AMOSTRAS, AMOSTRAS_TESTE, ESPERADO, TOL,
                      monta_specs, dados_da_amostra, serie_y, winsor, fit_full)
from eventos import Q2021
from ciclos import pval
from ipo import firmas_onda, ONDAS, Q1315

SEED = 20261
N_ALEAT = 1000
N_BOOT = 1000
PRE_H = (-4, -3, -2)
WAVE = firmas_onda(*ONDAS["entrada 2019Q1-2021Q4"])
# coeficientes de Ch_Caixa sem as entrantes (saida do ipo.py), para checar reproducao
REF_FILT = {("Restrita", 3): 0.0219, ("Restrita", 5): 0.0332, ("Restrita", 7): 0.0354,
            ("Total", 3): 0.0158, ("Total", 5): 0.0206, ("Total", 7): 0.0020}


# ---------------- construcao da amostra (y SEM winsor; o winsor entra depois do filtro) ----------------
def monta_base(am):
    x = monta_specs(AMOSTRAS[am]["com_q"])["ctrl_int"]["x"]
    return x, [v for v in x if v.endswith("_z")]


def junta_choque(d, zc):
    ti = d["t"].to_numpy() - G0
    d["Shock"], d["Shock_L1"] = S0[ti], SL10[ti]
    return d.dropna(subset=["y", "SETOR_ATIV", "Shock", "Shock_L1"] + zc).copy()


def interacoes(d, x):
    for v in x:
        if v in ZMAP:
            d[v] = d["Shock"] * d[ZMAP[v]]
    return d


def build_raw(am, h):
    x, zc = monta_base(am)
    d = dados_da_amostra(am)[["CNPJ_CIA", "t", "SETOR_ATIV"] + zc].merge(
        serie_y(h), on=["CNPJ_CIA", "t"], how="inner")
    return interacoes(junta_choque(d, zc), x), x


def within_coefs(X, y, codes):
    nf = codes.max() + 1
    cnt = np.maximum(np.bincount(codes, minlength=nf), 1).astype(float)
    dm = lambda a: a - (np.bincount(codes, weights=a, minlength=nf) / cnt)[codes]
    Xd = np.column_stack([dm(X[:, j]) for j in range(X.shape[1])])
    return np.linalg.lstsq(Xd, dm(y), rcond=None)[0]


def soma(r, nomes):
    est = float(r.params[nomes].sum())
    var = float(r.cov.loc[nomes, nomes].to_numpy().sum())
    se = math.sqrt(var)
    return est, se, pval(est / se)


# ======================= 1. PRE-TENDENCIA (media de h=-4,-3,-2) =======================
def teste_pre():
    rows = []
    for am in AMOSTRAS_TESTE:
        x, zc = monta_base(am)
        d = dados_da_amostra(am)[["CNPJ_CIA", "t", "SETOR_ATIV"] + zc]
        for h in PRE_H:
            d = d.merge(serie_y(h).rename(columns={"y": f"y{h}"}), on=["CNPJ_CIA", "t"], how="inner")
        d["y"] = d[[f"y{h}" for h in PRE_H]].mean(axis=1)
        d = interacoes(junta_choque(d, zc), x)
        d["y"] = winsor(d["y"])
        r = fit_full(d, x, min(PRE_H))
        rows.append(dict(amostra=am, coef_Ch_Caixa=r.params["Ch_Caixa"], se=r.std_errors["Ch_Caixa"],
                         p=r.pvalues["Ch_Caixa"], n=int(r.nobs), T=d["t"].nunique()))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


# ======================= 2. INTERACAO ENTRANTES x INCUMBENTES =======================
def add_regimes(d, x):
    t = d["t"].to_numpy()
    d["P1"] = np.isin(t, list(Q1315)).astype(float)
    d["P2"] = np.isin(t, list(Q2021)).astype(float)
    ex = []
    for nm in ("P1", "P2"):
        ex.append(nm)
        for v in x:
            if v in ZMAP:
                d[f"{v}_{nm}"] = d[v] * d[nm]
                ex.append(f"{v}_{nm}")
        d[f"Shock_{nm}"] = d["Shock"] * d[nm]
        ex.append(f"Shock_{nm}")
    return ex


def add_rec(d, com_p2):
    d["Shock_Rec"] = d["Shock"] * d["Rec"]
    d["Cash_Ratio_z_Rec"] = d["Cash_Ratio_z"] * d["Rec"]
    d["Ch_Caixa_Rec"] = d["Ch_Caixa"] * d["Rec"]
    ex = ["Shock_Rec", "Cash_Ratio_z_Rec", "Ch_Caixa_Rec"]
    if com_p2:
        d["P2_Rec"] = d["P2"] * d["Rec"]
        d["Shock_P2_Rec"] = d["Shock"] * d["P2"] * d["Rec"]
        d["Cash_Ratio_z_P2_Rec"] = d["Cash_Ratio_z"] * d["P2"] * d["Rec"]
        d["Ch_Caixa_P2_Rec"] = d["Ch_Caixa"] * d["P2"] * d["Rec"]
        ex += ["P2_Rec", "Shock_P2_Rec", "Cash_Ratio_z_P2_Rec", "Ch_Caixa_P2_Rec"]
    return ex


def teste_inter():
    rows = []
    for am in AMOSTRAS_TESTE:
        for h in HS:
            d, x = build_raw(am, h)
            d["y"] = winsor(d["y"])
            d["Rec"] = d["CNPJ_CIA"].isin(WAVE).astype(float)
            n_rec = int(d["Rec"].sum())
            # Modelo A: todos os periodos
            dA = d.copy()
            try:
                rA = fit_full(dA, x, h, extra=add_rec(dA, False))
                for nome, nomes in {"A: gradiente incumbentes": ["Ch_Caixa"],
                                    "A: entrantes menos incumbentes": ["Ch_Caixa_Rec"],
                                    "A: gradiente entrantes": ["Ch_Caixa", "Ch_Caixa_Rec"]}.items():
                    e, s, pv = soma(rA, nomes)
                    rows.append(dict(amostra=am, h=h, quantidade=nome, estimativa=e, se=s, p=pv, obs_entrantes=n_rec))
            except Exception as ex:
                print(f"[aviso] modelo A falhou em {am} h={h}: {ex}")
            # Modelo B: regimes + entrantes
            dB = d.copy()
            try:
                extra = add_regimes(dB, x) + add_rec(dB, True)
                rB = fit_full(dB, x, h, extra=extra)
                for nome, nomes in {
                        "B: calmo, incumbentes": ["Ch_Caixa"],
                        "B: calmo, entrantes menos incumbentes": ["Ch_Caixa_Rec"],
                        "B: 2021, incumbentes": ["Ch_Caixa", "Ch_Caixa_P2"],
                        "B: 2021, entrantes": ["Ch_Caixa", "Ch_Caixa_Rec", "Ch_Caixa_P2", "Ch_Caixa_P2_Rec"],
                        "B: 2021, entrantes menos incumbentes": ["Ch_Caixa_Rec", "Ch_Caixa_P2_Rec"]}.items():
                    e, s, pv = soma(rB, nomes)
                    rows.append(dict(amostra=am, h=h, quantidade=nome, estimativa=e, se=s, p=pv, obs_entrantes=n_rec))
            except Exception as ex:
                print(f"[aviso] modelo B falhou em {am} h={h}: {ex}")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


# ======================= 3. EXCLUSAO ALEATORIA DE FIRMAS =======================
def teste_aleat():
    rng, rows = np.random.default_rng(SEED), []
    for am in AMOSTRAS_TESTE:
        for h in HS:
            d, x = build_raw(am, h)
            cod = pd.factorize(d["CNPJ_CIA"])[0]
            X, y_raw, j = d[x].to_numpy(float), d["y"].to_numpy(float), x.index("Ch_Caixa")

            def coef(mask):
                yy = winsor(pd.Series(y_raw[mask])).to_numpy(float)
                return within_coefs(X[mask], yy, cod[mask])[j]

            setor = d.groupby("CNPJ_CIA")["SETOR_ATIV"].first()
            na_onda = setor.index.isin(list(WAVE))
            k_setor = setor[na_onda].value_counts()
            pool = setor[~na_onda]
            pool_setor = {s: g.index.to_numpy() for s, g in pool.groupby(pool)}
            c_cheio = coef(np.ones(len(d), bool))
            c_obs = coef(~d["CNPJ_CIA"].isin(WAVE).to_numpy())
            ok1 = abs(c_cheio - ESPERADO[(am, h)]) < TOL
            ok2 = abs(c_obs - REF_FILT[(am, h)]) < TOL
            print(f"[repro] {am:8s} h={h}: cheio {'OK' if ok1 else 'FALHA'} | sem entrantes {'OK' if ok2 else 'FALHA'}")
            if not (ok1 and ok2):
                sys.exit("Reproducao falhou: nao confie no placebo ate entender a diferenca.")
            nulos, n_obs_fora = [], []
            for _ in range(N_ALEAT):
                sel = [rng.choice(pool_setor[s], size=min(k, len(pool_setor[s])), replace=False)
                       for s, k in k_setor.items() if len(pool_setor.get(s, [])) > 0]
                mask = ~d["CNPJ_CIA"].isin(np.concatenate(sel)).to_numpy()
                nulos.append(coef(mask))
                n_obs_fora.append(len(d) - int(mask.sum()))
            a = np.asarray(nulos)
            rows.append(dict(amostra=am, h=h, coef_cheio=c_cheio, coef_sem_entrantes=c_obs,
                             nulo_media=a.mean(), nulo_dp=a.std(ddof=1), nulo_q05=np.quantile(a, .05),
                             p_queda=(1 + np.sum(a <= c_obs)) / (1 + len(a)),
                             obs_removidas_entrantes=int(len(d) - (~d["CNPJ_CIA"].isin(WAVE)).sum()),
                             obs_removidas_sorteio=float(np.mean(n_obs_fora))))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


# ======================= 4. BOOTSTRAP DE BLOCOS SEM AS ENTRANTES =======================
def prepara_d(d, x):
    t = d["t"].to_numpy()
    p1 = np.isin(t, list(Q1315)).astype(float)
    p2 = np.isin(t, list(Q2021)).astype(float)
    col = {v: d[v].to_numpy(float) for v in x}
    reg = {"P1": p1, "P2": p2, "Shock_P1": col["Shock"] * p1, "Shock_P2": col["Shock"] * p2}
    for v in x:
        if v in ZMAP:
            reg[v + "_P1"], reg[v + "_P2"] = col[v] * p1, col[v] * p2
    qs = np.unique(t)
    return dict(y=d["y"].to_numpy(float), codes=pd.factorize(d["CNPJ_CIA"])[0],
                Xb=np.column_stack([col[v] for v in x]), nb=list(x),
                Xr=np.column_stack([col[v] for v in x] + [reg[k] for k in reg]), nr=list(x) + list(reg),
                qs=qs, idx={q: np.where(t == q)[0] for q in qs})


def ajuste_boot(P, rows):
    c, y = P["codes"][rows], P["y"][rows]
    bb = dict(zip(P["nb"], within_coefs(P["Xb"][rows], y, c)))
    br = dict(zip(P["nr"], within_coefs(P["Xr"][rows], y, c)))
    a, b1, b2 = br["Ch_Caixa"], br["Ch_Caixa_P1"], br["Ch_Caixa_P2"]
    return {"Ch_Caixa (ctrl_int)": bb["Ch_Caixa"], "beta_calmo": a, "beta_2021": a + b2,
            "dif_ciclo1315_menos_2021": b1 - b2}


def sorteia(qs, L, rng):
    n = len(qs)
    starts = rng.integers(0, n - L + 1, size=math.ceil(n / L))
    return np.concatenate([qs[s:s + L] for s in starts])[:n]


def teste_boot():
    rng, out = np.random.default_rng(SEED + 1), []
    for am in AMOSTRAS_TESTE:
        for h in HS:
            d, x = build_raw(am, h)
            d = d[~d["CNPJ_CIA"].isin(WAVE)].copy()
            d["y"] = winsor(d["y"])
            P = prepara_d(d, x)
            obs = ajuste_boot(P, np.arange(len(P["y"])))
            chk = fit_full(d, x, h).params["Ch_Caixa"]
            if abs(obs["Ch_Caixa (ctrl_int)"] - chk) > 1e-8:
                sys.exit(f"Reproducao falhou em {am} h={h}")
            L, acc, t1 = h + 1, {k: [] for k in obs}, time.time()
            for _ in range(N_BOOT):
                sel = sorteia(P["qs"], L, rng)
                if not (np.isin(sel, list(Q1315)).any() and np.isin(sel, list(Q2021)).any()):
                    continue
                for k, v in ajuste_boot(P, np.concatenate([P["idx"][q] for q in sel])).items():
                    acc[k].append(v)
            for k in obs:
                a = np.asarray(acc[k])
                out.append(dict(amostra=am, h=h, bloco=f"L={L}", estatistica=k, estimativa=obs[k],
                                se_boot=a.std(ddof=1), ic95_inf=np.quantile(a, .025), ic95_sup=np.quantile(a, .975),
                                p_boot=max(2 * min(np.mean(a >= 0), np.mean(a <= 0)), 1 / (len(a) + 1)),
                                draws_validos=len(a)))
            print(f"[{am} h={h}] {len(acc['beta_calmo'])}/{N_BOOT} draws validos, {time.time() - t1:.0f}s")
    df = pd.DataFrame(out)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


TESTES = {"pre": teste_pre, "inter": teste_inter, "aleat": teste_aleat, "boot": teste_boot}

if __name__ == "__main__":
    pb.checar_baseline()
    print(f"[ipo2] firmas entrantes (2019Q1-2021Q4) no painel: {len(WAVE)}")
    for nome in (sys.argv[1:] or list(TESTES)):
        t1 = time.time()
        print(f"\n=== {nome} ===")
        df = TESTES[nome]()
        df.to_csv(f"ipo2_{nome}.csv", index=False)
        print(f"[{nome}] {time.time() - t1:.0f}s -> ipo2_{nome}.csv")