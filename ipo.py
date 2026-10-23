"""
ipo.py -- a onda de entradas/IPOs de 2019-21 explica o resultado de 2021?

Requer placebos.py, eventos.py e ciclos.py na MESMA pasta.
Uso:
    python ipo.py

Proxy de IPO/entrada: o primeiro trimestre em que a firma aparece no painel (T0). Firmas ja
presentes no primeiro trimestre do painel sao tratadas como incumbentes (observacao truncada a
esquerda). O painel nao tem data de IPO, entao T0 mistura IPOs, novos registros na CVM e firmas
que passaram a ter dados completos; trate como proxy.

Filtros testados (cada um re-estima, para Total e Restrita, h = 3, 5, 7):
  nenhuma                        referencia (tem de reproduzir os coeficientes originais)
  entrada 2019Q1-2021Q4          exclui firmas cujo T0 cai na janela (as firmas inteiras)
  entrada 2020Q1-2021Q4          idem, janela mais estreita
  entrada 2018Q1-2021Q4          idem, janela mais larga
  sem obs. jovens (<8 trim.)     exclui observacoes de firmas que entraram depois do inicio do
                                 painel enquanto tem menos de 8 trimestres desde T0

Para cada filtro: Ch_Caixa (ctrl_int, DK do original) e o modelo de regimes de ciclos.py
(beta_calmo, beta_2013-15, beta_2021 e a diferenca 2013-15 - 2021).
Observacao: os z-scores ja foram calculados por trimestre com todas as firmas; o filtro entra
depois, como nas exclusoes de ano do build() original. O desfecho e winsorizado apos o filtro.
Saidas: ipo_resultados.csv, ipo_diagnostico_entradas.csv
"""
import math
import sys
import time

import numpy as np
import pandas as pd

import placebos as pb
from placebos import (p, G0, S0, SL10, ZMAP, HS, AMOSTRAS, AMOSTRAS_TESTE, MIN_OBS, ESPERADO, TOL,
                      monta_specs, dados_da_amostra, serie_y, winsor, fit_full)
from eventos import comb, Q2021
from ciclos import trimestres, JANELAS, pval

Q1315 = trimestres(*JANELAS["2013Q2-2015Q2 (principal)"])
ONDAS = {"entrada 2019Q1-2021Q4": ("2019Q1", "2021Q4"),
         "entrada 2020Q1-2021Q4": ("2020Q1", "2021Q4"),
         "entrada 2018Q1-2021Q4": ("2018Q1", "2021Q4")}
IDADE_MIN = 8   # trimestres desde a entrada


# ---------------- primeira observacao de cada firma ----------------
def primeira_obs():
    q = p[p["obs_real"].astype(bool)] if "obs_real" in p.columns else p
    return q.groupby("CNPJ_CIA")["t"].min()


T0 = primeira_obs()
CENS = int(T0.min())          # firmas com T0 == CENS sao incumbentes (truncadas)


def ord_q(s):
    return pd.Period(s, freq="Q").ordinal


def firmas_onda(ini, fim):
    a, b = ord_q(ini), ord_q(fim)
    return set(T0.index[(T0 >= a) & (T0 <= b) & (T0 > CENS)])


# ---------------- build com filtro (replica passo-a-passo de placebos.build) ----------------
def build_f(am, h, filtro=None):
    x = monta_specs(AMOSTRAS[am]["com_q"])["ctrl_int"]["x"]
    zc = [v for v in x if v.endswith("_z")]
    d = dados_da_amostra(am)[["CNPJ_CIA", "t", "SETOR_ATIV"] + zc].merge(
        serie_y(h), on=["CNPJ_CIA", "t"], how="inner")
    ti = d["t"].to_numpy() - G0
    d["Shock"], d["Shock_L1"] = S0[ti], SL10[ti]
    d = d.dropna(subset=["y", "SETOR_ATIV", "Shock", "Shock_L1"] + zc).copy()
    n_antes, f_antes = len(d), set(d["CNPJ_CIA"])
    if filtro is not None:
        d = filtro(d)
    if len(d) < MIN_OBS:
        return None, x, 0, 0
    d["y"] = winsor(d["y"])
    for v in x:
        if v in ZMAP:
            d[v] = d["Shock"] * d[ZMAP[v]]
    return d, x, n_antes - len(d), len(f_antes - set(d["CNPJ_CIA"]))


def filtros():
    fs = {"nenhuma": None}
    for nome, (ini, fim) in ONDAS.items():
        F = firmas_onda(ini, fim)
        fs[nome] = (lambda d, F=F: d[~d["CNPJ_CIA"].isin(F)])
    def jovens(d):
        t0 = d["CNPJ_CIA"].map(T0)
        jovem = (t0 > CENS) & ((d["t"] - t0) < IDADE_MIN)
        return d[~jovem]
    fs[f"sem obs. jovens (<{IDADE_MIN} trim.)"] = jovens
    return fs


# ---------------- modelo de regimes (mesma logica de eventos.ajusta) ----------------
def modelo_regimes(d, x, h):
    d = d.copy()
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
    r = fit_full(d, x, h, extra=ex)
    a, b1, b2 = "Ch_Caixa", "Ch_Caixa_P1", "Ch_Caixa_P2"
    e1, _, p1 = comb(r, a, b1)
    e2, _, p2 = comb(r, a, b2)
    dif = r.params[b1] - r.params[b2]
    var = r.cov.loc[b1, b1] + r.cov.loc[b2, b2] - 2 * r.cov.loc[b1, b2]
    return dict(beta_calmo=r.params[a], p_calmo=r.pvalues[a], beta_ciclo1315=e1, p_ciclo=p1,
                beta_2021=e2, p_2021=p2, dif_1315_menos_2021=dif,
                p_igualdade=pval(dif / math.sqrt(var)))


# ---------------- diagnostico ----------------
def diagnostico():
    ent = T0[T0 > CENS]
    anos = pd.Series([pd.Period(ordinal=int(o), freq="Q").year for o in ent], name="ano_entrada")
    tab = anos.value_counts().sort_index().rename("firmas_novas").reset_index()
    print("\n--- Firmas por ano de primeira observacao (excluindo as presentes no inicio do painel) ---")
    print(tab.to_string(index=False))
    out = tab.copy()
    if "Cash_Ratio" in p.columns:
        F = firmas_onda(*ONDAS["entrada 2019Q1-2021Q4"])
        q21 = p[p["t"].isin(list(Q2021))].copy()
        q21["grupo"] = np.where(q21["CNPJ_CIA"].isin(F), "entrou 2019Q1-2021Q4", "demais")
        g = q21.groupby("grupo")["Cash_Ratio"].agg(["count", "mean", "median"])
        print("\n--- Cash_Ratio (nivel) nos trimestres de 2021, por grupo ---")
        print(g.to_string(float_format=lambda v: f"{v:.4f}"))
        g.reset_index().to_csv("ipo_diagnostico_cash.csv", index=False)
    out.to_csv("ipo_diagnostico_entradas.csv", index=False)


# ---------------- execucao ----------------
def checar_baseline():
    for (am, h), esp in ESPERADO.items():
        d, x, _, _ = build_f(am, h)
        cf = fit_full(d, x, h).params["Ch_Caixa"]
        ok = abs(cf - esp) < TOL
        print(f"[baseline build_f] {am:8s} h={h}: {cf:.5f} esperado={esp:.5f} {'OK' if ok else '<<< NAO REPRODUZ'}")
        if not ok:
            sys.exit("build_f nao reproduz o baseline.")


def rodar():
    fs, rows = filtros(), []
    for am in AMOSTRAS_TESTE:
        for h in HS:
            for nome, f in fs.items():
                d, x, n_obs_fora, n_firm_fora = build_f(am, h, f)
                if d is None:
                    continue
                r = fit_full(d, x, h)
                row = dict(amostra=am, h=h, filtro=nome, obs_excluidas=n_obs_fora,
                           firmas_excluidas=n_firm_fora, obs=int(r.nobs), T=d["t"].nunique(),
                           ch_caixa=r.params["Ch_Caixa"], se=r.std_errors["Ch_Caixa"],
                           p=r.pvalues["Ch_Caixa"])
                try:
                    row.update(modelo_regimes(d, x, h))
                except Exception as e:           # regime nao estimavel
                    print(f"[aviso] regimes falhou em {am} h={h} {nome}: {e}")
                rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


if __name__ == "__main__":
    pb.checar_baseline()
    checar_baseline()
    diagnostico()
    t1 = time.time()
    df = rodar()
    df.to_csv("ipo_resultados.csv", index=False)
    print(f"\n[{time.time() - t1:.0f}s] -> ipo_resultados.csv")