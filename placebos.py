"""
placebos.py -- testes placebo/robustez SEM alterar projecoes_locais.py

Como funciona: le o texto de projecoes_locais.py, executa tudo ATE o loop final
(que roda lp() em todas as amostras) e reaproveita p, serie_y, monta_specs, etc.
Uso (na pasta do projeto):
    python placebos.py                 # todos os testes
    python placebos.py perm loyo       # so alguns: perm loyo loqo lead pand cash
Saidas: placebo_<teste>.csv
"""
import contextlib, io, math, os, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

# ---------------- CONFIG ----------------
HS = (3, 5, 7)          # horizontes testados
N_PERM = 1000           # reps da permutacao aleatoria do choque
N_CASH = 500            # reps da permutacao do caixa
SEED = 12345
ALVOS = ("Ch_Caixa", "Shock")
AMOSTRAS_TESTE = ("Restrita", "Total")
# coeficientes de Ch_Caixa (ctrl_int, amostra cheia) do seu output, para checar a reproducao
ESPERADO = {("Restrita", 3): 0.04761, ("Restrita", 5): 0.05488, ("Restrita", 7): 0.06009,
            ("Total", 3): 0.04302, ("Total", 5): 0.04044, ("Total", 7): 0.03226}
TOL = 5e-4
# ----------------------------------------

os.chdir(Path(__file__).resolve().parent)
SRC = Path("projecoes_locais.py").read_text(encoding="utf-8")
i0 = SRC.find("def lp(")
i1 = SRC.find("\ntodos = []", i0)
if i0 < 0 or i1 < 0:
    sys.exit("Nao achei 'def lp(' / 'todos = []' em projecoes_locais.py")

ns = {"__name__": "pl_import", "__file__": str(Path("projecoes_locais.py").resolve())}
buf, t0 = io.StringIO(), time.time()
try:
    with contextlib.redirect_stdout(buf):
        exec(compile(SRC[:i1], "projecoes_locais.py", "exec"), ns)
except Exception:
    print(buf.getvalue()[-3000:])
    raise
print(f"[ok] projecoes_locais.py carregado sem rodar o loop final ({time.time() - t0:.0f}s)")

p = ns["p"]
serie_y, monta_specs, dados_da_amostra = ns["serie_y"], ns["monta_specs"], ns["dados_da_amostra"]
AMOSTRAS, cov_kwargs, winsor, ano_de_t = ns["AMOSTRAS"], ns["cov_kwargs"], ns["winsor"], ns["ano_de_t"]
MIN_OBS = ns["MIN_OBS"]

ZMAP = {"Ch_Caixa": "Cash_Ratio_z", "Ch_Alav": "Alavancagem_L1_z", "Ch_Tam": "Tamanho_z",
        "Ch_FCO": "Fluxo_Caixa_Op_Ativo_L1_z", "Ch_Q": "Tobin_Q_L1_z"}

G = np.arange(p["t"].min(), p["t"].max() + 1)       # grade trimestral continua
G0 = G[0]
S0 = p.groupby("t")["Shock"].first().reindex(G).to_numpy(float)


def lag(a):
    return np.r_[np.nan, a[:-1]]


SL10 = lag(S0)
V = np.where(np.isfinite(S0))[0]                    # posicoes com choque valido
rot = lambda o: str(pd.Period(ordinal=int(o), freq="Q"))


# ---------------- construcao da amostra de um (amostra, h) ----------------
def build(amostra, h, S=None, SL1=None, excl_years=(), excl_q=(), S2=None):
    """Replica o passo-a-passo de lp(): merge de y, dropna, filtros, winsor de y."""
    S = S0 if S is None else S
    SL1 = SL10 if SL1 is None else SL1
    x = monta_specs(AMOSTRAS[amostra]["com_q"])["ctrl_int"]["x"]
    zc = [v for v in x if v.endswith("_z")]
    d = dados_da_amostra(amostra)[["CNPJ_CIA", "t", "SETOR_ATIV"] + zc].merge(
        serie_y(h), on=["CNPJ_CIA", "t"], how="inner")
    ti = d["t"].to_numpy() - G0
    d["Shock"], d["Shock_L1"] = S[ti], SL1[ti]
    obrig = ["y", "SETOR_ATIV", "Shock", "Shock_L1"] + zc
    if S2 is not None:
        d["Shock_ld"] = S2[ti]
        obrig.append("Shock_ld")
    d = d.dropna(subset=obrig).copy()
    if len(excl_years):
        d = d[~np.isin(ano_de_t(d["t"]), list(excl_years))]
    if len(excl_q):
        d = d[~d["t"].isin(list(excl_q))]
    if len(d) < MIN_OBS:
        return None, x
    d["y"] = winsor(d["y"])
    for v in x:
        if v in ZMAP:
            d[v] = d["Shock"] * d[ZMAP[v]]
    if S2 is not None:
        d["Ch_Caixa_ld"] = d["Shock_ld"] * d["Cash_Ratio_z"]
    return d, x


def fit_full(d, x, h, extra=()):
    """PanelOLS com efeito fixo de firma e Driscoll-Kraay (mesmo cov_kwargs do original)."""
    dd = d.set_index(["CNPJ_CIA", "t"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PanelOLS(dd["y"], dd[list(x) + list(extra)], entity_effects=True).fit(
            **cov_kwargs(h, d["t"].nunique()))


class Fast:
    """Estimador within (so coeficientes) para muitas repeticoes."""

    def __init__(self, d, x):
        self.x = x
        self.ti = d["t"].to_numpy() - G0
        self.codes = pd.factorize(d["CNPJ_CIA"])[0]
        self.cnt = np.bincount(self.codes)
        self.z = {c: d[c].to_numpy(float) for c in d.columns if c.endswith("_z")}
        self.y = self.dm(d["y"].to_numpy(float))

    def dm(self, a):
        return a - (np.bincount(self.codes, weights=a) / self.cnt)[self.codes]

    def coefs(self, S, SL1, override=None):
        ov = override or {}
        s, sl = S[self.ti], SL1[self.ti]
        cols = []
        for v in self.x:
            if v == "Shock":
                c = s
            elif v == "Shock_L1":
                c = sl
            elif v in ZMAP:
                c = s * ov.get(ZMAP[v], self.z[ZMAP[v]])
            else:
                c = ov.get(v, self.z[v])
            cols.append(self.dm(c))
        b = np.linalg.lstsq(np.column_stack(cols), self.y, rcond=None)[0]
        return dict(zip(self.x, b))


def resumo_null(obs, null):
    null = np.asarray(null)
    return dict(obs=obs, null_media=null.mean(), null_dp=null.std(),
                q025=np.quantile(null, .025), q975=np.quantile(null, .975),
                p_perm=(1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null)))


# ---------------- 0. checagem de reproducao ----------------
def checar_baseline():
    ok = True
    for (am, h), esp in ESPERADO.items():
        d, x = build(am, h)
        r = fit_full(d, x, h)
        cf = r.params["Ch_Caixa"]
        cfast = Fast(d, x).coefs(S0, SL10)["Ch_Caixa"]
        flag = abs(cf - esp) < TOL and abs(cf - cfast) < 1e-8
        ok &= flag
        print(f"[baseline] {am:8s} h={h}: full={cf:.5f} fast={cfast:.5f} esperado={esp:.5f} "
              f"{'OK' if flag else '<<< NAO REPRODUZ'}")
    if not ok:
        sys.exit("Baseline nao reproduz: nao confie nos placebos ate entender a diferenca.")


# ---------------- 1. permutacao do choque ----------------
def teste_perm():
    rng, vals, rows = np.random.default_rng(SEED), S0[V], []
    for am in AMOSTRAS_TESTE:
        for h in HS:
            d, x = build(am, h)
            fz = Fast(d, x)
            obs = fz.coefs(S0, SL10)
            for esq in ("circular", "aleatoria"):
                g, null = np.full_like(S0, np.nan), []
                for k in (range(1, len(V)) if esq == "circular" else range(N_PERM)):
                    g[V] = np.roll(vals, k) if esq == "circular" else rng.permutation(vals)
                    null.append(fz.coefs(g, lag(g)))
                for a in ALVOS:
                    rows.append(dict(amostra=am, h=h, esquema=esq, alvo=a,
                                     **resumo_null(obs[a], [n[a] for n in null])))
    return pd.DataFrame(rows)


# ---------------- 2. leave-one-year-out ----------------
def teste_loyo():
    anos = sorted({int(ano_de_t(G[i])) for i in V})
    conj = [(a,) for a in anos] + [(2020, 2021)]
    rows = []
    for am in AMOSTRAS_TESTE:
        for h in HS:
            for ex in [()] + conj:
                d, x = build(am, h, excl_years=ex)
                if d is None:
                    continue
                r = fit_full(d, x, h)
                for a in ALVOS:
                    rows.append(dict(amostra=am, h=h, excluidos="nenhum" if not ex else "+".join(map(str, ex)),
                                     alvo=a, coef=r.params[a], se=r.std_errors[a], p=r.pvalues[a], n=int(r.nobs)))
    return pd.DataFrame(rows)


# ---------------- 3. leave-one-quarter-out ----------------
def teste_loqo():
    rows = []
    for am in AMOSTRAS_TESTE:
        for h in HS:
            d, x = build(am, h)
            obs = Fast(d, x).coefs(S0, SL10)
            for i in V:
                dq, xq = build(am, h, excl_q=(G[i],))
                if dq is None:
                    continue
                b = Fast(dq, xq).coefs(S0, SL10)
                for a in ALVOS:
                    rows.append(dict(amostra=am, h=h, trimestre_excluido=rot(G[i]), alvo=a,
                                     coef=b[a], coef_cheia=obs[a]))
    df = pd.DataFrame(rows)
    df["delta"] = df["coef"] - df["coef_cheia"]
    resumo = []
    for (am, h, a), g in df.groupby(["amostra", "h", "alvo"]):
        resumo.append(dict(amostra=am, h=h, alvo=a, coef_cheia=g["coef_cheia"].iloc[0],
                           min=g["coef"].min(), q_min=g.loc[g["coef"].idxmin(), "trimestre_excluido"],
                           max=g["coef"].max(), q_max=g.loc[g["coef"].idxmax(), "trimestre_excluido"],
                           mais_influente=g.loc[g["delta"].abs().idxmax(), "trimestre_excluido"],
                           inverte_sinal=int((np.sign(g["coef"]) != np.sign(g["coef_cheia"])).sum())))
    df.to_csv("placebo_loqo_detalhe.csv", index=False)
    return pd.DataFrame(resumo)


# ---------------- 4. choque adiantado (Shock_{t+k}) ----------------
def teste_lead():
    rows = []
    for am in AMOSTRAS_TESTE:
        for h in HS:
            for k in (1, 2, 3, 4):
                S2 = np.r_[S0[k:], np.full(k, np.nan)]
                d, x = build(am, h, S2=S2)
                r = fit_full(d, x, h, extra=("Shock_ld", "Ch_Caixa_ld"))
                rows.append(dict(amostra=am, h=h, k=k, ch_caixa_real=r.params["Ch_Caixa"], p_real=r.pvalues["Ch_Caixa"],
                                 ch_caixa_lead=r.params["Ch_Caixa_ld"], p_lead=r.pvalues["Ch_Caixa_ld"],
                                 shock_lead=r.params["Shock_ld"], p_shock_lead=r.pvalues["Shock_ld"], n=int(r.nobs)))
    return pd.DataFrame(rows)


# ---------------- 5. dummy de periodo (teste formal da diferenca) ----------------
def teste_pand():
    Q = lambda ys: {int(g) for g in G if int(ano_de_t(g)) in ys}
    regimes = {"2020-21": Q((2020, 2021)), "2021": Q((2021,)), "2020": Q((2020,)),
               "2021Q4": {pd.Period("2021Q4", freq="Q").ordinal}}
    rows = []
    for nome, qs in regimes.items():
        for am in AMOSTRAS_TESTE:
            for h in HS:
                d, x = build(am, h)
                d["Pand"] = d["t"].isin(qs).astype(float)
                d["Shock_P"] = d["Shock"] * d["Pand"]
                # com 1 único trimestre, Shock_P = constante * Pand (colinear): fica só Pand
                extra = ["Pand"] + ([] if len(qs) == 1 else ["Shock_P"])
                for v in x:
                    if v in ZMAP:
                        d[v + "_P"] = d[v] * d["Pand"]
                        extra.append(v + "_P")
                r = fit_full(d, x, h, extra=extra)
                a, b = "Ch_Caixa", "Ch_Caixa_P"
                var = r.cov.loc[a, a] + r.cov.loc[b, b] + 2 * r.cov.loc[a, b]
                soma = r.params[a] + r.params[b]
                z = soma / math.sqrt(var)
                rows.append(dict(regime=nome, amostra=am, h=h, beta_fora=r.params[a], p_fora=r.pvalues[a],
                                 diferenca=r.params[b], p_dif=r.pvalues[b], beta_no_regime=soma,
                                 p_regime=math.erfc(abs(z) / math.sqrt(2)), n_obs_regime=int(d["Pand"].sum())))
    return pd.DataFrame(rows)


# ---------------- 6. permutacao cross-section do caixa (dentro de setor x trimestre) ----------------
def teste_cash():
    rng, rows = np.random.default_rng(SEED + 1), []
    for am in AMOSTRAS_TESTE:
        for h in HS:
            d, x = build(am, h)
            fz = Fast(d, x)
            obs = fz.coefs(S0, SL10)["Ch_Caixa"]
            g = pd.factorize(d["SETOR_ATIV"].astype(str) + "|" + d["t"].astype(str))[0]
            base, zc, null = np.argsort(g, kind="stable"), fz.z["Cash_Ratio_z"], []
            for _ in range(N_CASH):
                perm = np.lexsort((rng.random(len(g)), g))
                new = np.empty_like(zc)
                new[base] = zc[perm]
                null.append(fz.coefs(S0, SL10, override={"Cash_Ratio_z": new})["Ch_Caixa"])
            rows.append(dict(amostra=am, h=h, alvo="Ch_Caixa", **resumo_null(obs, null)))
    return pd.DataFrame(rows)


TESTES = {"perm": teste_perm, "loyo": teste_loyo, "loqo": teste_loqo,
          "lead": teste_lead, "pand": teste_pand, "cash": teste_cash}

if __name__ == "__main__":
    escolhidos = sys.argv[1:] or list(TESTES)
    checar_baseline()
    for nome in escolhidos:
        t1 = time.time()
        print(f"\n=== {nome} ===")
        df = TESTES[nome]()
        df.to_csv(f"placebo_{nome}.csv", index=False)
        print(df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        print(f"[{nome}] {time.time() - t1:.0f}s -> placebo_{nome}.csv")