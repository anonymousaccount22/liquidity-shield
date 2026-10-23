"""
monte_carlo_lp.py -- Tamanho e poder do teste de Ch_Caixa (Shock x Cash_z) em Projecoes Locais
em painel quando o choque e concentrado em poucos episodios.

Autocontido: NAO usa seus CSVs nem os scripts do pipeline. So numpy/pandas.
Uso:
    python monte_carlo_lp.py                       # padrao: 500 repeticoes
    python monte_carlo_lp.py --reps 2000           # mais preciso (mais lento)
    python monte_carlo_lp.py --sig_lam 0.02 --N 300 --gammas 0 0.04

DGP (estilizado, ver comentarios):
  - T = 54 trimestres de estimacao (como o seu painel); choque Shock_t FIXO por cenario.
  - Caixa z_it: AR(1) (rho = 0.9) e padronizado por trimestre (z-score na cross-section).
  - Crescimento trimestral da firma i no trimestre s:
        A_is = lam_s * z_i,s-1  +  e_is            (lam_s ~ N(0, sig_lam^2) INDEPENDENTE do choque)
    lam_s e um fator macro que "carrega" no caixa (omitted macro x cash shock): o choque
    monetario NAO tem efeito algum quando gamma = 0.
  - Desfecho acumulado como no seu codigo: y_it(h) = soma_{s=t..t+h} A_is  (+ gamma*Shock_t*z_i,t-1).
  - Regressao: y ~ Shock_t + z_i,t-1 + Shock_t*z_i,t-1, efeito fixo de firma (sem outros controles).
  - H0 verdadeira: gamma = 0. Com gamma > 0 mede-se poder e cobertura.

Metodos de inferencia comparados para o coeficiente de Shock x z:
  DK(h+1)        Driscoll-Kraay, kernel Bartlett, bandwidth = h+1 (o seu padrao; sem correcao de
                 pequenas amostras)
  DK(2h+2)       idem com o dobro do bandwidth
  cluster-tempo  agrupamento no trimestre (bandwidth 0)
  boot-bloco     bootstrap de blocos moveis de trimestres (L = h+1) SOBRE OS MOMENTOS POR
                 TRIMESTRE (aproximacao rapida: o efeito fixo de firma e removido uma vez, na
                 amostra completa, e nao em cada reamostra como no seu bootstrap_blocos.py)

Cenarios de choque (T=54, mesma escala do seu choque, sd ~ 0.26):
  iid         choque iid, sem episodio dominante
  calibrada   bloco de 4 trimestres (2021) = 34.5% da soma de desvios quadrados, com 2021Q4 = 1.019
  extrema     um unico trimestre concentra ~60% da soma de desvios quadrados

Saida: monte_carlo_resultados.csv e tabela no terminal.
"""
import argparse
import time

import numpy as np
import pandas as pd

T = 54
IDX_2021 = [39, 40, 41, 42]          # 2021Q1..2021Q4 se a amostra comeca em 2011Q2
BLOCO_2021 = np.array([0.17, 0.28, 0.33, 1.019])   # soma 1.80 -> media anual 0.45


# ----------------------------------------------------------------------------
# Trajetorias do choque
# ----------------------------------------------------------------------------
def n_efetivo(x):
    s2 = np.asarray(x, float) ** 2
    w = s2 / s2.sum()
    return 1.0 / (w ** 2).sum()


def caminho_choque(tipo, seed=7):
    rng = np.random.default_rng(seed)
    z0 = rng.standard_normal(T)
    if tipo == "iid":
        s = 0.26 * z0
    elif tipo == "calibrada":
        def share(sd):
            s = sd * z0
            s[IDX_2021] = BLOCO_2021
            d2 = (s - s.mean()) ** 2
            return d2[IDX_2021].sum() / d2.sum()
        lo, hi = 0.01, 1.0                 # share cai com sd: busca por bissecao
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if share(mid) > 0.345 else (lo, mid)
        s = ((lo + hi) / 2) * z0
        s[IDX_2021] = BLOCO_2021
    elif tipo == "extrema":
        def share(sd):
            s = sd * z0
            s[IDX_2021[3]] = 1.2
            d2 = (s - s.mean()) ** 2
            return d2[IDX_2021[3]] / d2.sum()
        lo, hi = 0.01, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if share(mid) > 0.60 else (lo, mid)
        s = ((lo + hi) / 2) * z0
        s[IDX_2021[3]] = 1.2
    else:
        raise ValueError(tipo)
    return s


# ----------------------------------------------------------------------------
# Simulacao de um painel e estimacao
# ----------------------------------------------------------------------------
def gera_z(N, n_per, rho, rng):
    z = np.empty((N, n_per))
    z[:, 0] = rng.standard_normal(N)
    for s in range(1, n_per):
        z[:, s] = rho * z[:, s - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(N)
    z = (z - z.mean(0)) / z.std(0)       # padroniza por trimestre (cross-section)
    return z


def simula_painel(shock, h, N, sig_lam, sig_e, rho, phi, gamma, rng):
    """Devolve y (N,T) e regressores X (N,T,3) para o horizonte h."""
    n_per = T + h
    zl = gera_z(N, n_per, rho, rng)                    # zl[:, s] = caixa defasado do trimestre s
    lam = np.empty(n_per)
    lam[0] = sig_lam * rng.standard_normal()
    for s in range(1, n_per):                          # fator macro; phi = persistencia
        lam[s] = phi * lam[s - 1] + np.sqrt(1 - phi ** 2) * sig_lam * rng.standard_normal()
    A = lam[None, :] * zl + sig_e * rng.standard_normal((N, n_per))
    C = np.concatenate([np.zeros((N, 1)), np.cumsum(A, axis=1)], axis=1)
    t = np.arange(T)
    y = C[:, t + h + 1] - C[:, t]                      # soma de A de t ate t+h
    zt = zl[:, :T]
    y = y + gamma * shock[None, :] * zt
    X = np.stack([np.broadcast_to(shock[None, :], (N, T)), zt, shock[None, :] * zt], axis=2)
    return y, X


def estima(y, X, L_list, h, B, rng):
    N = y.shape[0]
    yd = y - y.mean(1, keepdims=True)
    Xd = X - X.mean(1, keepdims=True)
    Sxx = np.einsum("itk,itl->tkl", Xd, Xd)            # (T,3,3)
    Sxy = np.einsum("itk,it->tk", Xd, yd)              # (T,3)
    XtX, Xty = Sxx.sum(0), Sxy.sum(0)
    beta = np.linalg.solve(XtX, Xty)
    u = yd - np.einsum("itk,k->it", Xd, beta)
    g = np.einsum("itk,it->tk", Xd, u)                 # scores por trimestre
    XtXi = np.linalg.inv(XtX)
    out = {"beta": beta[2]}
    for nome, L in L_list:
        Om = g.T @ g
        for l in range(1, L + 1):
            w = 1 - l / (L + 1)
            G = g[l:].T @ g[:-l]
            Om += w * (G + G.T)
        V = XtXi @ Om @ XtXi
        out[nome] = np.sqrt(max(V[2, 2], 1e-30))
    # bootstrap de blocos de trimestres sobre os momentos
    Lb = h + 1
    n_blocos = int(np.ceil(T / Lb))
    starts = rng.integers(0, T - Lb + 1, size=(B, n_blocos))
    idx = (starts[:, :, None] + np.arange(Lb)[None, None, :]).reshape(B, -1)[:, :T]
    Sxx_b, Sxy_b = Sxx[idx].sum(1), Sxy[idx].sum(1)
    bb = np.linalg.solve(Sxx_b, Sxy_b[:, :, None])[:, :, 0][:, 2]
    out["boot"] = (np.quantile(bb, .025), np.quantile(bb, .975),
                   2 * min(np.mean(bb >= 0), np.mean(bb <= 0)))
    return out


def roda(args):
    rng = np.random.default_rng(args.seed)
    metodos = [("DK(h+1)", None), ("DK(2h+2)", None), ("cluster-tempo", 0)]
    linhas = []
    for cen in args.cenarios:
        shock = caminho_choque(cen)
        d2 = (shock - shock.mean()) ** 2
        print(f"\n=== cenario '{cen}': sd={shock.std():.3f}, n_efetivo={n_efetivo(shock - shock.mean()):.2f}, "
              f"maior trimestre = {d2.max() / d2.sum():.1%}, 2021 = {d2[IDX_2021].sum() / d2.sum():.1%}")
        for h in args.horizontes:
            L_list = [("DK(h+1)", h + 1), ("DK(2h+2)", 2 * h + 2), ("cluster-tempo", 0)]
            for gamma in args.gammas:
                t0 = time.time()
                acc = {m: dict(b=[], se=[], rej=[], cov=[]) for m, _ in L_list}
                acc["boot-bloco"] = dict(b=[], se=[], rej=[], cov=[])
                for _ in range(args.reps):
                    y, X = simula_painel(shock, h, args.N, args.sig_lam, args.sig_e,
                                         args.rho, args.phi, gamma, rng)
                    r = estima(y, X, L_list, h, args.B, rng)
                    for m, _l in L_list:
                        tz, tg = r["beta"] / r[m], (r["beta"] - gamma) / r[m]
                        acc[m]["b"].append(r["beta"]); acc[m]["se"].append(r[m])
                        acc[m]["rej"].append(abs(tz) > 1.96); acc[m]["cov"].append(abs(tg) <= 1.96)
                    lo, hi, p = r["boot"]
                    acc["boot-bloco"]["b"].append(r["beta"]); acc["boot-bloco"]["se"].append(np.nan)
                    acc["boot-bloco"]["rej"].append(p < 0.05)
                    acc["boot-bloco"]["cov"].append(lo <= gamma <= hi)
                for m, a in acc.items():
                    b = np.array(a["b"])
                    linhas.append(dict(
                        cenario=cen, n_efetivo=n_efetivo(shock - shock.mean()), h=h, gamma=gamma,
                        metodo=m, rejeicao_H0_zero=np.mean(a["rej"]), cobertura_95=np.mean(a["cov"]),
                        sd_verdadeiro_beta=b.std(ddof=1),
                        se_medio=np.nanmean(a["se"]) if m != "boot-bloco" else np.nan,
                        razao_se_sd=(np.nanmean(a["se"]) / b.std(ddof=1)) if m != "boot-bloco" else np.nan,
                        reps=args.reps))
                print(f"  h={h} gamma={gamma}: {time.time() - t0:.0f}s")
    return pd.DataFrame(linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--N", type=int, default=300)
    ap.add_argument("--B", type=int, default=299, help="repeticoes do bootstrap por simulacao")
    ap.add_argument("--sig_lam", type=float, default=0.02, help="sd do fator macro x caixa por trimestre")
    ap.add_argument("--sig_e", type=float, default=0.10, help="sd idiossincratico por trimestre")
    ap.add_argument("--rho", type=float, default=0.9, help="persistencia do caixa")
    ap.add_argument("--phi", type=float, default=0.0, help="persistencia do fator macro")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.04])
    ap.add_argument("--horizontes", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--cenarios", nargs="+", default=["iid", "calibrada", "extrema"])
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    df = roda(args)
    df.to_csv("monte_carlo_resultados.csv", index=False)
    pd.set_option("display.width", 200)
    t0 = df[df.gamma == 0].pivot_table(index=["cenario", "h"], columns="metodo",
                                       values="rejeicao_H0_zero")
    print("\nTAMANHO: taxa de rejeicao a 5% com gamma = 0 (nominal = 0.05)")
    print(t0.round(3).to_string())
    r = df[(df.gamma == 0) & (df.metodo != "boot-bloco")].pivot_table(
        index=["cenario", "h"], columns="metodo", values="razao_se_sd")
    print("\nRAZAO  SE medio / desvio-padrao verdadeiro de beta  (1 = EP correto; <1 = EP subestimado)")
    print(r.round(2).to_string())
    if any(g > 0 for g in args.gammas):
        pw = df[df.gamma > 0].pivot_table(index=["cenario", "h", "gamma"], columns="metodo",
                                          values="rejeicao_H0_zero")
        print("\nPODER: rejeicao de H0: gamma = 0 quando gamma > 0")
        print(pw.round(3).to_string())
    print("\n-> monte_carlo_resultados.csv")


if __name__ == "__main__":
    main()