"""
bootstrap_blocos.py -- inferencia por bootstrap de blocos de trimestres

Requer placebos.py, eventos.py e ciclos.py na MESMA pasta.
Uso:
    python bootstrap_blocos.py                       # B = 1000, amostra cheia (como antes)
    python bootstrap_blocos.py 300                   # B menor, para testar rapido
    python bootstrap_blocos.py 1000 --sem-entrantes  # NOVO: mesma inferencia SEM as entrantes

O que faz: reamostra TRIMESTRES (blocos moveis consecutivos), levando junto toda a cross-section
de firmas de cada trimestre sorteado, e reestima com efeito fixo de firma:
  (a) ctrl_int: coeficiente de Ch_Caixa;
  (b) modelo com regimes (dummies de 2013Q2-2015Q2 e 2021, como em ciclos.py):
      beta_calmo, beta_ciclo1315, beta_2021 e a diferenca (ciclo1315 - 2021).
Dois comprimentos de bloco: h+1 (igual ao bandwidth do DK) e 8 (sensibilidade).
Draws em que nenhum trimestre de 2013-15 ou de 2021 e sorteado sao descartados.
p_boot = 2*min(P(b*>=0), P(b*<=0)) (percentil, sem centralizar).

--sem-entrantes: o flag de entrante e calculado UMA vez, a partir do painel, antes da
reamostragem, e as firmas entrantes saem de todas as repeticoes. Antes de rodar, o script compara
os coeficientes pontuais com os valores de ipo_resultados.csv (ver ESPERADO_SEM_ENTRANTES) e
para se a diferenca for grande.
"""
import math
import sys
import time

import numpy as np
import pandas as pd

import placebos as pb
from placebos import build, ZMAP, HS, AMOSTRAS_TESTE, S0, SL10, Fast
from eventos import ajusta, W2021, Q2021
from ciclos import trimestres, JANELAS

args = [a for a in sys.argv[1:] if not a.startswith("--")]
SEM_ENTRANTES = "--sem-entrantes" in sys.argv
B = int(args[0]) if args else 1000
SEED = 20260
Q1315 = trimestres(*JANELAS["2013Q2-2015Q2 (principal)"])
ESQUEMAS = (("h+1", None), ("8", 8))

# --- definicao de entrante (AJUSTE AQUI se ipo.py definir de outro jeito) -----------------
PAINEL_CSV = "painel_regressao_q_bp.csv"
COL_FIRMA, COL_TRIM = "CNPJ_CIA", "Trimestre"
COL_OBS_REAL = "Cash_Ratio"            # observacao "real" = linha com esta coluna nao nula
ENTRADA_INI, ENTRADA_FIM = "2019Q1", "2021Q4"

# Valores do texto/auditoria (ipo_resultados.csv, Ch_Caixa sem entrantes), h = 3, 5, 7
ESPERADO_SEM_ENTRANTES = {
    ("Restrita", 3): 0.0219, ("Restrita", 5): 0.0332, ("Restrita", 7): 0.0354,
    ("Total", 3): 0.0158, ("Total", 5): 0.0206, ("Total", 7): 0.0020,
}
TOL_PARADA = 0.003    # diferenca maxima tolerada antes de parar
TOL_AVISO = 0.0005    # acima disso, apenas avisa


def flag_entrantes():
    """Conjunto de CNPJs cuja primeira observacao real cai em ENTRADA_INI..ENTRADA_FIM.
    Calculado uma vez, no painel, e mantido fixo em todas as repeticoes."""
    pan = pd.read_csv(PAINEL_CSV, usecols=[COL_FIRMA, COL_TRIM, COL_OBS_REAL])
    pan = pan.dropna(subset=[COL_OBS_REAL])
    primeira = pan.groupby(COL_FIRMA)[COL_TRIM].min()
    ent = primeira[(primeira >= ENTRADA_INI) & (primeira <= ENTRADA_FIM)].index
    print(f"[entrantes] {len(ent)} firmas com primeira observacao em {ENTRADA_INI}..{ENTRADA_FIM} "
          f"(de {len(primeira)} no painel)")
    return set(ent)


def prepara(am, h, excluir=None):
    d, x = build(am, h)
    if excluir:
        n0 = d[COL_FIRMA].nunique()
        d = d[~d[COL_FIRMA].isin(excluir)].reset_index(drop=True)
        print(f"   [{am} h={h}] entrantes removidas: {n0 - d[COL_FIRMA].nunique()} firmas; "
              f"{len(d)} obs.")
    t = d["t"].to_numpy()
    p1 = np.isin(t, list(Q1315)).astype(float)
    p2 = np.isin(t, list(Q2021)).astype(float)
    col = {v: d[v].to_numpy(float) for v in x}
    reg = {"P1": p1, "P2": p2, "Shock_P1": col["Shock"] * p1, "Shock_P2": col["Shock"] * p2}
    for v in x:
        if v in ZMAP:
            reg[v + "_P1"], reg[v + "_P2"] = col[v] * p1, col[v] * p2
    nb, nr = list(x), list(x) + list(reg)
    qs = np.unique(t)
    return dict(
        t=t, y=d["y"].to_numpy(float), codes=pd.factorize(d[COL_FIRMA])[0],
        Xb=np.column_stack([col[v] for v in nb]), nb=nb,
        Xr=np.column_stack([col[v] for v in x] + [reg[k] for k in reg]), nr=nr,
        qs=qs, idx={q: np.where(t == q)[0] for q in qs})


def within(X, y, codes, nf):
    cnt = np.maximum(np.bincount(codes, minlength=nf), 1).astype(float)
    dm = lambda a: a - (np.bincount(codes, weights=a, minlength=nf) / cnt)[codes]
    Xd = np.column_stack([dm(X[:, j]) for j in range(X.shape[1])])
    return np.linalg.lstsq(Xd, dm(y), rcond=None)[0]


def ajuste(P, rows):
    c, y = P["codes"][rows], P["y"][rows]
    nf = P["codes"].max() + 1
    bb = dict(zip(P["nb"], within(P["Xb"][rows], y, c, nf)))
    br = dict(zip(P["nr"], within(P["Xr"][rows], y, c, nf)))
    a, b1, b2 = br["Ch_Caixa"], br["Ch_Caixa_P1"], br["Ch_Caixa_P2"]
    return {"Ch_Caixa (ctrl_int)": bb["Ch_Caixa"], "beta_calmo": a,
            "beta_ciclo1315": a + b1, "beta_2021": a + b2, "dif_ciclo1315_menos_2021": b1 - b2}


def checar_reproducao():
    """Amostra cheia: reproduz os coeficientes originais (como na versao anterior)."""
    ok = True
    for am in AMOSTRAS_TESTE:
        for h in HS:
            P = prepara(am, h)
            est = ajuste(P, np.arange(len(P["y"])))
            d, x = build(am, h)
            c_fast = Fast(d, x).coefs(S0, SL10)["Ch_Caixa"]
            r, _ = ajusta(am, h, [("P1", np.isin(pb.G, list(Q1315)).astype(float)), ("P2", W2021)])
            f1 = abs(est["Ch_Caixa (ctrl_int)"] - c_fast) < 1e-8
            f2 = abs(est["dif_ciclo1315_menos_2021"] -
                     (r.params["Ch_Caixa_P1"] - r.params["Ch_Caixa_P2"])) < 1e-6
            ok &= f1 and f2
            print(f"[repro] {am:8s} h={h}: ctrl_int {'OK' if f1 else 'FALHA'} | regimes {'OK' if f2 else 'FALHA'}")
    if not ok:
        sys.exit("Reproducao falhou: nao confie no bootstrap ate entender a diferenca.")


def checar_sem_entrantes(excluir):
    """Compara Ch_Caixa (ctrl_int) sem entrantes com os valores de ipo_resultados.csv."""
    parar = False
    for am in AMOSTRAS_TESTE:
        for h in HS:
            P = prepara(am, h, excluir)
            est = ajuste(P, np.arange(len(P["y"])))["Ch_Caixa (ctrl_int)"]
            alvo = ESPERADO_SEM_ENTRANTES.get((am, h))
            if alvo is None:
                continue
            dif = abs(est - alvo)
            status = "OK" if dif <= TOL_AVISO else ("AVISO" if dif <= TOL_PARADA else "FALHA")
            parar |= status == "FALHA"
            print(f"[repro sem entrantes] {am:8s} h={h}: bootstrap-script {est:+.4f} | "
                  f"ipo_resultados {alvo:+.4f} | dif {dif:.4f} {status}")
    if parar:
        sys.exit("Coeficientes sem entrantes nao batem com ipo_resultados.csv. Causa provavel: "
                 "ordem entre exclusao, winsorizacao do y e z-scores (ipo.py pode excluir antes de "
                 "build). Alinhe flag_entrantes()/prepara() com ipo.py antes de confiar no bootstrap.")


def sorteia(qs, L, rng):
    n = len(qs)
    starts = rng.integers(0, n - L + 1, size=math.ceil(n / L))
    return np.concatenate([qs[s:s + L] for s in starts])[:n]


def resumo(nome, obs, draws):
    a = np.asarray(draws)
    return dict(estatistica=nome, estimativa=obs, se_boot=a.std(ddof=1),
                ic95_inf=np.quantile(a, .025), ic95_sup=np.quantile(a, .975),
                p_boot=max(2 * min(np.mean(a >= 0), np.mean(a <= 0)), 1 / (len(a) + 1)),
                draws_validos=len(a))


def rodar(excluir=None):
    rng, rows_out = np.random.default_rng(SEED), []
    filtro = "sem entrantes 2019Q1-2021Q4" if excluir else "nenhuma"
    for am in AMOSTRAS_TESTE:
        for h in HS:
            P = prepara(am, h, excluir)
            obs = ajuste(P, np.arange(len(P["y"])))
            for nome_esq, L in ESQUEMAS:
                L = h + 1 if L is None else L
                acc = {k: [] for k in obs}
                descartados = 0
                t1 = time.time()
                for _ in range(B):
                    sel = sorteia(P["qs"], L, rng)
                    if not (np.isin(sel, list(Q1315)).any() and np.isin(sel, list(Q2021)).any()):
                        descartados += 1
                        continue
                    rows = np.concatenate([P["idx"][q] for q in sel])
                    for k, v in ajuste(P, rows).items():
                        acc[k].append(v)
                for k in obs:
                    rows_out.append(dict(amostra=am, amostra_filtro=filtro, h=h,
                                         bloco=f"{nome_esq} (L={L})",
                                         draws_descartados=descartados,
                                         **resumo(k, obs[k], acc[k])))
                print(f"[{am} h={h} bloco={L} {filtro}] {len(acc['beta_calmo'])}/{B} validos, "
                      f"{descartados} descartados, {time.time() - t1:.0f}s")
    df = pd.DataFrame(rows_out)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


if __name__ == "__main__":
    pb.checar_baseline()
    if SEM_ENTRANTES:
        excl = flag_entrantes()
        checar_sem_entrantes(excl)
        df = rodar(excl)
        df.to_csv("bootstrap_blocos_sem_entrantes.csv", index=False)
        print("-> bootstrap_blocos_sem_entrantes.csv")
    else:
        checar_reproducao()
        df = rodar()
        df.to_csv("bootstrap_blocos.csv", index=False)
        print("-> bootstrap_blocos.csv")