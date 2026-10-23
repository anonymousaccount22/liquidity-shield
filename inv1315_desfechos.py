"""
inv1315_desfechos.py
Hipoteses A (artefato contabil) e B (caixa usado para pagar divida) para a
inversao de 2013Q2-2015Q2.

PARTE 1 - desfechos alternativos. Reestima o gradiente de caixa (fora da janela e
dentro da janela) trocando o desfecho:
   ativo_fixo   : desfecho original do artigo (log AF t+h menos log AF t-1)  [baseline]
   imobilizado  : log Imobilizado (sem intangivel)            -> A: artefato contabil?
   capex_dfc    : soma de Capex_DFC_Ativo de t a t+h (fluxo)   -> A: o fluxo confirma?
   emprestimos  : (Emprest_{t+h} - Emprest_{t-1}) / Ativo_{t-1} -> B: pagou divida?
   caixa        : (Caixa_{t+h} - Caixa_{t-1}) / Ativo_{t-1}     -> B: consumiu caixa?

PARTE 2 - split por terciles (calculados dentro da amostra, com media de
2012Q1-2013Q1, ou seja, ANTES da janela) de Alavancagem e Tamanho, para os
desfechos ativo_fixo, emprestimos e caixa. Estima o gradiente da janela em cada
extremo (tercil baixo vs alto).

ATENCAO na leitura de 'emprestimos' e 'caixa': o regressor e o NIVEL defasado de
Caixa/Ativo, entao ha reversao a media mecanica. Compare o gradiente DENTRO da
janela com o de FORA, nao o nivel em si.

Metodo: troca placebos.serie_y (que build() usa) pelo desfecho escolhido e usa uma
copia de eventos.ajusta, com P1 = janela e P2 = 2021.
Requer na raiz: placebos.py, eventos.py, painel_regressao_q_bp.csv
(e, para capex_dfc, painel_regressao_q_dfc_ortog.csv).
NAO modifica nenhum arquivo existente.

Uso:  python inv1315_desfechos.py
Saidas: inv1315_desfechos.csv, inv1315_split.csv
"""
import numpy as np
import pandas as pd

import placebos as pb
from placebos import build, fit_full, G, G0, ZMAP
from eventos import W2021, comb

AMOSTRAS_TESTE = ["Restrita", "Total"]
HORIZONTES = [3, 5, 7]
ORD = lambda s: pd.Period(s, freq="Q").ordinal
JAN = list(range(ORD("2013Q2"), ORD("2015Q2") + 1))
W1 = np.isin(G, JAN).astype(float)
PRE_INI, PRE_FIM = ORD("2012Q1"), ORD("2013Q1")

PAINEL = "painel_regressao_q_bp.csv"
DFC = "painel_regressao_q_dfc_ortog.csv"
SERIE_Y_ORIG = pb.serie_y

# ------------------------------------------------------------------ dados
COLS = ["CNPJ_CIA", "t", "Ativo_Total", "Caixa", "Imobilizado",
        "Emprestimos_CP", "Emprestimos_LP", "Alavancagem", "Tamanho"]
B = pd.read_csv(PAINEL, usecols=COLS, dtype={"CNPJ_CIA": str})
B = B.drop_duplicates(["CNPJ_CIA", "t"]).reset_index(drop=True)
B["t"] = B["t"].astype(pb.p["t"].dtype)
B["Emp"] = B["Emprestimos_CP"] + B["Emprestimos_LP"]
BASE = B[["CNPJ_CIA", "t"]]
_IDX = {c: B.set_index(["CNPJ_CIA", "t"])[c] for c in
        ["Ativo_Total", "Caixa", "Imobilizado", "Emp"]}


def look(col, desloc):
    """Valor de col em t+desloc para cada linha (CNPJ, t) de BASE."""
    key = pd.MultiIndex.from_arrays([B["CNPJ_CIA"], B["t"] + desloc])
    return _IDX[col].reindex(key).to_numpy(float)


def empacota(y):
    out = BASE.copy()
    out["y"] = y
    return out.dropna(subset=["y"]).reset_index(drop=True)


def y_imobilizado(h):
    with np.errstate(all="ignore"):
        f, a = look("Imobilizado", h), look("Imobilizado", -1)
        f, a = np.where(f > 0, f, np.nan), np.where(a > 0, a, np.nan)
        return empacota(np.log(f) - np.log(a))


def y_emprestimos(h):
    at = look("Ativo_Total", -1)
    return empacota((look("Emp", h) - look("Emp", -1)) / np.where(at > 0, at, np.nan))


def y_caixa(h):
    at = look("Ativo_Total", -1)
    return empacota((look("Caixa", h) - look("Caixa", -1)) / np.where(at > 0, at, np.nan))


def carrega_capex():
    try:
        c = pd.read_csv(DFC, usecols=["CNPJ_CIA", "Trimestre", "Capex_DFC_Ativo"],
                        dtype={"CNPJ_CIA": str})
        c["t"] = pd.PeriodIndex(c["Trimestre"].astype(str), freq="Q").asi8
    except Exception as e:
        cab = pd.read_csv(DFC, nrows=0).columns.tolist() if __import__("os").path.exists(DFC) else "arquivo ausente"
        print(f"[capex_dfc] nao consegui ler {DFC} ({e}). Colunas: {cab}. Desfecho pulado.")
        return None
    c = c.drop_duplicates(["CNPJ_CIA", "t"])
    s = c["Capex_DFC_Ativo"].dropna()
    print(f"[capex_dfc] Capex_DFC_Ativo: media={s.mean():.4f} mediana={s.median():.4f} "
          f"% negativos={100 * (s < 0).mean():.1f}%  (se a maioria for negativa, o capex "
          f"esta como saida e o SINAL do coeficiente inverte)")
    return c.pivot(index="CNPJ_CIA", columns="t", values="Capex_DFC_Ativo")


WIDE = carrega_capex()


def y_capex_dfc(h):
    w = WIDE.reindex(columns=np.arange(WIDE.columns.min(), WIDE.columns.max() + 1 + h))
    acc = w.copy()
    for k in range(1, h + 1):
        acc = acc + w.shift(-k, axis=1)       # NaN se faltar algum trimestre
    out = acc.stack().rename("y").reset_index()
    out.columns = ["CNPJ_CIA", "t", "y"]
    out["t"] = out["t"].astype(pb.p["t"].dtype)
    return out


DESFECHOS = {"ativo_fixo": SERIE_Y_ORIG, "imobilizado": y_imobilizado,
             "emprestimos": y_emprestimos, "caixa": y_caixa}
if WIDE is not None:
    DESFECHOS["capex_dfc"] = y_capex_dfc

# tercis pre-janela por firma
PRE = (B[(B["t"] >= PRE_INI) & (B["t"] <= PRE_FIM)]
       .groupby("CNPJ_CIA")[["Alavancagem", "Tamanho"]].mean())


# ------------------------------------------------------------------ estimacao
def ajusta_F(am, h, grupo=None):
    """Copia de eventos.ajusta (P1 = janela, P2 = 2021) com filtro opcional de firmas.
    grupo = (variavel_pre, 'baixo'|'alto'), tercis calculados dentro da amostra."""
    d, x = build(am, h)
    if d is None:
        return None
    if grupo is not None:
        var, lado = grupo
        v = PRE[var].reindex(d["CNPJ_CIA"].unique()).dropna()
        lo, hi = v.quantile([1 / 3, 2 / 3])
        sel = set(v.index[v <= lo]) if lado == "baixo" else set(v.index[v >= hi])
        d = d[d["CNPJ_CIA"].isin(sel)].copy()
        if len(d) < pb.MIN_OBS:
            return None
    Ws = [("P1", W1), ("P2", W2021)]
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


def estima(am, h, grupo=None):
    try:
        out = ajusta_F(am, h, grupo)
    except Exception as e:
        return None, f"erro: {e}"
    if out is None:
        return None, "sem amostra suficiente"
    r, d = out
    a, b = "Ch_Caixa", "Ch_Caixa_P1"
    est, se, pj = comb(r, a, b)
    return dict(beta_fora=r.params[a], p_fora=r.pvalues[a],
                dif_janela=r.params[b], p_dif=r.pvalues[b],
                beta_janela=est, se_janela=se, p_janela=pj, n=int(r.nobs)), ""


def main():
    import sys
    fmt = lambda v: f"{v:.4f}"
    so_capex = len(sys.argv) > 1 and sys.argv[1] == "capex"   # python inv1315_desfechos.py capex
    lista = ({k: v for k, v in DESFECHOS.items() if k in ("ativo_fixo", "capex_dfc")}
             if so_capex else DESFECHOS)

    # ---------- PARTE 1: desfechos ----------
    linhas = []
    for nome, fn in lista.items():
        pb.serie_y = fn
        for am in AMOSTRAS_TESTE:
            for h in HORIZONTES:
                res, msg = estima(am, h)
                if res is None:
                    print(f"[pula] {nome} {am} h={h}: {msg}")
                    continue
                linhas.append(dict(desfecho=nome, amostra=am, h=h, **res))
        print(f"ok desfecho {nome}")
    pb.serie_y = SERIE_Y_ORIG
    d1 = pd.DataFrame(linhas)
    d1.to_csv("inv1315_desfechos.csv", index=False)

    base = d1[(d1.desfecho == "ativo_fixo") & (d1.h == 3)]
    print("\nBASELINE ativo_fixo h=3 (esperado beta_janela ~ -0.0926 Restrita, ~ -0.2399 Total):")
    print(base[["amostra", "beta_janela", "p_janela"]].to_string(index=False, float_format=fmt))
    print("\n=== PARTE 1: gradiente por desfecho ===")
    print(d1[["desfecho", "amostra", "h", "beta_fora", "p_fora", "dif_janela", "p_dif",
              "beta_janela", "p_janela", "n"]].to_string(index=False, float_format=fmt))

    if so_capex:
        print("\n[modo capex] parte 2 nao executada.")
        return

    # ---------- PARTE 2: splits ----------
    linhas = []
    for nome in ["ativo_fixo", "emprestimos", "caixa"]:
        pb.serie_y = DESFECHOS[nome]
        for var in ["Alavancagem", "Tamanho"]:
            for lado in ["baixo", "alto"]:
                for am in AMOSTRAS_TESTE:
                    for h in HORIZONTES:
                        res, msg = estima(am, h, grupo=(var, lado))
                        if res is None:
                            print(f"[pula] {nome} {var} {lado} {am} h={h}: {msg}")
                            continue
                        linhas.append(dict(desfecho=nome, split=var, tercil=lado,
                                           amostra=am, h=h, **res))
        print(f"ok split {nome}")
    pb.serie_y = SERIE_Y_ORIG
    d2 = pd.DataFrame(linhas)
    d2.to_csv("inv1315_split.csv", index=False)
    print("\n=== PARTE 2: gradiente da JANELA por tercil pre-janela ===")
    print(d2[["desfecho", "split", "tercil", "amostra", "h", "beta_fora", "beta_janela",
              "se_janela", "p_janela", "n"]].to_string(index=False, float_format=fmt))
    print("\nSalvos: inv1315_desfechos.csv, inv1315_split.csv")


if __name__ == "__main__":
    main()