"""
inv1315_decomp.py
A inversao de 2013-15 esta no capex (fluxo) ou nas outras variacoes do ativo fixo?

Trabalha SOMENTE nas observacoes (firma, t) em que o capex da DFC existe para todos
os trimestres de t a t+h (amostra comum), para comparar desfechos no mesmo conjunto:
   af_log_comum     : desfecho original (log AF t+h menos log AF t-1), amostra comum
   delta_capex_bal  : soma de Delta_Capex (variacao do balanco / ativo) de t a t+h
   capex_dfc        : soma de Capex_DFC_Ativo (fluxo de caixa / ativo) de t a t+h
   outros_nao_capex : delta_capex_bal - capex_dfc
                      (depreciacao, baixas, impairment, cambio, aquisicoes, etc.)
Para cada desfecho estima o gradiente em (a) todas as firmas e (b) tercil alto de
Tamanho (onde a inversao se concentra).

Leitura:
 - af_log_comum e delta_capex_bal com inversao, capex_dfc sem inversao, e
   outros_nao_capex carregando a inversao -> a inversao esta em variacoes do ativo fixo
   que nao sao investimento (medida/contabilidade).
 - af_log_comum SEM inversao na amostra comum -> a DFC tem pouca cobertura/poder; nao
   da para concluir.
 - capex_dfc COM inversao -> a inversao e de investimento de fato.

Requer: inv1315_desfechos.py (importa dele), placebos.py, eventos.py e
painel_regressao_q_dfc_ortog.csv. NAO modifica nenhum arquivo existente.

Uso:  python inv1315_decomp.py
Saida: inv1315_decomp.csv
"""
import numpy as np
import pandas as pd

import placebos as pb
import inv1315_desfechos as D

C = pd.read_csv(D.DFC, usecols=["CNPJ_CIA", "Trimestre", "Delta_Capex", "Capex_DFC_Ativo"],
                dtype={"CNPJ_CIA": str})
C["t"] = pd.PeriodIndex(C["Trimestre"].astype(str), freq="Q").asi8
C = C.drop_duplicates(["CNPJ_CIA", "t"])
W_BAL = C.pivot(index="CNPJ_CIA", columns="t", values="Delta_Capex")
W_DFC = C.pivot(index="CNPJ_CIA", columns="t", values="Capex_DFC_Ativo")

ok = C.dropna(subset=["Delta_Capex", "Capex_DFC_Ativo"])
print(f"[checagem] Delta_Capex: media={ok['Delta_Capex'].mean():.4f}; "
      f"Capex_DFC_Ativo: media={ok['Capex_DFC_Ativo'].mean():.4f}; "
      f"corr={ok['Delta_Capex'].corr(ok['Capex_DFC_Ativo']):.3f} "
      f"(n={len(ok)} trimestres-firma)")

_CACHE = {}


def cum(w, h):
    g = w.reindex(columns=np.arange(w.columns.min(), w.columns.max() + 1 + h))
    acc = g.copy()
    for k in range(1, h + 1):
        acc = acc + g.shift(-k, axis=1)      # NaN se faltar algum trimestre
    return acc


def base_h(h):
    if h not in _CACHE:
        b, f = cum(W_BAL, h), cum(W_DFC, h)
        out = pd.concat([b.stack().rename("bal"), f.stack().rename("dfc")], axis=1)
        out = out.dropna().reset_index()
        out.columns = ["CNPJ_CIA", "t", "bal", "dfc"]
        out["t"] = out["t"].astype(pb.p["t"].dtype)
        out["outros"] = out["bal"] - out["dfc"]
        _CACHE[h] = out
    return _CACHE[h]


def mk(col):
    def f(h):
        b = base_h(h)
        return b[["CNPJ_CIA", "t", col]].rename(columns={col: "y"})
    return f


def y_af_comum(h):
    return D.SERIE_Y_ORIG(h).merge(base_h(h)[["CNPJ_CIA", "t"]], on=["CNPJ_CIA", "t"], how="inner")


DESFECHOS = {"af_log_comum": y_af_comum, "delta_capex_bal": mk("bal"),
             "capex_dfc": mk("dfc"), "outros_nao_capex": mk("outros")}


def main():
    fmt = lambda v: f"{v:.4f}"
    linhas = []
    for nome, fn in DESFECHOS.items():
        pb.serie_y = fn
        for grupo, rot in [(None, "todas"), (("Tamanho", "alto"), "tamanho_alto")]:
            for am in D.AMOSTRAS_TESTE:
                for h in D.HORIZONTES:
                    res, msg = D.estima(am, h, grupo)
                    if res is None:
                        print(f"[pula] {nome} {rot} {am} h={h}: {msg}")
                        continue
                    linhas.append(dict(desfecho=nome, firmas=rot, amostra=am, h=h, **res))
        print(f"ok {nome}")
    pb.serie_y = D.SERIE_Y_ORIG
    df = pd.DataFrame(linhas)
    df.to_csv("inv1315_decomp.csv", index=False)
    cols = ["desfecho", "firmas", "amostra", "h", "beta_fora", "p_fora", "dif_janela",
            "p_dif", "beta_janela", "se_janela", "p_janela", "n"]
    print("\n=== Amostra comum (DFC disponivel de t a t+h) ===")
    print(df[cols].to_string(index=False, float_format=fmt))
    print("\nSalvo: inv1315_decomp.csv")


if __name__ == "__main__":
    main()