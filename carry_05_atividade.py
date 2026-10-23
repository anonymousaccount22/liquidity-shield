"""
carry_05_atividade.py -- a inversao de 2013-15 e explicada por atividade fraca?
Requer internet (SGS/BCB). Reaproveita placebos.py/eventos.py. Nao altera nada.
Uso: python carry_05_atividade.py
"""
import numpy as np
import pandas as pd
import requests

import placebos as pb
from placebos import G, V, HS, AMOSTRAS_TESTE, rot
from eventos import ajusta, comb, W2021

COD_IBC = 24364     # CONFIRMAR no SGS: IBC-Br dessazonalizado (indice)
HOJE = pd.Timestamp("2026-10-17")


def baixa(cod, ini=2008, fim=2026, passo=5):
    partes = []
    for a in range(ini, fim + 1, passo):
        r = requests.get(
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{cod}/dados",
            params=dict(formato="json", dataInicial=f"01/01/{a}",
                        dataFinal=f"31/12/{min(a + passo - 1, fim)}"), timeout=60)
        r.raise_for_status()
        if r.json():
            partes.append(pd.DataFrame(r.json()))
    d = pd.concat(partes)
    d["data"] = pd.to_datetime(d["data"], format="%d/%m/%Y")
    d["valor"] = d["valor"].astype(float)
    s = d.drop_duplicates("data").sort_values("data").set_index("data")["valor"]
    return s[s.index <= HOJE]


def ords(ini, fim):
    return set(range(pd.Period(ini, freq="Q").ordinal, pd.Period(fim, freq="Q").ordinal + 1))


ibc = baixa(COD_IBC)
print("IBC-Br: primeiros/ultimos\n", ibc.head(2), "\n", ibc.tail(3))
per = ibc.index.to_period("Q")
q = ibc.groupby(per).agg(["mean", "count"])
q = q[q["count"] == 3]["mean"]                  # so trimestres completos
cresc = (q / q.shift(4) - 1) * 100              # crescimento anual (%)
cresc_L1 = cresc.shift(1)                       # atividade conhecida no inicio do trimestre
AT = pd.Series(cresc_L1.to_numpy(float),
               index=[p.ordinal for p in cresc_L1.index]).reindex(G).to_numpy(float)

ok = V[np.isfinite(AT[V])]
t1 = np.quantile(AT[ok], 1 / 3)
WL = np.where(np.isfinite(AT), (AT < t1).astype(float), np.nan)     # tercil de atividade baixa
W1 = np.isin(G, list(ords("2013Q2", "2015Q2"))).astype(float)
print(f"\nTrimestres com choque e atividade: {len(ok)} | corte do tercil baixo: {t1:.2f}% a.a.")
baixos = [int(G[i]) for i in ok if WL[i] == 1]
print("Atividade baixa:", " ".join(rot(o) for o in baixos))
print(f"Sobreposicao: {len(set(baixos) & ords('2013Q2', '2015Q2'))} de {len(baixos)} "
      f"trimestres de atividade baixa estao em 2013-15")


def estima():
    rows = []
    for am in AMOSTRAS_TESTE:
        for h in HS:
            sem = ajusta(am, h, [("P1", W1), ("P2", W2021)])
            com = ajusta(am, h, [("P1", W1), ("P2", W2021), ("P3", WL)])
            if sem is None or com is None:
                continue
            rs, ds = sem
            rc, dc = com
            e0, _, p0 = comb(rs, "Ch_Caixa", "Ch_Caixa_P1")
            e1, _, p1 = comb(rc, "Ch_Caixa", "Ch_Caixa_P1")
            rows.append(dict(amostra=am, h=h,
                             beta_2013_15_sem_ativ=e0, p_sem=p0,
                             beta_2013_15_com_ativ=e1, p_com=p1,
                             variacao=e1 - e0,
                             dif_ativ_baixa=rc.params["Ch_Caixa_P3"], p_ativ=rc.pvalues["Ch_Caixa_P3"],
                             n=int(rc.nobs), firmas=dc["CNPJ_CIA"].nunique(), T=dc["t"].nunique()))
    df = pd.DataFrame(rows)
    print("\n=== Gradiente de 2013-15 com e sem controle de atividade baixa ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


if __name__ == "__main__":
    pb.checar_baseline()
    estima().to_csv("carry_05_atividade.csv", index=False)