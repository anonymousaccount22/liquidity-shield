"""
carry_06_cambio.py -- a inversao de 2013-15 encolhe ao controlar por regime de depreciacao?
Requer internet. Reaproveita placebos.py/eventos.py. Nao altera nada.
"""
import numpy as np
import pandas as pd
import requests

import placebos as pb
from placebos import G, V, HS, AMOSTRAS_TESTE, rot
from eventos import ajusta, comb, W2021

COD_USD = 1            # CONFIRMAR no SGS: dolar americano (venda), diario
HOJE = pd.Timestamp("2026-10-17")


def baixa(cod, ini=2008, fim=2026, passo=5):
    partes = []
    for a in range(ini, fim + 1, passo):
        r = requests.get(f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{cod}/dados",
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


usd = baixa(COD_USD)
print("USD: primeiros/ultimos\n", usd.head(2), "\n", usd.tail(3))
fimq = usd.groupby(usd.index.to_period("Q")).last()      # cotacao de fim de trimestre
dep = (fimq / fimq.shift(4) - 1) * 100                   # depreciacao em 4 trimestres (%)
dep_L1 = dep.shift(1)                                    # conhecida no inicio do trimestre
DP = pd.Series(dep_L1.to_numpy(float), index=[p.ordinal for p in dep_L1.index]).reindex(G).to_numpy(float)

ok = V[np.isfinite(DP[V])]
t2 = np.quantile(DP[ok], 2 / 3)
WC = np.where(np.isfinite(DP), (DP >= t2).astype(float), np.nan)
W1 = np.isin(G, list(ords("2013Q2", "2015Q2"))).astype(float)
alto = [int(G[i]) for i in ok if WC[i] == 1]
print(f"\nTrimestres com choque e cambio: {len(ok)} | corte tercil alto: {t2:.1f}% em 4 trim.")
print("Depreciacao alta:", " ".join(rot(o) for o in alto))
print(f"Sobreposicao com 2013-15: {len(set(alto) & ords('2013Q2', '2015Q2'))} de {len(alto)}")


def estima():
    rows = []
    for am in AMOSTRAS_TESTE:
        for h in HS:
            sem = ajusta(am, h, [("P1", W1), ("P2", W2021)])
            com = ajusta(am, h, [("P1", W1), ("P2", W2021), ("P3", WC)])
            if sem is None or com is None:
                continue
            rs, _ = sem
            rc, dc = com
            e0, _, p0 = comb(rs, "Ch_Caixa", "Ch_Caixa_P1")
            e1, _, p1 = comb(rc, "Ch_Caixa", "Ch_Caixa_P1")
            rows.append(dict(amostra=am, h=h, beta_sem_cambio=e0, p_sem=p0,
                             beta_com_cambio=e1, p_com=p1, razao=e1 / e0,
                             dif_cambio=rc.params["Ch_Caixa_P3"], p_cambio=rc.pvalues["Ch_Caixa_P3"],
                             n=int(rc.nobs), firmas=dc["CNPJ_CIA"].nunique(), T=dc["t"].nunique(),
                             confirma=bool(abs(e1) < 0.5 * abs(e0) and rc.pvalues["Ch_Caixa_P3"] < 0.05)))
    df = pd.DataFrame(rows)
    print("\n=== Gradiente de 2013-15 com e sem controle de depreciacao ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


if __name__ == "__main__":
    pb.checar_baseline()
    estima().to_csv("carry_06_cambio.csv", index=False)