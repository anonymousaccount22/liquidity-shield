"""
carry_00b_juros.py -- monta a taxa real defasada por trimestre (SGS/BCB).
Saida: carry_juros_trimestral.csv. Nao toca em nenhum arquivo existente.
"""
import numpy as np
import pandas as pd
import requests

COD_SELIC_META = 432    # CONFIRMAR no SGS: meta Selic (% a.a.)
COD_IPCA_MENSAL = 433   # CONFIRMAR no SGS: IPCA, variacao mensal (%)
ANO_INI, ANO_FIM = 2009, 2026


def baixa(cod, passo=5):
    partes = []
    for ano in range(ANO_INI, ANO_FIM + 1, passo):
        r = requests.get(
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{cod}/dados",
            params=dict(formato="json", dataInicial=f"01/01/{ano}",
                        dataFinal=f"31/12/{min(ano + passo - 1, ANO_FIM)}"), timeout=60)
        r.raise_for_status()
        if r.json():
            partes.append(pd.DataFrame(r.json()))
    d = pd.concat(partes)
    d["data"] = pd.to_datetime(d["data"], format="%d/%m/%Y")
    d["valor"] = d["valor"].astype(float)
    return d.drop_duplicates("data").sort_values("data").set_index("data")["valor"]


selic = baixa(COD_SELIC_META)
ipca = baixa(COD_IPCA_MENSAL)
print("Selic meta - primeiros/ultimos:\n", selic.head(2), "\n", selic.tail(2))
print("IPCA mensal - ultimos:\n", ipca.tail(3))

# Selic no fim de cada trimestre (ultimo valor vigente)
sd = selic.asfreq("D").ffill()
selic_fim = sd.groupby(sd.index.to_period("Q")).last()

# IPCA acumulado em 12 meses, lido no ultimo mes de cada trimestre
m = (ipca / 100 + 1)
m.index = m.index.to_period("M")
acc12 = m.rolling(12).apply(np.prod, raw=True) - 1
acc12_q = acc12[acc12.index.month.isin([3, 6, 9, 12])]
acc12_q.index = acc12_q.index.asfreq("Q")

df = pd.DataFrame({"selic_fim": selic_fim, "ipca12m_fim": acc12_q * 100}).dropna()
df = df.reindex(pd.period_range(df.index.min(), df.index.max(), freq="Q"))
df["real_fim"] = ((1 + df["selic_fim"] / 100) / (1 + df["ipca12m_fim"] / 100) - 1) * 100
# DEFASADO: o que vale no inicio do trimestre t e o valor do fim de t-1
df["selic_L1"] = df["selic_fim"].shift(1)
df["real_L1"] = df["real_fim"].shift(1)
df["var_selic_no_trim"] = df["selic_fim"] - df["selic_L1"]
df.index.name = "trimestre"
df = df.reset_index()
df["trimestre"] = df["trimestre"].astype(str)
df.to_csv("carry_juros_trimestral.csv", index=False)
print(df.round(2).to_string(index=False))
print("\nLimitacao: taxa real EX-POST (IPCA 12m realizado), nao ex-ante (sem Focus).")