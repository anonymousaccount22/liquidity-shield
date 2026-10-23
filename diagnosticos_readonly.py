# diagnosticos_readonly.py — só lê painel_regressao_q_bp.csv
import numpy as np, pandas as pd

df = pd.read_csv("painel_regressao_q_bp.csv", low_memory=False)
try:
    tri = pd.PeriodIndex(df["Trimestre"].astype(str), freq="Q")
except Exception:
    tri = pd.to_datetime(df["Trimestre"]).dt.to_period("Q")
df["tri"] = tri
df["ano"] = pd.PeriodIndex(df["tri"], freq="Q").year

# E1: série do choque e nº de firmas por trimestre
sh = df.groupby("tri")["Shock_t"].first()
nf = df[df["Shock_t"].notna() & df["Cash_Ratio"].notna()].groupby("tri")["CNPJ_CIA"].nunique()
print(pd.DataFrame({"Shock": sh, "firmas": nf}).to_string())

# E2: variância do choque por ano e leave-out
s = sh.dropna()
d2 = (s - s.mean()) ** 2
print("\nParticipação na soma dos desvios² por ano:")
print((d2.groupby(s.index.year).sum() / d2.sum()).round(3))
for excl, lab in [((), "total"), ((2021,), "sem 2021"), ((2020,), "sem 2020"), ((2020, 2021), "sem 2020-21")]:
    x = s[~np.isin(s.index.year, excl)]
    print(f"{lab:12s} var={x.var():.5f}  n={len(x)}")

# E3: choque vs liquidez média (nível) e z médio
cr = df.groupby("tri")["Cash_Ratio"].mean()
z = df.groupby("tri")["Cash_Ratio"].transform(lambda v: (v - v.mean()) / v.std())
zm = z.groupby(df["tri"]).mean()
aux = pd.concat([sh, cr, zm], axis=1, keys=["Shock", "Cash_medio", "Cash_z_medio"]).dropna()
print("\nCorrelações:\n", aux.corr().round(3))

# E4: Cash_Ratio por ano
print("\n", df.groupby("ano")["Cash_Ratio"].describe(percentiles=[.01, .5, .99])[["count", "mean", "1%", "50%", "99%"]].round(4))