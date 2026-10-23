import pandas as pd

p = pd.read_csv("painel_final_q.csv")
tk = pd.read_csv("tickers.csv")
nomes = pd.read_csv("painel_final.csv", usecols=["CNPJ_CIA"]).drop_duplicates()

ext = p[(p["Tobin_Q"] > 20) | (p["Tobin_Q"] < 0.02)]
top = ext.groupby("CNPJ_CIA").size().sort_values(ascending=False).head(15).index

for c in top:
    t = tk[tk["CNPJ_CIA"] == c][["Ticker", "Classe"]].values.tolist()
    s = p[p["CNPJ_CIA"] == c].set_index("Trimestre")["Tobin_Q"]
    print(f"\n{c} | tickers: {t}")
    print(s.dropna().iloc[::6].round(3).to_string())