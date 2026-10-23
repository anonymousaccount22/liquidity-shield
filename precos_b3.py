import zipfile
import pandas as pd
from pathlib import Path

tk = set(pd.read_csv("tickers.csv")["Ticker"])
print(len(tk), "tickers na lista. Exemplo:", sorted(tk)[:5])

partes = []
for f in sorted(Path("dados_b3").glob("COTAHIST_A*.ZIP")):
    with zipfile.ZipFile(f) as z:
        nome = z.namelist()[0]
        with z.open(nome) as fh:
            linhas = fh.read().decode("latin1").splitlines()

    if f.name.endswith("A2010.ZIP"):
        print("Arquivo interno:", nome, "| linhas:", len(linhas))
        print("1ª linha:", linhas[0][:60])
        print("2ª linha:", linhas[1][:130])

    regs = []
    for ln in linhas:
        if ln[:2] != "01" or ln[24:27] != "010":
            continue
        if ln[10:12] not in ("02", "08"):
            continue
        tick = ln[12:24].strip()
        if tick not in tk:
            continue
        regs.append((ln[2:10], tick, int(ln[108:121]) / 100, int(ln[210:217])))

    d = pd.DataFrame(regs, columns=["DATA", "Ticker", "PREULT", "FATCOT"])
    d["Data"] = pd.to_datetime(d["DATA"], format="%Y%m%d", errors="coerce")
    d["Preco"] = d["PREULT"] / d["FATCOT"]
    partes.append(d[["Data", "Ticker", "Preco"]].dropna())
    print(f.name, len(d), "linhas")

px = pd.concat(partes).sort_values("Data")
px["Trimestre"] = px["Data"].dt.to_period("Q").astype(str)
q = px.groupby(["Ticker", "Trimestre"], as_index=False)["Preco"].last()
q[["Trimestre", "Ticker", "Preco"]].to_csv("precos_trimestrais.csv", index=False)
print(len(q), "linhas;", q["Ticker"].nunique(), "de", len(tk), "tickers com preço")
print(q[q["Ticker"] == "PETR4"].tail(4).to_string())