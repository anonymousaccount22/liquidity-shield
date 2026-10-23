import pandas as pd
from pathlib import Path

frames = [pd.read_csv(f, sep=";", encoding="latin1", dtype=str)
          for f in Path("dados_fca").rglob("*valor_mobiliario*.csv")]
vm = pd.concat(frames, ignore_index=True)
print("Colunas:", list(vm.columns))

cnpj_col = next(c for c in vm.columns if "CNPJ" in c.upper())
vm = vm.rename(columns={cnpj_col: "CNPJ_CIA"})

# Alguns campos trazem mais de um código; separa e explode
vm["Ticker"] = (vm["Codigo_Negociacao"].fillna("").str.upper().str.replace(r"[;,/]", " ", regex=True).str.split())
vm = vm.explode("Ticker")
vm = vm[vm["Ticker"].str.match(r"^[A-Z]{4}\d{1,2}$", na=False)].copy()

suf = vm["Ticker"].str[4:].astype(int)
vm["Classe"] = suf.map(lambda s: "ON" if s == 3 else "PN" if s in (4, 5, 6, 7, 8) else "UNIT" if s == 11 else None)
vm = vm.dropna(subset=["Classe"])

out = vm[["CNPJ_CIA", "Ticker", "Classe"]].drop_duplicates()
out.to_csv("tickers.csv", index=False)
print(f"{out['CNPJ_CIA'].nunique()} empresas, {len(out)} tickers")