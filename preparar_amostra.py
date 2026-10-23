import numpy as np
import pandas as pd

# ---------------- parâmetros (registre no artigo) ----------------
MIN_TRIM = 12                 # mínimo de trimestres por empresa
Q_MIN, Q_MAX = 0.02, 20       # faixa plausível do Tobin's Q
FRAC_MAX = 0.30               # exclui empresa com >30% do Q fora da faixa
WINS = (0.01, 0.99)           # winsorização

p = pd.read_csv("painel_final_q.csv")
p["Lucro_Op_Ativo"] = p["Lucro_Operacional"] / p["Ativo_Total_L1"].where(p["Ativo_Total_L1"] > 0)
p["Tamanho"] = np.log(p["Ativo_Total_L1"].where(p["Ativo_Total_L1"] > 0))
print(f"Início: {len(p)} linhas, {p['CNPJ_CIA'].nunique()} empresas")

# ---------------- 1) limpeza do Tobin's Q ----------------
tem_q = p["Tobin_Q"].notna()
fora = tem_q & ((p["Tobin_Q"] > Q_MAX) | (p["Tobin_Q"] < Q_MIN))

# fração de Q fora da faixa, só entre as observações com Q
frac = fora.groupby(p["CNPJ_CIA"]).sum() / tem_q.groupby(p["CNPJ_CIA"]).sum().replace(0, np.nan)
excluir = frac[frac > FRAC_MAX].index
print(f"{len(excluir)} empresas excluídas por Q implausível (>{FRAC_MAX:.0%} fora de [{Q_MIN}, {Q_MAX}])")
p = p[~p["CNPJ_CIA"].isin(excluir)].copy()

# nas demais, o Q fora da faixa vira NaN só naquele trimestre
fora = p["Tobin_Q"].notna() & ((p["Tobin_Q"] > Q_MAX) | (p["Tobin_Q"] < Q_MIN))
p.loc[fora, "Tobin_Q"] = np.nan
print(f"{fora.sum()} observações de Q anuladas (erro pontual)")

# ---------------- 2) mínimo de trimestres por empresa ----------------
n0, e0 = len(p), p["CNPJ_CIA"].nunique()
cont = p.groupby("CNPJ_CIA")["Trimestre"].transform("size")
p = p[cont >= MIN_TRIM].copy()
print(f"Corte de {MIN_TRIM} trimestres: {n0} -> {len(p)} linhas; {e0} -> {p['CNPJ_CIA'].nunique()} empresas")

# ---------------- 3) winsorização (amostra toda) ----------------
VARS = ["Delta_Capex", "Cash_Ratio", "Alavancagem", "Margem_Operacional",
    "Receita_Ativo", "Lucro_Op_Ativo", "Tobin_Q", "Tamanho"]
for v in VARS:
    lo, hi = p[v].quantile(list(WINS))
    p[v] = p[v].clip(lo, hi)

# ---------------- 4) saída ----------------
p = p.sort_values(["CNPJ_CIA", "Trimestre"]).reset_index(drop=True)
p.to_csv("painel_regressao.csv", index=False)                       # Q pode ser NaN
pq = p.dropna(subset=["Tobin_Q"])
pq.to_csv("painel_regressao_q.csv", index=False)                    # só com Q

print("\n=== Resumo ===")
print(p[VARS].describe(percentiles=[.01, .5, .99]).T.to_string())
print(f"\npainel_regressao.csv   : {len(p)} obs., {p['CNPJ_CIA'].nunique()} empresas")
print(f"painel_regressao_q.csv : {len(pq)} obs., {pq['CNPJ_CIA'].nunique()} empresas, "
      f"{pq['Trimestre'].min()} a {pq['Trimestre'].max()}")
print("Firmas com Q por ano:")
print(pq.groupby(pq["Trimestre"].str[:4])["CNPJ_CIA"].nunique().to_string())