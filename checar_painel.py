import pandas as pd

p = pd.read_csv("painel_final.csv")

print("\n=== 1) Firmas por trimestre ===")
print(p.groupby("Trimestre").size().to_string())

print("\n=== 2) Distribuição das variáveis (outliers) ===")
cols = ["Delta_Capex", "Cash_Ratio", "Alavancagem", "Margem_Operacional", "Receita_Ativo"]
print(p[cols].describe(percentiles=[.01, .05, .95, .99]).T.to_string())

print("\n=== 3) Trimestres por empresa (painel desbalanceado) ===")
print(p.groupby("CNPJ_CIA").size().describe().to_string())