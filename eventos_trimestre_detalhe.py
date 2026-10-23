import numpy as np, pandas as pd
from scipy import stats
d = pd.read_csv("eventos_trimestre_detalhe.csv")
d["ano"] = d["trimestre"].str[:4].astype(int)
for (am, h), g in d.groupby(["amostra", "h"]):
    g = g.sort_values("t").reset_index(drop=True)
    print(f"\n{am} h={h}  top5:", g.nlargest(5, "beta")[["trimestre", "beta"]].values.tolist())
    print("            bot5:", g.nsmallest(5, "beta")[["trimestre", "beta"]].values.tolist())
    for nome, m in {"todos": g.ano > 0, "sem2021": g.ano != 2021, "sem2020-21": ~g.ano.isin([2020, 2021])}.items():
        s = g[m]
        r = stats.pearsonr(s.choque, s.beta)[0]
        rs = stats.spearmanr(s.choque, s.beta)[0]
        print(f"  {nome:11s} pearson={r:.3f} spearman={rs:.3f} inclinacao={np.polyfit(s.choque, s.beta, 1)[0]:.4f}")
    b, s0 = g.beta.to_numpy(), g.choque.to_numpy()
    nul = [np.corrcoef(np.roll(s0, k), b)[0, 1] for k in range(1, len(g))]
    obs = np.corrcoef(s0, b)[0, 1]
    print(f"  perm circular: p={(1 + np.sum(np.abs(nul) >= abs(obs))) / (1 + len(nul)):.3f}")