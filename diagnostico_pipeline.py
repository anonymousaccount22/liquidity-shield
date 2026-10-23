import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

# ======================= CONFIGURAÇÃO =======================
DEP = "Capex_DFC_Ativo"  # "Delta_Capex" | "Capex_DFC_Ativo"
ACUM = True              # True = dependente = soma de DEP de t até t+h
EXCLUI_TRI = []  # trimestres de CHOQUE excluídos (mesma lógica do projecoes_locais.py)
FE_FIRMA = True
LIQ = "contemp"          # "contemp" | "pre"
DEFASA_CTRL = True
H_LIST = range(0, 9)            # comece só com o h=5; depois range(0, 9)
N_PERM = 200            # teste o tempo com 20 antes
SEED = 123
# ============================================================

ARQ = "painel_regressao_q_dfc.csv" if DEP == "Capex_DFC_Ativo" else "painel_regressao_q.csv"
p = pd.read_csv(ARQ)
p["t"] = pd.PeriodIndex(p["Trimestre"], freq="Q").astype(int)
p = p.sort_values(["CNPJ_CIA", "t"]).reset_index(drop=True)
p["ordem"] = p.groupby("CNPJ_CIA").cumcount()

def lag1(v):
    l = p[["CNPJ_CIA", "t", v]].copy()
    l["t"] += 1
    return p[["CNPJ_CIA", "t"]].merge(l, on=["CNPJ_CIA", "t"], how="left")[v].values

# choque padronizado na série trimestral completa (igual ao projecoes_locais.py)
tri = p.groupby("t").agg(Trimestre=("Trimestre", "first"), Choque_HFI=("Choque_HFI", "first"))
tri["Choque"] = (tri["Choque_HFI"] - tri["Choque_HFI"].mean()) / tri["Choque_HFI"].std()
p = p.merge(tri[["Choque"]], left_on="t", right_index=True, how="left")

p["Capex_L1"] = lag1(DEP)
p["Alav_c"] = lag1("Alavancagem") if DEFASA_CTRL else p["Alavancagem"]
p["Q_c"] = lag1("Tobin_Q") if DEFASA_CTRL else p["Tobin_Q"]
DEFASA_CTRL = globals().get("DEFASA_CTRL", True)
p["Lucro_c"] = lag1("Lucro_Op_Ativo") if DEFASA_CTRL else p["Lucro_Op_Ativo"]

if LIQ == "pre":
    for v in ["Cash_Ratio", "Alavancagem"]:
        p[v + "_u"] = p[v].where(p["ordem"] < 4).groupby(p["CNPJ_CIA"]).transform("mean")
else:
    p["Cash_Ratio_u"] = p["Cash_Ratio"]
    p["Alavancagem_u"] = p["Alav_c"]

def z(s): return (s - s.mean()) / s.std()
for v in ["Cash_Ratio_u", "Alavancagem_u", "Q_c", "Lucro_c"]:
    p[v + "_z"] = z(p[v])
CTRL = ["Cash_Ratio_u_z", "Alavancagem_u_z", "Q_c_z", "Lucro_c_z", "Capex_L1"]

reg = p.copy()
if LIQ == "pre":
    reg = reg[reg["ordem"] >= 4]
if EXCLUI_TRI:
    reg = reg[~reg["Trimestre"].isin(EXCLUI_TRI)]

# trimestres de choque que entram na regressão = conjunto a permutar
ts = np.sort(reg["t"].unique())
vals_por_t = tri["Choque"].reindex(ts).values
mapa_obs = dict(zip(ts, vals_por_t))

# dependente (lead ou soma t..t+h)
T_ALL = np.arange(p["t"].min(), p["t"].max() + 1)
piv_y = p.pivot(index="CNPJ_CIA", columns="t", values=DEP).reindex(columns=T_ALL)

def serie_y(h):
    if ACUM:
        m = np.sum([piv_y.shift(-k, axis=1).values for k in range(h + 1)], axis=0)
        m = pd.DataFrame(m, index=piv_y.index, columns=piv_y.columns)
    else:
        m = piv_y.shift(-h, axis=1)
    return m.stack().rename("y").reset_index()

base = {}
for h in H_LIST:
    d = reg.merge(serie_y(h), on=["CNPJ_CIA", "t"], how="inner") \
           .dropna(subset=["y"] + CTRL)
    d["t_val"] = d["t"]
    base[h] = d.set_index(["CNPJ_CIA", "t"])

def ajusta(h, mapa):
    d = base[h]
    ch = d["t_val"].map(mapa)
    X = pd.DataFrame({"Ch_Caixa": ch * d["Cash_Ratio_u_z"],
                      "Ch_Alav": ch * d["Alavancagem_u_z"]}, index=d.index).join(d[CTRL])
    m = PanelOLS(d["y"], X, entity_effects=FE_FIRMA, time_effects=True)
    return m.fit(cov_type="kernel", kernel="bartlett", bandwidth=h + 1)

rng = np.random.default_rng(SEED)
out = []
for h in H_LIST:
    r0 = ajusta(h, mapa_obs)
    c_obs, p_dk = r0.params["Ch_Caixa"], r0.pvalues["Ch_Caixa"]
    print(f"h={h}: observado = {c_obs:.5f} (p DK = {p_dk:.4f}, n = {int(r0.nobs)})", flush=True)
    c_perm, p_perm = [], []
    for i in range(N_PERM):
        mapa = dict(zip(ts, rng.permutation(vals_por_t)))
        r = ajusta(h, mapa)
        c_perm.append(r.params["Ch_Caixa"]); p_perm.append(r.pvalues["Ch_Caixa"])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{N_PERM}", flush=True)
    c_perm, p_perm = np.array(c_perm), np.array(p_perm)
    out.append({"h": h, "coef_obs": c_obs, "p_DK": p_dk,
                "p_aleat_bicaudal": (np.sum(np.abs(c_perm) >= abs(c_obs)) + 1) / (N_PERM + 1),
                "p_aleat_unilateral_neg": (np.sum(c_perm <= c_obs) + 1) / (N_PERM + 1),
                "rejeicao_5pct_DK_sob_placebo": (p_perm < 0.05).mean()})

res = pd.DataFrame(out)
print(res.round(4).to_string(index=False))
nome = f"diag_perm_{DEP}{'_acum' if ACUM else ''}{'_ex' + '-'.join(EXCLUI_TRI) if EXCLUI_TRI else ''}.csv"
res.to_csv(nome, index=False)
print("Salvo:", nome)