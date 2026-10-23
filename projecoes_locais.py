import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ======================= CONFIGURAÇÃO =======================
# Desfecho: y_{i,t+h} = log_Ativo_Fixo_{i,t+h} - log_Ativo_Fixo_{i,t-1}  (Delta_log_Ativo_Fixo)
# Pré-tendências: H_MIN = -4 (h = -1 é zero por construção e é pulado)
# Só pós-choque : H_MIN = 1
H_MIN, H_MAX = -4, 8
ARQ = "painel_regressao_q_bp.csv"
MIN_OBS = 200          # mínimo de observações para estimar um horizonte
H_PRE = (-4, -3, -2)   # horizontes do teste de pré-tendências

# Inferência (matriz de covariância):
#   "dk"   : Driscoll-Kraay (kernel Bartlett) com lags = |h| + 1  [PADRÃO]
#   "firma": cluster por firma
#   "dupla": cluster por firma e tempo (instável com T pequeno)
# Por que "dk" é o padrão: o Shock varia apenas no tempo (um valor por trimestre, comum a todas
# as firmas). Cluster por firma não captura a dependência cross-sectional gerada por esse
# regressor comum e tende a subestimar o erro-padrão de β1. Driscoll-Kraay é robusto a
# heterocedasticidade, autocorrelação serial E dependência entre firmas.
INFERENCIA = "dk"

WINSOR_P = (0.01, 0.99)   # limites da winsorização

# Escala e sinal do choque monetário
# Unidade em que 'Shock_t' vem no arquivo: "bp" (pontos-base) ou "pp" (pontos percentuais).
# Choque_HFI = Taxa(T+1) - Taxa(T) com a taxa em % a.a. => já está em PONTOS PERCENTUAIS.
UNIDADE_SHOCK_ORIGEM = "pp"
# "100bp"  : +1 unidade = +100 pb (+1 p.p.) de aperto  [PADRÃO]
# "zscore" : padroniza (média 0, desvio 1) -- convenção da versão anterior
ESCALA_SHOCK = "100bp"
# Convenção algébrica: Choque_HFI = DI(T+1) - DI(T) > 0  => juros sobem => APERTO.
# Se o sinal da sua série vier oposto, mude para True (inverte ANTES das interações).
inverter_sinal_choque = False

# Dupla amostra
# Roda também "Restrita_semQ": mesma amostra restrita, mas SEM o controle de Tobin's Q.
# Compará-la com "Total" isola o efeito da SELEÇÃO amostral; compará-la com "Restrita"
# isola o efeito de INCLUIR o Q. Desative se quiser economizar tempo.
RODAR_DIAGNOSTICO_SELECAO = True

# [NOVO] Controla pelo choque defasado (Shock_{t-1}) nas specs sem efeito fixo de tempo.
# Motivo: se o choque é autocorrelacionado (apertos em sequência), o choque de t-1 contamina
# as pré-tendências e o efeito em h>=0. Nas specs com FE setor×trimestre ele é colineares
# (varia só no tempo) e, por isso, NÃO entra nelas.
CONTROLE_SHOCK_L1 = True

# [NOVO] Robustez: anos cujas observações (data t da firma) são EXCLUÍDAS das regressões.
# O desfecho continua sendo construído com a série completa; só as linhas-base saem.
# Ex.: (2020, 2021) testa se o resultado depende da pandemia/aperto de 2021.
# ATENÇÃO: se usar, o SUFIXO muda e o main.py precisa apontar para o novo nome do arquivo.
EXCLUIR_ANOS: tuple = ()
# ============================================================

DESFECHO = "Delta_log_Ativo_Fixo"
SUFIXO = "_Delta_log_Ativo_Fixo_ctrlL1"
if EXCLUIR_ANOS:
    SUFIXO += "_sem" + "_".join(str(a) for a in EXCLUIR_ANOS)

if INFERENCIA not in ("dupla", "firma", "dk"):
    raise ValueError("INFERENCIA deve ser 'dupla', 'firma' ou 'dk'")
if UNIDADE_SHOCK_ORIGEM not in ("bp", "pp"):
    raise ValueError("UNIDADE_SHOCK_ORIGEM deve ser 'bp' ou 'pp'")
if ESCALA_SHOCK not in ("100bp", "zscore"):
    raise ValueError("ESCALA_SHOCK deve ser '100bp' ou 'zscore'")


# ---------------- FUNÇÕES AUXILIARES ----------------
def zscore(s):
    """Z-score; devolve NaN se não houver variação (evita divisão por zero)."""
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def winsor(s, lo=WINSOR_P[0], hi=WINSOR_P[1]):
    """Winsorização por clip nos quantis lo/hi (ignora NaN)."""
    s = pd.Series(s)
    if s.notna().sum() == 0:
        return s
    return s.clip(s.quantile(lo), s.quantile(hi))


def n_lags(h, n_t):
    """
    Lags de Newey-West/Bartlett proporcionais ao horizonte: lags = |h| + 1.
    O resíduo de uma projeção local a h passos tem estrutura MA(h-1) por construção
    (as janelas de y se sobrepõem), então h+1 lags é uma escolha conservadora que cobre essa
    autocorrelação mecânica. Para h < 0 (pré-tendências) usa-se |h| + 1, pois a janela
    [t+h, t-1] também se sobrepõe. Limitado a n_t - 1 (T pequeno).
    """
    return int(min(abs(h) + 1, n_t - 1))


def cov_kwargs(h, n_t):
    if INFERENCIA == "dupla":
        return dict(cov_type="clustered", cluster_entity=True, cluster_time=True)
    if INFERENCIA == "firma":
        # cluster por firma já é robusto a autocorrelação ARBITRÁRIA dentro da firma:
        # não há parâmetro de lags a ajustar
        return dict(cov_type="clustered", cluster_entity=True)
    # Driscoll-Kraay (linearmodels: cov_type="kernel"); bandwidth = número de lags
    return dict(cov_type="kernel", kernel="bartlett", bandwidth=n_lags(h, n_t))


def ano_de_t(t):
    """Ano a partir do ordinal trimestral (Period 'Q': ordinal 0 = 1970Q1)."""
    return 1970 + np.asarray(t) // 4


def prepara_choque(p):
    """
    Série trimestral do choque (um valor por trimestre), já na escala escolhida:
      1) converte a unidade de origem para pontos percentuais (1.0 = 100 pb);
      2) aplica a inversão de sinal opcional;
      3) (opcional) z-score.
    Com ESCALA_SHOCK='100bp', o coeficiente β1 de Shock é a resposta do desfecho a +100 pb.
    """
    tri = p.groupby("t")["Shock_t"].first().astype(float)
    if UNIDADE_SHOCK_ORIGEM == "bp":
        tri = tri / 100.0                 # pb -> pp  (100 pb = 1.0)
    if inverter_sinal_choque:
        tri = -tri
        print(">>> Sinal do choque INVERTIDO (inverter_sinal_choque=True).")

    # --- diagnósticos de unidade e de sinal ---
    sd = tri.std()
    print(f"Choque (unidade = 100 pb): média = {tri.mean():.4f} | desvio = {sd:.4f} | "
          f"min = {tri.min():.4f} | max = {tri.max():.4f} | T = {len(tri)}")
    if not (0.02 <= sd <= 1.5):
        warnings.warn(
            f"Desvio-padrão do choque = {sd:.4f} (em 100 pb) foge do esperado (~0,05-0,5). "
            f"Verifique UNIDADE_SHOCK_ORIGEM ('{UNIDADE_SHOCK_ORIGEM}').")
    anos = [pd.Period(ordinal=int(o), freq="Q").year for o in tri.index]
    print("Choque médio por ano (100 pb) -- anos de aperto da Selic (ex.: 2021-22) devem ter "
          "média positiva; anos de afrouxamento (ex.: 2023-24), negativa:")
    print(tri.groupby(anos).mean().round(4).to_string(), "\n")

    # [NOVO] autocorrelação do choque (trimestres consecutivos; NaN quebra a sequência)
    cont = tri.reindex(np.arange(tri.index.min(), tri.index.max() + 1))
    ac1, ac2 = cont.autocorr(1), cont.autocorr(2)
    print(f"Autocorrelação do choque: lag1 = {ac1:.3f} | lag2 = {ac2:.3f} "
          f"(|valor| > ~{2 / np.sqrt(cont.notna().sum()):.2f} sugere persistência relevante)\n")

    if ESCALA_SHOCK == "zscore":
        tri = (tri - tri.mean()) / tri.std()
    return tri.rename("Shock")


# ---------------- DADOS ----------------
p = pd.read_csv(ARQ, dtype={"CNPJ_CIA": str})

obrig = ["CNPJ_CIA", "log_Ativo_Fixo", "Shock_t", "SETOR_ATIV",
         "Cash_Ratio", "Alavancagem", "Tobin_Q", "Tamanho", "Fluxo_Caixa_Op_Ativo"]
faltam = [c for c in obrig if c not in p.columns]
if faltam:
    raise KeyError(f"Colunas ausentes em {ARQ}: {faltam}")

if "Trimestre" in p.columns:
    p["t"] = pd.PeriodIndex(p["Trimestre"], freq="Q").asi8
elif "t" not in p.columns:
    raise KeyError(f"{ARQ} precisa ter 'Trimestre' ou 't'.")
p["t"] = p["t"].astype("int64")
p = p.sort_values(["CNPJ_CIA", "t"]).reset_index(drop=True)
if p.duplicated(["CNPJ_CIA", "t"]).any():
    raise AssertionError("Há pares (CNPJ_CIA, t) duplicados na base.")

# log_Ativo_Fixo deve ser finito (log de não positivos já virou NaN em capex_bp.py)
p["log_Ativo_Fixo"] = p["log_Ativo_Fixo"].where(np.isfinite(p["log_Ativo_Fixo"]))
print(f"Desfecho: {DESFECHO} | obs. com log_Ativo_Fixo válido: "
      f"{int(p['log_Ativo_Fixo'].notna().sum())} de {len(p)} "
      f"({p.loc[p['log_Ativo_Fixo'].notna(), 'CNPJ_CIA'].nunique()} empresas)")
if EXCLUIR_ANOS:
    print(f">>> Robustez: observações dos anos {EXCLUIR_ANOS} excluídas das regressões.")

# Winsorização 1%/99% das variáveis de balanço em nível, ANTES dos lags e do z-score
NIVEL = ["Cash_Ratio", "Alavancagem", "Tobin_Q", "Tamanho", "Fluxo_Caixa_Op_Ativo"]
for v in NIVEL:
    p[v] = winsor(p[v])

# Remove defasagens já exportadas por outros scripts (evita _x/_y no merge abaixo);
# aqui elas são recalculadas a partir dos valores winsorizados
p = p.drop(columns=[c for c in ("Alavancagem_L1", "Tobin_Q_L1", "Fluxo_Caixa_Op_Ativo_L1")
                    if c in p.columns])

# Controles pré-determinados: tudo em t-1 (lag por MERGE em t+1: nunca pula trimestre)
for variable in ["Alavancagem", "Tobin_Q", "Fluxo_Caixa_Op_Ativo"]:
    lag = p[["CNPJ_CIA", "t", variable]].copy()
    lag["t"] += 1
    lag = lag.rename(columns={variable: variable + "_L1"})
    p = p.merge(lag, on=["CNPJ_CIA", "t"], how="left")

# Choque na escala/sinal escolhidos (ANTES de construir as interações)
tri = prepara_choque(p)
p = p.merge(tri, left_on="t", right_index=True, how="left")

# [NOVO] Choque defasado: série trimestral contínua -> shift(1) == trimestre t-1
tri_cont = tri.reindex(np.arange(tri.index.min(), tri.index.max() + 1))
shock_l1 = tri_cont.shift(1).rename("Shock_L1")
p = p.merge(shock_l1, left_on="t", right_index=True, how="left")

# Padronização cross-sectional dentro de cada trimestre
for v in ["Cash_Ratio", "Alavancagem_L1", "Tobin_Q_L1", "Tamanho", "Fluxo_Caixa_Op_Ativo_L1"]:
    p[v + "_z"] = p.groupby("t")[v].transform(zscore)

# Interações construídas APÓS o z-score trimestral
# (os z-scores usam o cross-section COMPLETO de cada trimestre, também para a amostra restrita:
#  assim as interações têm a mesma escala nas duas amostras e os coeficientes são comparáveis)
p["Ch_Caixa"] = p["Shock"] * p["Cash_Ratio_z"]
p["Ch_Alav"] = p["Shock"] * p["Alavancagem_L1_z"]
p["Ch_Q"] = p["Shock"] * p["Tobin_Q_L1_z"]
p["Ch_Tam"] = p["Shock"] * p["Tamanho_z"]                      # [NOVO]
p["Ch_FCO"] = p["Shock"] * p["Fluxo_Caixa_Op_Ativo_L1_z"]      # [NOVO]

# ---------------- DEFINIÇÃO DAS AMOSTRAS ----------------
# Restrita = observações com Tobin_Q(t-1) válido. Reduz a amostra, o que pode introduzir
# viés de seleção (firmas com preço/ações disponíveis tendem a ser maiores e mais líquidas).
# A amostra Total não usa Q e mantém todas as observações.
p["in_restrita"] = p["Tobin_Q_L1_z"].notna()
print(f"\nAmostra Total: {len(p)} obs. | Restrita (Tobin_Q_L1 válido): "
      f"{int(p['in_restrita'].sum())} obs. "
      f"({p.loc[p['in_restrita'], 'CNPJ_CIA'].nunique()} de {p['CNPJ_CIA'].nunique()} empresas)")

AMOSTRAS = {"Total": dict(com_q=False), "Restrita": dict(com_q=True)}
if RODAR_DIAGNOSTICO_SELECAO:
    AMOSTRAS["Restrita_semQ"] = dict(com_q=False)


def dados_da_amostra(nome):
    return p if nome == "Total" else p[p["in_restrita"]]


def monta_specs(com_q):
    """
    Especificações de uma amostra. Sem Q: tira Tobin_Q_L1_z dos controles e Ch_Q das interações.
    Interações com Tamanho e FCO entram sempre (testam se o canal do caixa é específico).
    Shock_L1 só entra onde não há FE de tempo (nas specs com FE setor×trimestre seria absorvido).
    """
    ctrl = ["Cash_Ratio_z", "Alavancagem_L1_z"] + (["Tobin_Q_L1_z"] if com_q else []) + \
           ["Tamanho_z", "Fluxo_Caixa_Op_Ativo_L1_z"]
    ctrl_t = (["Shock_L1"] if CONTROLE_SHOCK_L1 else []) + ctrl
    ints = ["Ch_Caixa", "Ch_Alav", "Ch_Tam", "Ch_FCO"] + (["Ch_Q"] if com_q else [])
    # tempo_fe=False: Shock identificado (efeito fixo só de firma)
    # tempo_fe=True : efeito fixo setor-trimestre absorve Shock => só interações
    return {
        "base":     dict(x_int=["Shock"], x=["Shock"] + ctrl_t, tempo_fe=False),
        "ctrl_int": dict(x_int=["Shock"] + ints, x=["Shock"] + ints + ctrl_t, tempo_fe=False),
        "setor_t":  dict(x_int=ints, x=ints + ctrl, tempo_fe=True),
    }


# ---------------- DIAGNÓSTICO DE SELEÇÃO (quem sai ao exigir Tobin's Q?) ----------------
def diagnostico_selecao(df):
    """Diferença padronizada de médias: firmas na amostra Restrita vs. fora dela."""
    vars_ = ["log_Ativo_Fixo", "Cash_Ratio", "Alavancagem_L1", "Tamanho", "Fluxo_Caixa_Op_Ativo_L1"]
    a, b = df[df["in_restrita"]], df[~df["in_restrita"]]
    linhas = []
    for v in vars_:
        sd = np.sqrt((a[v].var() + b[v].var()) / 2)
        linhas.append({"var": v, "media_restrita": a[v].mean(), "media_fora": b[v].mean(),
                       "dif_padronizada": (a[v].mean() - b[v].mean()) / sd if sd > 0 else np.nan,
                       "n_restrita": int(a[v].notna().sum()), "n_fora": int(b[v].notna().sum())})
    return pd.DataFrame(linhas)


sel = diagnostico_selecao(p)
print("\n===== Seleção: Restrita vs. fora da Restrita (|dif. padronizada| > 0,2 = relevante) =====")
print(sel.round(4).to_string(index=False))
sel.to_csv(f"diagnostico_selecao{SUFIXO}.csv", index=False)

# ---------------- split por mediana da alavancagem pré-choque ----------------
mediana_alav = p["Alavancagem_L1"].median()     # mediana da base completa (grupos consistentes)
p["grupo_alav"] = (
    pd.Series(np.where(p["Alavancagem_L1"] <= mediana_alav, "Baixa", "Alta"), index=p.index)
    .where(p["Alavancagem_L1"].notna())
)
print(f"\nInferência: {INFERENCIA}"
      + (" (lags = |h| + 1)" if INFERENCIA == "dk" else ""))
print(f"Controle Shock_L1: {CONTROLE_SHOCK_L1}")
print(f"Mediana Alavancagem(t-1) = {mediana_alav:.4f}")
print(p["grupo_alav"].value_counts(dropna=False).to_string(), "\n")


# ---------------- DIAGNÓSTICO VIF (spec CTRL_INT, por amostra) ----------------
def calcula_vif(X):
    X = X.assign(const=1.0)
    cols = list(X.columns)
    return pd.Series([variance_inflation_factor(X.values, i) for i in range(len(cols))],
                     index=cols).drop("const")


vifs = []
for nome in ("Total", "Restrita"):
    x = monta_specs(AMOSTRAS[nome]["com_q"])["ctrl_int"]["x"]
    Xv = dados_da_amostra(nome)[x + ["CNPJ_CIA"]].dropna()
    Xw = Xv[x] - Xv.groupby("CNPJ_CIA")[x].transform("mean")   # within-firma (o que identifica o FE)
    vifs.append(pd.DataFrame({"amostra": nome, "VIF_nivel": calcula_vif(Xv[x]),
                              "VIF_within_firma": calcula_vif(Xw)}))
    print(f"===== VIF (ctrl_int, amostra {nome}, n = {len(Xv)}) =====")
    print(vifs[-1][["VIF_nivel", "VIF_within_firma"]].round(3).to_string())
pd.concat(vifs).to_csv(f"vif_ctrl_int{SUFIXO}.csv")

# ---------------- SÉRIE DEPENDENTE ----------------
T_ALL = np.arange(p["t"].min(), p["t"].max() + 1)
piv_log_af = (p.pivot(index="CNPJ_CIA", columns="t", values="log_Ativo_Fixo")
              .reindex(columns=T_ALL))
piv_log_af.columns.name = "t"


def serie_y(h):
    """y = log AtivoFixo(t+h) - log AtivoFixo(t-1); h pode ser negativo.
    Colunas = grade trimestral contínua, então shift de colunas == deslocamento em trimestres
    (trimestre ausente vira NaN, nunca um trimestre errado)."""
    lead = piv_log_af.shift(-h, axis=1)
    baseline = piv_log_af.shift(1, axis=1)
    return (lead - baseline).stack().rename("y").reset_index()


# ---------------- PROJEÇÕES LOCAIS ----------------
RES_COLS = ["amostra", "subgrupo", "spec", "h", "var", "coef", "se", "p", "n", "n_firmas",
            "inferencia", "lags_dk"]


def lp(spec_nome, dados, amostra, subgrupo, specs):
    spec = specs[spec_nome]
    x, x_int = spec["x"], spec["x_int"]
    tag = f"[{amostra}|{subgrupo}|{spec_nome}]"
    res = []
    for h in range(H_MIN, H_MAX + 1):
        if h == -1:   # zero por construção (baseline = t-1)
            continue
        y = serie_y(h)
        d = dados.merge(y, on=["CNPJ_CIA", "t"], how="inner")
        d = d.dropna(subset=["y"] + x + ["SETOR_ATIV"]).copy()
        if EXCLUIR_ANOS:                                   # [NOVO] robustez por período
            d = d[~np.isin(ano_de_t(d["t"]), EXCLUIR_ANOS)]
        if len(d) < MIN_OBS:
            print(f"{tag} h={h}: poucas obs ({len(d)}), pulado")
            continue

        # Winsorização 1%/99% da variável dependente em cada horizonte
        d["y"] = winsor(d["y"])

        n_t = d["t"].nunique()
        if spec["tempo_fe"]:
            d["setor_t"] = d["SETOR_ATIV"].astype(str) + "|" + d["t"].astype(str)
        d = d.set_index(["CNPJ_CIA", "t"])

        if spec["tempo_fe"]:
            model = PanelOLS(d["y"], d[x], entity_effects=True, other_effects=d[["setor_t"]])
        else:
            model = PanelOLS(d["y"], d[x], entity_effects=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = model.fit(**cov_kwargs(h, n_t))
        if any("sqrt" in str(wi.message) for wi in w) or r.std_errors.isna().any():
            print(f"{tag} h={h}: AVISO - variância inválida (SE NaN) com inferencia='{INFERENCIA}'")

        nf = d.index.get_level_values(0).nunique()
        bw = n_lags(h, n_t) if INFERENCIA == "dk" else np.nan

        for v in x_int:
            res.append({"amostra": amostra, "subgrupo": subgrupo, "spec": spec_nome, "h": h,
                        "var": v, "coef": r.params[v], "se": r.std_errors[v], "p": r.pvalues[v],
                        "n": int(r.nobs), "n_firmas": nf, "inferencia": INFERENCIA, "lags_dk": bw})
        print(f"{tag} h={h}: n={int(r.nobs)}, empresas={nf}, T={n_t}"
              + (f", lags DK={bw}" if INFERENCIA == "dk" else ""))
    out = pd.DataFrame(res, columns=RES_COLS)
    # margens de erro para plotagem: IC 95% e 90%
    out["lo95"] = out["coef"] - 1.96 * out["se"]
    out["hi95"] = out["coef"] + 1.96 * out["se"]
    out["lo90"] = out["coef"] - 1.645 * out["se"]
    out["hi90"] = out["coef"] + 1.645 * out["se"]
    return out


# Roda as mesmas especificações em cada amostra
todos = []
for nome, cfg in AMOSTRAS.items():
    dados = dados_da_amostra(nome)
    specs = monta_specs(cfg["com_q"])
    print(f"\n########## AMOSTRA {nome}: {len(dados)} obs. | controle de Q: {cfg['com_q']} ##########")
    todos.append(lp("base", dados, nome, "Cheia", specs))                 # IRF de β1 (+ pré-tendências)
    for g in ["Baixa", "Alta"]:                                           # split por alavancagem
        todos.append(lp("base", dados[dados["grupo_alav"] == g], nome, g, specs))
    todos.append(lp("ctrl_int", dados, nome, "Cheia", specs))             # interações com controles
    todos.append(lp("setor_t", dados, nome, "Cheia", specs))              # FE setor×trimestre

res = pd.concat(todos, ignore_index=True)

# ---------------- EXPORTAÇÕES ----------------
# (a) IRFs consolidadas (todas as amostras/specs/variáveis) com erros-padrão e ICs
cols_irf = ["amostra", "subgrupo", "spec", "var", "h", "coef", "se", "p",
            "lo95", "hi95", "lo90", "hi90", "n", "n_firmas", "inferencia", "lags_dk"]
res[cols_irf].to_csv(f"irf_consolidado{SUFIXO}.csv", index=False)


# (b) Tabela lado a lado Total vs. Restrita (coeficiente e p-valor)
def tabela_comparativa(df, var, a1="Total", a2="Restrita"):
    d = df[(df["var"] == var) & (df["subgrupo"] == "Cheia") & (df["amostra"].isin([a1, a2]))]
    if d.empty:
        return pd.DataFrame()
    w = d.pivot_table(index=["spec", "h"], columns="amostra", values=["coef", "p", "se", "n"])
    w.columns = [f"{m}_{a}" for m, a in w.columns]
    w = w.reset_index()
    ordem = ([f"{m}_{a}" for a in (a1, a2) for m in ("coef", "p")] +
             [f"{m}_{a}" for m in ("se", "n") for a in (a1, a2)])
    ordem = [c for c in ordem if c in w.columns]
    if f"coef_{a1}" in w.columns and f"coef_{a2}" in w.columns:
        w["dif_coef"] = w[f"coef_{a2}"] - w[f"coef_{a1}"]
        ordem.append("dif_coef")
    w.insert(0, "var", var)
    return w[["var", "spec", "h"] + ordem]


comp = pd.concat([tabela_comparativa(res, v)
                  for v in ("Ch_Caixa", "Ch_Tam", "Ch_FCO", "Shock")], ignore_index=True)
comp.to_csv(f"comparacao_total_vs_restrita{SUFIXO}.csv", index=False)

print("\n===== Ch_Caixa: Total (sem Q) vs. Restrita (com Q) =====")
cc = comp[comp["var"] == "Ch_Caixa"]
print(cc.drop(columns="var").round(5).to_string(index=False) if not cc.empty
      else "(Ch_Caixa não estimado)")

for nome, df in [("BASE (β1)", res[(res["spec"] == "base") & (res["subgrupo"] == "Cheia")]),
                 ("CTRL_INT", res[res["spec"] == "ctrl_int"]),
                 ("SETOR_T", res[res["spec"] == "setor_t"])]:
    print(f"\n===== {nome} - {DESFECHO}: coeficientes =====")
    if df.empty:
        print("(sem horizontes estimados)")
        continue
    print(df.pivot_table(index="h", columns=["amostra", "var"], values="coef").round(5).to_string())
    print(f"===== {nome}: p-valores =====")
    print(df.pivot_table(index="h", columns=["amostra", "var"], values="p").round(4).to_string())


# ---------------- TESTE AUTOMÁTICO DE PRÉ-TENDÊNCIAS (h = -4, -3, -2) ----------------
def teste_pre_tendencia(df, rotulo, vars_teste):
    """β deve ser estatisticamente nulo antes do choque. Reporta por horizonte e por variável."""
    pre = df[df["h"].isin(H_PRE) & df["var"].isin(vars_teste)].copy()
    print(f"\n=== Pré-tendências [{rotulo}] - {DESFECHO} (h = {list(H_PRE)}) ===")
    if pre.empty:
        print("Nenhum horizonte pré-choque estimado (verifique H_MIN e MIN_OBS).")
        return pre
    pre["rejeita_5pct"] = pre["p"] < 0.05
    print(pre[["var", "h", "coef", "se", "p", "n", "rejeita_5pct"]].round(5).to_string(index=False))
    n_rej = int(pre["rejeita_5pct"].sum())
    print(f"Rejeições a 5%: {n_rej} de {len(pre)} testes "
          f"(esperado ao acaso sob H0: ~{0.05 * len(pre):.1f}).")
    return pre


VARS_CTRL_INT = ["Shock", "Ch_Caixa", "Ch_Alav", "Ch_Tam", "Ch_FCO", "Ch_Q"]
pre_all = []
for nome in AMOSTRAS:
    cheia = res[(res["amostra"] == nome) & (res["subgrupo"] == "Cheia")]
    pre_all.append(teste_pre_tendencia(cheia[cheia["spec"] == "base"],
                                       f"{nome} | base, Shock", ["Shock"])
                   .assign(amostra_teste=nome, spec_teste="base"))
    pre_all.append(teste_pre_tendencia(cheia[cheia["spec"] == "ctrl_int"],
                                       f"{nome} | ctrl_int", VARS_CTRL_INT)
                   .assign(amostra_teste=nome, spec_teste="ctrl_int"))
pd.concat(pre_all, ignore_index=True).to_csv(f"pre_tendencias{SUFIXO}.csv", index=False)


# ---------------- GRÁFICOS (amostras sobrepostas para comparação) ----------------
CORES = {"Total": "C0", "Restrita": "C3", "Restrita_semQ": "C7"}
NOMES = {"Shock": "β1 (Shock)", "Ch_Caixa": "Shock × Caixa",
         "Ch_Alav": "Shock × Alav(t-1)", "Ch_Q": "Shock × Q(t-1)",
         "Ch_Tam": "Shock × Tamanho", "Ch_FCO": "Shock × FCO(t-1)"}


def desenha(ax, s, label, cor, ls="-"):
    s = s.sort_values("h")
    if s.empty:
        return
    ax.plot(s["h"], s["coef"], marker="o", color=cor, ls=ls, label=label)
    ax.fill_between(s["h"], s["lo95"], s["hi95"], color=cor, alpha=0.12)


def formata(ax, titulo):
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(-0.5, color="gray", lw=0.8, ls=":")   # momento do choque (h=-1 é a referência)
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel("Horizonte (trimestres)")
    ax.legend(frameon=False)


def plot_comparativo(spec, vars_, arquivo):
    """Um painel por variável; as amostras (Total, Restrita, ...) ficam sobrepostas."""
    fig, axs = plt.subplots(1, len(vars_), figsize=(5.2 * len(vars_), 4), squeeze=False)
    for ax, v in zip(axs[0], vars_):
        for a, cor in CORES.items():
            s = res[(res["spec"] == spec) & (res["subgrupo"] == "Cheia")
                    & (res["amostra"] == a) & (res["var"] == v)]
            desenha(ax, s, a, cor, "--" if a == "Restrita_semQ" else "-")
        formata(ax, f"{NOMES[v]} - {DESFECHO}")
    plt.tight_layout(); plt.savefig(arquivo, dpi=150); plt.close(fig)


plot_comparativo("base", ["Shock"], f"lp_irf_base{SUFIXO}.png")
plot_comparativo("ctrl_int", ["Shock", "Ch_Caixa", "Ch_Alav", "Ch_Q"], f"lp_irf_interacoes{SUFIXO}.png")
plot_comparativo("ctrl_int", ["Ch_Caixa", "Ch_Tam", "Ch_FCO"], f"lp_irf_interacoes_extra{SUFIXO}.png")
plot_comparativo("setor_t", ["Ch_Caixa", "Ch_Alav", "Ch_Tam", "Ch_FCO", "Ch_Q"], f"lp_irf_setor_t{SUFIXO}.png")

# Split por alavancagem: um painel por amostra
fig, axs = plt.subplots(1, len(AMOSTRAS), figsize=(5.2 * len(AMOSTRAS), 4), squeeze=False)
for ax, a in zip(axs[0], AMOSTRAS):
    for g, cor in [("Baixa", "C2"), ("Alta", "C3")]:
        s = res[(res["spec"] == "base") & (res["amostra"] == a)
                & (res["subgrupo"] == g) & (res["var"] == "Shock")]
        desenha(ax, s, f"{g} alavancagem", cor)
    formata(ax, f"β1 por alavancagem - amostra {a}")
plt.tight_layout(); plt.savefig(f"lp_irf_split_alav{SUFIXO}.png", dpi=150); plt.close(fig)

print(f"\nSalvos: irf_consolidado{SUFIXO}.csv, comparacao_total_vs_restrita{SUFIXO}.csv, "
      f"diagnostico_selecao{SUFIXO}.csv, vif_ctrl_int{SUFIXO}.csv, pre_tendencias{SUFIXO}.csv "
      f"e gráficos lp_irf_*{SUFIXO}.png")