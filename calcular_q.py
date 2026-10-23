#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calcula o Tobin's Q (aproximado) para o painel firma-trimestre.

Saídas:
  - painel_final_q.csv           painel + Valor_Mercado, Tobin_Q_raw, Tobin_Q (winsorizado),
                                 log_Tobin_Q, Tobin_Q_z, log_Tobin_Q_z
  - suspeitos_splits.csv         saltos suspeitos na quantidade de ações (lista de checagem)
  - splits_ajustados.csv         decisão tomada para cada evento (ajustar / manter / sem PL)
  - estatisticas_tobin_q.csv     descritivas pré e pós-winsorização

Convenção: `Tobin_Q` (sem sufixo) é o Q WINSORIZADO (compatível com o uso a jusante);
o valor bruto fica em `Tobin_Q_raw`. Todas as variáveis aqui são contemporâneas (t);
para defasagens use merge em (CNPJ_CIA, t+1), como em script_painel.py.
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
# Winsorização cross-section (por trimestre)
WINS_LOWER, WINS_UPPER = 0.01, 0.99
WINS_MIN_N = 20                 # trimestres com menos obs. válidas não são winsorizados

# Lista de suspeitos de split (mesmos limiares do código original)
SUS_LIM_SUP, SUS_LIM_INF = 1.8, 0.55

# Evento tratado como split/grupamento: variação > 100% (razão > 2) ou queda equivalente (< 0.5),
# SEM variação proporcional do Patrimônio Líquido.
SPLIT_LIM_SUP, SPLIT_LIM_INF = 2.0, 0.5
# "Proporcional" = ln(PL_t/PL_t-1) / ln(Ações_t/Ações_t-1) >= PL_PROP_MIN.
# >= limiar -> emissão/recompra real (mantém); < limiar -> split/grupamento (ajusta).
PL_PROP_MIN = 0.5

# z-score: False = média/desvio da amostra inteira; True = padroniza dentro de cada trimestre
Z_POR_TRIMESTRE = False

CVM_ROOT = Path("dados_cvm")


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================
def carregar_pl(raiz: Path = CVM_ROOT) -> pd.DataFrame | None:
    """
    Patrimônio Líquido consolidado (conta 2.03 do BPP_con) por (CNPJ_CIA, Trimestre),
    em unidades monetárias. Devolve None se nenhum arquivo utilizável for encontrado.
    """
    cols = {"CNPJ_CIA", "DT_REFER", "CD_CONTA", "VL_CONTA", "ESCALA_MOEDA", "ORDEM_EXERC", "VERSAO"}
    frames = []
    for f in sorted(raiz.rglob("*.csv")):
        if "bpp_con" not in f.name.lower():
            continue
        try:
            d = pd.read_csv(f, sep=";", encoding="latin1", dtype=str,
                            usecols=lambda c: c in cols)
        except Exception as exc:
            print(f"Aviso: falha ao ler {f.name}: {exc}")
            continue
        d = d[d["CD_CONTA"].str.strip() == "2.03"]
        if "ORDEM_EXERC" in d.columns:           # só o exercício corrente ('ÚLTIMO')
            d = d[d["ORDEM_EXERC"].str.upper().str.contains("LTIMO", na=False)]
        if len(d):
            frames.append(d)
    if not frames:
        return None

    pl = pd.concat(frames, ignore_index=True)
    pl["PL"] = pd.to_numeric(pl["VL_CONTA"].str.replace(",", ".", regex=False), errors="coerce")
    if "ESCALA_MOEDA" in pl.columns:
        pl["PL"] *= pl["ESCALA_MOEDA"].str.upper().map({"MIL": 1_000.0}).fillna(1.0)
    pl["VERSAO"] = pd.to_numeric(pl.get("VERSAO", 0), errors="coerce").fillna(0)
    pl["DT_REFER"] = pd.to_datetime(pl["DT_REFER"], errors="coerce")
    pl = pl.dropna(subset=["DT_REFER", "PL"])
    pl["Trimestre"] = pl["DT_REFER"].dt.to_period("Q").astype(str)
    pl = (pl.sort_values(["CNPJ_CIA", "Trimestre", "VERSAO", "DT_REFER"])
            .drop_duplicates(["CNPJ_CIA", "Trimestre"], keep="last"))
    return pl[["CNPJ_CIA", "Trimestre", "PL"]]


def gerar_lista_suspeitos(acoes: pd.DataFrame, caminho: str = "suspeitos_splits.csv") -> pd.DataFrame:
    """Saltos grandes na quantidade total de ações (vs. a observação válida anterior)."""
    a = acoes.dropna(subset=["Q_TOT"]).sort_values(["CNPJ_CIA", "Trimestre"]).copy()
    g = a.groupby("CNPJ_CIA")
    a["Trim_ant"] = g["Trimestre"].shift()
    a["razao"] = a["Q_TOT"] / g["Q_TOT"].shift()
    sus = a[(a["razao"] > SUS_LIM_SUP) | (a["razao"] < SUS_LIM_INF)]
    sus = sus[["CNPJ_CIA", "Trimestre", "Trim_ant", "razao"]]
    sus.to_csv(caminho, index=False)
    print(len(sus), f"saltos suspeitos na qtd. de ações ({caminho})")
    return sus


def ajustar_splits(acoes: pd.DataFrame, pl: pd.DataFrame | None,
                   sus_csv: str = "suspeitos_splits.csv") -> pd.DataFrame:
    """
    Lê a lista de suspeitos e, para cada salto com razão > SPLIT_LIM_SUP ou < SPLIT_LIM_INF,
    verifica se o PL variou proporcionalmente:
      - proporcional (emissão/recompra real)  -> mantém a série;
      - não proporcional (split/grupamento)   -> ajusta RETROATIVAMENTE as quantidades
        anteriores ao evento pela razão do salto, tornando a série contínua na base pós-evento;
      - PL indisponível/<=0                   -> não ajusta (registrado como 'sem_PL').
    O relatório de decisões vai para splits_ajustados.csv.
    """
    sus = pd.read_csv(sus_csv, dtype={"CNPJ_CIA": str, "Trimestre": str, "Trim_ant": str})
    ev = sus[(sus["razao"] > SPLIT_LIM_SUP) | (sus["razao"] < SPLIT_LIM_INF)].copy()
    if ev.empty:
        print("Nenhum salto fora dos limites de split; nada a ajustar.")
        return acoes

    if pl is not None:
        ev = (ev.merge(pl.rename(columns={"PL": "PL_t"}), on=["CNPJ_CIA", "Trimestre"], how="left")
                .merge(pl.rename(columns={"Trimestre": "Trim_ant", "PL": "PL_ant"}),
                       on=["CNPJ_CIA", "Trim_ant"], how="left"))
        ok = (ev["PL_t"] > 0) & (ev["PL_ant"] > 0)
        ev["razao_PL"] = (ev["PL_t"] / ev["PL_ant"]).where(ok)
        ev["prop"] = np.log(ev["razao_PL"]) / np.log(ev["razao"])
    else:
        print("Aviso: BPP_con (conta 2.03) não encontrado; nenhum evento será ajustado.")
        ev["razao_PL"] = np.nan
        ev["prop"] = np.nan

    ev["acao"] = np.select([ev["prop"].isna(), ev["prop"] >= PL_PROP_MIN],
                           ["sem_PL", "manter_emissao"], default="ajustar")
    ev.to_csv("splits_ajustados.csv", index=False)
    print("Eventos de salto avaliados:", ev["acao"].value_counts().to_dict())

    aj = ev.loc[ev["acao"] == "ajustar", ["CNPJ_CIA", "Trimestre", "razao"]].rename(columns={"razao": "ev"})
    if aj.empty:
        return acoes

    out = (acoes.sort_values(["CNPJ_CIA", "Trimestre"])
                .merge(aj, on=["CNPJ_CIA", "Trimestre"], how="left"))
    out["ev"] = out["ev"].fillna(1.0)

    def fator_retro(s: pd.Series) -> pd.Series:
        # produto das razões dos eventos ESTRITAMENTE posteriores a cada linha
        rev = s[::-1].cumprod()[::-1]
        return rev.shift(-1).fillna(1.0)

    out["f_split"] = out.groupby("CNPJ_CIA")["ev"].transform(fator_retro)
    for c in ["Q_ON", "Q_PN", "Q_TOT"]:
        out[c] = out[c] * out["f_split"]
    print(out.loc[out["f_split"] != 1, "CNPJ_CIA"].nunique(), "empresas com split/grupamento ajustado")
    return out.drop(columns=["ev", "f_split"])


def winsorize_quarterly(s: pd.Series, lower: float = WINS_LOWER, upper: float = WINS_UPPER,
                        min_n: int = WINS_MIN_N) -> pd.Series:
    """Winsoriza uma série dentro de um grupo (trimestre); ignora grupos muito pequenos."""
    if s.notna().sum() < min_n:
        return s
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def zscore(s: pd.Series, por: pd.Series | None = None) -> pd.Series:
    """Z-score (desvio-padrão amostral); `por` padroniza dentro de grupos."""
    if por is None:
        z = (s - s.mean()) / s.std(ddof=1)
    else:
        g = s.groupby(por)
        z = (s - g.transform("mean")) / g.transform("std")
    return z.replace([np.inf, -np.inf], np.nan)


def resumo(s: pd.Series, nome: str) -> dict:
    s = s.dropna()
    return {"variavel": nome, "N": len(s), "media": s.mean(), "mediana": s.median(),
            "p1": s.quantile(0.01), "p99": s.quantile(0.99), "std": s.std(),
            "min": s.min(), "max": s.max()}


# =============================================================================
# DADOS DE ENTRADA
# =============================================================================
painel = pd.read_csv("painel_final.csv")
tk = pd.read_csv("tickers.csv")
px = pd.read_csv("precos_trimestrais.csv")

# ---------- 1) Ações 2011-2019: FRE (anual -> repete nos 4 trimestres) ----------
fre = []
for f in sorted(Path("dados_fre").rglob("*.csv")):
    if not re.search(r"capital_social_\d{4}\.csv$", f.name):
        continue
    d = pd.read_csv(f, sep=";", encoding="latin1", dtype=str)
    d["Ano"] = pd.to_datetime(d["Data_Referencia"], errors="coerce").dt.year
    d["Versao"] = pd.to_numeric(d["Versao"], errors="coerce").fillna(0)
    integ = d[d["Tipo_Capital"].str.contains("Integralizado", case=False, na=False)]
    fre.append(integ if len(integ) else d)
fre = pd.concat(fre, ignore_index=True)
for c in ["Quantidade_Acoes_Ordinarias", "Quantidade_Acoes_Preferenciais", "Quantidade_Total_Acoes"]:
    fre[c] = pd.to_numeric(fre[c], errors="coerce")
fre = (fre.sort_values(["CNPJ_Companhia", "Ano", "Versao"])
          .drop_duplicates(["CNPJ_Companhia", "Ano"], keep="last"))
fre = fre.rename(columns={"CNPJ_Companhia": "CNPJ_CIA", "Quantidade_Acoes_Ordinarias": "Q_ON",
                          "Quantidade_Acoes_Preferenciais": "Q_PN", "Quantidade_Total_Acoes": "Q_TOT"})
fre_q = pd.concat([fre.assign(Trimestre=fre["Ano"].astype(int).astype(str) + f"Q{q}")
                   for q in (1, 2, 3, 4)])[["CNPJ_CIA", "Trimestre", "Q_ON", "Q_PN", "Q_TOT"]]
fre_q = fre_q[fre_q["Trimestre"].str[:4].astype(int).between(2011, 2019)]
print("FRE:", fre["CNPJ_CIA"].nunique(), "empresas")

# ---------- 2) Ações 2020+: composição do capital (ITR + DFP) ----------
frames = [pd.read_csv(f, sep=";", encoding="latin1", dtype=str)
          for f in Path("dados_cvm").rglob("*composicao_capital*.csv")]
cap = pd.concat(frames, ignore_index=True)
print("Colunas da composição do capital:", list(cap.columns))


def achar(*pedacos):
    return next(c for c in cap.columns if all(p in c.upper() for p in pedacos))


c_on, c_pn, c_tot = achar("ORDIN", "INTEGR"), achar("PREF", "INTEGR"), achar("TOTAL", "INTEGR")
t_on, t_pn, t_tot = achar("ORDIN", "TESOURO"), achar("PREF", "TESOURO"), achar("TOTAL", "TESOURO")
print("Usando:", c_on, c_pn, c_tot, "| tesouraria:", t_on, t_pn, t_tot)

for c in (c_on, c_pn, c_tot, t_on, t_pn, t_tot):
    cap[c] = pd.to_numeric(cap[c], errors="coerce")
cap["DT_REFER"] = pd.to_datetime(cap["DT_REFER"], errors="coerce")
cap["VERSAO"] = pd.to_numeric(cap["VERSAO"], errors="coerce").fillna(0)
cap = (cap.dropna(subset=["DT_REFER"]).sort_values(["CNPJ_CIA", "DT_REFER", "VERSAO"])
          .drop_duplicates(["CNPJ_CIA", "DT_REFER"], keep="last"))
cap["TP"] = cap["DT_REFER"].dt.to_period("Q")
cap["Q_ON"] = cap[c_on] - cap[t_on].fillna(0)
cap["Q_PN"] = cap[c_pn] - cap[t_pn].fillna(0)
cap["Q_TOT"] = cap[c_tot] - cap[t_tot].fillna(0)
cap["Q_TOT"] = cap["Q_TOT"].fillna(cap["Q_ON"] + cap["Q_PN"].fillna(0))
cap = cap.drop_duplicates(["CNPJ_CIA", "TP"], keep="last")

grade = pd.MultiIndex.from_product(
    [cap["CNPJ_CIA"].unique(), pd.period_range("2020Q1", "2026Q2", freq="Q")],
    names=["CNPJ_CIA", "TP"])
cvm_q = (cap.set_index(["CNPJ_CIA", "TP"])[["Q_ON", "Q_PN", "Q_TOT"]].reindex(grade)
            .groupby(level="CNPJ_CIA").ffill(limit=3).dropna(how="all").reset_index())
cvm_q["Trimestre"] = cvm_q["TP"].astype(str)

# ---------- 3) Junta (CVM tem prioridade sobre o FRE) ----------
acoes = (pd.concat([fre_q, cvm_q[["CNPJ_CIA", "Trimestre", "Q_ON", "Q_PN", "Q_TOT"]]], ignore_index=True)
           .drop_duplicates(["CNPJ_CIA", "Trimestre"], keep="last"))
acoes["Q_TOT"] = acoes["Q_TOT"].fillna(acoes["Q_ON"] + acoes["Q_PN"].fillna(0))

# ---------- 3b) Corrige quebras de escala (x1000) em qualquer trimestre ----------
acoes = acoes.sort_values(["CNPJ_CIA", "Trimestre"]).reset_index(drop=True)
for c in ["Q_ON", "Q_PN", "Q_TOT"]:
    acoes.loc[acoes[c] <= 0, c] = np.nan          # quantidade zero/negativa vira NaN

r = acoes.groupby("CNPJ_CIA")["Q_TOT"].transform(lambda s: s / s.shift())
step = np.where((r > 0.0008) & (r < 0.0012), 1000.0,
        np.where((r > 800) & (r < 1200), 0.001, 1.0))
acoes["f"] = pd.Series(step, index=acoes.index).groupby(acoes["CNPJ_CIA"]).cumprod()
print(acoes.loc[acoes["f"] != 1, "CNPJ_CIA"].nunique(), "empresas com quebra de escala corrigida")
for c in ["Q_ON", "Q_PN", "Q_TOT"]:
    acoes[c] = acoes[c] * acoes["f"]
acoes = acoes.drop(columns="f")

# ---------- 3c) Splits/grupamentos: lista de suspeitos + teste do Patrimônio Líquido ----------
gerar_lista_suspeitos(acoes, "suspeitos_splits.csv")
acoes = ajustar_splits(acoes, carregar_pl(CVM_ROOT), "suspeitos_splits.csv")

# ---------- 4) Um ticker por empresa e classe (o com mais trimestres de preço) ----------
n = px.groupby("Ticker").size().rename("n")
esc = (tk.drop_duplicates(["CNPJ_CIA", "Ticker"]).merge(n, on="Ticker")
         .sort_values("n", ascending=False).drop_duplicates(["CNPJ_CIA", "Classe"]))
base = acoes.copy()
for cl in ("ON", "PN"):
    t = esc[esc["Classe"] == cl][["CNPJ_CIA", "Ticker"]].merge(px, on="Ticker")
    t = t[["CNPJ_CIA", "Trimestre", "Preco"]].rename(columns={"Preco": f"P_{cl}"})
    base = base.merge(t, on=["CNPJ_CIA", "Trimestre"], how="left")

ambas = base["P_ON"].notna() & base["P_PN"].notna() & (base["Q_PN"] > 0)
base["Valor_Mercado"] = np.where(
    ambas, base["P_ON"] * base["Q_ON"] + base["P_PN"] * base["Q_PN"],
    np.where(base["P_ON"].notna(), base["P_ON"] * base["Q_TOT"], base["P_PN"] * base["Q_TOT"]))

# ---------- 5) Tobin's Q: bruto -> winsorizado (cross-section) -> log -> z-scores ----------
p = painel.merge(base[["CNPJ_CIA", "Trimestre", "Valor_Mercado"]],
                 on=["CNPJ_CIA", "Trimestre"], how="left")
p["Tobin_Q_raw"] = (p["Valor_Mercado"] + p["Emprestimos_CP"] + p["Emprestimos_LP"]) / p["Ativo_Total"]
p["Tobin_Q_raw"] = p["Tobin_Q_raw"].replace([np.inf, -np.inf], np.nan)

# Winsorização 1%/99% DENTRO de cada trimestre
p["Tobin_Q"] = p.groupby("Trimestre")["Tobin_Q_raw"].transform(winsorize_quarterly)
n_wins = int((p["Tobin_Q"].notna() & (p["Tobin_Q"] != p["Tobin_Q_raw"])).sum())
print(f"Observações winsorizadas: {n_wins} de {int(p['Tobin_Q_raw'].notna().sum())}")

# Log do Q winsorizado (Q <= 0 vira NaN, por segurança)
p["log_Tobin_Q"] = np.log(p["Tobin_Q"].where(p["Tobin_Q"] > 0))

# Z-scores das duas versões
por = p["Trimestre"] if Z_POR_TRIMESTRE else None
p["Tobin_Q_z"] = zscore(p["Tobin_Q"], por)
p["log_Tobin_Q_z"] = zscore(p["log_Tobin_Q"], por)

# Relatório de estatísticas descritivas (pré e pós-winsorização)
rel = pd.DataFrame([
    resumo(p["Tobin_Q_raw"], "Tobin_Q_raw (pré-winsorização)"),
    resumo(p["Tobin_Q"], "Tobin_Q (pós-winsorização)"),
    resumo(p["log_Tobin_Q"], "log_Tobin_Q (pós-winsorização)"),
])
rel.to_csv("estatisticas_tobin_q.csv", index=False)
print("\n=== Estatísticas descritivas (estatisticas_tobin_q.csv) ===")
print(rel.round(4).to_string(index=False))

print(f"\nCobertura do Tobin_Q: {p['Tobin_Q'].notna().mean():.0%} das linhas")
p["Ano"] = p["Trimestre"].str[:4]
print(p.groupby("Ano")["Tobin_Q"].apply(lambda s: f"{s.notna().mean():.0%}").to_string())
p.drop(columns="Ano").to_csv("painel_final_q.csv", index=False)

# Diagnóstico de extremos sobre o Q BRUTO (o winsorizado já está limitado)
ext = p[(p["Tobin_Q_raw"] > 20) | (p["Tobin_Q_raw"] < 0.02)]
print(len(ext), "linhas com Q bruto extremo;", ext["CNPJ_CIA"].nunique(), "empresas")
print(ext.groupby("CNPJ_CIA").size().sort_values(ascending=False).head(15).to_string())