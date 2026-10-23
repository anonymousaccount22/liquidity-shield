from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from script_painel import read_csv_flex


ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "painel_regressao_q_bp.csv"
CAPEX_PANEL = ROOT / "painel_regressao_q_dfc_ortog.csv"
OUT = ROOT / "fechamento_letter.md"
HS = (3, 5, 7)
AMOSTRAS = ("Cheia", "Restrita")


def load_function(filename: str, function_name: str, namespace: dict):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name)
    isolated = ast.Module(body=[node], type_ignores=[])
    exec(compile(isolated, filename, "exec"), namespace)
    return namespace[function_name]


def fmt(value, digits=6):
    if pd.isna(value):
        return "NÃO RODADO"
    return f"{value:.{digits}f}".replace(".", ",")


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def winsor(series, low=0.01, high=0.99):
    if series.notna().sum() == 0:
        return series
    return series.clip(series.quantile(low), series.quantile(high))


def zscore(series):
    sd = series.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / sd


def prepare_panel():
    p = read_csv_flex(PANEL, dtype={"CNPJ_CIA": str})
    p["t"] = pd.PeriodIndex(p["Trimestre"].astype(str), freq="Q").asi8.astype("int64")
    p = p.sort_values(["CNPJ_CIA", "t"]).reset_index(drop=True)
    p["log_Ativo_Fixo"] = p["log_Ativo_Fixo"].where(np.isfinite(p["log_Ativo_Fixo"]))

    for col in ("Cash_Ratio", "Alavancagem", "Tobin_Q", "Tamanho", "Fluxo_Caixa_Op_Ativo"):
        p[col] = winsor(p[col])
    p = p.drop(columns=[c for c in ("Alavancagem_L1", "Tobin_Q_L1", "Fluxo_Caixa_Op_Ativo_L1")
                         if c in p.columns])
    for col in ("Alavancagem", "Tobin_Q", "Fluxo_Caixa_Op_Ativo"):
        lag = p[["CNPJ_CIA", "t", col]].copy()
        lag["t"] += 1
        p = p.merge(lag.rename(columns={col: col + "_L1"}), on=["CNPJ_CIA", "t"], how="left")

    shock = p.groupby("t")["Shock_t"].first().astype(float).rename("Shock")
    p = p.merge(shock, left_on="t", right_index=True, how="left")
    shock_cont = shock.reindex(np.arange(shock.index.min(), shock.index.max() + 1))
    p = p.merge(shock_cont.shift(1).rename("Shock_L1"), left_on="t", right_index=True, how="left")
    for col in ("Cash_Ratio", "Alavancagem_L1", "Tobin_Q_L1", "Tamanho", "Fluxo_Caixa_Op_Ativo_L1"):
        p[col + "_z"] = p.groupby("t")[col].transform(zscore)

    p["Ch_Caixa"] = p["Shock"] * p["Cash_Ratio_z"]
    p["Ch_Alav"] = p["Shock"] * p["Alavancagem_L1_z"]
    p["Ch_Q"] = p["Shock"] * p["Tobin_Q_L1_z"]
    p["Ch_Tam"] = p["Shock"] * p["Tamanho_z"]
    p["Ch_FCO"] = p["Shock"] * p["Fluxo_Caixa_Op_Ativo_L1_z"]
    p["in_restrita"] = p["Tobin_Q_L1_z"].notna()

    all_t = np.arange(p["t"].min(), p["t"].max() + 1)
    piv = p.pivot(index="CNPJ_CIA", columns="t", values="log_Ativo_Fixo").reindex(columns=all_t)
    piv.columns.name = "t"
    serie_y = load_function("projecoes_locais.py", "serie_y", {"np": np, "pd": pd,
                                                                  "piv_log_af": piv})
    specs = {
        "Cheia": ["Shock", "Ch_Caixa", "Ch_Alav", "Ch_Tam", "Ch_FCO", "Shock_L1",
                  "Cash_Ratio_z", "Alavancagem_L1_z", "Tamanho_z", "Fluxo_Caixa_Op_Ativo_L1_z"],
        "Restrita": ["Shock", "Ch_Caixa", "Ch_Alav", "Ch_Tam", "Ch_FCO", "Ch_Q", "Shock_L1",
                     "Cash_Ratio_z", "Alavancagem_L1_z", "Tobin_Q_L1_z", "Tamanho_z",
                     "Fluxo_Caixa_Op_Ativo_L1_z"],
    }
    samples = {}
    for am in AMOSTRAS:
        base = p if am == "Cheia" else p[p["in_restrita"]]
        for h in HS:
            y = serie_y(h)
            d = base.merge(y, on=["CNPJ_CIA", "t"], how="inner")
            d = d.dropna(subset=["y"] + specs[am] + ["SETOR_ATIV"]).copy()
            samples[(am, h)] = d
    return p, shock, serie_y, samples, specs


def shock_concentration(d):
    tri = d[["Trimestre", "t", "Shock_t"]].drop_duplicates("t").dropna(subset=["Shock_t"])
    values = tri["Shock_t"].to_numpy(float)
    dev2 = (values - values.mean()) ** 2
    total = dev2.sum()
    weights = dev2 / total if total else np.full(len(dev2), np.nan)
    years = tri["Trimestre"].astype(str).str[:4].to_numpy()
    quarters = tri["Trimestre"].astype(str).to_numpy()
    return (dev2[years == "2021"].sum() / total,
            dev2[quarters == "2021Q4"].sum() / total,
            1 / np.sum(weights ** 2))


def a1_a3(p, shock, serie_y, samples, specs):
    shock_tri = p[["Trimestre", "t", "Shock_t"]].drop_duplicates("t")
    observed = shock_tri.dropna(subset=["Shock_t"]).sort_values("t")
    rows_a1 = [["Painel atual", len(observed), observed.iloc[0]["Trimestre"], observed.iloc[-1]["Trimestre"]]]
    for am in AMOSTRAS:
        for h in HS:
            d = samples[(am, h)]
            used = sorted(d["t"].unique())
            rows_a1.append([f"{am}, h={h}", len(used),
                            str(pd.Period(ordinal=int(used[0]), freq="Q")) if used else "NÃO RODADO",
                            str(pd.Period(ordinal=int(used[-1]), freq="Q")) if used else "NÃO RODADO"])

    lost = sorted(set(observed["t"]) - set(samples[("Cheia", 3)]["t"].unique()))
    lost_text = (", ".join(str(pd.Period(ordinal=int(t), freq="Q")) for t in lost)
                 if lost else "Nenhum trimestre inteiro")
    explanation = ("Primeiro trimestre com choque cai por `Shock_L1` ausente; h não muda o conjunto de trimestres."
                   if lost and int(lost[0]) == int(observed.iloc[0]["t"])
                   else "A amostra não perde um trimestre inteiro comum; h afeta a disponibilidade do desfecho por firma.")

    rows_a2 = []
    for am in AMOSTRAS:
        for h in HS:
            d = samples[(am, h)].drop_duplicates("t")
            share_year, share_q4, neff = shock_concentration(d)
            rows_a2.append([am, h, len(d), fmt(share_year), fmt(share_q4), fmt(neff, 4)])

    valid_af = p["log_Ativo_Fixo"].notna() & np.isfinite(p["log_Ativo_Fixo"])
    rows_a3 = [
        ["Painel contábil bruto; pré-filtros institucionais", "NÃO RODADO", "NÃO RODADO",
         "Painel intermediário pré-filtros não está salvo; não reli toda a pasta CVM."],
        ["Sem financeiras", "NÃO RODADO", "NÃO RODADO", "Sem output por etapa."],
        ["Sem estatais", "NÃO RODADO", "NÃO RODADO", "Sem output por etapa."],
        ["Sem utilities", "NÃO RODADO", "NÃO RODADO", "Sem output por etapa."],
        ["Ativo fixo válido no painel salvo", p.loc[valid_af, "CNPJ_CIA"].nunique(), int(valid_af.sum()),
         "Contagem no painel de regressão já filtrado."],
        ["Amostra cheia no painel salvo", p["CNPJ_CIA"].nunique(), len(p), "painel_regressao_q_bp.csv"],
        ["Com Tobin's Q defasado", p.loc[p["in_restrita"], "CNPJ_CIA"].nunique(), int(p["in_restrita"].sum()),
         "Critério de amostra Restrita; firma-trimestres disponíveis no painel salvo."],
        ["Amostra restrita reportada", 286, "NÃO RODADO", "286 firmas consta em carry_00_contagem_N.csv; N de linhas dessa etapa não salvo."],
    ]
    for h in HS:
        for am in AMOSTRAS:
            d = samples[(am, h)]
            rows_a3.append([f"Usada na estimação, {am}, h={h}", d["CNPJ_CIA"].nunique(), len(d),
                            f"T={d['t'].nunique()}"])

    restrict_ids = set(p.loc[p["in_restrita"], "CNPJ_CIA"])
    missing_reason = []
    for h in HS:
        y = serie_y(h)
        candidate = p[p["in_restrita"]][["CNPJ_CIA", "t", "SETOR_ATIV"] + specs["Restrita"]]
        candidate = candidate.merge(y, on=["CNPJ_CIA", "t"], how="left")
        eligible = set(samples[("Restrita", h)]["CNPJ_CIA"])
        reasons = {"sem desfecho": 0, "sem regressor": 0, "sem setor": 0, "causas sobrepostas": 0}
        for firm in restrict_ids - eligible:
            fd = candidate[candidate["CNPJ_CIA"] == firm]
            with_y = fd[fd["y"].notna()]
            if with_y.empty:
                reasons["sem desfecho"] += 1
                continue
            has_sector = with_y["SETOR_ATIV"].notna()
            has_x = with_y[specs["Restrita"]].notna().all(axis=1)
            if not has_sector.any() and not has_x.any():
                reasons["causas sobrepostas"] += 1
            elif not has_sector.any():
                reasons["sem setor"] += 1
            elif not has_x.any():
                reasons["sem regressor"] += 1
            else:
                reasons["causas sobrepostas"] += 1
        missing_reason.append([h, len(restrict_ids - eligible), *(reasons[k] for k in reasons)])

    source = (ROOT / "script_painel.py").read_text(encoding="utf-8").lower()
    fiscal_exclusion = "Exclusão de firmas por ano fiscal não calendário: NÃO EXISTE no script_painel.py."
    if "exclusão de empresas por ano fiscal não calendário" in source or "ano fiscal nao calendario" in source:
        fiscal_exclusion = "Regra de exclusão por ano fiscal encontrada; requer conferência manual."
    return rows_a1, lost_text, explanation, rows_a2, rows_a3, missing_reason, fiscal_exclusion


def a4_capex(samples, p):
    if not CAPEX_PANEL.exists():
        return [], "NÃO RODADO: painel com `Capex_DFC_Ativo` não existe."
    cap = read_csv_flex(CAPEX_PANEL, dtype={"CNPJ_CIA": str})
    required = {"CNPJ_CIA", "Trimestre", "Capex_DFC_Ativo"}
    if not required.issubset(cap.columns):
        return [], "NÃO RODADO: faltam colunas de fluxo de CAPEX no painel DFC."
    cap["t"] = pd.PeriodIndex(cap["Trimestre"].astype(str), freq="Q").asi8.astype("int64")
    cap = cap.drop_duplicates(["CNPJ_CIA", "t"])
    wide = cap.pivot(index="CNPJ_CIA", columns="t", values="Capex_DFC_Ativo")
    pre_ini, pre_fim = pd.Period("2012Q1", freq="Q").ordinal, pd.Period("2013Q1", freq="Q").ordinal
    size = p[["CNPJ_CIA", "t", "Tamanho"]].drop_duplicates()
    pre_size = size[size["t"].between(pre_ini, pre_fim)].groupby("CNPJ_CIA")["Tamanho"].mean()
    rows = []
    for am in AMOSTRAS:
        for h in HS:
            d = samples[(am, h)].copy()
            complete = []
            for firm, t in d[["CNPJ_CIA", "t"]].itertuples(index=False, name=None):
                values = wide.reindex([firm]).reindex(columns=range(t, t + h + 1)).to_numpy()
                complete.append(values.size == h + 1 and np.isfinite(values).all())
            d["capex_window"] = complete
            ok = d[d["capex_window"]]
            v = pre_size.reindex(d["CNPJ_CIA"].unique()).dropna()
            threshold = v.quantile(2 / 3) if len(v) else np.nan
            top_firms = set(v.index[v >= threshold]) if pd.notna(threshold) else set()
            top = ok[ok["CNPJ_CIA"].isin(top_firms)]
            rows.extend([[am, h, "Amostra completa", ok["CNPJ_CIA"].nunique(), len(ok)],
                         [am, h, "Tercil superior de Tamanho", top["CNPJ_CIA"].nunique(), len(top)]])
    note = "Tercil conforme inv1315_desfechos.py: média de Tamanho por firma em 2012Q1–2013Q1; corte de 2/3 dentro da amostra."
    return rows, note


def a5_entrants(p):
    if "obs_real" not in p.columns:
        return [], "NÃO RODADO: coluna obs_real ausente para definir primeira observação real."
    observed = p[p["obs_real"].astype(bool)].copy()
    first = observed.groupby("CNPJ_CIA")["Trimestre"].min()
    censored_start = first.min()
    entrants = set(first[(first >= "2019Q1") & (first <= "2021Q4") & (first > censored_start)].index)
    q21 = observed[observed["Trimestre"].astype(str).str.startswith("2021")].copy()
    q21["grupo"] = np.where(q21["CNPJ_CIA"].isin(entrants), "Entrantes", "Não entrantes")
    rows = []
    for group, d in q21.groupby("grupo"):
        rows.append([group, d["CNPJ_CIA"].nunique(), len(d), fmt(d["Cash_Ratio"].mean())])
    note = "Entrante = primeira observação real em 2019Q1–2021Q4, excluídas firmas censuradas à esquerda; média de Cash_Ratio por firma-trimestre em 2021."
    return rows, note


def main():
    p, shock, serie_y, samples, specs = prepare_panel()
    a1, lost, why, a2, a3, reasons, fiscal = a1_a3(p, shock, serie_y, samples, specs)
    capex, capex_note = a4_capex(samples, p)
    entrants, entrants_note = a5_entrants(p)
    lines = [
        "# Fechamento Letter — Bloco A", "",
        "Painéis existentes: `painel_regressao_q_bp.csv` e `painel_regressao_q_dfc_ortog.csv`. "
        "A função `serie_y` foi carregada isoladamente de `projecoes_locais.py` para evitar executar o loop de estimações.", "",
        "## A1. Trimestres",
        table(["Amostra/h", "T usados", "Primeiro", "Último"], a1),
        f"\nTrimestres de choque não usados: {lost}. {why}", "",
        "## A2. Concentração na amostra final",
        "Base: `(Shock_t - média)^2` sobre os trimestres efetivamente usados em cada amostra/horizonte.",
        table(["Amostra", "h", "T", "Share 2021", "Share 2021Q4", "N efetivo"], a2), "",
        "## A3. Attrition",
        table(["Etapa", "Firmas", "Firma-trimestres", "Observação"], a3), "",
        "Motivo entre as firmas restritas que não chegam à amostra estimada:",
        table(["h", "Fora", "Sem desfecho", "Sem regressor", "Sem setor", "Sobrepostas"], reasons), "",
        fiscal, "",
        "## A4. CAPEX disponível em t..t+h",
        table(["Amostra", "h", "Subconjunto", "Firmas", "Observações"], capex) if capex else capex_note,
        capex_note if capex else "", "",
        "## A5. Entrantes em 2021",
        table(["Grupo", "Firmas únicas", "Firma-trimestres observados", "Cash_Ratio médio"], entrants)
        if entrants else entrants_note,
        entrants_note, "",
        "## Bloco B", "NÃO RODADO: execução limitada ao Bloco A conforme instrução.", "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("A1", a1)
    print("A2", a2)
    print("A3 reasons", reasons)
    print("A4", capex if capex else capex_note)
    print("A5", entrants if entrants else entrants_note)
    print("OUTPUT", OUT.name)


if __name__ == "__main__":
    main()