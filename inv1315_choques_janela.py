"""
inv1315_choques_janela.py
Diagnostico do choque: (1) recalculo por reuniao e conferencia com o painel (como antes);
(2) NOVO: concentracao do choque e numero efetivo de trimestres para 2013Q2-2015Q2, 2021, 2020
e para a amostra inteira de estimacao.

Le apenas CSVs ja existentes:
  - datas_copom.csv               (datas das reunioes)
  - swap_di_pre_360_diario.csv    (colunas Data, Taxa)
  - painel_regressao_q_bp.csv     (colunas Trimestre, Choque_HFI, Shock_t)

Saidas:
  inv1315_choques_reunioes.csv     (uma linha por reuniao)
  inv1315_choques_trimestres.csv   (uma linha por trimestre)
  inv_concentracao_janelas.csv     (NOVO: n efetivo e concentracao por janela)
  inv_concentracao_anos.csv        (NOVO: participacao anual na soma dos desvios quadrados)
NAO modifica nenhum arquivo existente.
"""
import pandas as pd
import numpy as np

COPOM_CSV = "datas_copom.csv"
SWAP_CSV = "swap_di_pre_360_diario.csv"
PAINEL_CSV = "painel_regressao_q_bp.csv"
JAN_INI, JAN_FIM = "2013Q2", "2015Q2"
CTX_INI, CTX_FIM = "2012Q4", "2016Q2"   # contexto em volta da janela

# janelas para o numero efetivo de trimestres (mesma formula para todas)
JANELAS_N = {
    "2013Q2-2015Q2": ("2013Q2", "2015Q2"),
    "2021": ("2021Q1", "2021Q4"),
    "2020": ("2020Q1", "2020Q4"),
}


def parse_datas(s):
    """Extrai a data de cada celula (ISO aaaa-mm-dd ou dd/mm/aaaa), ignorando lixo como ';;;;;'."""
    s = s.astype(str).str.strip()
    iso = s.str.extract(r"(\d{4}-\d{2}-\d{2})")[0]
    br = s.str.extract(r"(\d{2}/\d{2}/\d{4})")[0]
    out = pd.to_datetime(iso, format="%Y-%m-%d", errors="coerce")
    return out.fillna(pd.to_datetime(br, format="%d/%m/%Y", errors="coerce"))


def le_copom():
    df = pd.read_csv(COPOM_CSV, sep=";", dtype=str)
    print(f"[copom] colunas: {list(df.columns)}")
    col = df.columns[0]
    datas = parse_datas(df[col]).dropna().sort_values().drop_duplicates()
    print(f"[copom] usando coluna '{col}': {len(datas)} reunioes, "
          f"{datas.min().date()} a {datas.max().date()}")
    return list(datas)


def le_swap():
    df = pd.read_csv(SWAP_CSV, sep=None, engine="python", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    print(f"[swap] colunas: {list(df.columns)}")
    df["Data"] = parse_datas(df["Data"])
    df["Taxa"] = pd.to_numeric(df["Taxa"].astype(str).str.replace(",", ".", regex=False),
                               errors="coerce")
    df = df.dropna(subset=["Data", "Taxa"]).sort_values("Data")
    df = df[~df["Data"].duplicated(keep="last")]
    return df.set_index("Data")["Taxa"]


def choques_por_reuniao(datas, taxa):
    linhas = []
    idx = taxa.index
    for d in datas:
        antes = idx[idx <= d]
        depois = idx[idx > d]
        if len(antes) == 0 or len(depois) == 0:
            continue
        d0, d1 = antes[-1], depois[0]
        linhas.append(dict(
            reuniao=d, data_antes=d0, data_depois=d1,
            dias_antes=(d - d0).days, dias_depois=(d1 - d).days,
            taxa_antes=taxa.loc[d0], taxa_depois=taxa.loc[d1],
            choque=taxa.loc[d1] - taxa.loc[d0],
            Trimestre=str(pd.Period(d, freq="Q")),
        ))
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# NOVO: numero efetivo de trimestres e concentracao
# ---------------------------------------------------------------------------
def n_efetivo(serie):
    """Numero efetivo = 1 / soma(w^2), com w_t = x_t^2 / soma(x^2). Vale 1 se um trimestre
    concentra tudo e n se o peso e igual nos n trimestres."""
    s2 = np.asarray(serie, float) ** 2
    tot = s2.sum()
    if tot == 0 or len(s2) == 0:
        return np.nan, np.full(len(s2), np.nan)
    w = s2 / tot
    return 1.0 / (w ** 2).sum(), w


def concentracao(tri):
    """tri: DataFrame com Trimestre, Shock_t e coluna 'ord' (ordinal do trimestre).
    Devolve (tabela por janela, tabela por ano)."""
    ok = tri.dropna(subset=["Shock_t"]).sort_values("ord")
    # amostra de estimacao: Shock_L1 elimina o primeiro trimestre com choque
    est = ok.iloc[1:].copy()
    print(f"\n[concentracao] trimestres com Shock_t: {len(ok)}; "
          f"amostra de estimacao (sem o primeiro, por Shock_L1): T = {len(est)}  (esperado 54)")

    media = est["Shock_t"].mean()
    est["dev2"] = (est["Shock_t"] - media) ** 2
    est["ano"] = est["Trimestre"].str[:4].astype(int)
    ss_tot = est["dev2"].sum()

    anos = (est.groupby("ano")["dev2"].sum() / ss_tot).rename("participacao_ss").reset_index()
    print("\n[anos] participacao na soma dos desvios quadrados do choque (amostra de estimacao):")
    print(anos.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    linhas = []
    for nome, (a, b) in JANELAS_N.items():
        oa, ob = pd.Period(a, "Q").ordinal, pd.Period(b, "Q").ordinal
        j = est[est["ord"].between(oa, ob)]
        if len(j) == 0:
            continue
        n_raw, w = n_efetivo(j["Shock_t"])
        n_dev, _ = n_efetivo(j["Shock_t"] - media)
        i_max = int(np.nanargmax(w))
        linhas.append(dict(
            janela=nome, n_trimestres=len(j),
            n_efetivo_shock2=n_raw,            # mesma formula do diagnostico de 2013-15
            n_efetivo_desvio2=n_dev,           # variante com o choque centrado na media da amostra
            participacao_ss_amostra=j["dev2"].sum() / ss_tot,
            trimestre_dominante=j["Trimestre"].iloc[i_max],
            peso_dominante=w[i_max],
        ))
    # amostra inteira
    n_raw, w = n_efetivo(est["Shock_t"])
    n_dev, w2 = n_efetivo(est["Shock_t"] - media)
    i_max = int(np.nanargmax(w2))
    linhas.append(dict(
        janela="amostra inteira", n_trimestres=len(est),
        n_efetivo_shock2=n_raw, n_efetivo_desvio2=n_dev,
        participacao_ss_amostra=1.0,
        trimestre_dominante=est["Trimestre"].iloc[i_max], peso_dominante=w2[i_max]))
    jan = pd.DataFrame(linhas)
    print("\n[janelas] numero efetivo de trimestres e concentracao:")
    print(jan.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return jan, anos


def main():
    datas = le_copom()
    taxa = le_swap()
    reu = choques_por_reuniao(datas, taxa)
    reu.to_csv("inv1315_choques_reunioes.csv", index=False)

    soma = (reu.groupby("Trimestre")
               .agg(n_reunioes=("choque", "size"),
                    soma_recalc=("choque", "sum"),
                    maior_abs=("choque", lambda s: s.abs().max()))
               .reset_index())

    pan = pd.read_csv(PAINEL_CSV, usecols=["Trimestre", "Choque_HFI", "Shock_t"])
    pan = pan.drop_duplicates("Trimestre")
    tri = pan.merge(soma, on="Trimestre", how="left")
    tri["dif_recalc_vs_painel"] = tri["soma_recalc"] - tri["Choque_HFI"]
    tri["ord"] = tri["Trimestre"].map(lambda s: pd.Period(s, freq="Q").ordinal)
    tri = tri.sort_values("ord").reset_index(drop=True)

    o_ini, o_fim = pd.Period(JAN_INI, "Q").ordinal, pd.Period(JAN_FIM, "Q").ordinal
    na_jan = tri["ord"].between(o_ini, o_fim)
    q2 = tri["Shock_t"] ** 2
    tri["na_janela"] = na_jan
    tri["peso_na_janela"] = np.where(na_jan, q2 / q2[na_jan].sum(), np.nan)
    tri.drop(columns="ord").to_csv("inv1315_choques_trimestres.csv", index=False)

    # ---- validacao ----
    chk = tri.dropna(subset=["soma_recalc", "Choque_HFI"])
    print("\n[validacao] |soma recalculada - Choque_HFI do painel|:")
    print(f"  maximo = {chk['dif_recalc_vs_painel'].abs().max():.6f} em {len(chk)} trimestres")
    if chk["dif_recalc_vs_painel"].abs().max() > 1e-3:
        print("  ATENCAO: nao bate. Pode haver diferenca na regra de datas "
              "(<= vs <) ou na unidade. Veja inv1315_choques_trimestres.csv.")

    # ---- tabela da janela ----
    o_ctx = (pd.Period(CTX_INI, "Q").ordinal, pd.Period(CTX_FIM, "Q").ordinal)
    ctx = tri[tri["ord"].between(*o_ctx)]
    cols = ["Trimestre", "na_janela", "n_reunioes", "Choque_HFI", "Shock_t", "peso_na_janela"]
    print("\n[janela e contexto]")
    print(ctx[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    jan = tri[na_jan]
    peso_ord = jan.sort_values("peso_na_janela", ascending=False)
    cum = peso_ord["peso_na_janela"].cumsum().values
    print("\n[concentracao 2013-15] trimestres ordenados pelo peso (Shock_t^2):")
    print(peso_ord[["Trimestre", "Shock_t", "peso_na_janela"]]
          .assign(acumulado=cum).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    n_eff = 1.0 / (jan["peso_na_janela"] ** 2).sum()
    print(f"\nNumero efetivo de trimestres na janela (1/soma(peso^2)) = {n_eff:.2f} de {len(jan)}")

    # ---- reunioes da janela ----
    r_jan = reu[reu["Trimestre"].map(
        lambda s: o_ini <= pd.Period(s, freq="Q").ordinal <= o_fim)]
    print("\n[reunioes dentro da janela]")
    print(r_jan[["reuniao", "Trimestre", "taxa_antes", "taxa_depois", "choque",
                 "dias_antes", "dias_depois"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- NOVO: concentracao em todas as janelas e na amostra inteira ----
    tab_jan, tab_anos = concentracao(tri)
    tab_jan.to_csv("inv_concentracao_janelas.csv", index=False)
    tab_anos.to_csv("inv_concentracao_anos.csv", index=False)

    print("\nSalvos: inv1315_choques_reunioes.csv, inv1315_choques_trimestres.csv, "
          "inv_concentracao_janelas.csv, inv_concentracao_anos.csv")


if __name__ == "__main__":
    main()