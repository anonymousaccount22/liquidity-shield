"""
Teste F: separa choques monetarios "puros" de choques "de informacao"
(logica de Jarocinski-Karadi) usando o sinal da reacao da bolsa/cambio

Regra: juros sobem e ativo de risco cai (sinais opostos)  -> PURO
       juros sobem e ativo de risco sobe (mesmo sinal)     -> INFORMACAO
Para o cambio (USD/BRL): aperto puro aprecia o real (USD/BRL cai, sinal oposto
ao do swap); mesma regra de "sinais opostos = puro".

ENTRADAS (ajuste a secao CONFIG):
  1) ARQ_SWAP: serie diaria do Swap DI-Pre 360d (a mesma que o seu
     coletor_swap.py gera). Colunas: data, taxa (em % a.a.).
  2) ARQ_REUNIOES: uma coluna com a data da DECISAO do COPOM (dia em que a
     decisao e anunciada, normalmente a quarta-feira, apos o fechamento).

SAIDAS:
  f_reunioes_classificadas.csv  (uma linha por reuniao)
  f_choques_trimestrais.csv     (Trimestre, Shock_puro, Shock_info, ...)

Ibovespa e USD/BRL vem do SGS/BCB. Confira os codigos antes de usar.
"""
import yfinance as yf
import sys
import numpy as np
import pandas as pd
import requests

# ----------------------------- CONFIG ---------------------------------
ARQ_SWAP = "swap_di_pre_360_diario.csv"
COL_DATA_SWAP = "Data"
COL_TAXA_SWAP = "Taxa"
ARQ_REUNIOES = "datas_copom.csv"
COL_DATA_REUNIAO = "Data"

SGS_IBOV = 7        # Ibovespa, fechamento (CONFIRMAR no site do SGS)
SGS_USDBRL = 1      # dolar comercial venda (CONFIRMAR no site do SGS)

# Janela: fechamento do dia da decisao -> fechamento do proximo dia util.
# Isso so faz sentido se a decisao sai APOS o fechamento. Se o seu
# Choque_HFI usa outra janela, ajuste aqui para ficar IGUAL a ele.
USA_PROXIMO_DIA = True

LIMIAR_PB = 3.0     # |dSwap| abaixo disso (em pb) = neutro
INICIO, FIM = "2010-12-01", "2026-12-31"
# ----------------------------------------------------------------------


def baixa_sgs(codigo, inicio, fim, passo_anos=5):
    """O SGS limita consultas diarias a janelas de ~10 anos: baixa em blocos."""
    ini, fim_ts = pd.Timestamp(inicio), pd.Timestamp(fim)
    partes = []
    cur = ini
    while cur <= fim_ts:
        nxt = min(cur + pd.DateOffset(years=passo_anos) - pd.Timedelta(days=1), fim_ts)
        url = (f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"
               f"?formato=json&dataInicial={cur:%d/%m/%Y}&dataFinal={nxt:%d/%m/%Y}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        js = r.json()
        if js:
            partes.append(pd.DataFrame(js))
        cur = nxt + pd.Timedelta(days=1)
    if not partes:
        raise RuntimeError(f"SGS {codigo} sem dados")
    df = pd.concat(partes, ignore_index=True)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df.drop_duplicates("data").set_index("data")["valor"].sort_index().dropna()


def variacao_janela(serie, datas, em_log):
    """Para cada data de decisao d: pre = ultimo fechamento <= d;
    pos = fechamento seguinte (se USA_PROXIMO_DIA) ou o proprio d."""
    idx = serie.index
    out = []
    for d in datas:
        pos = idx.searchsorted(d, side="right") - 1
        if pos < 0 or pos + 1 >= len(idx):
            out.append(np.nan)
            continue
        pre = serie.iloc[pos]
        post = serie.iloc[pos + 1] if USA_PROXIMO_DIA else serie.iloc[pos]
        if not USA_PROXIMO_DIA:  # janela = d-1 -> d
            pre = serie.iloc[pos - 1] if pos >= 1 else np.nan
        out.append(100 * np.log(post / pre) if em_log else post - pre)
    return np.array(out)


def classifica(d_swap_pb, d_ref, limiar):
    if np.isnan(d_swap_pb) or np.isnan(d_ref):
        return "sem_dado"
    if abs(d_swap_pb) < limiar:
        return "neutro"
    return "puro" if np.sign(d_swap_pb) != np.sign(d_ref) else "info"


def trimestre(ts):
    return f"{ts.year}Q{ts.quarter}"

def baixa_yf(ticker, ini, fim):
    d = yf.download(ticker, start=ini, end=fim, progress=False, auto_adjust=False)
    s = d["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s.dropna().sort_index()
    if len(s) < 1000:
        raise RuntimeError(f"{ticker}: so {len(s)} observacoes, confira o download")
    return s


def main():
    swap = pd.read_csv(ARQ_SWAP)
    swap[COL_DATA_SWAP] = pd.to_datetime(swap[COL_DATA_SWAP])
    swap = swap.set_index(COL_DATA_SWAP)[COL_TAXA_SWAP].sort_index().dropna()

    reun = pd.read_csv(ARQ_REUNIOES, sep=None, engine="python", encoding="utf-8-sig")
    reun.columns = reun.columns.str.strip()
    print("Colunas de datas_copom.csv:", reun.columns.tolist())
    print(reun.head(3))
    col = COL_DATA_REUNIAO if COL_DATA_REUNIAO in reun.columns else reun.columns[0]
    s = reun[col].dropna().astype(str).str.strip()
    s = s[s.str.match(r"^\d{4}-\d{2}-\d{2}")]
    datas = pd.to_datetime(s, format="%Y-%m-%d").sort_values().reset_index(drop=True)
    print("Reunioes lidas:", len(datas), "|", datas.min().date(), "->", datas.max().date())

    print("Baixando Ibovespa e USD/BRL do SGS...")
    ibov = baixa_yf("^BVSP", INICIO, FIM)
    brl = baixa_yf("BRL=X", INICIO, FIM)

    df = pd.DataFrame({"data_decisao": datas})
    df["dSwap_pp"] = variacao_janela(swap, datas, em_log=False)   # pp = 100 pb
    df["dSwap_pb"] = 100 * df["dSwap_pp"]
    df["dIbov_pct"] = variacao_janela(ibov, datas, em_log=True)
    df["dUSDBRL_pct"] = variacao_janela(brl, datas, em_log=True)

    df["classe_ibov"] = [classifica(a, b, LIMIAR_PB) for a, b in zip(df.dSwap_pb, df.dIbov_pct)]
    df["classe_brl"] = [classifica(a, b, LIMIAR_PB) for a, b in zip(df.dSwap_pb, df.dUSDBRL_pct)]
    df["Trimestre"] = df["data_decisao"].apply(trimestre)
    # --- choque "limpo": remove a parte do swap correlacionada com bolsa/cambio ---
    ok = df[["dSwap_pp", "dIbov_pct", "dUSDBRL_pct"]].dropna()
    X1 = np.column_stack([np.ones(len(ok)), ok["dIbov_pct"]])
    b1 = np.linalg.lstsq(X1, ok["dSwap_pp"], rcond=None)[0]
    X2 = np.column_stack([np.ones(len(ok)), ok["dIbov_pct"], ok["dUSDBRL_pct"]])
    b2 = np.linalg.lstsq(X2, ok["dSwap_pp"], rcond=None)[0]
    print("Coef. dSwap(pp) ~ dIbov(%):", b1[1].round(4),
      "| ~ dIbov, dBRL:", b2[1:].round(4))
    df["Shock_limpo_ibov"] = df["dSwap_pp"] - b1[1] * df["dIbov_pct"]
    df["Shock_limpo_ibov_brl"] = (df["dSwap_pp"] - b2[1] * df["dIbov_pct"]
                              - b2[2] * df["dUSDBRL_pct"])
    df.to_csv("f_reunioes_classificadas.csv", index=False)

    # ---- contagens (importante: N pequeno na janela 2013Q2-2015Q2) ----
    jan = df[(df.Trimestre >= "2013Q2") & (df.Trimestre <= "2015Q2")]
    print("\nReunioes por classe (Ibovespa) - total:")
    print(df["classe_ibov"].value_counts().to_string())
    print("\nReunioes por classe (Ibovespa) - janela 2013Q2-2015Q2:")
    print(jan["classe_ibov"].value_counts().to_string())
    print("\nReunioes por classe (USD/BRL) - janela 2013Q2-2015Q2:")
    print(jan["classe_brl"].value_counts().to_string())
    print("\nConcordancia Ibov x BRL (janela):",
          (jan["classe_ibov"] == jan["classe_brl"]).mean().round(2))

    # ---- agregacao trimestral (unidade: pp = 100 pb, igual ao seu Shock) ----
    def agrega(col_classe):
        g = df.copy()
        g["puro"] = np.where(g[col_classe] == "puro", g["dSwap_pp"], 0.0)
        g["info"] = np.where(g[col_classe] == "info", g["dSwap_pp"], 0.0)
        g["n_puro"] = (g[col_classe] == "puro").astype(int)
        g["n_info"] = (g[col_classe] == "info").astype(int)
        return g.groupby("Trimestre")[["puro", "info", "n_puro", "n_info"]].sum()

    a = agrega("classe_ibov").add_suffix("_ibov")
    b = agrega("classe_brl").add_suffix("_brl")
    tot = df.groupby("Trimestre")["dSwap_pp"].sum().rename("Shock_total_pp")
    tri = pd.concat([tot, a, b], axis=1).reset_index()
    tri = tri.rename(columns={
        "puro_ibov": "Shock_puro_ibov", "info_ibov": "Shock_info_ibov",
        "puro_brl": "Shock_puro_brl", "info_brl": "Shock_info_brl"})
    lim = df.groupby("Trimestre")[["Shock_limpo_ibov", "Shock_limpo_ibov_brl"]].sum().reset_index()
    tri = tri.merge(lim, on="Trimestre", how="left")
    tri.to_csv("f_choques_trimestrais.csv", index=False)
    print("\nSalvos: f_reunioes_classificadas.csv, f_choques_trimestrais.csv")
    print("\nCOMPARE Shock_total_pp com o seu Choque_HFI (antes da ortogonalizacao).")
    print("Se nao baterem, a janela/serie diferem e a classificacao nao e comparavel.")


if __name__ == "__main__":
    sys.exit(main())