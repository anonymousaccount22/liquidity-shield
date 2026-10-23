"""
gerar_tabela1.py -- monta a Tabela 1 (gradiente Ch_Caixa por episodio) e calcula a
concentracao do choque, SO a partir de CSVs ja existentes. Nao reestima nada.

Uso: coloque este arquivo na pasta dos CSVs e rode:  python gerar_tabela1.py
Saidas: tabela1.csv (e impressao no terminal).

ATENCAO: confirme os nomes de arquivo e de coluna na secao CONFIG.
- ciclos_igualdade.csv: colunas conferidas (cabecalho visto).
- irf_consolidado...csv: colunas vistas so na auditoria (amostra, subgrupo, spec, var, h, coef, se, p, n, n_firmas).
- ciclos_exclusao.csv: cabecalho NAO foi visto; ajuste COLS_EXCL se o script avisar.
- painel_regressao_q_bp.csv: assume colunas Trimestre e Shock_t.
"""
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

# ------------------------------- CONFIG -------------------------------------
PASTA = Path(".")
F_IRF = PASTA / "irf_consolidado_Delta_log_Ativo_Fixo_ctrlL1.csv"
F_CICLOS = PASTA / "ciclos_igualdade.csv"
F_EXCL = PASTA / "ciclos_exclusao.csv"
F_PAINEL = PASTA / "painel_regressao_q_bp.csv"
HORIZONTES = [3, 5, 7]
AMOSTRAS = ["Restrita", "Total"]
JANELA = "2013Q2-2015Q2 (principal)"
# ciclos_exclusao.csv: nomes de coluna e o valor que significa "excluiu so 2021"
COLS_EXCL = dict(amostra="amostra", h="h", excluidos="excluidos", coef="coef", p="p")
VALOR_SO_2021 = "sem 2021"
# ----------------------------------------------------------------------------


def se_de_beta_p(beta, p):
    """SE implicito por aproximacao normal: |beta| / z(p). Aproximado."""
    if pd.isna(p) or not (0 < p < 1):
        return np.nan
    return abs(beta) / NormalDist().inv_cdf(1 - p / 2)


def linha(amostra, rotulo, h, coef, se, p, n=np.nan, n_firmas=np.nan, nota=""):
    return dict(amostra=amostra, linha=rotulo, h=h, coef=coef, se=se, p=p,
                n_obs=n, n_firmas=n_firmas, nota=nota)


def pooled():
    df = pd.read_csv(F_IRF)
    m = ((df["subgrupo"] == "Cheia") & (df["spec"] == "ctrl_int") &
         (df["var"] == "Ch_Caixa") & (df["h"].isin(HORIZONTES)) &
         (df["amostra"].isin(AMOSTRAS)))
    return [linha(r.amostra, "Pooled", r.h, r.coef, r.se, r.p, r.n, r.n_firmas)
            for r in df[m].itertuples()]


def episodios():
    df = pd.read_csv(F_CICLOS)
    df = df[(df["janela"] == JANELA) & (df["h"].isin(HORIZONTES))]
    out = []
    for r in df.itertuples():
        out.append(linha(r.amostra, "2021 sozinho", r.h, r.beta_2021,
                         se_de_beta_p(r.beta_2021, r.p_2021), r.p_2021,
                         r.n_obs_2021, nota="se_derivado"))
        out.append(linha(r.amostra, "2013-15 sozinho", r.h, r.beta_ciclo,
                         se_de_beta_p(r.beta_ciclo, r.p_ciclo), r.p_ciclo,
                         r.n_obs_ciclo, nota="se_derivado"))
        out.append(linha(r.amostra, "Calmo (sem 2021 e sem 2013-15)", r.h, r.beta_calmo,
                         se_de_beta_p(r.beta_calmo, r.p_calmo), r.p_calmo,
                         nota="se_derivado; N nao disponivel neste arquivo"))
        out.append(linha(r.amostra, "Dif. 2013-15 menos 2021", r.h,
                         r.dif_ciclo_menos_2021, r.se_dif, r.p_igualdade))
    return out


def exclusao_2021():
    if not F_EXCL.exists():
        print(f"[aviso] {F_EXCL} nao encontrado; linha 'Sem 2021' omitida.")
        return []
    df = pd.read_csv(F_EXCL)
    faltam = [c for c in COLS_EXCL.values() if c not in df.columns]
    if faltam:
        print(f"[aviso] colunas ausentes em {F_EXCL.name}: {faltam}")
        print(f"        colunas existentes: {list(df.columns)}")
        print("        ajuste COLS_EXCL e VALOR_SO_2021 na secao CONFIG.")
        return []
    c = COLS_EXCL
    m = (df[c["excluidos"]].astype(str) == VALOR_SO_2021) & df[c["h"]].isin(HORIZONTES)
    if m.sum() == 0:
        print(f"[aviso] nenhuma linha com {c['excluidos']} == '{VALOR_SO_2021}'. "
              f"Valores encontrados: {sorted(df[c['excluidos']].astype(str).unique())[:10]}")
        return []
    return [linha(r[c["amostra"]], "Sem 2021", r[c["h"]], r[c["coef"]], np.nan, r[c["p"]],
                  nota="de ciclos_exclusao.csv; SE nao lida") for _, r in df[m].iterrows()]


def concentracao():
    """Participacao de cada ano na soma dos desvios quadrados do choque (trimestres com choque)."""
    if not F_PAINEL.exists():
        print(f"[aviso] {F_PAINEL} nao encontrado; concentracao nao calculada.")
        return
    df = pd.read_csv(F_PAINEL, usecols=["Trimestre", "Shock_t"])
    q = df.drop_duplicates("Trimestre").dropna(subset=["Shock_t"]).copy()
    q["d2"] = (q["Shock_t"] - q["Shock_t"].mean()) ** 2
    q["ano"] = q["Trimestre"].astype(str).str[:4]
    tot = q["d2"].sum()
    print(f"\nCONCENTRACAO DO CHOQUE ({len(q)} trimestres com choque, sd={q['Shock_t'].std():.3f})")
    print((q.groupby("ano")["d2"].sum() / tot).round(3).to_string())
    w = q["d2"] / tot
    print(f"\nTrimestre com maior peso: {q.loc[w.idxmax(), 'Trimestre']} = {w.max():.1%}")
    print(f"Numero efetivo de trimestres (1/soma w^2): {1 / (w ** 2).sum():.2f}")
    print("Obs.: usa desvios em relacao a media de todos os trimestres com choque. "
          "Se o seu 34,5% vem de outra definicao, compare antes de citar.")


def main():
    linhas = pooled() + exclusao_2021() + episodios()
    t = pd.DataFrame(linhas).sort_values(["amostra", "linha", "h"]).reset_index(drop=True)
    t.to_csv("tabela1.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.4g}")
    print(t.to_string(index=False))
    print("\n-> tabela1.csv")
    concentracao()


if __name__ == "__main__":
    main()