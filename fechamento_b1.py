from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from linearmodels.panel import PanelOLS
from fechamento_letter import prepare_panel, fmt, table, winsor, ROOT, HS, AMOSTRAS

OUT = ROOT / "fechamento_b1.md"
Q = lambda s: pd.Period(s, freq="Q").ordinal
E21 = np.arange(Q("2021Q1"), Q("2021Q4") + 1)
E1315 = np.arange(Q("2013Q2"), Q("2015Q2") + 1)
EBOTH = np.union1d(E21, E1315)

REF_POOL = {("Restrita", 3): .047615, ("Restrita", 5): .054878, ("Restrita", 7): .060087,
            ("Cheia", 3): .043019, ("Cheia", 5): .040438, ("Cheia", 7): .032255}
REF_21 = {("Restrita", 3): .082318, ("Restrita", 5): .094153, ("Restrita", 7): .095459,
          ("Cheia", 3): .090112, ("Cheia", 5): .094937, ("Cheia", 7): .091526}
REF_1315 = {("Restrita", 3): -.092599, ("Restrita", 5): -.097387, ("Restrita", 7): -.109232,
            ("Cheia", 3): -.239919, ("Cheia", 5): -.315714, ("Cheia", 7): -.296161}
REF_DIF = {("Restrita", 3): -.174917, ("Restrita", 5): -.191540, ("Restrita", 7): -.204691,
           ("Cheia", 3): -.330032, ("Cheia", 5): -.410651, ("Cheia", 7): -.387688}


def touches(t, E, h):
    idx = np.searchsorted(np.sort(E), t - 1, side="left")
    ok = idx < len(E)
    out = np.zeros(len(t), dtype=bool)
    out[ok] = np.sort(E)[idx[ok]] <= (t[ok] + h)
    return out


def fit(d, cols, h):
    d = d.copy()
    d["y"] = winsor(d["y"])          # igual ao projecoes_locais.py (por horizonte, após filtrar)
    x = d.set_index(["CNPJ_CIA", "t"]).sort_index()
    T = x.index.get_level_values(1).nunique()
    r = PanelOLS(x["y"], x[cols], entity_effects=True, drop_absorbed=True,
                 check_rank=False).fit(cov_type="kernel", kernel="bartlett",
                                       bandwidth=min(h + 1, T - 1))
    return r, len(x), x.index.get_level_values(0).nunique()


def one(r, v):
    b, se = float(r.params[v]), float(np.sqrt(r.cov.loc[v, v]))
    return b, se, 2 * (1 - norm.cdf(abs(b / se)))


def comb(r, a, b):
    est = float(r.params[a] + r.params[b])
    var = r.cov.loc[a, a] + r.cov.loc[b, b] + 2 * r.cov.loc[a, b]
    se = float(np.sqrt(var))
    return est, se, 2 * (1 - norm.cdf(abs(est / se)))


def dif(r, a, b):
    est = float(r.params[a] - r.params[b])
    var = r.cov.loc[a, a] + r.cov.loc[b, b] - 2 * r.cov.loc[a, b]
    se = float(np.sqrt(var))
    return est, se, 2 * (1 - norm.cdf(abs(est / se)))


def joint(d, cols, h, vars_):
    d = d.copy()
    w1, w2 = d["t"].isin(E1315).astype(float), d["t"].isin(E21).astype(float)
    d["P1"], d["P2"] = w1, w2
    extra = ["P1", "P2"]
    for v in vars_:
        d[v + "_P1"], d[v + "_P2"] = d[v] * w1, d[v] * w2
        extra += [v + "_P1", v + "_P2"]
    r, n, f = fit(d, cols + extra, h)
    return {"calmo": one(r, "Ch_Caixa"), "13-15": comb(r, "Ch_Caixa", "Ch_Caixa_P1"),
            "2021": comb(r, "Ch_Caixa", "Ch_Caixa_P2"),
            "dif": dif(r, "Ch_Caixa_P1", "Ch_Caixa_P2"), "n": n, "f": f}


def R(am, h, linha, ver, t, n, f):
    if t is None:
        return [am, h, linha, ver, "NÃO RODADO", "", "", n, f]
    return [am, h, linha, ver, fmt(t[0], 4), fmt(t[1], 4), fmt(t[2], 4), n, f]


def main():
    p, shock, serie_y, samples, specs = prepare_panel()
    rows, valid = [], []
    for am in AMOSTRAS:
        for h in HS:
            d, cols = samples[(am, h)], specs[am]
            t = d["t"].to_numpy()
            in21, in1315 = d["t"].isin(E21).to_numpy(), d["t"].isin(E1315).to_numpy()
            vars_ = ["Shock", "Ch_Caixa", "Ch_Alav", "Ch_Tam", "Ch_FCO"] + (["Ch_Q"] if am == "Restrita" else [])

            r, n, f = fit(d, cols, h)
            pool = one(r, "Ch_Caixa")
            rows.append(R(am, h, "Pooled", "-", pool, n, f))

            for nome, base_m, strict_m in [
                ("Excluindo 2021", ~in21, ~touches(t, E21, h)),
                ("Excluindo 2013-15", ~in1315, ~touches(t, E1315, h))]:
                for ver, m in [("atual (só t)", base_m), ("estrita ([t-1,t+h])", strict_m)]:
                    rr, nn, ff = fit(d[m], cols, h)
                    rows.append(R(am, h, nome, ver, one(rr, "Ch_Caixa"), nn, ff))

            in_ep = in21 | in1315
            ja = joint(d, cols, h, vars_)
            js = joint(d[in_ep | ~touches(t, EBOTH, h)], cols, h, vars_)
            for ver, j in [("atual (só t)", ja), ("estrita ([t-1,t+h])", js)]:
                for k, nome in [("calmo", "Joint: calmo"), ("13-15", "Joint: 2013-15"),
                                ("2021", "Joint: 2021"), ("dif", "Joint: 13-15 menos 2021")]:
                    rows.append(R(am, h, nome, ver, j[k], j["n"], j["f"]))
            valid.append([am, h, fmt(pool[0], 4), fmt(REF_POOL[(am, h)], 4),
                          fmt(ja["2021"][0], 4), fmt(REF_21[(am, h)], 4),
                          fmt(ja["13-15"][0], 4), fmt(REF_1315[(am, h)], 4),
                          fmt(ja["dif"][0], 4), fmt(REF_DIF[(am, h)], 4)])
            print(am, h, "ok", flush=True)

    md = ["# B1 v2 - Janelas sobrepostas", "", "## Validação (reestimado vs. paper)",
          table(["Amostra", "h", "Pooled", "Ref", "2021", "Ref", "13-15", "Ref", "Dif", "Ref"], valid), "",
          "SE: se os seis pooled não batem até a 3a casa, PARE e não use as linhas abaixo.", "",
          "## Resultados",
          table(["Amostra", "h", "Linha", "Versão", "Coef", "SE DK", "p", "N obs", "Firmas"], rows), "",
          "Nota: 'estrita' descarta observações calmas cujo [t-1,t+h] toca 2013-15 ou 2021; "
          "observações dentro dos episódios são mantidas."]
    OUT.write_text("\n".join(md), encoding="utf-8")
    print("OUTPUT", OUT.name)


if __name__ == "__main__":
    main()