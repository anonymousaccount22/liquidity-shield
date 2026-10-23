from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_quarterly(path: Path, value_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Série ausente: {path}. Forneça CSV trimestral com Trimestre e {value_column}."
        )
    df = pd.read_csv(path, sep=None, engine="python")
    required = {"Trimestre", value_column}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} deve conter as colunas {sorted(required)}.")
    df = df[["Trimestre", value_column]].copy()
    df["t"] = pd.PeriodIndex(df["Trimestre"], freq="Q").astype(int)
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
    if df["t"].duplicated().any():
        raise ValueError(f"{path} deve ter no máximo uma observação por trimestre.")
    return df.sort_values("t")


def orthogonalize_shock(
    panel: pd.DataFrame, vix: pd.DataFrame, fiscal_risk: pd.DataFrame
) -> pd.DataFrame:
    """Residualiza o choque trimestral contra ΔVIX e o nível do risco fiscal."""
    hfi = (panel[["Trimestre", "Choque_HFI"]].drop_duplicates("Trimestre").copy())
    hfi["t"] = pd.PeriodIndex(hfi["Trimestre"], freq="Q").astype(int)
    vix = vix.copy()
    vix["VIX_L1"] = vix["VIX"].shift(1)
    vix["t_l1"] = vix["t"].shift(1)
    vix["Delta_VIX"] = (vix["VIX"] - vix["VIX_L1"]).where(vix["t"] - vix["t_l1"] == 1)
    z = hfi.merge(vix[["t", "Delta_VIX"]], on="t", how="left")
    z = z.merge(fiscal_risk[["t", "Risco_Fiscal"]], on="t", how="left")
    z["Choque_HFI"] = pd.to_numeric(z["Choque_HFI"], errors="coerce")
    sample = z[["Choque_HFI", "Delta_VIX", "Risco_Fiscal"]].dropna()
    if len(sample) <= 3:
        raise ValueError("São necessárias pelo menos quatro observações trimestrais completas.")

    y = sample["Choque_HFI"].to_numpy(dtype=float)
    x = np.column_stack((np.ones(len(sample)), sample["Delta_VIX"], sample["Risco_Fiscal"]))
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError("Regressores de ortogonalização sem posto completo.")
    residuals = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]

    r2 = 1 - residuals.var() / y.var()
    print(f"1º estágio: N={len(y)} | R²={r2:.3f} | desvio {y.std():.4f} -> resíduo {residuals.std():.4f}")
    print("Trimestres sem resíduo:", sorted(z.loc[~z.index.isin(sample.index), "Trimestre"]))
    print(f"VIX até {vix['t'].max()} | Risco fiscal até {fiscal_risk['t'].max()} (ordinais; "
          f"{pd.Period(ordinal=int(vix['t'].max()), freq='Q')} e "
          f"{pd.Period(ordinal=int(fiscal_risk['t'].max()), freq='Q')})")

    z["Shock_t"] = np.nan
    z.loc[sample.index, "Shock_t"] = residuals
    out = panel.merge(z[["Trimestre", "Shock_t"]], on="Trimestre", how="left")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cria Shock_t por OLS de Choque_HFI em ΔVIX e risco fiscal trimestral."
    )
    parser.add_argument("--panel", type=Path, default=Path("painel_final_q.csv"))
    parser.add_argument("--vix", type=Path, default=Path("vix_trimestral.csv"))
    parser.add_argument("--fiscal", type=Path, default=Path("risco_fiscal_trimestral.csv"))
    parser.add_argument("--output", type=Path, default=Path("painel_regressao_q_ortog.csv"))
    args = parser.parse_args()

    panel = pd.read_csv(args.panel)
    vix = read_quarterly(args.vix, "VIX")
    fiscal = read_quarterly(args.fiscal, "Risco_Fiscal")
    out = orthogonalize_shock(panel, vix, fiscal)
    out.to_csv(args.output, index=False)
    print(f"Painel com Shock_t salvo em {args.output}")
    print(f"Trimestres com resíduo: {out.loc[out['Shock_t'].notna(), 'Trimestre'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())