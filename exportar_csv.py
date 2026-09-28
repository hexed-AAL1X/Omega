#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "datasets" / "parquet" / "model_table.parquet"
OUT = ROOT / "datasets" / "csv" / "model_table.csv"

COLUMNS = [
    "project_id", "donor", "wb_region", "wb_income", "fila_ok", "in_financing_window",
    "start_year", "duration_years", "commitment_usd",
    "gdp_pc_ppp", "gdp_pc", "gdp_pc_growth", "elec_access", "child_mort", "unemp",
    "urban_pct", "internet_pct", "poverty", "literacy",
    "n_aid_projects", "n_donors", "usd_total_net", "usd_total_gross", "usd_ods17",
    "n_projects_ods17", "n_decommitments", "success",
]


def main() -> int:
    if not MODEL.is_file():
        print(f"Falta {MODEL}. Ejecuta EDA.ipynb primero.")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(MODEL).select(COLUMNS)
    df.write_csv(OUT, null_value="")
    print(f"{OUT.relative_to(ROOT)}: {df.height:,} filas, {df.width} columnas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
