#!/usr/bin/env python3

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
IMG = ROOT / "docs" / "img"

PPD_CSV = DATASETS / "ppd" / "PPD2_Jan_21_2022.csv"
FIN_CSV = DATASETS / "financing_sdgs" / "FinancingtotheSDGsDataset_v1.0.csv"
WB_CSV = DATASETS / "worldbank_wdi" / "WDICSV.csv"
MODEL = DATASETS / "parquet" / "model_table.parquet"
MODEL_WINDOW = DATASETS / "parquet" / "model_table_2000_2013.parquet"
WDI_WINDOW = DATASETS / "parquet" / "wdi_2000_2013.parquet"

PPD_COLS = [
    "project_id", "donor", "countryname_WB", "country_code_WB", "start_date",
    "startyear", "wb_startyear", "project_duration", "wb_netcommitment", "six_overall_rating",
]
NON_COUNTRY = re.compile(
    r"(?i)regional|unspecified|bilateral|no value|multi-country|unallocated|global|madct|ex-yugoslav"
)
MONO = {"family": "monospace", "size": 10}


def cargar_ppd() -> pl.DataFrame:
    ppd = pl.read_csv(
        PPD_CSV, columns=PPD_COLS, schema_overrides={c: pl.Utf8 for c in PPD_COLS}, infer_schema_length=0
    )
    fecha = pl.coalesce([
        pl.col("start_date").str.to_date("%m/%d/%Y", strict=False),
        pl.col("start_date").str.to_date("%-m/%-d/%y", strict=False),
        pl.col("start_date").str.to_date("%Y-%m-%d", strict=False),
    ]).dt.year()
    wb_year = pl.col("wb_startyear").cast(pl.Int64, strict=False)
    return ppd.with_columns(
        pl.col("donor").str.strip_chars(),
        pl.col("six_overall_rating").cast(pl.Float64, strict=False).alias("rating"),
        pl.col("project_duration").cast(pl.Float64, strict=False).alias("dias"),
        pl.col("wb_netcommitment").cast(pl.Float64, strict=False).alias("usd"),
        pl.coalesce([
            pl.col("startyear").cast(pl.Int64, strict=False),
            fecha,
            pl.when(wb_year.is_between(1950, 2020)).then(wb_year),
        ]).alias("start_year"),
    )


def embudo(ppd: pl.DataFrame, model: pl.DataFrame) -> list[tuple[str, int]]:
    donantes = ppd.group_by("donor").len().filter(pl.col("len") >= 10).get_column("donor")
    paso1 = ppd.filter(pl.col("donor").is_in(donantes.to_list()))
    paso2 = paso1.unique(subset=["project_id"], keep="first")
    paso3 = paso2.filter(pl.col("rating").is_not_null() & pl.col("start_year").is_not_null())
    return [
        ("PPD crudo", ppd.height),
        ("Donantes con >= 10 proyectos", paso1.height),
        ("Sin project_id duplicado", paso2.height),
        ("Con rating y año de inicio", paso3.height),
        ("Tabla limpia (model_table)", model.height),
        ("fila_ok (país real, 1990-2017)", int(model["fila_ok"].sum())),
    ]


def problemas(ppd: pl.DataFrame, model: pl.DataFrame) -> list[tuple[str, int, int]]:
    pais = pl.col("countryname_WB").fill_null("")
    dur = model["duration_years"]
    return [
        ("project_id duplicado", int(ppd["project_id"].is_duplicated().sum()),
         int(model["project_id"].is_duplicated().sum())),
        ("rating nulo", int(ppd["rating"].is_null().sum()), int(model["rating"].is_null().sum())),
        ("año de inicio nulo", int(ppd["start_year"].is_null().sum()), int(model["start_year"].is_null().sum())),
        ("duración <= 0 días (hay negativos)", ppd.filter(pl.col("dias") <= 0).height, int((dur <= 0).sum())),
        ("duración 1-40 (unidad dudosa)", ppd.filter(pl.col("dias").is_between(1, 40)).height, 0),
        ("duración > 25 años", ppd.filter(pl.col("dias") / 365.25 > 25).height, int((dur > 25).sum())),
        ("compromiso USD <= 0", ppd.filter(pl.col("usd") <= 0).height, int((model["commitment_usd"] <= 0).sum())),
        ("espacios raros en país (NBSP)", ppd.filter(pais.str.contains("[\u00a0\u2007\u202f]")).height, 0),
        ("país no real (regional, global…)", ppd.filter(pais.str.contains(NON_COUNTRY.pattern)).height,
         model.filter(pl.col("iso3").is_in(["WLD", "SAS", "EAS", "ECS", "MEA"])).height),
    ]


def img_datasets() -> None:
    fin = pl.scan_csv(FIN_CSV, infer_schema_length=0)
    wb = pl.scan_csv(WB_CSV, infer_schema_length=0)
    ppd_n = pl.scan_csv(PPD_CSV, infer_schema_length=0)
    filas = [
        ("PPD2 (AidData)", ppd_n.select(pl.len()).collect().item(), len(ppd_n.collect_schema()), "1956-2016"),
        ("Financing to the SDGs", fin.select(pl.len()).collect().item(), len(fin.collect_schema()), "2000-2013"),
        ("World Development Indicators", wb.select(pl.len()).collect().item(), len(wb.collect_schema()), "1960-2025"),
    ]
    for nombre, ruta, anios in (
        ("wdi_2000_2013 (formato largo)", WDI_WINDOW, "2000-2013"),
        ("model_table (limpia)", MODEL, "1956-2016"),
        ("model_table_2000_2013", MODEL_WINDOW, "2000-2013"),
    ):
        tabla = pl.read_parquet(ruta)
        filas.append((nombre, tabla.height, tabla.width, anios))

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    tabla = ax.table(
        cellText=[[n, f"{r:,}", str(c), a] for n, r, c, a in filas],
        colLabels=["Dataset", "Registros", "Columnas", "Años"],
        colWidths=[0.4, 0.2, 0.15, 0.2],
        loc="center", cellLoc="left", colLoc="left",
    )
    tabla.auto_set_font_size(False)
    for (fila, _), celda in tabla.get_celld().items():
        celda.set_text_props(fontproperties=matplotlib.font_manager.FontProperties(**MONO))
        if fila == 0:
            celda.set_facecolor("#2d3e50")
            celda.get_text().set_color("white")
    tabla.scale(1, 1.6)
    ax.set_title("Datasets usados y número de registros", fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMG / "01_datasets_registros.png", dpi=150)
    plt.close(fig)


def img_embudo(pasos: list[tuple[str, int]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    nombres = [p[0] for p in pasos][::-1]
    valores = [p[1] for p in pasos][::-1]
    barras = ax.barh(nombres, valores, color="#3b7dd8")
    for b, v in zip(barras, valores):
        ax.text(v + 150, b.get_y() + b.get_height() / 2, f"{v:,}", va="center", fontdict=MONO)
    ax.set_xlim(0, max(valores) * 1.15)
    ax.set_xlabel("Registros")
    ax.set_title("Embudo de limpieza del PPD (registros que quedan en cada paso)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMG / "02_embudo_limpieza.png", dpi=150)
    plt.close(fig)


def img_problemas(filas: list[tuple[str, int, int]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.axis("off")
    tabla = ax.table(
        cellText=[[n, f"{a:,}", f"{d:,}", "OK" if d == 0 else "REVISAR"] for n, a, d in filas],
        colLabels=["Problema detectado", "Antes (crudo)", "Después", "Estado"],
        colWidths=[0.5, 0.18, 0.14, 0.14],
        loc="center", cellLoc="left", colLoc="left",
    )
    tabla.auto_set_font_size(False)
    for (fila, col), celda in tabla.get_celld().items():
        celda.set_text_props(fontproperties=matplotlib.font_manager.FontProperties(**MONO))
        if fila == 0:
            celda.set_facecolor("#2d3e50")
            celda.get_text().set_color("white")
        elif col == 3:
            celda.set_facecolor("#d4f4dd" if celda.get_text().get_text() == "OK" else "#f8d7da")
    tabla.scale(1, 1.5)
    ax.set_title("Control de calidad: problemas antes y después de limpiar", fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMG / "03_problemas_antes_despues.png", dpi=150)
    plt.close(fig)


def img_duracion(ppd: pl.DataFrame, model: pl.DataFrame) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    crudo = ppd["dias"].drop_nulls().to_numpy()
    a1.hist(crudo, bins=60, color="#d9534f")
    a1.set_title("Antes: project_duration crudo (días, con negativos)")
    a1.set_xlabel("valor crudo")
    a2.hist(model["duration_years"].drop_nulls().to_numpy(), bins=40, color="#5cb85c")
    a2.set_title("Después: duration_years (0-25 años)")
    a2.set_xlabel("años")
    fig.tight_layout()
    fig.savefig(IMG / "04_duracion_antes_despues.png", dpi=150)
    plt.close(fig)


def img_target(model: pl.DataFrame) -> None:
    conteo = model.group_by("success").len().sort("success")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    etiquetas = ["Fracaso (rating < 4)", "Éxito (rating >= 4)"]
    barras = ax.bar(etiquetas, conteo["len"].to_list(), color=["#d9534f", "#5cb85c"])
    total = model.height
    for b, v in zip(barras, conteo["len"].to_list()):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,} ({100 * v / total:.1f}%)", ha="center", va="bottom", fontdict=MONO)
    ax.set_ylabel("Proyectos")
    ax.set_title("Variable objetivo `success` en la tabla limpia", fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMG / "05_variable_objetivo.png", dpi=150)
    plt.close(fig)


def main() -> None:
    for p in (PPD_CSV, FIN_CSV, WB_CSV, MODEL, MODEL_WINDOW, WDI_WINDOW):
        if not p.is_file():
            raise SystemExit(f"Falta {p}. Corre download_datasets.py y el notebook EDA.ipynb primero.")
    IMG.mkdir(parents=True, exist_ok=True)
    ppd = cargar_ppd()
    model = pl.read_parquet(MODEL)

    pasos = embudo(ppd, model)
    filas = problemas(ppd, model)
    for nombre, n in pasos:
        print(f"{nombre:<34} {n:>8,}")
    print()
    for nombre, antes, despues in filas:
        print(f"{nombre:<34} {antes:>8,} -> {despues:,}")

    img_datasets()
    img_embudo(pasos)
    img_problemas(filas)
    img_duracion(ppd, model)
    img_target(model)
    print(f"\nImágenes en {IMG}")


if __name__ == "__main__":
    main()
