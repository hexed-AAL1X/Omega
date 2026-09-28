#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parent
RES = ROOT / "resultados"
IMG = ROOT / "docs" / "img"

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

AZUL = "#1f4e79"
NARANJA = "#c55a11"
GRIS = "#7f7f7f"


def guardar(fig, nombre: str) -> None:
    fig.tight_layout()
    fig.savefig(IMG / nombre, dpi=160)
    plt.close(fig)
    print("OK", IMG / nombre)


def speedup_y_eficiencia(df: pl.DataFrame, titulo: str, nombre: str) -> None:
    conc = df.filter(pl.col("modo") == "concurrente").sort("workers")
    w = conc["workers"].to_list()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(w, w, "--", color=GRIS, label="Ideal (lineal)")
    a1.plot(w, conc["speedup"].to_list(), "o-", color=AZUL, lw=2, label="Medido")
    for x, s in zip(w, conc["speedup"].to_list()):
        a1.annotate(f"{s:.2f}x", (x, s), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    a1.set_xscale("log", base=2)
    a1.set_xticks(w, [str(v) for v in w])
    a1.set_xlabel("Workers (goroutines)")
    a1.set_ylabel("Speedup = Tseq / Tconc")
    a1.set_title("Speedup")
    a1.legend()
    a2.bar([str(v) for v in w], conc["eficiencia"].to_list(), color=NARANJA)
    a2.axhline(1, ls="--", color=GRIS)
    a2.set_ylim(0, 1.1)
    a2.set_xlabel("Workers (goroutines)")
    a2.set_ylabel("Eficiencia = Speedup / Workers")
    a2.set_title("Eficiencia")
    fig.suptitle(titulo, fontweight="bold")
    guardar(fig, nombre)


def tiempos(df: pl.DataFrame, titulo: str, nombre: str) -> None:
    df = df.with_columns(
        pl.when(pl.col("modo") == "secuencial").then(pl.lit("Sec.")).otherwise(pl.col("workers").cast(pl.Utf8) + " w").alias("etq")
    )
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colores = [GRIS if m == "secuencial" else AZUL for m in df["modo"].to_list()]
    ax.bar(df["etq"].to_list(), df["media_recortada_s"].to_list(), yerr=df["desv_s"].to_list(), color=colores, capsize=4)
    for i, t in enumerate(df["media_recortada_s"].to_list()):
        ax.text(i, t, f"{t:.2f}s", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Media recortada (s)")
    ax.set_title(titulo, fontweight="bold")
    guardar(fig, nombre)


def uso_recursos(df: pl.DataFrame, nombre: str) -> None:
    df = df.with_columns(
        pl.when(pl.col("modo") == "secuencial").then(pl.lit("Sec.")).otherwise(pl.col("workers").cast(pl.Utf8) + " w").alias("etq")
    )
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.bar(df["etq"].to_list(), df["uso_cpu_pct"].to_list(), color=AZUL)
    a1.set_ylim(0, 105)
    a1.set_ylabel("Uso de CPU (% del total de núcleos)")
    a1.set_title("Uso de CPU")
    a2.bar(df["etq"].to_list(), df["heap_pico_mb"].to_list(), color=NARANJA)
    a2.set_ylabel("Heap pico (MB)")
    a2.set_title("Memoria")
    fig.suptitle("Pipeline Financing (1,25 M filas): uso de recursos", fontweight="bold")
    guardar(fig, nombre)


def tamano_pipeline(df: pl.DataFrame, nombre: str) -> None:
    seq = df.filter(pl.col("modo") == "secuencial").sort("filas")
    con = df.filter(pl.col("modo") == "concurrente").sort("filas")
    w = con["workers"][0]
    x = [f / 1e6 for f in seq["filas"].to_list()]
    fig, a1 = plt.subplots(figsize=(8, 4.2))
    a1.plot(x, seq["media_recortada_s"].to_list(), "o-", color=GRIS, lw=2, label="Secuencial")
    a1.plot(x, con["media_recortada_s"].to_list(), "o-", color=AZUL, lw=2, label=f"Concurrente ({w} workers)")
    a1.set_xlabel("Registros procesados (millones)")
    a1.set_ylabel("Media recortada (s)")
    a2 = a1.twinx()
    a2.plot(x, con["speedup"].to_list(), "s--", color=NARANJA, label="Speedup")
    a2.set_ylabel("Speedup", color=NARANJA)
    a2.grid(False)
    l1, e1 = a1.get_legend_handles_labels()
    l2, e2 = a2.get_legend_handles_labels()
    a1.legend(l1 + l2, e1 + e2, loc="upper left")
    a1.set_title("Pipeline Financing: escalabilidad por volumen de datos", fontweight="bold")
    guardar(fig, nombre)


def escalabilidad_rf(df: pl.DataFrame, nombre: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, exp, col, xlabel in [
        (axes[0], "tamano_datos", "filas_train", "Filas de entrenamiento"),
        (axes[1], "numero_arboles", "arboles", "Número de árboles"),
    ]:
        sub = df.filter(pl.col("experimento") == exp)
        seq = sub.filter(pl.col("modo") == "secuencial").sort(col)
        con = sub.filter(pl.col("modo") == "concurrente").sort(col)
        w = con["workers"][0]
        x = seq[col].to_list()
        ax.plot(x, seq["media_recortada_s"].to_list(), "o-", color=GRIS, lw=2, label="Secuencial")
        ax.plot(x, con["media_recortada_s"].to_list(), "o-", color=AZUL, lw=2, label=f"Concurrente ({w} workers)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Media recortada (s)")
        ax2 = ax.twinx()
        ax2.plot(x, con["speedup"].to_list(), "s--", color=NARANJA, label="Speedup")
        ax2.set_ylabel("Speedup", color=NARANJA)
        ax2.set_ylim(0, max(con["speedup"].to_list()) * 1.3)
        ax2.grid(False)
        l1, e1 = ax.get_legend_handles_labels()
        l2, e2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, e1 + e2, loc="upper left", fontsize=9)
    axes[0].set_title("Variando el tamaño de datos (50 árboles)")
    axes[1].set_title("Variando el número de árboles (train completo)")
    fig.suptitle("Random Forest: escalabilidad", fontweight="bold")
    guardar(fig, nombre)


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    rf = pl.read_csv(RES / "benchmark.csv")
    speedup_y_eficiencia(rf, "Random Forest: entrenamiento de 50 árboles", "06_rf_speedup_eficiencia.png")
    tiempos(rf, "Random Forest: tiempo de entrenamiento por configuración", "07_rf_tiempos.png")

    pw = RES / "pipeline_workers.csv"
    if pw.exists():
        p = pl.read_csv(pw)
        speedup_y_eficiencia(p, "Pipeline Financing (1 252 036 registros)", "08_pipeline_speedup_eficiencia.png")
        uso_recursos(p, "09_pipeline_recursos.png")
    pt = RES / "pipeline_tamano.csv"
    if pt.exists():
        tamano_pipeline(pl.read_csv(pt), "10_pipeline_escalabilidad.png")
    es = RES / "escalabilidad.csv"
    if es.exists():
        escalabilidad_rf(pl.read_csv(es), "11_rf_escalabilidad.png")


if __name__ == "__main__":
    main()
