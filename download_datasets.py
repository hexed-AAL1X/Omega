#!/usr/bin/env python3
"""Descarga y deja los 3 datasets en datasets/, igual que el layout local."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"

SOURCES = [
    {
        "name": "PPD",
        "url": "https://docs.aiddata.org/ad4/datasets/PPD2_archive_Jan21_2022.zip",
        "zip_name": "PPD2_archive_Jan21_2022.zip",
        "extract_dir": "ppd",
        "keep": (
            "PPD2_Jan_21_2022.csv",
            "PPD2codebook_Jan_21_2022.pdf",
        ),
    },
    {
        "name": "Financing to the SDGs",
        "url": (
            "https://github.com/AidData-WM/public_datasets/releases/download/"
            "v1.0/Financing_to_the_Sustainable_Development_Goals_Dataset_version_1_0.zip"
        ),
        "zip_name": "Financing_to_SDGs_v1.0.zip",
        "extract_dir": "financing_sdgs",
        "keep": (
            "FinancingtotheSDGsDataset_v1.0.csv",
            "Readme_FinancingtotheSDGsDataset_v1.0.pdf",
        ),
    },
    {
        "name": "World Bank SDG",
        "url": "https://databank.worldbank.org/data/download/SDG_CSV.zip",
        "zip_name": "SDG_CSV.zip",
        "extract_dir": "worldbank_sdg",
        "keep": (
            "SDGData.csv",
            "SDGCountry.csv",
            "SDGSeries.csv",
            "SDGCountry-Series.csv",
            "SDGSeries-Time.csv",
            "SDGFootNote.csv",
        ),
    },
]


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 Nexora-dataset-downloader"})
    print(f"  bajando {dest.name} ...")
    with urlopen(req, timeout=120) as resp, dest.open("wb") as out:
        total = resp.headers.get("Content-Length")
        total_n = int(total) if total and total.isdigit() else None
        done = 0
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total_n:
                pct = done * 100 / total_n
                print(f"\r  {pct:5.1f}%  {done / 1e6:.1f} / {total_n / 1e6:.1f} MB", end="", flush=True)
            else:
                print(f"\r  {done / 1e6:.1f} MB", end="", flush=True)
    print()


def extract_selected(zip_path: Path, dest_dir: Path, keep: tuple[str, ...]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(keep)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if name not in wanted or info.is_dir():
                continue
            target = dest_dir / name
            print(f"  extrayendo {name}")
            with zf.open(info) as src, target.open("wb") as out:
                while True:
                    chunk = src.read(1024 * 256)
                    if not chunk:
                        break
                    out.write(chunk)


def already_extracted(dest_dir: Path, keep: tuple[str, ...]) -> bool:
    return dest_dir.is_dir() and all((dest_dir / name).is_file() for name in keep)


def main() -> int:
    parser = argparse.ArgumentParser(description="Descarga PPD, Financing SDGs y World Bank SDG.")
    parser.add_argument("--force", action="store_true", help="Vuelve a bajar y extraer aunque ya existan.")
    args = parser.parse_args()

    DATASETS.mkdir(parents=True, exist_ok=True)
    print(f"Destino: {DATASETS}\n")

    for src in SOURCES:
        print(f"== {src['name']} ==")
        zip_path = DATASETS / src["zip_name"]
        dest_dir = DATASETS / src["extract_dir"]

        if not args.force and already_extracted(dest_dir, src["keep"]):
            print("  ya está extraído, salto (usa --force para repetir)\n")
            continue

        if args.force or not zip_path.is_file():
            download(src["url"], zip_path)
        else:
            print(f"  ZIP ya existe: {zip_path.name}")

        extract_selected(zip_path, dest_dir, src["keep"])
        print()

    print("Listo. Estructura:")
    print("  datasets/ppd/")
    print("  datasets/financing_sdgs/")
    print("  datasets/worldbank_sdg/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
