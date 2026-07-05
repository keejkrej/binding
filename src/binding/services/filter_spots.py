from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from binding.core.paths import filtered_spots_output_path


@dataclass(frozen=True)
class FilterSpotsResult:
    output_dir: Path
    file_count: int
    total_in: int
    total_out: int


def filter_spot_frame(
    rows: list[dict[str, str]],
    *,
    min_intensity: float | None,
    max_intensity: float | None,
    min_fwhm: float | None,
    max_fwhm: float | None,
    min_probability: float | None,
) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for row in rows:
        if min_intensity is not None and float(row["intensity"]) < min_intensity:
            continue
        if max_intensity is not None and float(row["intensity"]) > max_intensity:
            continue
        if "fwhm" in row and row["fwhm"] != "":
            fwhm = float(row["fwhm"])
            if min_fwhm is not None and fwhm < min_fwhm:
                continue
            if max_fwhm is not None and fwhm > max_fwhm:
                continue
        if min_probability is not None and float(row["probability"]) < min_probability:
            continue
        filtered.append(row)
    return filtered


def read_spot_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Spot CSV {path} has no header")
        rows = list(reader)
        return list(reader.fieldnames), rows


def write_spot_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_filter_spots(
    input_dir: Path,
    *,
    output: Path,
    min_intensity: float | None,
    max_intensity: float | None,
    min_fwhm: float | None,
    max_fwhm: float | None,
    min_probability: float | None,
) -> FilterSpotsResult:
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")

    total_in = 0
    total_out = 0
    for csv_path in csv_files:
        fieldnames, rows = read_spot_csv(csv_path)
        required = {"y", "x", "intensity", "probability"}
        if not required.issubset(fieldnames):
            raise ValueError(
                f"{csv_path} must contain y, x, intensity, and probability columns"
            )

        filtered_rows = filter_spot_frame(
            rows,
            min_intensity=min_intensity,
            max_intensity=max_intensity,
            min_fwhm=min_fwhm,
            max_fwhm=max_fwhm,
            min_probability=min_probability,
        )
        output_path = output / filtered_spots_output_path(csv_path).name
        write_spot_csv(output_path, fieldnames, filtered_rows)
        total_in += len(rows)
        total_out += len(filtered_rows)

    return FilterSpotsResult(
        output_dir=output,
        file_count=len(csv_files),
        total_in=total_in,
        total_out=total_out,
    )