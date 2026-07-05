from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlotResult:
    output_path: Path
    row_count: int
    bins: int
    topk: int | None


def histogram_output_path(input_file: Path) -> Path:
    return input_file.with_name(f"{input_file.stem}_histograms.png")


def read_measurements(input_file: Path) -> list[tuple[float, float]]:
    measurements: list[tuple[float, float]] = []

    with open(input_file, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"volume", "total_intensity"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Analysis CSV must contain volume and total_intensity columns"
            )

        for row in reader:
            measurements.append((float(row["volume"]), float(row["total_intensity"])))

    if not measurements:
        raise ValueError("Analysis CSV has no measurement rows")

    return measurements


def run_plot(
    input_file: Path,
    *,
    bins: int,
    topk: int | None,
) -> PlotResult:
    measurements = read_measurements(input_file)

    if topk is not None:
        measurements = sorted(
            measurements,
            key=lambda measurement: measurement[0],
            reverse=True,
        )[:topk]

    volumes = [measurement[0] for measurement in measurements]
    total_intensities = [measurement[1] for measurement in measurements]

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output_path = histogram_output_path(input_file)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].hist(volumes, bins=bins, color="#2563eb", edgecolor="white")
    axes[0].set_title("Volume")
    axes[0].set_xlabel("volume")
    axes[0].set_ylabel("count")

    axes[1].hist(total_intensities, bins=bins, color="#dc2626", edgecolor="white")
    axes[1].set_title("Total intensity")
    axes[1].set_xlabel("total_intensity")
    axes[1].set_ylabel("count")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return PlotResult(
        output_path=output_path,
        row_count=len(volumes),
        bins=bins,
        topk=topk,
    )