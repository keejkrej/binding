"""Regenerate fig5 with an EARLY (sparse) time frame for panel A.

This is a copy of plot_lnp.py with the auto_select_time function replaced so
that it picks an early time index (small spot_count) instead of the dense
~70%-of-max frame. It writes both PNG and SVG to the paper figs dir.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from matplotlib import patches
from skimage.measure import find_contours

from binding.commands.filter_spots import read_spot_csv
from binding.core import compute_cell_mask, load_roi_stack

PANEL_LABEL_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
MEDIAN_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 12


def to_plot_time(values: list[float], unit: str) -> list[float]:
    if unit == "min":
        return [value / 60.0 for value in values]
    if unit == "sec":
        return values
    raise ValueError(f"Unsupported time unit: {unit}")


def read_counts_csv(path: Path) -> tuple[list[int], dict[int, tuple[list[float], list[int]]]]:
    grouped: dict[int, tuple[list[float], list[int]]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"roi", "time_real", "spot_count"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Counts CSV must contain roi, time_real, and spot_count columns")
        for row in reader:
            roi = int(row["roi"])
            times, counts = grouped.setdefault(roi, ([], []))
            times.append(float(row["time_real"]))
            counts.append(int(row["spot_count"]))

    if not grouped:
        raise ValueError("Counts CSV has no rows")

    for roi in grouped:
        times, counts = grouped[roi]
        order = sorted(range(len(times)), key=lambda index: times[index])
        grouped[roi] = (
            [times[index] for index in order],
            [counts[index] for index in order],
        )

    return sorted(grouped), grouped


def auto_select_roi(grouped: dict[int, tuple[list[float], list[int]]]) -> int:
    return max(
        grouped,
        key=lambda roi: grouped[roi][1][-1] if grouped[roi][1] else 0,
    )


def auto_select_time_early(counts_by_roi: dict[int, tuple[list[float], list[int]]], roi: int) -> int:
    """Pick an EARLY, SPARSE time frame.

    The original auto_select_time picked ~70% of max spot_count (dense).
    For the paper we want to show an early, sparse frame so the reader can
    see individual LNP binding events. We pick the first time index where
    spot_count reaches a small threshold (~10% of max) so there are clearly
    visible but sparse spots. Falls back to the first nonzero count, or 0.
    """
    _, counts = counts_by_roi[roi]
    if not counts:
        return 0
    max_count = max(counts)
    if max_count <= 0:
        return 0
    target = max(1, int(max_count * 0.10))  # ~10% of max, at least 1
    for index, count in enumerate(counts):
        if count >= target:
            return index
    return 0


def intensity_to_radius(intensity: float) -> float:
    """Map spot intensity to circle radius (linear: 2000 -> 2 px, 16000 -> 16 px)."""
    s = float(intensity)
    return float(np.clip(2.0 + 14.0 * (s - 2000.0) / 14000.0, 2.0, 16.0))


def resolve_spot_csv(filtered_dir: Path, roi: int, time_index: int) -> Path:
    path = filtered_dir / f"roi{roi:02d}_time{time_index:09d}_filtered.csv"
    if not path.exists():
        raise ValueError(f"Filtered spot CSV not found: {path}")
    return path


def add_panel_label(axis, label: str) -> None:
    axis.text(
        -0.14,
        1.04,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        color="black",
        clip_on=False,
    )


def render_fluorescence(axis, image: np.ndarray) -> None:
    axis.imshow(
        image,
        cmap="inferno",
        vmin=float(image.min()),
        vmax=float(image.max()),
        interpolation="nearest",
    )
    axis.set_facecolor("black")
    axis.axis("off")


def render_detections(
    axis,
    image: np.ndarray,
    rows: list[dict[str, str]],
    contour_coords: list[np.ndarray] | None = None,
) -> None:
    """Render fig B purely: black background + cell contour from BF + circles sized by spot intensity."""
    h, w = image.shape
    background = np.zeros((h, w), dtype=np.float32)
    axis.imshow(background, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axis.set_facecolor("black")
    axis.set_xlim(-0.5, w - 0.5)
    axis.set_ylim(h - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.axis("off")

    if contour_coords:
        for cont in contour_coords:
            if len(cont) > 1:
                axis.plot(cont[:, 1], cont[:, 0], color="#00f0ff", linewidth=1.8, alpha=0.95)

    for row in rows:
        x = float(row["x"])
        y = float(row["y"])
        intens = float(row.get("intensity", 5000.0))
        radius = intensity_to_radius(intens)
        axis.add_patch(
            patches.Circle(
                (x, y),
                radius=radius,
                fill=False,
                edgecolor="#ffeb3b",
                linewidth=1.3,
            )
        )


def render_early(
    input_dir: Path,
    counts_csv: Path,
    filtered_dir: Path,
    output_png: Path,
    output_svg: Path,
    position: int = 0,
    channel: int = 1,
    time_unit: str = "min",
) -> None:
    rois, grouped = read_counts_csv(counts_csv)
    selected_roi = auto_select_roi(grouped)
    selected_time = auto_select_time_early(grouped, selected_roi)

    fluo_stack = load_roi_stack(input_dir, position, selected_roi, channel)
    bf_stack = load_roi_stack(input_dir, position, selected_roi, 0)
    image = fluo_stack[selected_time]
    bf_frame = np.asarray(bf_stack[selected_time], dtype=np.float64)
    mask = compute_cell_mask(bf_frame)
    contour_coords = find_contours(mask.astype(float), level=0.5)
    spot_csv = resolve_spot_csv(filtered_dir, selected_roi, selected_time)
    _, spot_rows = read_spot_csv(spot_csv)

    if time_unit not in {"sec", "min"}:
        raise ValueError("--time-unit must be 'sec' or 'min'")

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white")
    axis_d, axis_e, axis_f = axes
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.14, top=0.88, wspace=0.28)

    render_fluorescence(axis_d, np.asarray(image))
    add_panel_label(axis_d, "A")
    render_detections(axis_e, np.asarray(image), spot_rows, contour_coords=list(contour_coords))
    add_panel_label(axis_e, "B")

    median_times: list[float] | None = None
    median_counts: list[float] | None = None
    for roi_index in rois:
        times, counts = grouped[roi_index]
        plot_times = to_plot_time(times, time_unit)
        axis_f.plot(
            plot_times,
            counts,
            color="#b8b8b8",
            linewidth=1.0,
            alpha=0.8,
        )
        if median_times is None:
            median_times = plot_times
            median_counts = [float(value) for value in counts]
        else:
            for index, count in enumerate(counts):
                median_counts[index] += float(count)

    if median_times is not None and median_counts is not None:
        median_counts = [value / len(rois) for value in median_counts]
        axis_f.plot(
            median_times,
            median_counts,
            color="#d7263d",
            linewidth=2.0,
            label="median",
        )
        axis_f.text(
            0.03,
            0.97,
            "median",
            transform=axis_f.transAxes,
            color="#d7263d",
            fontsize=MEDIAN_LABEL_FONTSIZE,
            ha="left",
            va="top",
        )

    x_label = "time (min)" if time_unit == "min" else "time (s)"
    axis_f.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    axis_f.set_ylabel("LNP count", fontsize=AXIS_LABEL_FONTSIZE)
    axis_f.set_facecolor("white")
    axis_f.spines["top"].set_visible(False)
    axis_f.spines["right"].set_visible(False)
    axis_f.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE, pad=6)
    add_panel_label(axis_f, "C")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, facecolor="white")
    fig.savefig(output_svg, format="svg", facecolor="white")
    plt.close(fig)

    max_count = max(grouped[selected_roi][1]) if grouped[selected_roi][1] else 0
    print(
        f"Saved {output_png} and {output_svg} "
        f"with roi={selected_roi}, time={selected_time}, "
        f"spots={len(spot_rows)}, max_count={max_count}, cells={len(rois)}"
    )


if __name__ == "__main__":
    DATA_DIR = Path("/home/jack/data/lisca_review/fig5/20260324_1")
    COUNTS_CSV = DATA_DIR / "results" / "spot_counts_position000_channel001.csv"
    FILTERED_DIR = DATA_DIR / "results" / "filtered"
    OUT_PNG = Path("/home/jack/workspace/lisca-paper/figs/fig5.png")
    OUT_SVG = Path("/home/jack/workspace/lisca-paper/figs/fig5.svg")
    render_early(
        input_dir=DATA_DIR,
        counts_csv=COUNTS_CSV,
        filtered_dir=FILTERED_DIR,
        output_png=OUT_PNG,
        output_svg=OUT_SVG,
        position=0,
        channel=1,
        time_unit="min",
    )
