"""Regenerate fig5 as a four-row figure.

Row A: three stages (early/middle/late) of the fluorescence image of the ROI
(rhodamine-labeled lipid channel).
Row B: the same three stages rendered as white background + BF-derived cell
contour + intensity-scaled LNP circles.
Row C: blank white placeholders (i, ii, iii) reserved for future panels.
Row D: dual-axis time course, one line per cell (low opacity) plus the
across-cell median (opaque) on each axis -- left axis: LNP count per cell
(red); right axis: median LNP intensity per cell (blue), with vertical phase
boundaries at 30 and 80 min (adsorption, clustering, saturation).

Writes both PNG and SVG to the paper figs dir.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from matplotlib import patches
from matplotlib.lines import Line2D
from skimage.measure import find_contours

from binding.services.filter_spots import read_spot_csv
from binding.core import compute_cell_mask, load_roi_stack

PANEL_LABEL_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
MEDIAN_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 12

# Panel B (white background) colors: chosen for contrast against white,
# distinct from the black-background yellow/cyan scheme used elsewhere.
CONTOUR_COLOR = "#0b2545"  # dark navy cell contour
SPOT_COLOR = "#b3001c"  # dark red spot outline

# Panel C: count (left axis) in red, LNP intensity (right axis) in blue; the
# per-cell traces reuse the same hue at low opacity so the two axes stay
# visually distinct while each keeps a single consistent color identity.
COUNT_COLOR = "#d7263d"
INTENSITY_COLOR = "#1f77b4"
PER_CELL_ALPHA = 0.25
PHASE_BOUNDARIES_MIN = (30.0, 80.0)
PHASE_LABELS = ("I", "II", "III")
PHASE_NAMES = ("adsorption", "clustering", "saturation")
PHASE_LINE_COLOR = "#555555"


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


def select_frame_fraction_time(n_frames: int, fraction: float) -> int:
    """Pick a time index at a given fraction of the ROI's total frame count.

    Used for the "middle" and "late" time points so the three sub-panels are
    spread across the full recording rather than clustered near where
    spot_count first crosses a threshold.
    """
    if n_frames <= 1:
        return 0
    return int(round(fraction * (n_frames - 1)))


def intensity_to_radius(intensity: float) -> float:
    """Map spot intensity to circle radius (linear: 2000 -> 2 px, 16000 -> 16 px)."""
    s = float(intensity)
    return float(np.clip(2.0 + 14.0 * (s - 2000.0) / 14000.0, 2.0, 16.0))


def resolve_spot_csv(filtered_dir: Path, roi: int, time_index: int) -> Path:
    return filtered_dir / f"roi{roi:02d}_time{time_index:09d}_filtered.csv"


def read_spot_intensities(filtered_dir: Path, roi: int, time_index: int) -> list[float]:
    path = resolve_spot_csv(filtered_dir, roi, time_index)
    if not path.exists():
        return []
    _, rows = read_spot_csv(path)
    return [float(row["intensity"]) for row in rows if row.get("intensity") not in (None, "")]


def per_cell_median_intensity_series(filtered_dir: Path, roi: int, times: list[float]) -> np.ndarray:
    """Per-time median LNP intensity for one cell; NaN where no spots were detected."""
    series = np.full(len(times), np.nan, dtype=float)
    for index in range(len(times)):
        intensities = read_spot_intensities(filtered_dir, roi, index)
        if intensities:
            series[index] = float(np.median(intensities))
    return series


def add_panel_label(axis, label: str, *, x: float = -0.14) -> None:
    axis.text(
        x,
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


def add_panel_border(axis, *, color: str = "black", linewidth: float = 0.8) -> None:
    x0, x1 = axis.get_xlim()
    y0, y1 = axis.get_ylim()
    xmin, xmax = min(x0, x1), max(x0, x1)
    ymin, ymax = min(y0, y1), max(y0, y1)
    axis.add_patch(
        patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            clip_on=True,
            zorder=10,
        )
    )


def show_all_spines(axis, *, color: str = "black", linewidth: float = 0.8) -> None:
    for side in ("top", "right", "bottom", "left"):
        axis.spines[side].set_visible(True)
        axis.spines[side].set_color(color)
        axis.spines[side].set_linewidth(linewidth)


def render_fluorescence(axis, image: np.ndarray, *, title: str | None = None) -> None:
    axis.imshow(
        image,
        cmap="inferno",
        vmin=float(image.min()),
        vmax=float(image.max()),
        interpolation="nearest",
    )
    axis.set_facecolor("black")
    axis.axis("off")
    add_panel_border(axis)
    if title:
        axis.text(
            0.5,
            -0.06,
            title,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=TICK_LABEL_FONTSIZE,
            color="black",
        )


def render_detections_white(
    axis,
    image_shape: tuple[int, int],
    rows: list[dict[str, str]],
    contour_coords: list[np.ndarray] | None,
    *,
    title: str | None = None,
) -> None:
    """Panel B sub-image: white background + cell contour + intensity-scaled spot circles."""
    h, w = image_shape
    background = np.ones((h, w), dtype=np.float32)
    axis.imshow(background, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axis.set_facecolor("white")
    axis.set_xlim(-0.5, w - 0.5)
    axis.set_ylim(h - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.axis("off")
    add_panel_border(axis)

    if contour_coords:
        for cont in contour_coords:
            if len(cont) > 1:
                axis.plot(cont[:, 1], cont[:, 0], color=CONTOUR_COLOR, linewidth=1.8, alpha=0.95)

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
                edgecolor=SPOT_COLOR,
                linewidth=1.3,
            )
        )

    if title:
        axis.text(
            0.5,
            -0.06,
            title,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=TICK_LABEL_FONTSIZE,
            color="black",
        )


def render_placeholder(
    axis,
    image_shape: tuple[int, int],
    *,
    title: str | None = None,
) -> None:
    """Blank white panel matching the image-panel footprint of rows A and B."""
    h, w = image_shape
    background = np.ones((h, w), dtype=np.float32)
    axis.imshow(background, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axis.set_facecolor("white")
    axis.set_xlim(-0.5, w - 0.5)
    axis.set_ylim(h - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.axis("off")
    add_panel_border(axis)
    if title:
        axis.text(
            0.5,
            -0.06,
            title,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=TICK_LABEL_FONTSIZE,
            color="black",
        )


def add_phase_markers(axis, x_max: float) -> None:
    """Mark LNP adsorption phases with vertical lines and region labels."""
    import matplotlib.transforms as transforms

    panel_x = transforms.blended_transform_factory(axis.transData, axis.transAxes)
    for boundary in PHASE_BOUNDARIES_MIN:
        axis.axvline(
            boundary,
            color=PHASE_LINE_COLOR,
            linewidth=1.2,
            linestyle="--",
            zorder=1,
        )
    bounds = [0.0, *PHASE_BOUNDARIES_MIN, x_max]
    for index, (roman, name) in enumerate(zip(PHASE_LABELS, PHASE_NAMES)):
        x_center = 0.5 * (bounds[index] + bounds[index + 1])
        axis.text(
            x_center,
            0.97,
            roman,
            transform=panel_x,
            ha="center",
            va="top",
            fontsize=TICK_LABEL_FONTSIZE,
            fontweight="bold",
            color=PHASE_LINE_COLOR,
            clip_on=False,
        )
        axis.text(
            x_center,
            0.90,
            name,
            transform=panel_x,
            ha="center",
            va="top",
            fontsize=TICK_LABEL_FONTSIZE - 1,
            color=PHASE_LINE_COLOR,
            clip_on=False,
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

    n_frames = len(grouped[selected_roi][1])
    mid_time = select_frame_fraction_time(n_frames, 0.5)
    late_time = select_frame_fraction_time(n_frames, 0.9)
    stage_times = [
        (selected_time, "early", "i"),
        (mid_time, "middle", "ii"),
        (late_time, "late", "iii"),
    ]

    fluo_stack = load_roi_stack(input_dir, position, selected_roi, channel)
    bf_stack = load_roi_stack(input_dir, position, selected_roi, 0)

    if time_unit not in {"sec", "min"}:
        raise ValueError("--time-unit must be 'sec' or 'min'")

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(11.5, 15.5), facecolor="white")
    gs = fig.add_gridspec(
        4,
        3,
        height_ratios=[1.0, 1.0, 1.0, 0.85],
        hspace=0.4,
        wspace=0.12,
        left=0.09,
        right=0.90,
        bottom=0.06,
        top=0.96,
    )
    axis_a_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axis_b_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    axis_c_axes = [fig.add_subplot(gs[2, i]) for i in range(3)]
    axis_d = fig.add_subplot(gs[3, :])

    # ---- Row A: three stages of the raw fluorescence image (rhodamine-labeled LNP) ----
    for axis, (time_index, title, sub_label) in zip(axis_a_axes, stage_times):
        render_fluorescence(axis, np.asarray(fluo_stack[time_index]), title=title)
    add_panel_label(axis_a_axes[0], "A")
    for axis, (_, _, sub_label) in zip(axis_a_axes, stage_times):
        add_panel_label(axis, sub_label, x=0.04)

    # ---- Row B: same three stages, white background + contour + spot circles ----
    for axis, (time_index, title, sub_label) in zip(axis_b_axes, stage_times):
        bf_frame = np.asarray(bf_stack[time_index], dtype=np.float64)
        mask = compute_cell_mask(bf_frame)
        contour_coords = list(find_contours(mask.astype(float), level=0.5))

        spot_csv = resolve_spot_csv(filtered_dir, selected_roi, time_index)
        if spot_csv.exists():
            _, spot_rows = read_spot_csv(spot_csv)
        else:
            spot_rows = []  # no filtered CSV for this frame; draw contour only

        render_detections_white(
            axis,
            np.asarray(fluo_stack[time_index]).shape,
            spot_rows,
            contour_coords=contour_coords,
            title=title,
        )
    add_panel_label(axis_b_axes[0], "B")
    for axis, (_, _, sub_label) in zip(axis_b_axes, stage_times):
        add_panel_label(axis, sub_label, x=0.04)

    # ---- Row C: blank placeholders (same footprint as rows A/B) ----
    image_shape = np.asarray(fluo_stack[0]).shape
    for axis, (_, title, sub_label) in zip(axis_c_axes, stage_times):
        render_placeholder(axis, image_shape, title=title)
    add_panel_label(axis_c_axes[0], "C")
    for axis, (_, _, sub_label) in zip(axis_c_axes, stage_times):
        add_panel_label(axis, sub_label, x=0.04)

    # ---- Row D: dual-axis, one line per cell (low opacity) + across-cell median (opaque) ----
    reference_times = grouped[rois[0]][0]
    plot_times = to_plot_time(reference_times, time_unit)
    n_times = len(reference_times)

    count_matrix = np.array([grouped[roi_index][1] for roi_index in rois], dtype=float)
    for row in count_matrix:
        axis_d.plot(plot_times, row, color=COUNT_COLOR, linewidth=1.0, alpha=PER_CELL_ALPHA)
    median_counts = np.median(count_matrix, axis=0)
    axis_d.plot(plot_times, median_counts, color=COUNT_COLOR, linewidth=2.0)

    x_label = "time (min)" if time_unit == "min" else "time (s)"
    axis_d.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    axis_d.set_ylabel("LNP count per cell", fontsize=AXIS_LABEL_FONTSIZE, color=COUNT_COLOR)
    axis_d.set_facecolor("white")
    show_all_spines(axis_d)
    axis_d.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE, pad=6)
    axis_d.tick_params(axis="y", labelcolor=COUNT_COLOR)
    add_panel_label(axis_d, "D")
    add_phase_markers(axis_d, float(plot_times[-1]) if plot_times else PHASE_BOUNDARIES_MIN[-1] + 10.0)

    intensity_matrix = np.array(
        [per_cell_median_intensity_series(filtered_dir, roi_index, reference_times) for roi_index in rois]
    )
    axis_d2 = axis_d.twinx()
    show_all_spines(axis_d2)
    for row in intensity_matrix:
        axis_d2.plot(plot_times, row, color=INTENSITY_COLOR, linewidth=1.0, alpha=PER_CELL_ALPHA)
    median_intensity = np.nanmedian(intensity_matrix, axis=0)
    axis_d2.plot(plot_times, median_intensity, color=INTENSITY_COLOR, linewidth=2.0)
    axis_d2.set_ylabel(
        "median LNP intensity per cell (a.u.)", fontsize=AXIS_LABEL_FONTSIZE, color=INTENSITY_COLOR
    )
    axis_d2.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE, labelcolor=INTENSITY_COLOR)

    legend_handles = [
        Line2D([0], [0], color=COUNT_COLOR, linewidth=2.0, label="LNP count (median)"),
        Line2D([0], [0], color=INTENSITY_COLOR, linewidth=2.0, label="LNP intensity (median)"),
    ]
    axis_d.legend(
        handles=legend_handles,
        fontsize=LEGEND_FONTSIZE,
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        framealpha=0.9,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, facecolor="white")
    fig.savefig(output_svg, format="svg", facecolor="white")
    plt.close(fig)

    max_count = max(grouped[selected_roi][1]) if grouped[selected_roi][1] else 0
    print(
        f"Saved {output_png} and {output_svg} "
        f"with roi={selected_roi}, times(early/mid/late)={selected_time}/{mid_time}/{late_time}, "
        f"max_count={max_count}, cells={len(rois)}"
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
