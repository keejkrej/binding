from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from matplotlib import patches
from skimage.measure import find_contours

from binding.commands.filter_spots import read_spot_csv
from binding.core import compute_cell_mask, load_roi_stack

FILTERED_SPOT_FILE_RE = re.compile(
    r"^roi(?P<roi>\d+)_time(?P<time>\d+)_filtered\.csv$",
    re.IGNORECASE,
)

PANEL_LABEL_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
MEDIAN_LABEL_FONTSIZE = 14


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


def auto_select_time(counts_by_roi: dict[int, tuple[list[float], list[int]]], roi: int) -> int:
    _, counts = counts_by_roi[roi]
    if not counts:
        return 0
    target = max(counts) * 0.7
    return next(
        (index for index, count in enumerate(counts) if count >= target),
        len(counts) - 1,
    )


def intensity_to_radius(intensity: float) -> float:
    """Map spot intensity to circle radius (linear: 2000 -> 2 px, 16000 -> 16 px)."""
    s = float(intensity)
    return float(np.clip(2.0 + 14.0 * (s - 2000.0) / 14000.0, 2.0, 16.0))


def _load_spots_for_roi(filtered_dir: Path, roi: int, n_times: int) -> dict[int, list[dict[str, str]]]:
    spots: dict[int, list[dict[str, str]]] = {}
    for ti in range(n_times):
        try:
            p = resolve_spot_csv(filtered_dir, roi, ti)
            _, rows = read_spot_csv(p)
            spots[ti] = rows
        except Exception:
            spots[ti] = []
    return spots


def generate_b_movie(
    input_dir: Path,
    filtered_dir: Path,
    position: int,
    roi: int,
    output_path: Path,
    channel: int = 1,
    fps: float = 12.0,
) -> None:
    """Create mp4 of figB (BF-derived contour + intensity-scaled spot circles) over time for one ROI."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation

    fluo_stack = load_roi_stack(input_dir, position, roi, channel)
    bf_stack = load_roi_stack(input_dir, position, roi, 0)
    n_times = int(fluo_stack.shape[0])
    spots_by_t = _load_spots_for_roi(filtered_dir, roi, n_times)
    h, w = fluo_stack.shape[1:3] if fluo_stack.ndim == 3 else fluo_stack.shape[-2:]

    dpi = 72
    fig, ax = plt.subplots(figsize=(max(3.0, w / dpi), max(3.0, h / dpi)), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.margins(0)

    def _draw_frame(ti: int) -> None:
        ax.clear()
        fluo = np.asarray(fluo_stack[ti])
        bf = np.asarray(bf_stack[ti], dtype=np.float64)
        mask = compute_cell_mask(bf)
        conts = find_contours(mask.astype(float), level=0.5)
        # Pure black background - no raw fluorescence overlay
        bg = np.zeros((h, w), dtype=np.float32)
        ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_facecolor("black")
        for cont in conts:
            if len(cont) > 1:
                ax.plot(cont[:, 1], cont[:, 0], color="#00f0ff", linewidth=1.6, alpha=0.95)
        for row in spots_by_t.get(ti, []):
            x = float(row["x"])
            y = float(row["y"])
            r = intensity_to_radius(float(row.get("intensity", 5000.0)))
            ax.add_patch(patches.Circle((x, y), radius=r, fill=False, edgecolor="#ffeb3b", linewidth=1.2))
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0.01, 0.99, f"roi{roi:02d} t={ti}", transform=ax.transAxes, color="#cccccc", fontsize=7, va="top")

    def update(ti: int):
        _draw_frame(ti)
        return []

    _draw_frame(0)  # init view
    ani = FuncAnimation(fig, update, frames=n_times, interval=1000.0 / fps, blit=False, repeat=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(output_path), writer="ffmpeg", fps=fps, dpi=dpi, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"])
    plt.close(fig)


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
    """Render fig B purely: black background + cell contour from BF + circles sized by spot intensity.
    No raw fluorescence data overlay.
    """
    # Pure black background via zero image (ensures black pixels, not just facecolor)
    h, w = image.shape
    background = np.zeros((h, w), dtype=np.float32)
    axis.imshow(background, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axis.set_facecolor("black")
    axis.set_xlim(-0.5, w - 0.5)
    axis.set_ylim(h - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.axis("off")

    # cell contour (cyan)
    if contour_coords:
        for cont in contour_coords:
            if len(cont) > 1:
                axis.plot(cont[:, 1], cont[:, 0], color="#00f0ff", linewidth=1.8, alpha=0.95)

    # spots as circles, radius from intensity
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


def plot_lnp(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=True,
            file_okay=False,
            help="Dataset directory with roi/ crops.",
        ),
    ],
    counts_csv: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Spot counts CSV produced by binding spot-counts.",
        ),
    ],
    filtered_dir: Annotated[
        Path,
        typer.Option(
            "--filtered-dir",
            exists=True,
            file_okay=False,
            help="Directory containing filtered spot CSV files.",
        ),
    ] = ...,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output PNG path.",
        ),
    ] = Path("fig5_panels.png"),
    position: Annotated[
        int,
        typer.Option("--position", "-p", help="Position index for ROI loading."),
    ] = 0,
    channel: Annotated[
        int,
        typer.Option("--channel", "-c", help="Fluorescence channel index."),
    ] = 1,
    roi: Annotated[
        int | None,
        typer.Option("--roi", help="ROI index for panels d and e."),
    ] = None,
    time: Annotated[
        int | None,
        typer.Option("--time", "-t", help="Time index for panels d and e."),
    ] = None,
    time_unit: Annotated[
        str,
        typer.Option(
            "--time-unit",
            help="Time axis unit. Counts CSV stores time_real in seconds.",
        ),
    ] = "min",
    movies: Annotated[
        Path | None,
        typer.Option(
            "--movies",
            help="Directory to write per-ROI figB mp4 movies (one per roi). If set, generates movies of B viz for all ROIs after the static figure.",
        ),
    ] = None,
) -> None:
    try:
        rois, grouped = read_counts_csv(counts_csv)
        selected_roi = roi if roi is not None else auto_select_roi(grouped)
        selected_time = time if time is not None else auto_select_time(grouped, selected_roi)
        fluo_stack = load_roi_stack(input_dir, position, selected_roi, channel)
        bf_stack = load_roi_stack(input_dir, position, selected_roi, 0)
        image = fluo_stack[selected_time]
        bf_frame = np.asarray(bf_stack[selected_time], dtype=np.float64)
        mask = compute_cell_mask(bf_frame)
        # find_contours returns (row, col) i.e. (y, x) coords
        contour_coords = find_contours(mask.astype(float), level=0.5)
        spot_csv = resolve_spot_csv(filtered_dir, selected_roi, selected_time)
        _, spot_rows = read_spot_csv(spot_csv)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if time_unit not in {"sec", "min"}:
        raise typer.BadParameter("--time-unit must be 'sec' or 'min'")

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

    x_label = "t (min)" if time_unit == "min" else "t (s)"
    axis_f.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    axis_f.set_ylabel("n", fontsize=AXIS_LABEL_FONTSIZE)
    axis_f.set_facecolor("white")
    axis_f.spines["top"].set_visible(False)
    axis_f.spines["right"].set_visible(False)
    axis_f.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE, pad=6)
    add_panel_label(axis_f, "C")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)

    typer.echo(
        f"Saved {output} with roi={selected_roi}, time={selected_time}, "
        f"spots={len(spot_rows)}, cells={len(rois)}"
    )

    if movies is not None:
        movies = Path(movies)
        for r in rois:
            out_mp4 = movies / f"roi{r:02d}_figB.mp4"
            try:
                generate_b_movie(input_dir, filtered_dir, position, r, out_mp4, channel=channel)
            except Exception as exc:
                typer.echo(f"Warning: failed movie for roi {r}: {exc}")
