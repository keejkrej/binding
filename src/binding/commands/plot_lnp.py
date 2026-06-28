from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from matplotlib import patches


from binding.commands.filter_spots import read_spot_csv
from binding.core import load_roi_stack

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


def render_detections(axis, image_shape: tuple[int, int], rows: list[dict[str, str]]) -> None:
    background = np.zeros(image_shape, dtype=np.float32)
    axis.imshow(background, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axis.set_xlim(-0.5, image_shape[1] - 0.5)
    axis.set_ylim(image_shape[0] - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.axis("off")

    for row in rows:
        x = float(row["x"])
        y = float(row["y"])
        radius = 4.0
        if "fwhm" in row and row["fwhm"] != "":
            radius = max(3.0, float(row["fwhm"]) * 1.5)
        axis.add_patch(
            patches.Circle(
                (x, y),
                radius=radius,
                fill=False,
                edgecolor="#f4f4f4",
                linewidth=1.4,
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
) -> None:
    try:
        rois, grouped = read_counts_csv(counts_csv)
        selected_roi = roi if roi is not None else auto_select_roi(grouped)
        selected_time = time if time is not None else auto_select_time(grouped, selected_roi)
        image = load_roi_stack(input_dir, position, selected_roi, channel)[selected_time]
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
    render_detections(axis_e, image.shape, spot_rows)
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