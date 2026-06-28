from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from binding.core import load_time_map, spot_counts_output_path

SPOT_FILE_RE = re.compile(r"^roi(?P<roi>\d+)_time(?P<time>\d+)\.csv$", re.IGNORECASE)
FILTERED_SPOT_FILE_RE = re.compile(
    r"^roi(?P<roi>\d+)_time(?P<time>\d+)_filtered\.csv$",
    re.IGNORECASE,
)


def parse_spot_file(path: Path) -> tuple[int, int]:
    for pattern in (FILTERED_SPOT_FILE_RE, SPOT_FILE_RE):
        match = pattern.match(path.name)
        if match is not None:
            return int(match.group("roi")), int(match.group("time"))
    raise ValueError(f"Unrecognized spot CSV filename: {path.name}")


def read_spot_coordinates(path: Path) -> np.ndarray:
    coordinates: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "x" not in reader.fieldnames or "y" not in reader.fieldnames:
            raise ValueError(f"Spot CSV {path} must contain x and y columns")
        for row in reader:
            coordinates.append((float(row["x"]), float(row["y"])))
    if not coordinates:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(coordinates, dtype=np.float64)


def count_new_spots(
    previous: np.ndarray,
    current: np.ndarray,
    match_distance: float,
) -> tuple[int, np.ndarray]:
    if current.size == 0:
        return 0, previous
    if previous.size == 0:
        return int(current.shape[0]), current

    updated = [tuple(point) for point in previous]
    new_count = 0
    for point in current:
        distances = np.hypot(previous[:, 0] - point[0], previous[:, 1] - point[1])
        if float(distances.min()) <= match_distance:
            continue
        updated.append((float(point[0]), float(point[1])))
        new_count += 1

    if not updated:
        return 0, np.empty((0, 2), dtype=np.float64)
    return new_count, np.asarray(updated, dtype=np.float64)


def collect_spot_files(input_dir: Path) -> dict[int, dict[int, Path]]:
    grouped: dict[int, dict[int, Path]] = {}
    for path in sorted(input_dir.glob("*.csv")):
        try:
            roi, time_index = parse_spot_file(path)
        except ValueError:
            continue
        grouped.setdefault(roi, {})[time_index] = path
    if not grouped:
        raise ValueError(f"No spot CSV files found in {input_dir}")
    return grouped


def spot_counts(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=True,
            file_okay=False,
            help="Directory containing filtered Spotiflow CSV files.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for the counts CSV.",
        ),
    ] = Path("."),
    position: Annotated[
        int,
        typer.Option("--position", "-p", help="Position index for output naming."),
    ] = 0,
    channel: Annotated[
        int,
        typer.Option("--channel", "-c", help="Channel index for output naming."),
    ] = 1,
    time_interval: Annotated[
        float,
        typer.Option("--time-interval", help="Seconds between frames when no time map is given."),
    ] = 40.0,
    time_map: Annotated[
        Path | None,
        typer.Option(
            "--time-map",
            exists=True,
            dir_okay=False,
            help="Optional CSV with t and t_real columns.",
        ),
    ] = None,
    cumulative: Annotated[
        bool,
        typer.Option(
            "--cumulative/--per-frame",
            help="Report cumulative unique spot counts over time.",
        ),
    ] = True,
    match_distance: Annotated[
        float,
        typer.Option("--match-distance", help="Pixel distance for matching spots across frames."),
    ] = 5.0,
) -> None:
    try:
        grouped = collect_spot_files(input_dir)
        all_times = sorted({time_index for by_time in grouped.values() for time_index in by_time})
        if time_map is not None:
            time_real_by_index = load_time_map(time_map)
        else:
            time_real_by_index = {
                time_index: time_index * time_interval for time_index in all_times
            }
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    rows: list[dict[str, object]] = []
    for roi in sorted(grouped):
        known = np.empty((0, 2), dtype=np.float64)
        cumulative_count = 0
        for time_index in all_times:
            path = grouped[roi].get(time_index)
            current = read_spot_coordinates(path) if path is not None else np.empty((0, 2), dtype=np.float64)
            frame_count = int(current.shape[0])

            if cumulative:
                new_count, known = count_new_spots(known, current, match_distance)
                cumulative_count += new_count
                count_value = cumulative_count
            else:
                count_value = frame_count

            if time_index not in time_real_by_index:
                raise typer.BadParameter(f"time map missing entry for time={time_index}")

            rows.append(
                {
                    "roi": roi,
                    "time": time_index,
                    "time_real": time_real_by_index[time_index],
                    "spot_count": count_value,
                    "frame_spots": frame_count,
                }
            )

    output_path = spot_counts_output_path(output, position, channel)
    output.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["roi", "time", "time_real", "spot_count", "frame_spots"],
        )
        writer.writeheader()
        writer.writerows(rows)

    typer.echo(
        f"Saved {output_path} with rows={len(rows)}, rois={len(grouped)}, "
        f"times={len(all_times)}, cumulative={cumulative}"
    )