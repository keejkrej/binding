from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from binding.core.paths import spot_counts_output_path
from binding.core.roi import load_time_map

SPOT_FILE_RE = re.compile(r"^roi(?P<roi>\d+)_time(?P<time>\d+)\.csv$", re.IGNORECASE)
FILTERED_SPOT_FILE_RE = re.compile(
    r"^roi(?P<roi>\d+)_time(?P<time>\d+)_filtered\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SpotCountsResult:
    output_path: Path
    row_count: int
    roi_count: int
    time_count: int
    cumulative: bool


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


def run_spot_counts(
    input_dir: Path,
    *,
    output: Path,
    position: int,
    channel: int,
    time_interval: float,
    time_map: Path | None,
    cumulative: bool,
    match_distance: float,
) -> SpotCountsResult:
    grouped = collect_spot_files(input_dir)
    all_times = sorted({time_index for by_time in grouped.values() for time_index in by_time})
    if time_map is not None:
        time_real_by_index = load_time_map(time_map)
    else:
        time_real_by_index = {
            time_index: time_index * time_interval for time_index in all_times
        }

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
                raise ValueError(f"time map missing entry for time={time_index}")

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

    return SpotCountsResult(
        output_path=output_path,
        row_count=len(rows),
        roi_count=len(grouped),
        time_count=len(all_times),
        cumulative=cumulative,
    )