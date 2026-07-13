"""Link filtered Spotiflow detections across time and flag LNP clustering (merge) events."""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

FILTERED_SPOT_FILE_RE = re.compile(
    r"^roi(?P<roi>\d+)_time(?P<time>\d+)_filtered\.csv$",
    re.IGNORECASE,
)

LARGE_COST = 1e6


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    intensity: float


@dataclass
class Track:
    track_id: int
    x: float
    y: float
    intensity: float
    birth_time: int
    last_time: int
    alive: bool = True


def read_detections(path: Path) -> list[Detection]:
    rows: list[Detection] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return rows
        for row in reader:
            intensity = float(row.get("intensity", 0.0) or 0.0)
            rows.append(Detection(float(row["x"]), float(row["y"]), intensity))
    return rows


def collect_files(input_dir: Path) -> dict[int, dict[int, Path]]:
    grouped: dict[int, dict[int, Path]] = {}
    for path in sorted(input_dir.glob("*_filtered.csv")):
        match = FILTERED_SPOT_FILE_RE.match(path.name)
        if match is None:
            continue
        roi = int(match.group("roi"))
        time_index = int(match.group("time"))
        grouped.setdefault(roi, {})[time_index] = path
    if not grouped:
        raise ValueError(f"No filtered spot CSVs in {input_dir}")
    return grouped


def match_tracks(
    tracks: list[Track],
    detections: list[Detection],
    *,
    max_distance: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    active = [index for index, track in enumerate(tracks) if track.alive]
    if not active or not detections:
        return [], active, list(range(len(detections)))

    cost = np.full((len(active), len(detections)), LARGE_COST, dtype=np.float64)
    for row, track_index in enumerate(active):
        track = tracks[track_index]
        for col, det in enumerate(detections):
            dist = np.hypot(track.x - det.x, track.y - det.y)
            if dist <= max_distance:
                cost[row, col] = dist

    row_ind, col_ind = linear_sum_assignment(cost)
    pairs: list[tuple[int, int]] = []
    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    for row, col in zip(row_ind, col_ind):
        if cost[row, col] >= LARGE_COST:
            continue
        track_index = active[row]
        pairs.append((track_index, col))
        used_tracks.add(track_index)
        used_dets.add(col)

    unmatched_tracks = [index for index in active if index not in used_tracks]
    unmatched_dets = [index for index in range(len(detections)) if index not in used_dets]
    return pairs, unmatched_tracks, unmatched_dets


def find_merge_events(
    previous: list[Detection],
    current: list[Detection],
    *,
    merge_distance: float,
    min_sources: int = 2,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    if not previous or not current:
        return events

    prev_xy = np.array([(det.x, det.y) for det in previous], dtype=np.float64)
    for target_index, det in enumerate(current):
        distances = np.hypot(prev_xy[:, 0] - det.x, prev_xy[:, 1] - det.y)
        source_indices = np.flatnonzero(distances <= merge_distance).tolist()
        if len(source_indices) < min_sources:
            continue
        source_intensity = float(sum(previous[i].intensity for i in source_indices))
        events.append(
            {
                "target_index": target_index,
                "source_indices": source_indices,
                "source_count": len(source_indices),
                "x": det.x,
                "y": det.y,
                "intensity": det.intensity,
                "source_intensity_sum": source_intensity,
                "intensity_gain": det.intensity - source_intensity,
            }
        )
    return events


def track_roi(
    files_by_time: dict[int, Path],
    *,
    time_interval: float,
    max_distance: float,
    merge_distance: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    times = sorted(files_by_time)
    tracks: list[Track] = []
    next_id = 0
    track_rows: list[dict[str, object]] = []
    merge_rows: list[dict[str, object]] = []

    previous_detections: list[Detection] = []
    previous_time = times[0]

    for time_index in times:
        detections = read_detections(files_by_time[time_index])
        pairs, unmatched_tracks, unmatched_dets = match_tracks(
            tracks,
            detections,
            max_distance=max_distance,
        )

        for track_index, det_index in pairs:
            track = tracks[track_index]
            det = detections[det_index]
            track.x = det.x
            track.y = det.y
            track.intensity = det.intensity
            track.last_time = time_index
            track_rows.append(
                {
                    "track_id": track.track_id,
                    "time": time_index,
                    "time_real": time_index * time_interval,
                    "x": det.x,
                    "y": det.y,
                    "intensity": det.intensity,
                    "event": "continue",
                }
            )

        for track_index in unmatched_tracks:
            tracks[track_index].alive = False
            track_rows.append(
                {
                    "track_id": tracks[track_index].track_id,
                    "time": time_index,
                    "time_real": time_index * time_interval,
                    "x": tracks[track_index].x,
                    "y": tracks[track_index].y,
                    "intensity": tracks[track_index].intensity,
                    "event": "lost",
                }
            )

        for det_index in unmatched_dets:
            det = detections[det_index]
            tracks.append(
                Track(
                    track_id=next_id,
                    x=det.x,
                    y=det.y,
                    intensity=det.intensity,
                    birth_time=time_index,
                    last_time=time_index,
                )
            )
            track_rows.append(
                {
                    "track_id": next_id,
                    "time": time_index,
                    "time_real": time_index * time_interval,
                    "x": det.x,
                    "y": det.y,
                    "intensity": det.intensity,
                    "event": "birth",
                }
            )
            next_id += 1

        if previous_detections:
            for event in find_merge_events(
                previous_detections,
                detections,
                merge_distance=merge_distance,
            ):
                merge_rows.append(
                    {
                        "time": time_index,
                        "time_real": time_index * time_interval,
                        "prev_time": previous_time,
                        "prev_time_real": previous_time * time_interval,
                        **event,
                    }
                )

        previous_detections = detections
        previous_time = time_index

    return track_rows, merge_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filtered_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--time-interval", type=float, default=4.0)
    parser.add_argument("--max-distance", type=float, default=5.0)
    parser.add_argument("--merge-distance", type=float, default=8.0)
    parser.add_argument(
        "--min-intensity-gain",
        type=float,
        default=500.0,
        help="Minimum intensity increase vs. sum of source spots to keep a merge event.",
    )
    parser.add_argument("--roi", type=int, default=None)
    args = parser.parse_args()

    grouped = collect_files(args.filtered_dir)
    all_track_rows: list[dict[str, object]] = []
    all_merge_rows: list[dict[str, object]] = []
    all_merge_strict: list[dict[str, object]] = []

    rois = [args.roi] if args.roi is not None else sorted(grouped)
    for roi in rois:
        track_rows, merge_rows = track_roi(
            grouped[roi],
            time_interval=args.time_interval,
            max_distance=args.max_distance,
            merge_distance=args.merge_distance,
        )
        for row in track_rows:
            row["roi"] = roi
        for row in merge_rows:
            row["roi"] = roi
        all_track_rows.extend(track_rows)
        all_merge_rows.extend(merge_rows)
        strict_count = 0
        for row in merge_rows:
            if (
                int(row["source_count"]) >= 2
                and float(row["intensity_gain"]) >= args.min_intensity_gain
            ):
                all_merge_strict.append(row)
                strict_count += 1
        print(
            f"roi {roi:02d}: tracks={len({r['track_id'] for r in track_rows})}, "
            f"merge_candidates={len(merge_rows)}, strict_merges={strict_count}"
        )

    write_csv(
        args.output / "spot_tracks.csv",
        all_track_rows,
        ["roi", "track_id", "time", "time_real", "x", "y", "intensity", "event"],
    )
    write_csv(
        args.output / "cluster_merge_events.csv",
        all_merge_rows,
        [
            "roi",
            "time",
            "time_real",
            "prev_time",
            "prev_time_real",
            "target_index",
            "source_count",
            "source_indices",
            "x",
            "y",
            "intensity",
            "source_intensity_sum",
            "intensity_gain",
        ],
    )
    write_csv(
        args.output / "cluster_merge_events_strict.csv",
        all_merge_strict,
        [
            "roi",
            "time",
            "time_real",
            "prev_time",
            "prev_time_real",
            "target_index",
            "source_count",
            "source_indices",
            "x",
            "y",
            "intensity",
            "source_intensity_sum",
            "intensity_gain",
        ],
    )
    print(
        f"Wrote {args.output / 'spot_tracks.csv'}, cluster_merge_events.csv, "
        f"cluster_merge_events_strict.csv ({len(all_merge_strict)} strict merges)"
    )


if __name__ == "__main__":
    main()
