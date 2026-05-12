from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from binding.core import labeled_output_path


def _read_dict_point(item: dict[str, object], keys: tuple[str, str, str]) -> tuple[int, int, int] | None:
    x_key, y_key, z_key = keys
    try:
        return (
            int(round(float(item[x_key]))),
            int(round(float(item[y_key]))),
            int(round(float(item[z_key]))),
        )
    except (TypeError, ValueError, KeyError):
        return None


def _parse_coordinate_row(
    values: tuple[str, ...],
    header: tuple[str, ...],
) -> tuple[int, int, int] | None:
    columns = {name: index for index, name in enumerate(header)}
    if all(key in columns for key in ("x", "y", "z")):
        idx = (columns["x"], columns["y"], columns["z"])
        return _row_to_zyx(values, idx)

    if all(key in columns for key in ("z", "y", "x")):
        idx = (columns["z"], columns["y"], columns["x"])
        return _row_to_zyx(values, idx)

    if len(values) >= 3:
        return _row_to_zyx(values, (0, 1, 2))
    return None


def _row_to_zyx(row: tuple[str, ...], index: tuple[int, int, int]) -> tuple[int, int, int] | None:
    try:
        x = int(round(float(row[index[0]])))
        y = int(round(float(row[index[1]])))
        z = int(round(float(row[index[2]])))
    except (TypeError, ValueError, IndexError):
        return None
    return z, y, x


def _normalize_seed_points(
    rows: Iterable[tuple[str, ...]],
    header: tuple[str, ...],
) -> list[tuple[int, int, int]]:
    points: list[tuple[int, int, int]] = []
    for row in rows:
        point = _parse_coordinate_row(row, header)
        if point is None:
            continue
        points.append(point)
    return points


def read_seed_points(seed_file: Path) -> list[tuple[int, int, int]]:
    suffix = seed_file.suffix.lower()
    if suffix == ".npy":
        coords = np.load(seed_file)
        if coords.ndim != 2 or coords.shape[1] < 3:
            raise ValueError("Seed .npy must be a 2D array with at least 3 columns")
        coords = np.asarray(coords[:, :3], dtype=float)
        return [(int(round(x)), int(round(y)), int(round(z))) for z, y, x in coords]

    if suffix in {".json", ".js", ".geojson"}:
        with seed_file.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            points: list[tuple[int, int, int]] = []
            for item in payload:
                point = _read_dict_point(item, ("x", "y", "z"))
                if point is None:
                    point = _read_dict_point(item, ("z", "y", "x"))
                if point is None:
                    continue
                z, y, x = point
                points.append((int(round(z)), int(round(y)), int(round(x))))
            return points
        if isinstance(payload, list) and payload and isinstance(payload[0], (list, tuple)):
            rows: list[tuple[str, ...]] = [tuple(str(v) for v in point[:3]) for point in payload if len(point) >= 3]  # type: ignore[arg-type]
            return [
                (
                    int(round(float(z))),
                    int(round(float(y))),
                    int(round(float(x))),
                )
                for z, y, x in rows
            ]
        if isinstance(payload, dict) and "spots" in payload and isinstance(payload["spots"], list):
            points: list[tuple[int, int, int]] = []
            for item in payload["spots"]:
                if not isinstance(item, dict):
                    continue
                point = _read_dict_point(item, ("x", "y", "z"))
                if point is None:
                    point = _read_dict_point(item, ("z", "y", "x"))
                if point is None:
                    continue
                z, y, x = point
                points.append((z, y, x))
            return points
        raise ValueError("Unsupported JSON seed format")

    delimiter = "\t" if suffix == ".tsv" else ","
    with seed_file.open("r", encoding="utf-8") as fh:
        rows_raw = list(csv.reader(fh, delimiter=delimiter))
    if not rows_raw:
        return []

    first_row = tuple(value.strip().lower() for value in rows_raw[0])
    has_header = any(cell in {"x", "y", "z", "z", "y", "x", "id"} for cell in first_row)
    if has_header:
        header = first_row
        data_rows = [tuple(value.strip() for value in row) for row in rows_raw[1:] if row]
    else:
        header = ("x", "y", "z")
        data_rows = [tuple(value.strip() for value in row) for row in rows_raw if row]

    return _normalize_seed_points(data_rows, header)


def seed_markers_from_coordinates(
    shape: tuple[int, int, int],
    points: list[tuple[int, int, int]],
) -> np.ndarray:
    markers = np.zeros(shape, dtype=np.int32)
    marker_id = 1
    for z, y, x in points:
        if not (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]):
            continue
        markers[z, y, x] = marker_id
        marker_id += 1
    return markers


def watershed_components(
    binary_stack: np.ndarray,
    seed_file: Path,
) -> tuple[np.ndarray, int]:
    from scipy import ndimage
    from skimage.segmentation import watershed

    mask = np.asarray(binary_stack, dtype=bool)
    seed_points = read_seed_points(seed_file)
    if not seed_points:
        raise ValueError(f"Seed file {seed_file} did not contain any valid point coordinates")

    markers = seed_markers_from_coordinates(mask.shape, seed_points)
    if int(markers.max()) == 0:
        raise ValueError(f"Seed file {seed_file} did not provide any in-bounds points for mask shape {mask.shape}")

    distance = ndimage.distance_transform_edt(mask)
    labels = watershed(-distance, markers, mask=mask)
    return labels.astype(np.int32, copy=False), int(labels.max())


def label(
    input_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Binary .npy stack to label.",
        ),
    ],
    seeds: Annotated[
        Path,
        typer.Option(
            "--seeds",
            exists=True,
            dir_okay=False,
            help="Seed coordinate file (CSV/TXT/JSON/NPY) generated by spotiflow.",
        ),
    ],
) -> None:
    binary_stack = np.load(input_file)
    labels, count = watershed_components(
        binary_stack,
        seed_file=seeds,
    )
    output_path = labeled_output_path(input_file)
    np.save(output_path, labels)

    typer.echo(
        f"Saved {output_path} with shape={labels.shape}, "
        f"dtype={labels.dtype}, components={count}, "
        f"method=seeded-watershed, seed_file={seeds}"
    )
