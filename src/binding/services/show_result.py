from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from binding.core.frames import load_voxel_scale


@dataclass(frozen=True)
class ShowResultView:
    input_file: Path
    shape: tuple[int, ...]
    dtype: np.dtype
    component_count: int
    inside_count: int
    surface_count: int
    outside_count: int
    voxel_scale: tuple[float, float, float] | None
    metadata: Path | None


def read_mean_intensities(input_file: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    with open(input_file, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "id" not in reader.fieldnames:
            raise ValueError("Analysis CSV must contain an id column")

        has_mean = "mean_intensity" in reader.fieldnames
        has_total_and_volume = {"total_intensity", "volume"}.issubset(reader.fieldnames)
        if not has_mean and not has_total_and_volume:
            raise ValueError(
                "Analysis CSV must contain mean_intensity or total_intensity and volume"
            )

        for row in reader:
            particle_id = int(row["id"])
            if has_mean:
                values[particle_id] = float(row["mean_intensity"])
            else:
                volume = float(row["volume"])
                values[particle_id] = float(row["total_intensity"]) / volume

    if not values:
        raise ValueError("Analysis CSV has no measurement rows")
    return values


def read_membrane_distances(input_file: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    with open(input_file, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"id", "distance_to_membrane_um"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Membrane result CSV must contain id and distance_to_membrane_um columns"
            )

        for row in reader:
            values[int(row["id"])] = float(row["distance_to_membrane_um"])

    if not values:
        raise ValueError("Membrane result CSV has no rows")
    return values


def normalized_brightness(values: dict[int, float]) -> dict[int, float]:
    min_value = min(values.values())
    max_value = max(values.values())
    if max_value == min_value:
        return {particle_id: 1.0 for particle_id in values}

    return {
        particle_id: 0.25 + 0.75 * ((value - min_value) / (max_value - min_value))
        for particle_id, value in values.items()
    }


def result_colormap(
    max_label: int,
    mean_intensities: dict[int, float],
    membrane_distances: dict[int, float],
    thickness_um: float,
):
    from napari.utils.colormaps import DirectLabelColormap

    brightness = normalized_brightness(mean_intensities)
    colors = defaultdict(lambda: np.array((0.0, 0.0, 0.0, 0.0)))
    colors[0] = np.array((0.0, 0.0, 0.0, 0.0))

    for label_id in range(1, max_label + 1):
        if label_id not in brightness or label_id not in membrane_distances:
            continue

        value = brightness[label_id]
        distance = membrane_distances[label_id]
        if distance > thickness_um:
            colors[label_id] = np.array((value, 0.12 * value, 0.08 * value, 1.0))
        elif distance < -thickness_um:
            colors[label_id] = np.array((0.12 * value, value, 0.18 * value, 1.0))
        else:
            colors[label_id] = np.array((0.08 * value, 0.32 * value, value, 1.0))

    return DirectLabelColormap(color_dict=colors, name="membrane-result")


def run_show_result(
    input_file: Path,
    *,
    analysis: Path,
    membrane_result: Path,
    metadata: Path | None,
    thickness_um: float,
) -> ShowResultView:
    labels = np.load(input_file, mmap_mode="r")
    voxel_scale = load_voxel_scale(metadata) if metadata is not None else None
    mean_intensities = read_mean_intensities(analysis)
    membrane_distances = read_membrane_distances(membrane_result)

    max_label = int(labels.max())
    surface_count = sum(abs(distance) <= thickness_um for distance in membrane_distances.values())
    inside_count = sum(distance < -thickness_um for distance in membrane_distances.values())
    outside_count = sum(distance > thickness_um for distance in membrane_distances.values())

    import napari

    viewer = napari.Viewer()
    viewer.add_labels(
        labels,
        name=input_file.stem,
        colormap=result_colormap(
            max_label,
            mean_intensities,
            membrane_distances,
            thickness_um,
        ),
        metadata={
            "source": str(input_file),
            "analysis": str(analysis),
            "membrane_result": str(membrane_result),
            "metadata": str(metadata) if metadata is not None else None,
            "voxel_size_um_zyx": voxel_scale,
        },
        scale=voxel_scale,
        units=("um", "um", "um") if voxel_scale is not None else None,
    )
    viewer.dims.ndisplay = 3
    napari.run()

    return ShowResultView(
        input_file=input_file,
        shape=labels.shape,
        dtype=labels.dtype,
        component_count=max_label,
        inside_count=inside_count,
        surface_count=surface_count,
        outside_count=outside_count,
        voxel_scale=voxel_scale,
        metadata=metadata,
    )