from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from binding.core.frames import load_stack
from binding.core.paths import analysis_output_path


@dataclass(frozen=True)
class AnalyzeResult:
    output_path: Path
    row_count: int
    source_shape: tuple[int, ...]
    mask_shape: tuple[int, ...]


def write_analysis_csv(
    output_path: Path,
    image_stack: np.ndarray,
    labels: np.ndarray,
) -> int:
    if image_stack.shape != labels.shape:
        raise ValueError(
            f"Image stack shape {image_stack.shape} does not match mask shape {labels.shape}"
        )
    if labels.ndim != 3:
        raise ValueError(f"Expected a 3D labeled mask, got shape {labels.shape}")
    if int(labels.min()) < 0:
        raise ValueError("Labeled mask must not contain negative ids")

    max_label = int(labels.max())
    if max_label == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["id", "volume", "total_intensity", "centroid_x", "centroid_y", "centroid_z"]
            )
        return 0

    depth, height, width = labels.shape
    volume = np.zeros(max_label + 1, dtype=np.int64)
    total_intensity = np.zeros(max_label + 1, dtype=np.float64)
    sum_x = np.zeros(max_label + 1, dtype=np.float64)
    sum_y = np.zeros(max_label + 1, dtype=np.float64)
    sum_z = np.zeros(max_label + 1, dtype=np.float64)

    x_coords = np.tile(np.arange(width, dtype=np.float64), height)
    y_coords = np.repeat(np.arange(height, dtype=np.float64), width)

    for z in range(depth):
        label_values = np.asarray(labels[z]).ravel()
        image_values = np.asarray(image_stack[z]).ravel()
        counts = np.bincount(label_values, minlength=max_label + 1)

        volume += counts
        total_intensity += np.bincount(
            label_values,
            weights=image_values,
            minlength=max_label + 1,
        )
        sum_x += np.bincount(
            label_values,
            weights=x_coords,
            minlength=max_label + 1,
        )
        sum_y += np.bincount(
            label_values,
            weights=y_coords,
            minlength=max_label + 1,
        )
        sum_z += counts * z

    ids = np.flatnonzero(volume[1:]) + 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["id", "volume", "total_intensity", "centroid_x", "centroid_y", "centroid_z"]
        )
        for label_id in ids:
            writer.writerow(
                [
                    int(label_id),
                    int(volume[label_id]),
                    f"{total_intensity[label_id]:.17g}",
                    f"{sum_x[label_id] / volume[label_id]:.17g}",
                    f"{sum_y[label_id] / volume[label_id]:.17g}",
                    f"{sum_z[label_id] / volume[label_id]:.17g}",
                ]
            )

    return int(len(ids))


def run_analyze(
    input_dir: Path,
    *,
    position: int,
    channel: int,
    time: int,
    mask: Path,
    output: Path,
) -> AnalyzeResult:
    image_stack = load_stack(input_dir, position, channel, time)
    labels = np.load(mask, mmap_mode="r")
    output_path = analysis_output_path(output, position, channel, time)
    row_count = write_analysis_csv(output_path, image_stack, labels)
    return AnalyzeResult(
        output_path=output_path,
        row_count=row_count,
        source_shape=image_stack.shape,
        mask_shape=labels.shape,
    )