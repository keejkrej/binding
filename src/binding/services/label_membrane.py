from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from binding.core.paths import labeled_output_path


@dataclass(frozen=True)
class LabelMembraneResult:
    output_path: Path
    shape: tuple[int, ...]
    dtype: np.dtype
    component_count: int
    kept_volume: int


def solidify_xy_planes(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    solidified = np.zeros(mask.shape, dtype=bool)
    for z in range(mask.shape[0]):
        solidified[z] = ndimage.binary_fill_holes(mask[z])
    return solidified


def label_membrane_mask(
    binary_stack: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, int, int]:
    from scipy import ndimage

    mask = np.asarray(binary_stack, dtype=bool)
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    if iterations:
        opened = ndimage.binary_opening(mask, structure=structure, iterations=iterations)
        closed = ndimage.binary_closing(opened, structure=structure, iterations=iterations)
    else:
        closed = mask

    labels, component_count = ndimage.label(closed, structure=structure)
    if component_count == 0:
        return np.zeros(mask.shape, dtype=np.int32), 0, 0

    volumes = np.bincount(labels.ravel())
    largest_label = int(np.argmax(volumes[1:]) + 1)
    largest = labels == largest_label
    solidified = solidify_xy_planes(largest)
    solidified = ndimage.binary_fill_holes(solidified, structure=structure)
    output = solidified.astype(np.int32)
    return output, int(component_count), int(output.sum())


def run_label_membrane(input_file: Path, *, iterations: int) -> LabelMembraneResult:
    binary_stack = np.load(input_file)
    labels, component_count, volume = label_membrane_mask(binary_stack, iterations)
    output_path = labeled_output_path(input_file)
    np.save(output_path, labels)
    return LabelMembraneResult(
        output_path=output_path,
        shape=labels.shape,
        dtype=labels.dtype,
        component_count=component_count,
        kept_volume=volume,
    )