from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from binding.core.frames import load_stack, load_voxel_scale
from binding.core.paths import binary_output_path


@dataclass(frozen=True)
class BinarizeResult:
    output_path: Path
    shape: tuple[int, ...]
    dtype: np.dtype
    threshold: float
    voxel_scale: tuple[float, float, float] | None


def choose_threshold_in_napari(
    stack: np.ndarray,
    name: str,
    voxel_scale: tuple[float, float, float] | None,
) -> float:
    import napari

    viewer = napari.Viewer()
    layer = viewer.add_image(
        stack,
        name=name,
        scale=voxel_scale,
        units=("um", "um", "um") if voxel_scale is not None else None,
    )
    viewer.dims.ndisplay = 3
    napari.run()
    return float(layer.contrast_limits[0])


def run_binarize(
    input_dir: Path,
    *,
    position: int,
    channel: int,
    time: int,
    output: Path,
    threshold: float | None,
    metadata: Path | None,
) -> BinarizeResult:
    stack = load_stack(input_dir, position, channel, time)
    voxel_scale = load_voxel_scale(metadata) if metadata is not None else None

    layer_name = f"Pos{position} C{channel} T{time}"
    resolved_threshold = (
        choose_threshold_in_napari(stack, layer_name, voxel_scale)
        if threshold is None
        else threshold
    )
    binary_stack = stack > resolved_threshold

    output.mkdir(parents=True, exist_ok=True)
    output_path = binary_output_path(output, position, channel, time)
    np.save(output_path, binary_stack)

    return BinarizeResult(
        output_path=output_path,
        shape=binary_stack.shape,
        dtype=binary_stack.dtype,
        threshold=resolved_threshold,
        voxel_scale=voxel_scale,
    )