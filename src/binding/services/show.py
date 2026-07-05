from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from binding.core.frames import load_stack, load_voxel_scale


@dataclass(frozen=True)
class ShowResult:
    position: int
    channel: int
    time: int
    shape: tuple[int, ...]
    dtype: np.dtype
    voxel_scale: tuple[float, float, float] | None
    input_dir: Path
    metadata: Path | None


def run_show(
    input_dir: Path,
    *,
    position: int,
    channel: int,
    time: int,
    metadata: Path | None,
) -> ShowResult:
    stack = load_stack(input_dir, position, channel, time)
    voxel_scale = load_voxel_scale(metadata) if metadata is not None else None

    import napari

    viewer = napari.Viewer()
    viewer.add_image(
        stack,
        name=f"Pos{position} C{channel} T{time}",
        metadata={
            "position": position,
            "channel": channel,
            "time": time,
            "source": str(input_dir),
            "axis_order": "ZYX",
            "metadata": str(metadata) if metadata is not None else None,
            "voxel_size_um_zyx": voxel_scale,
        },
        scale=voxel_scale,
        units=("um", "um", "um") if voxel_scale is not None else None,
    )
    viewer.dims.ndisplay = 3
    napari.run()

    return ShowResult(
        position=position,
        channel=channel,
        time=time,
        shape=stack.shape,
        dtype=stack.dtype,
        voxel_scale=voxel_scale,
        input_dir=input_dir,
        metadata=metadata,
    )