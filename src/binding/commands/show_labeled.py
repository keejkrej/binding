from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from binding.core import load_voxel_scale

LABEL_PALETTE: tuple[tuple[float, float, float, float], ...] = (
    (0.93, 0.18, 0.29, 1.0),
    (0.10, 0.58, 0.95, 1.0),
    (0.22, 0.74, 0.36, 1.0),
    (0.97, 0.68, 0.13, 1.0),
    (0.58, 0.35, 0.86, 1.0),
    (0.00, 0.73, 0.78, 1.0),
    (0.94, 0.39, 0.16, 1.0),
    (0.89, 0.33, 0.72, 1.0),
)


def alternating_label_colormap(max_label: int):
    from napari.utils.colormaps import DirectLabelColormap

    colors = defaultdict(lambda: np.array((1.0, 1.0, 1.0, 1.0)))
    colors[0] = np.array((0.0, 0.0, 0.0, 0.0))
    for label_index in range(1, max_label + 1):
        colors[label_index] = np.array(
            LABEL_PALETTE[(label_index - 1) % len(LABEL_PALETTE)]
        )

    return DirectLabelColormap(color_dict=colors, name="alternating-labels")


def show_labeled(
    input_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Labeled .npy stack to show.",
        ),
    ],
    metadata: Annotated[
        Path | None,
        typer.Option(
            "--metadata",
            exists=True,
            dir_okay=False,
            help="metadata.json containing normalized.pixel_size_um x/y/z values.",
        ),
    ] = None,
) -> None:
    labels = np.load(input_file, mmap_mode="r")
    voxel_scale = load_voxel_scale(metadata) if metadata is not None else None
    max_label = int(labels.max())

    typer.echo(
        f"Loaded {input_file}: shape={labels.shape}, dtype={labels.dtype}, "
        f"components={max_label}"
    )
    if voxel_scale is not None:
        typer.echo(f"Voxel scale (z, y, x): {voxel_scale} um")

    import napari

    viewer = napari.Viewer()
    viewer.add_labels(
        labels,
        name=input_file.stem,
        colormap=alternating_label_colormap(max_label),
        metadata={
            "source": str(input_file),
            "metadata": str(metadata) if metadata is not None else None,
            "voxel_size_um_zyx": voxel_scale,
        },
        scale=voxel_scale,
        units=("um", "um", "um") if voxel_scale is not None else None,
    )
    viewer.dims.ndisplay = 3
    napari.run()
