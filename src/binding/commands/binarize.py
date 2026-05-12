from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from binding.core import binary_output_path, load_stack, load_voxel_scale


def choose_threshold_in_napari(
    stack: np.ndarray,
    name: str,
    voxel_scale: tuple[float, float, float] | None,
) -> float:
    import napari

    typer.echo(
        "Adjust the image contrast minimum in napari, then close the napari window "
        "to use that minimum as the binarization threshold."
    )
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


def binarize(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Folder containing Pos*/img_channel... TIFFs from convert.",
        ),
    ],
    position: Annotated[int, typer.Option("--position", "-p", help="Position index to binarize.")] = 0,
    channel: Annotated[int, typer.Option("--channel", "-c", help="Channel index to binarize.")] = 0,
    time: Annotated[int, typer.Option("--time", "-t", help="Time index to binarize.")] = 0,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory where the encoded .npy output will be written.",
        ),
    ] = Path("."),
    threshold: Annotated[
        float | None,
        typer.Option(
            "--threshold",
            help="Intensity threshold. Defaults to napari contrast minimum after close.",
        ),
    ] = None,
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
    try:
        stack = load_stack(input_dir, position, channel, time)
        voxel_scale = load_voxel_scale(metadata) if metadata is not None else None
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if voxel_scale is not None:
        typer.echo(f"Voxel scale (z, y, x): {voxel_scale} um")

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

    typer.echo(
        f"Saved {output_path} with shape={binary_stack.shape}, "
        f"dtype={binary_stack.dtype}, threshold={resolved_threshold}"
    )
