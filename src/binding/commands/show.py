from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from binding.core import load_stack, load_voxel_scale


def show(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Folder containing Pos*/img_channel... TIFFs from convert.",
        ),
    ],
    position: Annotated[int, typer.Option("--position", "-p", help="Position index to show.")] = 0,
    channel: Annotated[int, typer.Option("--channel", "-c", help="Channel index to show.")] = 0,
    time: Annotated[int, typer.Option("--time", "-t", help="Time index to show.")] = 0,
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

    typer.echo(
        f"Loaded position={position}, channel={channel}, time={time}: "
        f"shape={stack.shape}, dtype={stack.dtype}"
    )
    if voxel_scale is not None:
        typer.echo(f"Voxel scale (z, y, x): {voxel_scale} um")

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
