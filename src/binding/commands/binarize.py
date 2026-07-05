from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from binding.app import app
from binding.services.binarize import run_binarize


@app.command()
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
        if threshold is None:
            typer.echo(
                "Adjust the image contrast minimum in napari, then close the napari window "
                "to use that minimum as the binarization threshold."
            )
        result = run_binarize(
            input_dir,
            position=position,
            channel=channel,
            time=time,
            output=output,
            threshold=threshold,
            metadata=metadata,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if result.voxel_scale is not None:
        typer.echo(f"Voxel scale (z, y, x): {result.voxel_scale} um")

    typer.echo(
        f"Saved {result.output_path} with shape={result.shape}, "
        f"dtype={result.dtype}, threshold={result.threshold}"
    )