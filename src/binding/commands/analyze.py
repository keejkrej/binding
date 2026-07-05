from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from binding.app import app
from binding.services.analyze import run_analyze


@app.command()
def analyze(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Folder containing Pos*/img_channel... TIFFs from convert.",
        ),
    ],
    position: Annotated[int, typer.Option("--position", "-p", help="Position index to analyze.")] = 0,
    channel: Annotated[int, typer.Option("--channel", "-c", help="Channel index to analyze.")] = 0,
    time: Annotated[int, typer.Option("--time", "-t", help="Time index to analyze.")] = 0,
    mask: Annotated[
        Path,
        typer.Option(
            "--mask",
            exists=True,
            dir_okay=False,
            help="Labeled .npy mask to measure.",
        ),
    ] = ...,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory where the analysis CSV will be written.",
        ),
    ] = Path("."),
) -> None:
    try:
        result = run_analyze(
            input_dir,
            position=position,
            channel=channel,
            time=time,
            mask=mask,
            output=output,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        f"Saved {result.output_path} with rows={result.row_count}, "
        f"source_shape={result.source_shape}, mask_shape={result.mask_shape}"
    )