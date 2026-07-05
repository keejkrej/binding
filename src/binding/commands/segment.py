from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from binding.app import app
from binding.services.segment import run_segment


@app.command()
def segment(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Folder containing Pos*/img_channel... TIFFs from convert.",
        ),
    ],
    position: Annotated[int, typer.Option("--position", "-p", help="Position index to segment.")] = 0,
    channel: Annotated[int, typer.Option("--channel", "-c", help="Channel index to segment.")] = 0,
    time: Annotated[int, typer.Option("--time", "-t", help="Time index to segment.")] = 0,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory where the labeled .npy output will be written.",
        ),
    ] = Path("."),
    min_z_planes: Annotated[
        int,
        typer.Option(
            "--min-z-planes",
            help="Minimum number of z planes a linked component must occupy. Use 0 to disable filtering.",
        ),
    ] = 0,
) -> None:
    try:
        result = run_segment(
            input_dir,
            position=position,
            channel=channel,
            time=time,
            output=output,
            min_z_planes=min_z_planes,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        f"Saved {result.output_path} with shape={result.shape}, dtype={result.dtype}, "
        f"components={result.component_count}, removed_z_short={result.removed_count}, "
        f"min_z_planes={result.min_z_planes}, threshold_yen_iqr_replaced={result.threshold_replaced_count}, "
        f"threshold_yen_min={result.threshold_yen_min:.17g}, "
        f"threshold_yen_max={result.threshold_yen_max:.17g}"
    )