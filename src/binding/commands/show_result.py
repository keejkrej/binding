from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from binding.app import app
from binding.services.show_result import run_show_result


@app.command(name="show-result")
def show_result(
    input_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Labeled particle .npy stack to show.",
        ),
    ],
    analysis: Annotated[
        Path,
        typer.Option(
            "--analysis",
            exists=True,
            dir_okay=False,
            help="Particle analysis CSV with intensity measurements.",
        ),
    ],
    membrane_result: Annotated[
        Path,
        typer.Option(
            "--membrane-result",
            exists=True,
            dir_okay=False,
            help="Membrane result CSV with distance_to_membrane_um.",
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
    thickness_um: Annotated[
        float,
        typer.Option(
            "--thickness-um",
            min=0.0,
            help="Membrane thickness band in micrometers.",
        ),
    ] = 0.1,
) -> None:
    try:
        result = run_show_result(
            input_file,
            analysis=analysis,
            membrane_result=membrane_result,
            metadata=metadata,
            thickness_um=thickness_um,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        f"Loaded {result.input_file}: shape={result.shape}, dtype={result.dtype}, "
        f"components={result.component_count}, inside={result.inside_count}, "
        f"surface={result.surface_count}, outside={result.outside_count}"
    )
    if result.voxel_scale is not None:
        typer.echo(f"Voxel scale (z, y, x): {result.voxel_scale} um")