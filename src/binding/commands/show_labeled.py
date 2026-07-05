from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from binding.app import app
from binding.services.show_labeled import run_show_labeled


@app.command()
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
    result = run_show_labeled(input_file, metadata=metadata)

    typer.echo(
        f"Loaded {result.input_file}: shape={result.shape}, dtype={result.dtype}, "
        f"components={result.component_count}"
    )
    if result.voxel_scale is not None:
        typer.echo(f"Voxel scale (z, y, x): {result.voxel_scale} um")