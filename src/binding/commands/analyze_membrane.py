from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from binding.app import app
from binding.services.analyze_membrane import run_analyze_membrane


@app.command(name="analyze-membrane")
def analyze_membrane(
    input_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Particle analysis CSV with id and centroid columns.",
        ),
    ],
    membrane_mask: Annotated[
        Path,
        typer.Option(
            "--membrane-mask",
            exists=True,
            dir_okay=False,
            help="Solidified membrane labeled/binary .npy mask.",
        ),
    ],
    metadata: Annotated[
        Path,
        typer.Option(
            "--metadata",
            exists=True,
            dir_okay=False,
            help="metadata.json containing normalized.pixel_size_um x/y/z values.",
        ),
    ],
) -> None:
    try:
        result = run_analyze_membrane(
            input_file,
            membrane_mask=membrane_mask,
            metadata=metadata,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        f"Saved {result.output_path} with rows={result.row_count}, "
        f"distance_unit=um"
    )