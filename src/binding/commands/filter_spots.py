from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from binding.app import app
from binding.services.filter_spots import run_filter_spots


@app.command(name="filter-spots")
def filter_spots(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=True,
            file_okay=False,
            help="Directory containing Spotiflow CSV files.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory for filtered CSV files.",
        ),
    ] = Path("filtered"),
    min_intensity: Annotated[
        float | None,
        typer.Option("--min-intensity", help="Minimum spot intensity."),
    ] = 2500.0,
    max_intensity: Annotated[
        float | None,
        typer.Option("--max-intensity", help="Maximum spot intensity."),
    ] = None,
    min_fwhm: Annotated[
        float | None,
        typer.Option("--min-fwhm", help="Minimum spot FWHM."),
    ] = 2.0,
    max_fwhm: Annotated[
        float | None,
        typer.Option("--max-fwhm", help="Maximum spot FWHM."),
    ] = 6.0,
    min_probability: Annotated[
        float | None,
        typer.Option("--min-probability", help="Minimum detection probability."),
    ] = 0.4,
) -> None:
    try:
        result = run_filter_spots(
            input_dir,
            output=output,
            min_intensity=min_intensity,
            max_intensity=max_intensity,
            min_fwhm=min_fwhm,
            max_fwhm=max_fwhm,
            min_probability=min_probability,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        f"Filtered {result.file_count} files into {result.output_dir}: "
        f"spots {result.total_in} -> {result.total_out}"
    )