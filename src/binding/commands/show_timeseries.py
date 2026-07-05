from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from binding.app import app
from binding.services.show_timeseries import run_show_timeseries


@app.command(name="show-timeseries")
def show_timeseries(
    input_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Timeseries CSV produced by binding timeseries.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output PNG path. Defaults to <input>_timeseries.png.",
        ),
    ] = None,
    use_time_real: Annotated[
        bool,
        typer.Option(
            "--use-time-real/--use-time-index",
            help="Use time_real on the x-axis when present in the CSV.",
        ),
    ] = True,
) -> None:
    try:
        result = run_show_timeseries(
            input_file,
            output=output,
            use_time_real=use_time_real,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        f"Saved {result.output_path} with roi_sizes={result.roi_sizes}, "
        f"x_axis={result.x_axis}, rows={result.row_count}"
    )