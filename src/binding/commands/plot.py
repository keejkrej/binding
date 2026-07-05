from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from binding.app import app
from binding.services.plot import run_plot


@app.command()
def plot(
    input_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Analysis CSV with volume and total_intensity columns.",
        ),
    ],
    bins: Annotated[
        int,
        typer.Option(
            "--bins",
            min=1,
            help="Number of histogram bins.",
        ),
    ] = 50,
    topk: Annotated[
        int | None,
        typer.Option(
            "--topk",
            min=1,
            help="Only plot the top K objects by volume.",
        ),
    ] = None,
) -> None:
    try:
        result = run_plot(input_file, bins=bins, topk=topk)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        f"Saved {result.output_path} with rows={result.row_count}, bins={result.bins}, topk={result.topk}"
    )