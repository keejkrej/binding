from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from binding.app import app
from binding.services.label_membrane import run_label_membrane


@app.command(name="label-membrane")
def label_membrane(
    input_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Binary membrane .npy stack to reduce to one labeled object.",
        ),
    ],
    iterations: Annotated[
        int,
        typer.Option(
            "--iterations",
            min=0,
            help="Opening and closing iterations before choosing the largest component.",
        ),
    ] = 1,
) -> None:
    result = run_label_membrane(input_file, iterations=iterations)

    typer.echo(
        f"Saved {result.output_path} with shape={result.shape}, dtype={result.dtype}, "
        f"components_before_filter={result.component_count}, kept_volume={result.kept_volume}, "
        f"method=opening-closing-largest-solidify-xy"
    )