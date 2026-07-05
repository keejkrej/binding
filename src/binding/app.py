from __future__ import annotations

import typer

app = typer.Typer(
    add_completion=False,
    help="Inspect converted microscope TIFF folders and visualize stacks.",
)


@app.callback()
def cli() -> None:
    """Inspect converted microscope TIFF folders and visualize stacks."""