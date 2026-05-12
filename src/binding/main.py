from __future__ import annotations

import typer

from binding.commands.analyze import analyze
from binding.commands.binarize import binarize
from binding.commands.label import label
from binding.commands.show import show
from binding.commands.show_labeled import show_labeled

app = typer.Typer(
    add_completion=False,
    help="Inspect converted microscope TIFF folders and visualize stacks.",
)


@app.callback()
def cli() -> None:
    """Inspect converted microscope TIFF folders and visualize stacks."""


app.command()(binarize)
app.command()(analyze)
app.command()(show_labeled)
app.command()(label)
app.command()(show)


def main() -> None:
    app(prog_name="binding")


if __name__ == "__main__":
    main()
