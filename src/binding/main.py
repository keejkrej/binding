from __future__ import annotations

import typer

from binding.commands.analyze import analyze
from binding.commands.analyze_membrane import analyze_membrane
from binding.commands.binarize import binarize
from binding.commands.label import label
from binding.commands.label_membrane import label_membrane
from binding.commands.spotiflow import spotiflow
from binding.commands.plot import plot
from binding.commands.show import show
from binding.commands.show_labeled import show_labeled
from binding.commands.show_result import show_result

app = typer.Typer(
    add_completion=False,
    help="Inspect converted microscope TIFF folders and visualize stacks.",
)


@app.callback()
def cli() -> None:
    """Inspect converted microscope TIFF folders and visualize stacks."""


app.command()(binarize)
app.command()(analyze)
app.command(name="analyze-membrane")(analyze_membrane)
app.command()(show_labeled)
app.command()(label)
app.command(name="label-membrane")(label_membrane)
app.command()(spotiflow)
app.command()(show)
app.command(name="show-result")(show_result)
app.command()(plot)


def main() -> None:
    app(prog_name="binding")


if __name__ == "__main__":
    main()
