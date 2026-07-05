from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from binding.app import app
from binding.services.spot_counts import run_spot_counts


@app.command(name="spot-counts")
def spot_counts(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=True,
            file_okay=False,
            help="Directory containing filtered Spotiflow CSV files.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for the counts CSV.",
        ),
    ] = Path("."),
    position: Annotated[
        int,
        typer.Option("--position", "-p", help="Position index for output naming."),
    ] = 0,
    channel: Annotated[
        int,
        typer.Option("--channel", "-c", help="Channel index for output naming."),
    ] = 1,
    time_interval: Annotated[
        float,
        typer.Option("--time-interval", help="Seconds between frames when no time map is given."),
    ] = 40.0,
    time_map: Annotated[
        Path | None,
        typer.Option(
            "--time-map",
            exists=True,
            dir_okay=False,
            help="Optional CSV with t and t_real columns.",
        ),
    ] = None,
    cumulative: Annotated[
        bool,
        typer.Option(
            "--cumulative/--per-frame",
            help="Report cumulative unique spot counts over time.",
        ),
    ] = True,
    match_distance: Annotated[
        float,
        typer.Option("--match-distance", help="Pixel distance for matching spots across frames."),
    ] = 5.0,
) -> None:
    try:
        result = run_spot_counts(
            input_dir,
            output=output,
            position=position,
            channel=channel,
            time_interval=time_interval,
            time_map=time_map,
            cumulative=cumulative,
            match_distance=match_distance,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        f"Saved {result.output_path} with rows={result.row_count}, rois={result.roi_count}, "
        f"times={result.time_count}, cumulative={result.cumulative}"
    )