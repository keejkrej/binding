from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import tifffile
import typer

from binding.core import load_stack


def _spotiflow_command() -> list[str]:
    command = shutil.which("spotiflow-predict")
    if command is not None:
        return [command]
    return [sys.executable, "-m", "spotiflow.cli.predict"]


def spotiflow(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=True,
            file_okay=False,
            help="Directory containing raw microscopy data.",
        ),
    ],
    position: Annotated[
        int,
        typer.Option("--position", "-p", help="Position index to process."),
    ] = 0,
    channel: Annotated[
        int,
        typer.Option("--channel", "-c", help="Channel index to process."),
    ] = 0,
    time: Annotated[
        int,
        typer.Option("--time", "-t", help="Time index to process."),
    ] = 0,
) -> None:
    output_path = input_dir / (
        f"spotiflow_position{position:03d}_channel{channel:03d}_time{time:09d}.csv"
    )

    try:
        stack = load_stack(input_dir, position, channel, time)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        f"Loaded position={position}, channel={channel}, time={time}: "
        f"shape={stack.shape}, dtype={stack.dtype}"
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        stack_path = tmp_dir / output_path.with_suffix(".tif").name
        tifffile.imwrite(stack_path, stack)

        command = _spotiflow_command() + [
            str(stack_path),
            "--pretrained-model",
            "smfish_3d",
            "--out-dir",
            str(tmp_dir),
        ]

        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                "spotiflow command failed; ensure spotiflow is installed and the command/options are valid"
            )

        generated_path = stack_path.with_suffix(".csv")
        if not generated_path.exists():
            raise RuntimeError(
                f"spotiflow completed but did not produce expected CSV {generated_path}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(generated_path), output_path)
    typer.echo(f"Saved spotiflow output to {output_path}")
