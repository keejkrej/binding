from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import numpy as np
import tifffile
import typer
from tqdm import tqdm

from binding.core import (
    available_times,
    list_rois,
    load_roi_stack,
    load_stack,
    spotiflow_roi_output_path,
)
from binding.spotiflow_utils import load_spotiflow_model, predict_spots, write_spot_csv


def _spotiflow_command() -> list[str]:
    command = shutil.which("spotiflow-predict")
    if command is not None:
        return [command]
    return [sys.executable, "-m", "spotiflow.cli.predict"]


def _predict_stack_with_cli(
    stack: np.ndarray,
    output_path: Path,
    *,
    model_name: str,
    estimate_params: bool,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        stack_path = tmp_dir / output_path.with_suffix(".tif").name
        tifffile.imwrite(stack_path, stack)

        command = _spotiflow_command() + [
            str(stack_path),
            "--pretrained-model",
            model_name,
            "--out-dir",
            str(tmp_dir),
        ]
        if estimate_params:
            command.append("--estimate-params")
            command.append("True")

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


def _resolve_roi_times(
    input_dir: Path,
    position: int,
    channel: int,
    roi: int | None,
    all_rois: bool,
    all_times: bool,
    time: int,
) -> tuple[list[int], list[int]]:
    if all_rois:
        rois = list_rois(input_dir, position)
    elif roi is not None:
        rois = [roi]
    else:
        raise typer.BadParameter("ROI mode requires --roi or --all-rois")

    if all_times:
        times = available_times(input_dir, position, channel)
        if not times:
            stack = load_roi_stack(input_dir, position, rois[0], channel)
            times = list(range(stack.shape[0]))
    else:
        times = [time]

    return rois, times


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
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory where Spotiflow CSV files will be written.",
        ),
    ] = Path("spotiflow"),
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Pretrained Spotiflow model name."),
    ] = "general",
    estimate_params: Annotated[
        bool,
        typer.Option("--estimate-params/--no-estimate-params", help="Estimate spot intensity and FWHM."),
    ] = False,
    device: Annotated[
        str,
        typer.Option("--device", "-d", help="Device for Spotiflow prediction."),
    ] = "auto",
    roi: Annotated[
        int | None,
        typer.Option("--roi", help="Single cell ROI index to process."),
    ] = None,
    all_rois: Annotated[
        bool,
        typer.Option("--all-rois", help="Process every cell ROI."),
    ] = False,
    all_times: Annotated[
        bool,
        typer.Option("--all-times", help="Process every available time index."),
    ] = False,
    use_roi_stacks: Annotated[
        bool,
        typer.Option(
            "--roi-stacks/--stack",
            help="Use roi/Pos*/Roi*.tif crops instead of full-field stacks.",
        ),
    ] = False,
) -> None:
    roi_mode = use_roi_stacks or all_rois or roi is not None

    if roi_mode:
        try:
            rois, times = _resolve_roi_times(
                input_dir,
                position,
                channel,
                roi,
                all_rois,
                all_times,
                time,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

        model_obj = load_spotiflow_model(model)
        total_spots = 0
        for roi_index in rois:
            stack = load_roi_stack(input_dir, position, roi_index, channel)
            for time_index in tqdm(times, desc=f"ROI {roi_index:02d}", unit="frame"):
                if time_index >= stack.shape[0]:
                    raise typer.BadParameter(
                        f"time={time_index} is out of range for ROI {roi_index} with {stack.shape[0]} frames"
                    )
                frame = np.asarray(stack[time_index])
                spots = predict_spots(
                    model_obj,
                    frame,
                    estimate_params=estimate_params,
                    device=device,
                )
                output_path = spotiflow_roi_output_path(output, roi_index, time_index)
                write_spot_csv(output_path, spots)
                total_spots += len(spots)

        typer.echo(
            f"Saved Spotiflow output to {output} for rois={rois}, times={len(times)}, "
            f"total_spots={total_spots}, model={model}"
        )
        return

    output_path = output / (
        f"spotiflow_position{position:03d}_channel{channel:03d}_time{time:09d}.csv"
    )
    resolved_model = "smfish_3d" if model == "general" else model

    try:
        stack = load_stack(input_dir, position, channel, time)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        f"Loaded position={position}, channel={channel}, time={time}: "
        f"shape={stack.shape}, dtype={stack.dtype}, model={resolved_model}"
    )

    if stack.ndim == 2 or (stack.ndim == 3 and stack.shape[0] == 1):
        image = stack[0] if stack.ndim == 3 else stack
        model_obj = load_spotiflow_model(model)
        spots = predict_spots(
            model_obj,
            np.asarray(image),
            estimate_params=estimate_params,
            device=device,
        )
        write_spot_csv(output_path, spots)
    else:
        _predict_stack_with_cli(
            stack,
            output_path,
            model_name=resolved_model,
            estimate_params=estimate_params,
        )

    typer.echo(f"Saved spotiflow output to {output_path}")