from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from binding.core.frames import available_times, load_stack
from binding.core.paths import spotiflow_roi_output_path
from binding.core.roi import list_rois, load_roi_stack
from binding.core.spotiflow import load_spotiflow_model, predict_spots, write_spot_csv


@dataclass(frozen=True)
class SpotiflowRoiResult:
    output_dir: Path
    rois: list[int]
    time_count: int
    total_spots: int
    model: str


@dataclass(frozen=True)
class SpotiflowStackResult:
    output_path: Path
    position: int
    channel: int
    time: int
    shape: tuple[int, ...]
    dtype: np.dtype
    model: str


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


def resolve_roi_times(
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
        raise ValueError("ROI mode requires --roi or --all-rois")

    if all_times:
        times = available_times(input_dir, position, channel)
        if not times:
            stack = load_roi_stack(input_dir, position, rois[0], channel)
            times = list(range(stack.shape[0]))
    else:
        times = [time]

    return rois, times


def run_spotiflow_roi(
    input_dir: Path,
    *,
    position: int,
    channel: int,
    time: int,
    output: Path,
    model: str,
    estimate_params: bool,
    device: str,
    roi: int | None,
    all_rois: bool,
    all_times: bool,
) -> SpotiflowRoiResult:
    rois, times = resolve_roi_times(
        input_dir,
        position,
        channel,
        roi,
        all_rois,
        all_times,
        time,
    )

    from tqdm import tqdm

    model_obj = load_spotiflow_model(model)
    total_spots = 0
    for roi_index in rois:
        stack = load_roi_stack(input_dir, position, roi_index, channel)
        for time_index in tqdm(times, desc=f"ROI {roi_index:02d}", unit="frame"):
            if time_index >= stack.shape[0]:
                raise ValueError(
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

    return SpotiflowRoiResult(
        output_dir=output,
        rois=rois,
        time_count=len(times),
        total_spots=total_spots,
        model=model,
    )


def run_spotiflow_stack(
    input_dir: Path,
    *,
    position: int,
    channel: int,
    time: int,
    output: Path,
    model: str,
    estimate_params: bool,
    device: str,
) -> SpotiflowStackResult:
    output_path = output / (
        f"spotiflow_position{position:03d}_channel{channel:03d}_time{time:09d}.csv"
    )
    resolved_model = "smfish_3d" if model == "general" else model
    stack = load_stack(input_dir, position, channel, time)

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

    return SpotiflowStackResult(
        output_path=output_path,
        position=position,
        channel=channel,
        time=time,
        shape=stack.shape,
        dtype=stack.dtype,
        model=resolved_model,
    )