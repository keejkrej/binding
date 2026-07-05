from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from binding.core.types import FRAME_RE, Frame


def parse_frame(path: Path) -> Frame | None:
    match = FRAME_RE.match(path.name)
    if match is None:
        return None

    return Frame(
        path=path,
        position=int(match.group("position")),
        channel=int(match.group("channel")),
        time=int(match.group("time")),
        z=int(match.group("z")),
    )


def find_frames(root: Path) -> list[Frame]:
    frames: list[Frame] = []
    for pos_dir in sorted(root.glob("Pos*")):
        if not pos_dir.is_dir():
            continue
        for path in sorted(pos_dir.glob("*.tif*")):
            frame = parse_frame(path)
            if frame is not None:
                frames.append(frame)
    return frames


def available_times(root: Path, position: int, channel: int) -> list[int]:
    frames = find_frames(root)
    return sorted(
        {
            frame.time
            for frame in frames
            if frame.position == position and frame.channel == channel
        }
    )


def available_summary(frames: list[Frame]) -> str:
    positions = sorted({frame.position for frame in frames})
    channels = sorted({frame.channel for frame in frames})
    times = sorted({frame.time for frame in frames})
    return (
        f"available positions={positions}, channels={channels}, "
        f"times={times}"
    )


def load_stack(root: Path, position: int, channel: int, time: int) -> np.ndarray:
    frames = find_frames(root)
    if not frames:
        raise ValueError(
            f"No converted TIFF frames found under {root}. "
            "Expected Pos*/img_channel###_position###_time#########_z###.tif"
        )

    selected = [
        frame
        for frame in frames
        if frame.position == position and frame.channel == channel and frame.time == time
    ]
    if not selected:
        raise ValueError(
            f"No frames found for position={position}, channel={channel}, time={time}; "
            f"{available_summary(frames)}"
        )

    by_z: dict[int, Path] = {}
    for frame in selected:
        if frame.z in by_z:
            raise ValueError(f"Duplicate z={frame.z} frame for selection: {frame.path}")
        by_z[frame.z] = frame.path

    planes = [tifffile.imread(by_z[z]) for z in sorted(by_z)]
    first_shape = planes[0].shape
    mismatched = [plane.shape for plane in planes if plane.shape != first_shape]
    if mismatched:
        raise ValueError(
            f"Cannot stack planes with different shapes; first={first_shape}, "
            f"mismatched={mismatched[0]}"
        )

    return np.stack(planes, axis=0)


def load_voxel_scale(metadata_path: Path) -> tuple[float, float, float]:
    with open(metadata_path, encoding="utf-8") as fh:
        metadata = json.load(fh)

    try:
        pixel_size = metadata["normalized"]["pixel_size_um"]
        x = float(pixel_size["x"])
        y = float(pixel_size["y"])
        z = float(pixel_size["z"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Metadata must contain normalized.pixel_size_um with numeric x, y, z values"
        ) from exc

    return z, y, x