from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

import numpy as np
import tifffile
import typer

app = typer.Typer(
    add_completion=False,
    help="Inspect converted microscope TIFF folders and visualize stacks.",
)

FRAME_RE = re.compile(
    r"^img_channel(?P<channel>\d+)_position(?P<position>\d+)_"
    r"time(?P<time>\d+)_z(?P<z>\d+)\.tiff?$",
    re.IGNORECASE,
)


class LabelMethod(str, Enum):
    cca = "cca"
    watershed = "watershed"


@app.callback()
def cli() -> None:
    """Inspect converted microscope TIFF folders and visualize stacks."""


@dataclass(frozen=True)
class Frame:
    path: Path
    position: int
    channel: int
    time: int
    z: int


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


def binary_output_path(output_dir: Path, position: int, channel: int, time: int) -> Path:
    return output_dir / (
        f"binary_position{position:03d}_channel{channel:03d}_time{time:09d}.npy"
    )


def labeled_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_labeled.npy")


def analysis_output_path(output_dir: Path, position: int, channel: int, time: int) -> Path:
    return output_dir / (
        f"analysis_position{position:03d}_channel{channel:03d}_time{time:09d}.csv"
    )


def connected_components(binary_stack: np.ndarray) -> tuple[np.ndarray, int]:
    from scipy import ndimage

    labels, count = ndimage.label(binary_stack)
    return labels, int(count)


def watershed_components(binary_stack: np.ndarray, min_distance: int) -> tuple[np.ndarray, int]:
    from scipy import ndimage
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    mask = np.asarray(binary_stack, dtype=bool)
    distance = ndimage.distance_transform_edt(mask)
    coordinates = peak_local_max(
        distance,
        labels=mask,
        min_distance=min_distance,
        exclude_border=False,
    )
    markers = np.zeros(mask.shape, dtype=np.int32)
    if coordinates.size == 0:
        return connected_components(mask)

    markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1, dtype=np.int32)
    labels = watershed(-distance, markers, mask=mask)
    return labels.astype(np.int32, copy=False), int(labels.max())


LABEL_PALETTE: tuple[tuple[float, float, float, float], ...] = (
    (0.93, 0.18, 0.29, 1.0),
    (0.10, 0.58, 0.95, 1.0),
    (0.22, 0.74, 0.36, 1.0),
    (0.97, 0.68, 0.13, 1.0),
    (0.58, 0.35, 0.86, 1.0),
    (0.00, 0.73, 0.78, 1.0),
    (0.94, 0.39, 0.16, 1.0),
    (0.89, 0.33, 0.72, 1.0),
)


def alternating_label_colormap(max_label: int):
    from napari.utils.colormaps import DirectLabelColormap

    colors = defaultdict(lambda: np.array((1.0, 1.0, 1.0, 1.0)))
    colors[0] = np.array((0.0, 0.0, 0.0, 0.0))
    for label_index in range(1, max_label + 1):
        colors[label_index] = np.array(
            LABEL_PALETTE[(label_index - 1) % len(LABEL_PALETTE)]
        )

    return DirectLabelColormap(color_dict=colors, name="alternating-labels")


def choose_threshold_in_napari(
    stack: np.ndarray,
    name: str,
    voxel_scale: tuple[float, float, float] | None,
) -> float:
    import napari

    typer.echo(
        "Adjust the image contrast minimum in napari, then close the napari window "
        "to use that minimum as the binarization threshold."
    )
    viewer = napari.Viewer()
    layer = viewer.add_image(
        stack,
        name=name,
        scale=voxel_scale,
        units=("um", "um", "um") if voxel_scale is not None else None,
    )
    viewer.dims.ndisplay = 3
    napari.run()
    return float(layer.contrast_limits[0])


@app.command()
def binarize(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Folder containing Pos*/img_channel... TIFFs from convert.",
        ),
    ],
    position: Annotated[int, typer.Option("--position", "-p", help="Position index to binarize.")] = 0,
    channel: Annotated[int, typer.Option("--channel", "-c", help="Channel index to binarize.")] = 0,
    time: Annotated[int, typer.Option("--time", "-t", help="Time index to binarize.")] = 0,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory where the encoded .npy output will be written.",
        ),
    ] = Path("."),
    threshold: Annotated[
        float | None,
        typer.Option(
            "--threshold",
            help="Intensity threshold. Defaults to napari contrast minimum after close.",
        ),
    ] = None,
    metadata: Annotated[
        Path | None,
        typer.Option(
            "--metadata",
            exists=True,
            dir_okay=False,
            help="metadata.json containing normalized.pixel_size_um x/y/z values.",
        ),
    ] = None,
) -> None:
    try:
        stack = load_stack(input_dir, position, channel, time)
        voxel_scale = load_voxel_scale(metadata) if metadata is not None else None
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if voxel_scale is not None:
        typer.echo(f"Voxel scale (z, y, x): {voxel_scale} um")

    layer_name = f"Pos{position} C{channel} T{time}"
    resolved_threshold = (
        choose_threshold_in_napari(stack, layer_name, voxel_scale)
        if threshold is None
        else threshold
    )
    binary_stack = stack > resolved_threshold

    output.mkdir(parents=True, exist_ok=True)
    output_path = binary_output_path(output, position, channel, time)
    np.save(output_path, binary_stack)

    typer.echo(
        f"Saved {output_path} with shape={binary_stack.shape}, "
        f"dtype={binary_stack.dtype}, threshold={resolved_threshold}"
    )


def write_analysis_csv(
    output_path: Path,
    image_stack: np.ndarray,
    labels: np.ndarray,
) -> int:
    if image_stack.shape != labels.shape:
        raise ValueError(
            f"Image stack shape {image_stack.shape} does not match mask shape {labels.shape}"
        )
    if labels.ndim != 3:
        raise ValueError(f"Expected a 3D labeled mask, got shape {labels.shape}")
    if int(labels.min()) < 0:
        raise ValueError("Labeled mask must not contain negative ids")

    max_label = int(labels.max())
    if max_label == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["id", "volume", "total_intensity", "centroid_x", "centroid_y", "centroid_z"]
            )
        return 0

    depth, height, width = labels.shape
    volume = np.zeros(max_label + 1, dtype=np.int64)
    total_intensity = np.zeros(max_label + 1, dtype=np.float64)
    sum_x = np.zeros(max_label + 1, dtype=np.float64)
    sum_y = np.zeros(max_label + 1, dtype=np.float64)
    sum_z = np.zeros(max_label + 1, dtype=np.float64)

    x_coords = np.tile(np.arange(width, dtype=np.float64), height)
    y_coords = np.repeat(np.arange(height, dtype=np.float64), width)

    for z in range(depth):
        label_values = np.asarray(labels[z]).ravel()
        image_values = np.asarray(image_stack[z]).ravel()
        counts = np.bincount(label_values, minlength=max_label + 1)

        volume += counts
        total_intensity += np.bincount(
            label_values,
            weights=image_values,
            minlength=max_label + 1,
        )
        sum_x += np.bincount(
            label_values,
            weights=x_coords,
            minlength=max_label + 1,
        )
        sum_y += np.bincount(
            label_values,
            weights=y_coords,
            minlength=max_label + 1,
        )
        sum_z += counts * z

    ids = np.flatnonzero(volume[1:]) + 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["id", "volume", "total_intensity", "centroid_x", "centroid_y", "centroid_z"]
        )
        for label_id in ids:
            writer.writerow(
                [
                    int(label_id),
                    int(volume[label_id]),
                    f"{total_intensity[label_id]:.17g}",
                    f"{sum_x[label_id] / volume[label_id]:.17g}",
                    f"{sum_y[label_id] / volume[label_id]:.17g}",
                    f"{sum_z[label_id] / volume[label_id]:.17g}",
                ]
            )

    return int(len(ids))


@app.command()
def analyze(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Folder containing Pos*/img_channel... TIFFs from convert.",
        ),
    ],
    position: Annotated[int, typer.Option("--position", "-p", help="Position index to analyze.")] = 0,
    channel: Annotated[int, typer.Option("--channel", "-c", help="Channel index to analyze.")] = 0,
    time: Annotated[int, typer.Option("--time", "-t", help="Time index to analyze.")] = 0,
    mask: Annotated[
        Path,
        typer.Option(
            "--mask",
            exists=True,
            dir_okay=False,
            help="Labeled .npy mask to measure.",
        ),
    ] = ...,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory where the analysis CSV will be written.",
        ),
    ] = Path("."),
) -> None:
    try:
        image_stack = load_stack(input_dir, position, channel, time)
        labels = np.load(mask, mmap_mode="r")
        output_path = analysis_output_path(output, position, channel, time)
        row_count = write_analysis_csv(output_path, image_stack, labels)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        f"Saved {output_path} with rows={row_count}, "
        f"source_shape={image_stack.shape}, mask_shape={labels.shape}"
    )


@app.command()
def show_labeled(
    input_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Labeled .npy stack to show.",
        ),
    ],
    metadata: Annotated[
        Path | None,
        typer.Option(
            "--metadata",
            exists=True,
            dir_okay=False,
            help="metadata.json containing normalized.pixel_size_um x/y/z values.",
        ),
    ] = None,
) -> None:
    labels = np.load(input_file, mmap_mode="r")
    voxel_scale = load_voxel_scale(metadata) if metadata is not None else None
    max_label = int(labels.max())

    typer.echo(
        f"Loaded {input_file}: shape={labels.shape}, dtype={labels.dtype}, "
        f"components={max_label}"
    )
    if voxel_scale is not None:
        typer.echo(f"Voxel scale (z, y, x): {voxel_scale} um")

    import napari

    viewer = napari.Viewer()
    viewer.add_labels(
        labels,
        name=input_file.stem,
        colormap=alternating_label_colormap(max_label),
        metadata={
            "source": str(input_file),
            "metadata": str(metadata) if metadata is not None else None,
            "voxel_size_um_zyx": voxel_scale,
        },
        scale=voxel_scale,
        units=("um", "um", "um") if voxel_scale is not None else None,
    )
    viewer.dims.ndisplay = 3
    napari.run()


@app.command()
def label(
    input_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Binary .npy stack to label.",
        ),
    ],
    method: Annotated[
        LabelMethod,
        typer.Option(
            "--method",
            help="Labeling method to use.",
        ),
    ],
    min_distance: Annotated[
        int,
        typer.Option(
            "--min-distance",
            min=1,
            help="Minimum peak spacing for watershed markers.",
        ),
    ] = 5,
) -> None:
    binary_stack = np.load(input_file)
    labels, count = (
        watershed_components(binary_stack, min_distance)
        if method == LabelMethod.watershed
        else connected_components(binary_stack)
    )
    output_path = labeled_output_path(input_file)
    np.save(output_path, labels)

    typer.echo(
        f"Saved {output_path} with shape={labels.shape}, "
        f"dtype={labels.dtype}, components={count}, "
        f"method={method.value}"
    )


@app.command()
def show(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Folder containing Pos*/img_channel... TIFFs from convert.",
        ),
    ],
    position: Annotated[int, typer.Option("--position", "-p", help="Position index to show.")] = 0,
    channel: Annotated[int, typer.Option("--channel", "-c", help="Channel index to show.")] = 0,
    time: Annotated[int, typer.Option("--time", "-t", help="Time index to show.")] = 0,
    metadata: Annotated[
        Path | None,
        typer.Option(
            "--metadata",
            exists=True,
            dir_okay=False,
            help="metadata.json containing normalized.pixel_size_um x/y/z values.",
        ),
    ] = None,
) -> None:
    try:
        stack = load_stack(input_dir, position, channel, time)
        voxel_scale = load_voxel_scale(metadata) if metadata is not None else None
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        f"Loaded position={position}, channel={channel}, time={time}: "
        f"shape={stack.shape}, dtype={stack.dtype}"
    )
    if voxel_scale is not None:
        typer.echo(f"Voxel scale (z, y, x): {voxel_scale} um")

    import napari

    viewer = napari.Viewer()
    viewer.add_image(
        stack,
        name=f"Pos{position} C{channel} T{time}",
        metadata={
            "position": position,
            "channel": channel,
            "time": time,
            "source": str(input_dir),
            "axis_order": "ZYX",
            "metadata": str(metadata) if metadata is not None else None,
            "voxel_size_um_zyx": voxel_scale,
        },
        scale=voxel_scale,
        units=("um", "um", "um") if voxel_scale is not None else None,
    )
    viewer.dims.ndisplay = 3
    napari.run()


def main() -> None:
    app(prog_name="binding")


if __name__ == "__main__":
    main()
