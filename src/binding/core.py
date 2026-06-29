from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

FRAME_RE = re.compile(
    r"^img_channel(?P<channel>\d+)_position(?P<position>\d+)_"
    r"time(?P<time>\d+)_z(?P<z>\d+)\.tiff?$",
    re.IGNORECASE,
)


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


def roi_dir(input_dir: Path, position: int) -> Path:
    return input_dir / "roi" / f"Pos{position}"


def load_roi_index(input_dir: Path, position: int) -> dict:
    index_path = roi_dir(input_dir, position) / "index.json"
    if not index_path.exists():
        raise ValueError(f"ROI index not found at {index_path}")

    with open(index_path, encoding="utf-8") as fh:
        return json.load(fh)


def list_rois(input_dir: Path, position: int) -> list[int]:
    index = load_roi_index(input_dir, position)
    return sorted(int(entry["roi"]) for entry in index["rois"])


def roi_stack_path(input_dir: Path, position: int, roi: int) -> Path:
    index = load_roi_index(input_dir, position)
    for entry in index["rois"]:
        if int(entry["roi"]) == roi:
            return roi_dir(input_dir, position) / entry["fileName"]
    raise ValueError(f"ROI {roi} not found for position={position}")


def load_roi_stack(
    input_dir: Path,
    position: int,
    roi: int,
    channel: int,
) -> np.ndarray:
    index = load_roi_index(input_dir, position)
    roi_entry = next(
        (entry for entry in index["rois"] if int(entry["roi"]) == roi),
        None,
    )
    if roi_entry is None:
        raise ValueError(f"ROI {roi} not found for position={position}")

    shape = tuple(int(value) for value in roi_entry["shape"])
    if len(shape) != 5:
        raise ValueError(f"Expected TCZYX ROI shape, got {shape}")

    time_count, channel_count, z_count, _, _ = shape
    if not 0 <= channel < channel_count:
        raise ValueError(
            f"Channel {channel} is out of range for ROI {roi} with {channel_count} channels"
        )

    raw = tifffile.imread(roi_stack_path(input_dir, position, roi))
    expected_pages = time_count * channel_count * z_count
    if raw.ndim == 3 and raw.shape[0] == expected_pages:
        volume = raw.reshape(time_count, channel_count, z_count, raw.shape[1], raw.shape[2])
    elif raw.ndim == 5:
        volume = raw
    else:
        raise ValueError(
            f"ROI {roi} has unsupported layout: raw shape {raw.shape}, expected {shape}"
        )

    return np.asarray(volume[:, channel, 0], dtype=raw.dtype)


def load_time_map(path: Path) -> dict[int, float]:
    mapping: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "t" not in reader.fieldnames:
            raise ValueError("Time map CSV must contain a t column")
        real_key = "t_real" if "t_real" in reader.fieldnames else "t"
        for row in reader:
            mapping[int(row["t"])] = float(row[real_key])
    if not mapping:
        raise ValueError(f"Time map CSV {path} has no rows")
    return mapping


def build_time_map(time_count: int, interval_sec: float) -> dict[int, float]:
    return {time_index: time_index * interval_sec for time_index in range(time_count)}


def spotiflow_roi_output_path(output_dir: Path, roi: int, time: int) -> Path:
    return output_dir / f"roi{roi:02d}_time{time:09d}.csv"


def filtered_spots_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_filtered.csv")


def spot_counts_output_path(output_dir: Path, position: int, channel: int) -> Path:
    return output_dir / f"spot_counts_position{position:03d}_channel{channel:03d}.csv"


# --- Cell mask inference from BF (channel 0) using transfection mask-channel algorithm ---
# See transfection/core/segment.py : variation filter -> gaussian -> otsu -> fill holes


def _box_mean_2d(image: np.ndarray, *, radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError(f"Variation radius must be >= 0, got {radius}")
    if radius == 0:
        return image.astype(np.float64, copy=False)

    window = radius * 2 + 1
    padded = np.pad(image.astype(np.float64, copy=False), ((radius, radius), (radius, radius)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    summed = (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    )
    return summed / float(window * window)


def variation_filter_2d(image: np.ndarray, *, radius: int) -> np.ndarray:
    values = image.astype(np.float64, copy=False)
    mean = _box_mean_2d(values, radius=radius)
    mean_square = _box_mean_2d(values * values, radius=radius)
    variance = np.maximum(mean_square - mean * mean, 0.0)
    return np.sqrt(variance)


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(np.ceil(sigma * 3.0)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _convolve_axis_reflect(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    if pad == 0:
        return image
    pad_width = [(0, 0)] * image.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(image, pad_width, mode="edge")
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), axis, padded)


def gaussian_filter_2d(image: np.ndarray, *, sigma: float) -> np.ndarray:
    if sigma < 0:
        raise ValueError(f"Gaussian sigma must be >= 0, got {sigma}")
    kernel = _gaussian_kernel_1d(sigma)
    smoothed = _convolve_axis_reflect(image.astype(np.float64, copy=False), kernel, axis=0)
    return _convolve_axis_reflect(smoothed, kernel, axis=1)


def otsu_threshold(image: np.ndarray, *, bins: int = 256) -> float:
    values = image[np.isfinite(image)].astype(np.float64, copy=False)
    if values.size == 0:
        raise ValueError("Cannot compute Otsu threshold for an empty image")

    min_value = float(values.min())
    max_value = float(values.max())
    if min_value == max_value:
        return min_value

    hist, edges = np.histogram(values, bins=bins, range=(min_value, max_value))
    centers = (edges[:-1] + edges[1:]) * 0.5
    weight_foreground = np.cumsum(hist).astype(np.float64)
    weight_background = float(values.size) - weight_foreground
    intensity_sum = np.cumsum(hist * centers)
    total_intensity_sum = intensity_sum[-1]

    valid = (weight_foreground > 0) & (weight_background > 0)
    if not np.any(valid):
        return min_value

    mean_foreground = np.zeros_like(centers)
    mean_background = np.zeros_like(centers)
    mean_foreground[valid] = intensity_sum[valid] / weight_foreground[valid]
    mean_background[valid] = (total_intensity_sum - intensity_sum[valid]) / weight_background[valid]
    variance = np.zeros_like(centers)
    variance[valid] = (
        weight_foreground[valid]
        * weight_background[valid]
        * np.square(mean_foreground[valid] - mean_background[valid])
    )
    return float(centers[int(np.argmax(variance))])


def fill_binary_holes_2d(mask: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape={mask_bool.shape}")

    background = ~mask_bool
    exterior = np.zeros(mask_bool.shape, dtype=bool)
    stack: list[tuple[int, int]] = []

    height, width = mask_bool.shape
    for x in range(width):
        if background[0, x]:
            stack.append((0, x))
        if height > 1 and background[height - 1, x]:
            stack.append((height - 1, x))
    for y in range(height):
        if background[y, 0]:
            stack.append((y, 0))
        if width > 1 and background[y, width - 1]:
            stack.append((y, width - 1))

    while stack:
        y, x = stack.pop()
        if exterior[y, x] or not background[y, x]:
            continue
        exterior[y, x] = True
        if y > 0:
            stack.append((y - 1, x))
        if y + 1 < height:
            stack.append((y + 1, x))
        if x > 0:
            stack.append((y, x - 1))
        if x + 1 < width:
            stack.append((y, x + 1))

    holes = background & ~exterior
    return mask_bool | holes


def segment_frame(
    frame: np.ndarray,
    *,
    variation_radius: int,
    gaussian_sigma: float,
) -> np.ndarray:
    varied = variation_filter_2d(frame, radius=variation_radius)
    smoothed = gaussian_filter_2d(varied, sigma=gaussian_sigma)
    threshold = otsu_threshold(smoothed)
    return fill_binary_holes_2d(smoothed > threshold)


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    labels, component_count = ndimage.label(mask, structure=structure)
    if component_count == 0:
        return np.asarray(mask, dtype=bool)
    volumes = np.bincount(labels.ravel())
    largest_label = int(np.argmax(volumes[1:]) + 1)
    return labels == largest_label


def solidify_cell_mask(mask: np.ndarray, *, closing_iterations: int = 4) -> np.ndarray:
    """Merge fragmented segmentation into one filled convex cell mask."""
    from scipy import ndimage
    from skimage.morphology import convex_hull_image

    structure = ndimage.generate_binary_structure(2, 1)
    closed = ndimage.binary_closing(
        np.asarray(mask, dtype=bool),
        structure=structure,
        iterations=closing_iterations,
    )
    component = _largest_connected_component(closed)
    if not component.any():
        return component
    return convex_hull_image(component)


def compute_cell_mask(
    frame: np.ndarray,
    *,
    variation_radius: int = 2,
    gaussian_sigma: float = 1.0,
) -> np.ndarray:
    """Compute binary cell mask from a BF (or mask channel) 2D frame using the transfection algorithm."""
    raw_mask = segment_frame(
        np.asarray(frame, dtype=np.float64),
        variation_radius=variation_radius,
        gaussian_sigma=gaussian_sigma,
    )
    return solidify_cell_mask(raw_mask)
