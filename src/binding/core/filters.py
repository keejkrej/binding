from __future__ import annotations

import numpy as np


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