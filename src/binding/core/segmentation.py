from __future__ import annotations

import numpy as np

from binding.core.filters import (
    fill_binary_holes_2d,
    gaussian_filter_2d,
    otsu_threshold,
    variation_filter_2d,
)


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