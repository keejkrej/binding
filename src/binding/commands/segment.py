from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from binding.core import binary_output_path, labeled_output_path, load_stack


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size + 1))
        self.rank = [0] * (size + 1)

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]

        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent

        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return

        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


def replace_iqr_outlier_thresholds(thresholds: list[float]) -> tuple[list[float], int]:
    values = np.asarray(thresholds, dtype=np.float64)
    q1, q3 = np.percentile(values, [25, 75])
    median = float(np.median(values))
    outlier = (values < q1) | (values > q3)

    adjusted = values.copy()
    adjusted[outlier] = median
    return adjusted.tolist(), int(np.count_nonzero(outlier))


def threshold_yen_planes(stack: np.ndarray) -> tuple[np.ndarray, list[float], int]:
    from skimage.filters import threshold_yen

    if stack.ndim != 3:
        raise ValueError(f"Expected a 3D stack, got shape {stack.shape}")
    if stack.shape[0] == 0:
        raise ValueError("Expected at least one z plane")

    mask = np.zeros(stack.shape, dtype=bool)
    thresholds: list[float] = []
    for z in range(stack.shape[0]):
        thresholds.append(float(threshold_yen(stack[z])))

    thresholds, replaced_count = replace_iqr_outlier_thresholds(thresholds)
    for z, threshold in enumerate(thresholds):
        mask[z] = stack[z] > threshold
    return mask, thresholds, replaced_count


def link_plane_masks(mask: np.ndarray) -> tuple[np.ndarray, int]:
    from scipy import ndimage

    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}")

    plane_labels = np.zeros(mask.shape, dtype=np.int32)
    structure = ndimage.generate_binary_structure(2, 2)
    next_label = 1
    for z in range(mask.shape[0]):
        local_labels, count = ndimage.label(mask[z], structure=structure)
        if count == 0:
            continue
        plane_labels[z] = np.where(local_labels > 0, local_labels + next_label - 1, 0)
        next_label += int(count)

    max_label = next_label - 1
    if max_label == 0:
        return plane_labels, 0

    links = UnionFind(max_label)
    for z in range(mask.shape[0] - 1):
        current = plane_labels[z]
        following = plane_labels[z + 1]
        overlap = (current > 0) & (following > 0)
        if not np.any(overlap):
            continue

        pairs = np.unique(
            np.column_stack((current[overlap], following[overlap])),
            axis=0,
        )
        for left, right in pairs:
            links.union(int(left), int(right))

    remap = np.zeros(max_label + 1, dtype=np.int32)
    root_to_label: dict[int, int] = {}
    next_output_label = 1
    for label_id in range(1, max_label + 1):
        root = links.find(label_id)
        if root not in root_to_label:
            root_to_label[root] = next_output_label
            next_output_label += 1
        remap[label_id] = root_to_label[root]

    labels = remap[plane_labels]
    return labels.astype(np.int32, copy=False), next_output_label - 1


def filter_labels_by_z_planes(
    labels: np.ndarray,
    min_z_planes: int,
) -> tuple[np.ndarray, int, int]:
    if min_z_planes < 0:
        raise ValueError("--min-z-planes must be at least 0")

    max_label = int(labels.max())
    if min_z_planes == 0:
        return labels, max_label, 0
    if max_label == 0:
        return labels, 0, 0

    z_plane_count = np.zeros(max_label + 1, dtype=np.int32)
    for z in range(labels.shape[0]):
        label_ids = np.unique(labels[z])
        label_ids = label_ids[label_ids > 0]
        z_plane_count[label_ids] += 1

    keep = z_plane_count >= min_z_planes
    kept_ids = np.flatnonzero(keep)
    remap = np.zeros(max_label + 1, dtype=np.int32)
    remap[kept_ids] = np.arange(1, len(kept_ids) + 1, dtype=np.int32)

    filtered = remap[labels]
    return filtered.astype(np.int32, copy=False), int(len(kept_ids)), int(max_label - len(kept_ids))


def segment(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Folder containing Pos*/img_channel... TIFFs from convert.",
        ),
    ],
    position: Annotated[int, typer.Option("--position", "-p", help="Position index to segment.")] = 0,
    channel: Annotated[int, typer.Option("--channel", "-c", help="Channel index to segment.")] = 0,
    time: Annotated[int, typer.Option("--time", "-t", help="Time index to segment.")] = 0,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory where the labeled .npy output will be written.",
        ),
    ] = Path("."),
    min_z_planes: Annotated[
        int,
        typer.Option(
            "--min-z-planes",
            help="Minimum number of z planes a linked component must occupy. Use 0 to disable filtering.",
        ),
    ] = 0,
) -> None:
    try:
        stack = load_stack(input_dir, position, channel, time)
        mask, thresholds, threshold_replaced_count = threshold_yen_planes(stack)
        labels, count = link_plane_masks(mask)
        labels, count, removed_count = filter_labels_by_z_planes(labels, min_z_planes)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    output.mkdir(parents=True, exist_ok=True)
    output_path = labeled_output_path(binary_output_path(output, position, channel, time))
    np.save(output_path, labels)

    typer.echo(
        f"Saved {output_path} with shape={labels.shape}, dtype={labels.dtype}, "
        f"components={count}, removed_z_short={removed_count}, "
        f"min_z_planes={min_z_planes}, threshold_yen_iqr_replaced={threshold_replaced_count}, "
        f"threshold_yen_min={min(thresholds):.17g}, "
        f"threshold_yen_max={max(thresholds):.17g}"
    )
