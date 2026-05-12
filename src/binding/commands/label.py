from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from binding.core import labeled_output_path


class LabelMethod(str, Enum):
    cca = "cca"
    watershed = "watershed"


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
