from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from binding.core import labeled_output_path


def solidify_xy_planes(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    solidified = np.zeros(mask.shape, dtype=bool)
    for z in range(mask.shape[0]):
        solidified[z] = ndimage.binary_fill_holes(mask[z])
    return solidified


def label_membrane_mask(
    binary_stack: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, int, int]:
    from scipy import ndimage

    mask = np.asarray(binary_stack, dtype=bool)
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    if iterations:
        opened = ndimage.binary_opening(mask, structure=structure, iterations=iterations)
        closed = ndimage.binary_closing(opened, structure=structure, iterations=iterations)
    else:
        closed = mask

    labels, component_count = ndimage.label(closed, structure=structure)
    if component_count == 0:
        return np.zeros(mask.shape, dtype=np.int32), 0, 0

    volumes = np.bincount(labels.ravel())
    largest_label = int(np.argmax(volumes[1:]) + 1)
    largest = labels == largest_label
    solidified = solidify_xy_planes(largest)
    solidified = ndimage.binary_fill_holes(solidified, structure=structure)
    output = solidified.astype(np.int32)
    return output, int(component_count), int(output.sum())


def label_membrane(
    input_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Binary membrane .npy stack to reduce to one labeled object.",
        ),
    ],
    iterations: Annotated[
        int,
        typer.Option(
            "--iterations",
            min=0,
            help="Opening and closing iterations before choosing the largest component.",
        ),
    ] = 1,
) -> None:
    binary_stack = np.load(input_file)
    labels, component_count, volume = label_membrane_mask(binary_stack, iterations)
    output_path = labeled_output_path(input_file)
    np.save(output_path, labels)

    typer.echo(
        f"Saved {output_path} with shape={labels.shape}, dtype={labels.dtype}, "
        f"components_before_filter={component_count}, kept_volume={volume}, "
        f"method=opening-closing-largest-solidify-xy"
    )
