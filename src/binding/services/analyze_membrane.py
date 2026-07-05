from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from binding.core.frames import load_voxel_scale


@dataclass(frozen=True)
class AnalyzeMembraneResult:
    output_path: Path
    row_count: int


def membrane_output_path(input_file: Path) -> Path:
    return input_file.with_name(f"{input_file.stem}_membrane.csv")


def read_particle_centroids(input_file: Path) -> list[tuple[int, float, float, float]]:
    particles: list[tuple[int, float, float, float]] = []
    with open(input_file, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"id", "centroid_x", "centroid_y", "centroid_z"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Analysis CSV must contain id, centroid_x, centroid_y, centroid_z columns"
            )

        for row in reader:
            particles.append(
                (
                    int(row["id"]),
                    float(row["centroid_x"]),
                    float(row["centroid_y"]),
                    float(row["centroid_z"]),
                )
            )

    return particles


def centroid_indices(
    x: float,
    y: float,
    z: float,
    shape: tuple[int, ...],
) -> tuple[int, int, int]:
    zi = int(round(z))
    yi = int(round(y))
    xi = int(round(x))
    zi = min(max(zi, 0), shape[0] - 1)
    yi = min(max(yi, 0), shape[1] - 1)
    xi = min(max(xi, 0), shape[2] - 1)
    return zi, yi, xi


def write_membrane_distances(
    output_path: Path,
    particles: list[tuple[int, float, float, float]],
    membrane_mask: np.ndarray,
    voxel_scale: tuple[float, float, float],
) -> int:
    from scipy import ndimage

    if membrane_mask.ndim != 3:
        raise ValueError(f"Expected a 3D membrane mask, got shape {membrane_mask.shape}")

    membrane = np.asarray(membrane_mask) > 0
    if not np.any(membrane):
        raise ValueError("Membrane mask contains no foreground voxels")

    structure = ndimage.generate_binary_structure(membrane.ndim, 1)
    eroded = ndimage.binary_erosion(membrane, structure=structure, border_value=0)
    surface = membrane & ~eroded
    if not np.any(surface):
        raise ValueError("Membrane mask has no detectable surface voxels")

    distance = ndimage.distance_transform_edt(~surface, sampling=voxel_scale)
    distance[membrane & ~surface] *= -1
    distance[surface] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "distance_to_membrane_um"])
        for particle_id, x, y, z in particles:
            zi, yi, xi = centroid_indices(x, y, z, membrane.shape)
            writer.writerow([particle_id, f"{distance[zi, yi, xi]:.17g}"])

    return len(particles)


def run_analyze_membrane(
    input_file: Path,
    *,
    membrane_mask: Path,
    metadata: Path,
) -> AnalyzeMembraneResult:
    particles = read_particle_centroids(input_file)
    membrane = np.load(membrane_mask, mmap_mode="r")
    voxel_scale = load_voxel_scale(metadata)
    output_path = membrane_output_path(input_file)
    row_count = write_membrane_distances(output_path, particles, membrane, voxel_scale)
    return AnalyzeMembraneResult(output_path=output_path, row_count=row_count)