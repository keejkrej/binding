"""Rebuild per-cell ROI stacks from full-field Pos0 TIFFs using existing bbox/index.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm

from binding.core.frames import available_times, load_stack


def rebuild_rois(
    data_dir: Path,
    *,
    position: int = 0,
    source_index: Path | None = None,
) -> None:
    index_path = source_index or (data_dir / "roi" / f"Pos{position}" / "index.json")
    if not index_path.exists():
        raise FileNotFoundError(f"ROI index not found: {index_path}")

    with index_path.open(encoding="utf-8") as fh:
        index = json.load(fh)

    times = available_times(data_dir, position, channel=0)
    if not times:
        raise ValueError(f"No timepoints under {data_dir}/Pos{position}")

    roi_dir = data_dir / "roi" / f"Pos{position}"
    roi_dir.mkdir(parents=True, exist_ok=True)

    for entry in index["rois"]:
        roi_id = int(entry["roi"])
        bbox = entry["bbox"]
        x0 = int(bbox["x"])
        y0 = int(bbox["y"])
        width = int(bbox["w"])
        height = int(bbox["h"])
        channel_count = int(index.get("channelCount", 2))
        z_count = 1

        frames: list[np.ndarray] = []
        for time_index in tqdm(times, desc=f"Roi{roi_id}", unit="t"):
            channels = [
                load_stack(data_dir, position, channel, time_index)[0]
                for channel in range(channel_count)
            ]
            crop = np.stack(
                [plane[y0 : y0 + height, x0 : x0 + width] for plane in channels],
                axis=0,
            )
            frames.append(crop)

        stack = np.stack(frames, axis=0)[:, :, np.newaxis, :, :]
        out_path = roi_dir / entry["fileName"]
        flat = stack.reshape(-1, stack.shape[-2], stack.shape[-1])
        tifffile.imwrite(out_path, flat, photometric="minisblack")

        entry["shape"] = [len(times), channel_count, z_count, height, width]
        print(f"Wrote {out_path} shape={entry['shape']}")

    out_index = roi_dir / "index.json"
    with out_index.open("w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    print(f"Updated {out_index} ({len(times)} timepoints)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--position", type=int, default=0)
    parser.add_argument(
        "--source-index",
        type=Path,
        default=None,
        help="Existing index.json with bbox entries (defaults to data_dir/roi/PosN/index.json).",
    )
    args = parser.parse_args()
    rebuild_rois(args.data_dir, position=args.position, source_index=args.source_index)


if __name__ == "__main__":
    main()
