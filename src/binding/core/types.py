from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

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