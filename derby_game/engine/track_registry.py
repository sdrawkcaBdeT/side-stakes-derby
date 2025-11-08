from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .data_models import Track
from .svg_loader import load_svg_track


def _default_svg_directory() -> Path:
    return Path(__file__).resolve().parents[2] / "racetracks" / "svg"


@dataclass(frozen=True)
class TrackDescriptor:
    track_id: str
    name: str
    svg_path: Path
    nominal_distance: Optional[float] = None


class TrackRegistry:
    """Loads and caches racetrack geometry from SVG files."""

    def __init__(self, svg_directory: Optional[Path] = None) -> None:
        self.svg_directory = Path(svg_directory) if svg_directory else _default_svg_directory()
        self._cache: Dict[str, Track] = {}
        self._descriptors: List[TrackDescriptor] = []
        self._index_directory()

    def _index_directory(self) -> None:
        if not self.svg_directory.exists():
            return
        for svg_file in sorted(self.svg_directory.glob("*.svg")):
            name = svg_file.stem
            track_id = name.lower().replace(" ", "_")
            distance = _parse_distance_from_name(name)
            self._descriptors.append(TrackDescriptor(track_id=track_id, name=name, svg_path=svg_file, nominal_distance=distance))

    def load(self, track_id: str) -> Track:
        key = track_id.lower()
        if key in self._cache:
            return self._cache[key]

        descriptor = next((d for d in self._descriptors if d.track_id == key), None)
        if descriptor is None:
            raise KeyError(f"Track '{track_id}' not found in {self.svg_directory}")

        track = load_svg_track(descriptor.svg_path, track_id=descriptor.track_id, name=descriptor.name)
        self._cache[key] = track
        return track

    def find_by_distance(self, distance: float, tolerance: float = 10.0) -> Optional[Track]:
        """Returns the track whose nominal distance is within the tolerance."""
        candidates: List[Tuple[float, TrackDescriptor]] = []
        for descriptor in self._descriptors:
            if descriptor.nominal_distance is None:
                continue
            delta = abs(descriptor.nominal_distance - distance)
            if delta <= tolerance:
                candidates.append((delta, descriptor))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        track = self.load(candidates[0][1].track_id)
        return track

    def list_tracks(self) -> Iterable[TrackDescriptor]:
        return list(self._descriptors)


_DISTANCE_PATTERN = re.compile(r"^(?P<value>\d+(?:\.\d+)?)m", re.IGNORECASE)


def _parse_distance_from_name(name: str) -> Optional[float]:
    match = _DISTANCE_PATTERN.match(name)
    if not match:
        return None
    return float(match.group("value"))


DEFAULT_TRACK_REGISTRY = TrackRegistry()
