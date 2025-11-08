from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .data_models import Track, TrackSlope


def slope_gradient_at(track: Track, position: float) -> float:
    """
    Returns slope gradient (% * 100) for the given 1D position.

    Tracks that do not define slopes simply return 0.
    """
    if not track.slopes:
        return 0.0

    for slope in track.slopes:
        if slope.start_pos <= position <= slope.end_pos:
            return slope.gradient
    return 0.0


def section_length(track: Track, sections: int = 24) -> float:
    if track.distance <= 0 or sections <= 0:
        return 0.0
    return track.distance / sections
