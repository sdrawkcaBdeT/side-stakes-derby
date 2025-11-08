from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return (dx * dx + dy * dy) ** 0.5


def _lerp(a: Tuple[float, float], b: Tuple[float, float], t: float) -> Tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


@dataclass
class Polyline:
    """Arc-length aware polyline helper used for SVG track data."""

    points: Sequence[Tuple[float, float]]
    cumulative_lengths: Sequence[float]
    length: float

    @classmethod
    def from_points(cls, points: Iterable[Tuple[float, float]]) -> "Polyline":
        pts: List[Tuple[float, float]] = list(points)
        if len(pts) < 2:
            raise ValueError("Polyline requires at least two points")

        cumulative: List[float] = [0.0]

        for idx in range(1, len(pts)):
            cumulative.append(cumulative[-1] + _distance(pts[idx - 1], pts[idx]))

        return cls(points=tuple(pts), cumulative_lengths=tuple(cumulative), length=cumulative[-1])

    def position_at(self, distance: float) -> Tuple[float, float]:
        """Returns the coordinate at a distance along the polyline."""
        if distance <= 0:
            return self.points[0]
        if distance >= self.length:
            return self.points[-1]

        idx, t = self._segment_parameters(distance)
        return _lerp(self.points[idx - 1], self.points[idx], t)

    def sample_evenly(self, count: int) -> List[Tuple[float, float]]:
        if count < 2:
            raise ValueError("Sample count must be at least 2")
        step = self.length / (count - 1)
        return [self.position_at(step * idx) for idx in range(count)]

    def tangent_at(self, distance: float) -> Tuple[float, float]:
        """Returns the unit tangent vector at the given arc-length."""
        idx, _ = self._segment_parameters(distance)

        tangent = (
            self.points[idx][0] - self.points[idx - 1][0],
            self.points[idx][1] - self.points[idx - 1][1],
        )

        magnitude = math.hypot(*tangent)
        if magnitude == 0.0:
            # Fallback: search outward for the nearest non-zero segment.
            for offset in range(1, len(self.points)):
                lower = max(0, idx - 1 - offset)
                upper = min(len(self.points) - 1, idx - 1 + offset)
                if lower < idx - 1:
                    tangent = (
                        self.points[idx - 1][0] - self.points[lower][0],
                        self.points[idx - 1][1] - self.points[lower][1],
                    )
                    magnitude = math.hypot(*tangent)
                    if magnitude:
                        break
                if upper > idx - 1:
                    tangent = (
                        self.points[upper][0] - self.points[idx - 1][0],
                        self.points[upper][1] - self.points[idx - 1][1],
                    )
                    magnitude = math.hypot(*tangent)
                    if magnitude:
                        break
        if magnitude == 0.0:
            return (1.0, 0.0)
        return (tangent[0] / magnitude, tangent[1] / magnitude)

    def curvature_at(self, distance: float, window: float = 1.0) -> float:
        """Approximate curvature magnitude (radians per meter) at the given arc-length."""
        if self.length <= 0.0:
            return 0.0
        d0 = max(0.0, distance - window)
        d1 = min(self.length, distance)
        d2 = min(self.length, distance + window)
        p0 = self.position_at(d0)
        p1 = self.position_at(d1)
        p2 = self.position_at(d2)
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        len1 = math.hypot(*v1)
        len2 = math.hypot(*v2)
        if len1 < 1e-6 or len2 < 1e-6:
            return 0.0
        v1n = (v1[0] / len1, v1[1] / len1)
        v2n = (v2[0] / len2, v2[1] / len2)
        dot = max(-1.0, min(1.0, v1n[0] * v2n[0] + v1n[1] * v2n[1]))
        angle = math.acos(dot)
        arc = len1 + len2
        if arc == 0.0:
            return 0.0
        return angle / arc

    def normal_at(self, distance: float) -> Tuple[float, float]:
        """Returns the unit normal vector (rotated left) at the given arc-length."""
        tx, ty = self.tangent_at(distance)
        return (-ty, tx)

    def _segment_parameters(self, distance: float) -> Tuple[int, float]:
        """Return segment index and interpolation factor for a given distance."""
        if distance <= 0:
            return 1, 0.0
        if distance >= self.length:
            return len(self.points) - 1, 1.0

        lo = 0
        hi = len(self.cumulative_lengths) - 1

        while lo < hi:
            mid = (lo + hi) // 2
            if self.cumulative_lengths[mid] < distance:
                lo = mid + 1
            else:
                hi = mid

        idx = max(1, lo)
        prev_dist = self.cumulative_lengths[idx - 1]
        next_dist = self.cumulative_lengths[idx]
        seg_length = next_dist - prev_dist
        if seg_length == 0.0:
            return idx, 0.0

        t = (distance - prev_dist) / seg_length
        return idx, max(0.0, min(1.0, t))
