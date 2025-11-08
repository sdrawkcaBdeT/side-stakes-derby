"""
Race engine package implementing the 2.5D hybrid simulation.

The package is split into data models, physics kernels, spatial logic,
and parsing utilities. Higher level orchestration code composes these
pieces for the master race loop.
"""

from .course import slope_gradient_at  # noqa: F401
from .data_models import (  # noqa: F401
    AptitudeGrade,
    Aptitudes,
    LaneGuide,
    RacerProfile,
    RacerState,
    RacerStats,
    Strategy,
    Track,
    TrackBoundary,
    TrackCorner,
    TrackSlope,
    TrackStraight,
)
from .svg_loader import load_svg_track  # noqa: F401
from .telemetry import TelemetryCollector, TelemetryFrame, TelemetryRacerFrame  # noqa: F401
from .race_loop import HybridRaceLoop  # noqa: F401

__all__ = [
    "slope_gradient_at",
    "AptitudeGrade",
    "Aptitudes",
    "LaneGuide",
    "RacerProfile",
    "RacerState",
    "RacerStats",
    "Strategy",
    "Track",
    "TrackBoundary",
    "TrackCorner",
    "TrackSlope",
    "TrackStraight",
    "TelemetryCollector",
    "TelemetryFrame",
    "TelemetryRacerFrame",
    "HybridRaceLoop",
    "load_svg_track",
]
