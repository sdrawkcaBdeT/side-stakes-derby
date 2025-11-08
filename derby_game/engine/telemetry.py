from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple


@dataclass
class TelemetryRacerFrame:
    racer_id: int
    name: str
    lane_position: float
    world_position: Tuple[float, float]
    pos: float
    distance_delta: float
    speed_delta: float
    speed: float
    target_speed: float
    accel: float
    hp: float
    hp_delta: float
    stamina_drain: float
    ai_mode: str
    status_flags: Dict[str, bool]
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TelemetryFrame:
    tick: int
    time: float
    phase: str
    leading_segment: str
    racers: List[TelemetryRacerFrame] = field(default_factory=list)


class TelemetryCollector:
    def __init__(self) -> None:
        self.frames: List[TelemetryFrame] = []

    def record_frame(self, frame: TelemetryFrame) -> None:
        self.frames.append(frame)

    def export(self) -> Sequence[TelemetryFrame]:
        return tuple(self.frames)

    def clear(self) -> None:
        self.frames.clear()
