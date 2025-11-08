from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


class Strategy(Enum):
    """Race macro strategy mapping to legacy system identifiers."""

    FRONT_RUNNER = "front_runner"
    PACE_CHASER = "pace_chaser"
    LATE_SURGER = "late_surger"
    END_CLOSER = "end_closer"

    @classmethod
    def from_legacy(cls, strategy_id: int) -> "Strategy":
        mapping = {
            1: cls.FRONT_RUNNER,
            2: cls.PACE_CHASER,
            3: cls.LATE_SURGER,
            4: cls.END_CLOSER,
        }
        if strategy_id not in mapping:
            raise ValueError(f"Unknown legacy strategy id: {strategy_id}")
        return mapping[strategy_id]


class AptitudeGrade(Enum):
    """Distance and surface aptitude grades."""

    S = "S"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"

    @classmethod
    def from_str(cls, value: str) -> "AptitudeGrade":
        try:
            return cls(value.upper())
        except ValueError as exc:
            raise ValueError(f"Unknown aptitude grade: {value}") from exc


@dataclass(frozen=True)
class RacerStats:
    speed: float
    acceleration: float
    stamina: float
    grit: float
    cognition: float
    focus: float
    luck: float


@dataclass(frozen=True)
class Aptitudes:
    distance: AptitudeGrade
    surface: AptitudeGrade


@dataclass(frozen=True)
class Skill:
    """Placeholder for active/passive skills until detailed system lands."""

    identifier: str
    level: int = 1
    data: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RacerProfile:
    racer_id: int
    name: str
    stats: RacerStats
    strategy: Strategy
    aptitudes: Aptitudes
    skills: Sequence[Skill] = field(default_factory=tuple)


@dataclass(frozen=True)
class TrackSlope:
    start_pos: float
    length: float
    gradient: float

    @property
    def end_pos(self) -> float:
        return self.start_pos + self.length


@dataclass(frozen=True)
class TrackCorner:
    start_pos: float
    length: float

    @property
    def end_pos(self) -> float:
        return self.start_pos + self.length


@dataclass(frozen=True)
class TrackStraight:
    start_pos: float
    end_pos: float

    @property
    def length(self) -> float:
        return self.end_pos - self.start_pos


@dataclass(frozen=True)
class TrackBoundary:
    """Represents a polyline or spline describing a track boundary."""

    name: str
    points: Sequence[Tuple[float, float]]


@dataclass(frozen=True)
class TrackSpline:
    """Central racing line; interpolation handled by spatial module."""

    points: Sequence[Tuple[float, float]]
    length: float


@dataclass(frozen=True)
class LaneGuide:
    """Offset spline parallel to the racing line."""

    side: str  # 'left' or 'right'
    index: int
    spline: TrackSpline


@dataclass(frozen=True)
class Track:
    track_id: str
    name: str
    distance: float
    spline: TrackSpline
    track_width: float = 0.0
    surface: str = "unknown"
    turn: str = "unknown"
    slopes: Sequence[TrackSlope] = field(default_factory=tuple)
    corners: Sequence[TrackCorner] = field(default_factory=tuple)
    straights: Sequence[TrackStraight] = field(default_factory=tuple)
    boundary_inner: Optional[TrackBoundary] = None
    boundary_outer: Optional[TrackBoundary] = None
    lane_guides: Sequence[LaneGuide] = field(default_factory=tuple)
    lane_length_deltas: Dict[str, Dict[int, float]] = field(default_factory=dict)


@dataclass
class RNGContainer:
    """Seeded RNGs aligned with the design doc."""

    main_seed: int
    ai_seed: int
    battle_seed: int

    main_rng: Optional[random.Random] = field(init=False, default=None)
    ai_rng: Optional[random.Random] = field(init=False, default=None)
    battle_rng: Optional[random.Random] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.main_rng = random.Random(self.main_seed)
        self.ai_rng = random.Random(self.ai_seed)
        self.battle_rng = random.Random(self.battle_seed)


@dataclass
class RacerState:
    """Mutable per-tick racer state, mirroring design doc structure."""

    profile: RacerProfile
    pos: float = 0.0
    world_position: Tuple[float, float] = (0.0, 0.0)
    current_speed: float = 0.0
    target_speed: float = 0.0
    accel: float = 0.0
    min_speed: float = 0.0
    current_hp: float = 0.0
    max_hp: float = 0.0
    lane_position: float = 0.0
    lateral_speed: float = 0.0
    target_lane: float = 0.0
    collider_radius: float = 0.75
    phase: str = "opening"
    section: int = 1
    visible_racers: List[int] = field(default_factory=list)
    visible_overtake_targets: List[int] = field(default_factory=list)
    overtake_targets_meta: List[Dict[str, float]] = field(default_factory=list)
    candidate_lanes: List[float] = field(default_factory=list)
    ai_mode: str = "normal"
    status_flags: Dict[str, bool] = field(
        default_factory=lambda: {
            "isLastSpurt": False,
            "isStartDash": False,
            "isBlockedFront": False,
            "isBlockedSide": False,
            "isBlockedSideLeft": False,
            "isBlockedSideRight": False,
            "isRailContact": False,
            "gritPush": False,
            "isFinished": False,
            "isExhausted": False,
            "isStartDelayed": False,
            "hasSpurtPlan": False,
            "isStaminaKeep": False,
        }
    )
    rng: Optional[RNGContainer] = None
    blocking_racer_id: Optional[int] = None
    blocking_speed: Optional[float] = None
    blocking_distance: float = 0.0
    blocking_timer: float = 0.0
    section_random_section: int = 0
    section_random_offset: float = 0.0
    force_in_roll: float = 0.0
    moved_last_tick: bool = False
    downhill_mode: bool = False
    overtake_timer: float = 0.0
    lane_mode: str = "normal"
    pk_mode: str = "normal"
    pk_timer: float = 0.0
    pk_cooldown: float = 0.0
    pk_check_timer: float = 0.0
    pk_threshold: float = 0.0
    pk_mode: str = "normal"
    pk_timer: float = 0.0
    pk_cooldown: float = 0.0
    pk_check_timer: float = 0.0
    debug_log: Dict[str, Any] = field(default_factory=dict)
    start_delay: float = 0.0
    start_delay_remaining: float = 0.0
    speed_noise_timer: float = 0.0
    speed_noise_value: float = 0.0
    spurt_multiplier: float = 1.0
    spurt_plan_committed: bool = False
    pacing_timer: float = 0.0
    pacing_speed_factor: float = 1.0
    battle_cooldowns: Dict[str, float] = field(default_factory=dict)
