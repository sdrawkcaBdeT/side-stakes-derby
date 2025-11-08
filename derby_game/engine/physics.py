from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

from derby_game.config import BALANCE_CONFIG

from .constants import HORSE_LANE_WIDTH, MAX_STAT_VALUE
from .data_models import AptitudeGrade, RacerProfile, RacerState, Strategy, Track

# --- Constants sourced from the design document ---

PHASE_ORDER = ("opening", "middle", "final")


class Phase(Enum):
    OPENING = 0
    MIDDLE = 1
    FINAL = 2

    @classmethod
    def from_string(cls, value: str) -> "Phase":
        value = value.lower()
        if value not in ("opening", "middle", "final"):
            raise ValueError(f"Unknown phase '{value}'")
        return cls(PHASE_ORDER.index(value))


def _race_engine_config() -> Dict[str, Dict[str, float]]:
    if not isinstance(BALANCE_CONFIG, dict):
        return {}
    config = BALANCE_CONFIG.get("race_engine")
    if not isinstance(config, dict):
        return {}
    return config


_RACE_ENGINE_CONFIG = _race_engine_config()
_PHASE_LABELS = {
    Phase.OPENING: ("opening", "early", "early_race"),
    Phase.MIDDLE: ("middle", "mid", "mid_race"),
    Phase.FINAL: ("final", "late", "late_race", "last_spurt"),
}


def _strategy_key_variants(strategy: Strategy) -> Tuple[str, ...]:
    base = strategy.name.lower()
    human = strategy.name.replace("_", " ").title()
    return (
        base,
        base.replace("_", ""),
        base.replace("_", " "),
        strategy.name,
        human,
        human.replace(" ", ""),
    )


def _config_strategy_table(
    config_key: str, default_table: Dict[Strategy, Dict[Phase, float]]
) -> Dict[Strategy, Dict[Phase, float]]:
    config_tables = _RACE_ENGINE_CONFIG.get(config_key, {}) if _RACE_ENGINE_CONFIG else {}
    result: Dict[Strategy, Dict[Phase, float]] = {}
    for strategy, fallback in default_table.items():
        entry = None
        if isinstance(config_tables, dict):
            for key in _strategy_key_variants(strategy):
                if key in config_tables:
                    entry = config_tables[key]
                    break
        strategy_map: Dict[Phase, float] = {}
        for phase, fallback_value in fallback.items():
            value = None
            if isinstance(entry, dict):
                for alias in _PHASE_LABELS[phase]:
                    if alias in entry:
                        value = entry[alias]
                        break
            strategy_map[phase] = float(value) if value is not None else fallback_value
        result[strategy] = strategy_map
    return result


DEFAULT_STRATEGY_SPEED_MOD = {
    Strategy.FRONT_RUNNER: {
        Phase.OPENING: 1.0,
        Phase.MIDDLE: 0.98,
        Phase.FINAL: 0.962,
    },
    Strategy.PACE_CHASER: {
        Phase.OPENING: 0.978,
        Phase.MIDDLE: 0.991,
        Phase.FINAL: 0.975,
    },
    Strategy.LATE_SURGER: {
        Phase.OPENING: 0.938,
        Phase.MIDDLE: 0.998,
        Phase.FINAL: 0.994,
    },
    Strategy.END_CLOSER: {
        Phase.OPENING: 0.931,
        Phase.MIDDLE: 1.0,
        Phase.FINAL: 1.0,
    },
}

STRATEGY_SPEED_MOD = _config_strategy_table("strategy_phase_speed", DEFAULT_STRATEGY_SPEED_MOD)

STRATEGY_HP_MOD = {
    Strategy.FRONT_RUNNER: 0.95,
    Strategy.PACE_CHASER: 0.89,
    Strategy.LATE_SURGER: 1.0,
    Strategy.END_CLOSER: 0.995,
}

DEFAULT_STRATEGY_ACCEL_MOD = {
    Strategy.FRONT_RUNNER: {Phase.OPENING: 1.0, Phase.MIDDLE: 1.0, Phase.FINAL: 0.996},
    Strategy.PACE_CHASER: {Phase.OPENING: 0.985, Phase.MIDDLE: 1.0, Phase.FINAL: 0.996},
    Strategy.LATE_SURGER: {Phase.OPENING: 0.975, Phase.MIDDLE: 1.0, Phase.FINAL: 1.0},
    Strategy.END_CLOSER: {Phase.OPENING: 0.945, Phase.MIDDLE: 1.0, Phase.FINAL: 0.997},
}

STRATEGY_ACCEL_MOD = _config_strategy_table("strategy_phase_accel", DEFAULT_STRATEGY_ACCEL_MOD)

PHASE_DECELERATION = {
    Phase.OPENING: -1.2,
    Phase.MIDDLE: -0.8,
    Phase.FINAL: -1.0,
}

DISTANCE_SPEED_MOD = [1.05, 1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1]
DISTANCE_ACCEL_MOD = [1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.5, 0.4]
SURFACE_ACCEL_MOD = [1.05, 1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1]

START_DASH_ACCEL_BONUS = 24.0
START_DASH_TARGET_FACTOR = 0.85

BASE_ACCEL = 0.0006
UPHILL_BASE_ACCEL = 0.0004

LAST_SPURT_GRIT_BONUS_COEF = 0.0001


@dataclass
class PhysicsContext:
    """Inputs that do not live on the RacerState but influence stepping."""

    phase: Phase
    section: int
    slope_gradient: float = 0.0
    pacing_modifier: float = 0.0
    skill_speed_bonus: float = 0.0
    skill_accel_bonus: float = 0.0
    segment_type: str = "straight"
    lane_position: float = 0.0
    external_speed_cap: Optional[float] = None
    curvature: float = 0.0
    move_lane_bonus: float = 0.0
    downhill_bonus: float = 0.0
    downhill_mode: bool = False


def _grade_index(grade: AptitudeGrade) -> int:
    order = [AptitudeGrade.S, AptitudeGrade.A, AptitudeGrade.B, AptitudeGrade.C, AptitudeGrade.D, AptitudeGrade.E, AptitudeGrade.F, AptitudeGrade.G]
    return order.index(grade)


def _distance_speed_factor(grade: AptitudeGrade) -> float:
    return DISTANCE_SPEED_MOD[_grade_index(grade)]


def _distance_accel_factor(grade: AptitudeGrade) -> float:
    return DISTANCE_ACCEL_MOD[_grade_index(grade)]


def _surface_accel_factor(grade: AptitudeGrade) -> float:
    return SURFACE_ACCEL_MOD[_grade_index(grade)]


def _base_course_speed(track: Track) -> float:
    return 20.0 - (track.distance - 2000.0) / 1000.0


class PhysicsKernel:
    """Implements the forward (1D) physics step."""

    def __init__(self, track: Track):
        self.track = track
        self.base_speed = _base_course_speed(track)
        self._lane_penalties = self._build_lane_penalties(track)

    def initialise_state(self, state: RacerState) -> None:
        """Populate HP-related fields for a newly spawned racer."""
        profile = state.profile
        strategy = profile.strategy
        stamina = profile.stats.stamina

        max_hp = 0.8 * STRATEGY_HP_MOD[strategy] * stamina + self.track.distance
        state.max_hp = max_hp
        if state.current_hp <= 0.0:
            state.current_hp = max_hp

        state.min_speed = self._calculate_min_speed(profile)

    def step(self, state: RacerState, context: PhysicsContext, dt: float) -> None:
        """Advance the racer's forward physics by dt seconds."""
        profile = state.profile
        stats = profile.stats
        phase = context.phase

        target_speed, target_components = self._calculate_target_speed(state, context)
        exhaustion_penalty = 0.0
        exhaustion_brake = 0.0
        hp_ratio = state.current_hp / state.max_hp if state.max_hp > 0 else 0.0
        exhausted = state.current_hp <= 0.0
        state.status_flags["isExhausted"] = exhausted
        if exhausted:
            fatigue_ratio = min(1.0, 1.0 - min(max(hp_ratio, 0.0), 1.0))
            penalty_factor = 0.35 + 0.35 * fatigue_ratio
            original_target = target_speed
            target_speed = max(state.min_speed, original_target * (1.0 - penalty_factor))
            exhaustion_penalty = target_speed - original_target
            target_components["exhaustion_penalty"] = exhaustion_penalty

        accel, accel_components = self._calculate_acceleration(state, target_speed, context)
        if exhausted:
            exhaustion_brake = 0.6 + 0.6 * (1.0 - min(max(hp_ratio, 0.0), 1.0))
            accel -= exhaustion_brake
            accel_components["exhaustion"] = -exhaustion_brake

        # Apply start dash boost if applicable
        if state.status_flags.get("isStartDash", False) and state.current_speed < START_DASH_TARGET_FACTOR * self.base_speed:
            accel += START_DASH_ACCEL_BONUS
        else:
            state.status_flags["isStartDash"] = False

        # Update speed using basic kinematics
        new_speed = state.current_speed + accel * dt

        # Clamp to minimum speed floor and external caps
        new_speed = max(new_speed, state.min_speed)
        if context.external_speed_cap is not None:
            new_speed = min(new_speed, context.external_speed_cap)
        if exhausted:
            exhausted_cap = max(state.min_speed * 0.4, target_speed * 0.85)
            new_speed = min(new_speed, exhausted_cap)

        state.target_speed = target_speed
        state.accel = accel
        state.current_speed = max(0.0, new_speed)

        # Drain stamina and track HP
        drain = self._calculate_stamina_drain(state, context, dt)
        state.current_hp = max(0.0, state.current_hp - drain)
        state.debug_log["target_components"] = target_components
        state.debug_log["accel_components"] = accel_components
        state.debug_log["stamina_drain"] = drain
        if exhausted:
            state.debug_log["exhaustion"] = {
                "penalty": exhaustion_penalty,
                "brake": -exhaustion_brake,
            }
        else:
            state.debug_log.pop("exhaustion", None)

    # --- Helpers ---------------------------------------------------------

    def _calculate_target_speed(self, state: RacerState, context: PhysicsContext) -> Tuple[float, Dict[str, float]]:
        profile = state.profile
        stats = profile.stats
        strategy = profile.strategy
        phase = context.phase

        distance_factor = _distance_speed_factor(profile.aptitudes.distance)
        base = self.base_speed * STRATEGY_SPEED_MOD[strategy][phase]
        strategy_bias = STRATEGY_PHASE_BIAS.get(strategy, {}).get(phase, 0.0) * self.base_speed

        final_leg_bonus = 0.0
        if phase is Phase.FINAL or state.status_flags.get("isLastSpurt", False):
            final_leg_bonus = math.sqrt(500.0 * stats.speed) * distance_factor * 0.002
            final_leg_bonus += math.pow(450.0 * stats.grit, 0.597) * LAST_SPURT_GRIT_BONUS_COEF
            if stats.stamina > 1200.0:
                final_leg_bonus += self._stamina_limit_break(stats.stamina, distance_factor)

        pacing_bonus = self.base_speed * context.pacing_modifier
        corner_penalty = 0.0
        if context.segment_type == "corner":
            corner_penalty = 0.02 * max(abs(context.lane_position) - 0.5, 0.0) * self.base_speed
        lane_penalty = self._lane_penalty(context.lane_position)
        lane_penalty_term = lane_penalty * self.base_speed * (1.4 if context.segment_type == "corner" else 1.0)

        total = (
            base
            + final_leg_bonus
            + pacing_bonus
            + context.skill_speed_bonus
            + context.move_lane_bonus
            + context.downhill_bonus
            + strategy_bias
            - corner_penalty
            - lane_penalty_term
        )
        total = max(0.0, total)
        components = {
            "base": base,
            "final_leg_bonus": final_leg_bonus,
            "pacing_bonus": pacing_bonus,
            "skills": context.skill_speed_bonus,
            "move_lane_bonus": context.move_lane_bonus,
            "downhill_bonus": context.downhill_bonus,
            "strategy_bias": strategy_bias,
            "corner_penalty": -corner_penalty,
            "lane_penalty": -lane_penalty_term,
        }
        components["strategy_multiplier"] = STRATEGY_SPEED_MOD[strategy][phase]
        multiplier = getattr(state, "spurt_multiplier", 1.0)
        if state.status_flags.get("isLastSpurt") and multiplier and multiplier != 1.0:
            total *= multiplier
            components["spurt_multiplier"] = multiplier

        pacing_factor = getattr(state, "pacing_speed_factor", 1.0)
        if pacing_factor != 1.0:
            total *= pacing_factor
            components["pacing_factor"] = pacing_factor
        return total, components

    def _calculate_acceleration(
        self,
        state: RacerState,
        target_speed: float,
        context: PhysicsContext,
    ) -> Tuple[float, Dict[str, float]]:
        profile = state.profile
        stats = profile.stats
        strategy = profile.strategy
        phase = context.phase

        if state.current_speed > target_speed:
            value = PHASE_DECELERATION[phase]
            return value, {"deceleration": value}

        base_value = UPHILL_BASE_ACCEL if context.slope_gradient > 0 else BASE_ACCEL
        accel = (
            base_value
            * math.sqrt(500.0 * stats.acceleration)
            * STRATEGY_ACCEL_MOD[strategy][phase]
            * _surface_accel_factor(profile.aptitudes.surface)
            * _distance_accel_factor(profile.aptitudes.distance)
        )

        if context.slope_gradient != 0.0:
            accel -= (context.slope_gradient / 10000.0) * 200.0 / max(stats.acceleration, 1.0)

        total = accel + context.skill_accel_bonus
        components = {
            "base": accel,
            "skills": context.skill_accel_bonus,
            "slope_penalty": (context.slope_gradient / 10000.0) * 200.0 / max(stats.acceleration, 1.0)
            if context.slope_gradient != 0.0
            else 0.0,
        }
        components["strategy_multiplier"] = STRATEGY_ACCEL_MOD[strategy][phase]

        return total, components

    def _calculate_stamina_drain(self, state: RacerState, context: PhysicsContext, dt: float) -> float:
        stats = state.profile.stats
        grit_mod = 1.0
        if context.phase is Phase.FINAL or state.status_flags.get("isLastSpurt", False):
            grit_mod = 1.0 + 200.0 / math.sqrt(max(600.0 * stats.grit, 1.0))

        focus_ratio = min(state.profile.stats.focus / MAX_STAT_VALUE, 1.0)
        focus_mod = 1.05 - 0.1 * focus_ratio

        speed_delta = max(state.current_speed - self.base_speed + 12.0, 0.0)
        drain_per_sec = 20.0 * (speed_delta ** 2) / 144.0 * grit_mod * focus_mod
        lane_penalty = self._lane_penalty(context.lane_position)
        lane_scale = 1.6 if context.segment_type == "corner" else 1.0
        drain_per_sec *= 1.0 + lane_penalty * 4.0 * lane_scale
        if context.downhill_mode:
            drain_per_sec *= 0.4

        return drain_per_sec * dt

    def _calculate_min_speed(self, profile: RacerProfile) -> float:
        stats = profile.stats
        return 0.85 * self.base_speed + math.sqrt(200.0 * stats.grit) * 0.001

    @staticmethod
    def _stamina_limit_break(stamina: float, distance_factor: float) -> float:
        return math.sqrt(stamina - 1200.0) * 0.0085 * distance_factor

    def _build_lane_penalties(self, track: Track) -> Dict[int, float]:
        penalties: Dict[int, float] = {}
        base_length = max(track.distance, 1.0)
        lane_deltas = getattr(track, "lane_length_deltas", {}).get("right", {})
        for index, delta in lane_deltas.items():
            if index < 0:
                continue
            penalties[index] = max(0.0, delta / base_length)
        return penalties

    def _lane_penalty(self, lane_position: float) -> float:
        if lane_position <= 0.0:
            return 0.0
        idx = int(round(lane_position))
        if idx <= 0:
            return 0.0
        penalty = self._lane_penalties.get(idx)
        if penalty is not None:
            return penalty
        track_width = max(self.track.track_width, HORSE_LANE_WIDTH)
        return max(0.0, lane_position * HORSE_LANE_WIDTH / track_width)
STRATEGY_PHASE_BIAS = {
    Strategy.FRONT_RUNNER: {
        Phase.OPENING: 0.018,
        Phase.MIDDLE: 0.0,
        Phase.FINAL: -0.015,
    },
    Strategy.PACE_CHASER: {
        Phase.OPENING: -0.01,
        Phase.MIDDLE: 0.012,
        Phase.FINAL: 0.0,
    },
    Strategy.LATE_SURGER: {
        Phase.OPENING: -0.02,
        Phase.MIDDLE: 0.0,
        Phase.FINAL: 0.018,
    },
    Strategy.END_CLOSER: {
        Phase.OPENING: -0.03,
        Phase.MIDDLE: -0.005,
        Phase.FINAL: 0.025,
    },
}
