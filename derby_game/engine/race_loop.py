from __future__ import annotations

import math
import random
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .course import slope_gradient_at
from .data_models import LaneGuide, RacerProfile, RacerState, RNGContainer, Strategy, Track
from .geometry import Polyline
from .constants import HORSE_LANE_WIDTH, MAX_STAT_VALUE
from .physics import Phase, PhysicsContext, PhysicsKernel
from .spatial import SpatialEngine
from .telemetry import TelemetryCollector, TelemetryFrame, TelemetryRacerFrame

PACING_COMFORT = {
    Strategy.PACE_CHASER: (3.0, 5.0),
    Strategy.LATE_SURGER: (6.5, 7.0),
    Strategy.END_CLOSER: (7.5, 8.0),
}

FRONTRUNNER_GAP = (1.0, 6.0)
STRATEGY_COMPETE_COEF = {
    Strategy.FRONT_RUNNER: 1.0,
    Strategy.PACE_CHASER: 1.05,
    Strategy.LATE_SURGER: 1.1,
    Strategy.END_CLOSER: 1.15,
}

FORCE_IN_MODIFIERS = {
    Strategy.FRONT_RUNNER: 0.02,
    Strategy.PACE_CHASER: 0.01,
    Strategy.LATE_SURGER: 0.01,
    Strategy.END_CLOSER: 0.03,
}

DISABLE_POSITION_KEEPING = os.getenv("DERBY_DISABLE_POSITION_KEEP", "0") == "1"

PK_SPEED_FACTORS = {
    "speed_up": 1.08,
    "overtake": 1.06,
    "pace_up": 1.06,
    "pace_down_open": 0.90,
    "pace_down_mid": 0.93,
    "pace_up_ex": 2.0,
}

PK_DURATION_SCALE = {
    "speed_up": 0.8,
    "overtake": 1.0,
    "pace_up": 1.0,
    "pace_down": 1.2,
    "pace_up_ex": 0.6,
}

PK_SECTION_LIMIT = 10

MACRO_PACING_BONUS = {
    "pace_up": 0.04,
    "pace_up_ex": 0.05,
    "pace_down": -0.07,
    "stamina_keep": -0.08,
    "push_lead": 0.025,
    "relax_lead": -0.03,
    "chase_lead": 0.035,
    "normal": 0.0,
}

START_LANE_SPACING_LANES = 2.5 / HORSE_LANE_WIDTH


@dataclass
class TickRacerSnapshot:
    racer_id: int
    name: str
    pos: float
    speed: float
    hp: float
    lane: float
    section: int
    phase: Phase
    is_finished: bool


@dataclass
class TickSnapshot:
    time: float
    racers: List[TickRacerSnapshot] = field(default_factory=list)


@dataclass
class BattleResult:
    speed_bonus: float = 0.0
    accel_bonus: float = 0.0
    stamina_multiplier: float = 1.0


class HybridRaceLoop:
    """Thin orchestration layer combining the physics kernel with spline geometry."""

    def __init__(
        self,
        track: Track,
        racers: Sequence[RacerProfile],
        dt: float = 1.0 / 15.0,
        telemetry: Optional[TelemetryCollector] = None,
        rng_seed: int = 42,
    ) -> None:
        if not track.spline.points:
            raise ValueError("Track spline must provide sampled points.")

        self.track = track
        self.dt = dt
        self.physics = PhysicsKernel(track)
        self._polyline = Polyline.from_points(track.spline.points)
        self._outer_polyline = (
            Polyline.from_points(track.boundary_outer.points) if track.boundary_outer else None
        )
        self._lane_polylines = self._build_lane_polylines(track.lane_guides)
        self._normal_sign = self._determine_normal_sign()
        self._course_factor = max(0.5, 0.0008 * (max(self.track.distance, 0.0) - 1000.0) + 1.0)
        self.spatial = SpatialEngine(track)
        self.telemetry = telemetry
        self.tick_index = 0
        self._results_persisted = False

        self._states: List[RacerState] = []
        base_seed = rng_seed

        total_racers = len(racers)
        collider_lanes_default = 0.75 / HORSE_LANE_WIDTH
        default_spacing = max(1.0, (2.0 * collider_lanes_default) + 0.5)
        lane_spacing = default_spacing
        if total_racers > 1:
            max_span = max(1e-3, self.spatial.max_lane_position)
            lane_spacing = min(default_spacing, max_span / (total_racers - 1))

        for idx, profile in enumerate(racers):
            state = RacerState(profile=profile)
            rng = RNGContainer(
                main_seed=base_seed + idx * 11 + 1,
                ai_seed=base_seed + idx * 11 + 2,
                battle_seed=base_seed + idx * 11 + 3,
            )
            state.rng = rng
            state.status_flags["isStartDash"] = True
            focus_ratio = min(state.profile.stats.focus / MAX_STAT_VALUE, 1.0)
            max_delay = max(0.0, 0.15 * (1.0 - 0.75 * focus_ratio))
            min_delay = max(0.0, max_delay * 0.25)
            delay_span = max(0.0, max_delay - min_delay)
            roll = rng.main_rng.random() if rng.main_rng else random.random()
            state.start_delay = min_delay + delay_span * roll
            state.start_delay_remaining = state.start_delay
            base_force = FORCE_IN_MODIFIERS.get(profile.strategy, 0.01)
            random_force = (rng.main_rng.random() if rng and rng.main_rng else 0.05) * 0.1
            state.force_in_roll = random_force + base_force
            collider_lanes = state.collider_radius / HORSE_LANE_WIDTH
            lane_position = idx * START_LANE_SPACING_LANES
            max_lane = max(0.0, self.spatial.max_lane_position - collider_lanes - 0.25)
            lane_position = max(0.0, min(max_lane, lane_position))
            state.lane_position = lane_position
            state.target_lane = lane_position
            self.physics.initialise_state(state)
            self._states.append(state)

        self.time_elapsed = 0.0
        self.finished_order: List[int] = []
        self._section_length = track.distance / 24.0 if track.distance > 0 else 0.0

    @property
    def states(self) -> Sequence[RacerState]:
        return self._states

    def tick(self) -> TickSnapshot:
        ordered_states = sorted(self._states, key=lambda s: s.pos, reverse=True)
        lead_state = ordered_states[0] if ordered_states else None
        lead_pos = lead_state.pos if lead_state else 0.0
        second_pos = ordered_states[1].pos if len(ordered_states) > 1 else lead_pos
        lead_ratio = lead_pos / self.track.distance if self.track.distance else 0.0

        collect = self.telemetry is not None
        previous_hp = {state.profile.racer_id: state.current_hp for state in self._states} if collect else {}
        previous_positions = {state.profile.racer_id: state.pos for state in self._states} if collect else {}
        previous_speeds = {state.profile.racer_id: state.current_speed for state in self._states} if collect else {}

        for rank, state in enumerate(ordered_states, start=1):
            state.status_flags["order"] = rank
            state.status_flags["distanceToLead"] = lead_pos - state.pos

        global_phase = self._phase_from_ratio(lead_ratio)

        self.spatial.update_perception(self._states)
        self.spatial.update_ai_targets(self._states, self.dt)
        self._decay_battle_cooldowns(self._states)

        for state in ordered_states:
            if state.status_flags.get("isFinished"):
                continue

            racer_ratio = state.pos / self.track.distance if self.track.distance else 0.0
            racer_phase = self._phase_from_ratio(racer_ratio)
            if global_phase is Phase.FINAL:
                state.status_flags["isLastSpurt"] = True
                self._ensure_spurt_plan(state)

            segment_type = self._segment_type(state.pos)
            section_index = self._section_for(state.pos)
            pacing_modifier = self._compute_pacing_modifier(state, lead_pos, second_pos, ordered_states, global_phase)
            if self._apply_start_delay(state):
                state.section = section_index
                state.phase = racer_phase.name.lower()
                state.status_flags["segmentType"] = segment_type
                state.world_position = self._world_position(state.pos, state.lane_position)
                continue

            battle = self._update_battles(state, ordered_states, segment_type, global_phase)
            noise_bonus = self._speed_noise(state)
            section_bonus = self._section_random_bonus(state, section_index)
            force_bonus = self._force_in_bonus(state, racer_phase, state.lane_position)
            move_lane_bonus = self._move_lane_bonus(state)
            slope_gradient = slope_gradient_at(self.track, state.pos)
            downhill_bonus, downhill_mode = self._update_downhill_state(state, slope_gradient)

            context = PhysicsContext(
                phase=racer_phase,
                section=section_index,
                slope_gradient=slope_gradient,
                pacing_modifier=pacing_modifier,
                segment_type=segment_type,
                lane_position=state.lane_position,
                external_speed_cap=self.spatial.speed_cap(state),
                skill_speed_bonus=noise_bonus + battle.speed_bonus + section_bonus + force_bonus + downhill_bonus,
                skill_accel_bonus=battle.accel_bonus,
                curvature=self._curvature_at(state.pos),
                move_lane_bonus=move_lane_bonus,
                downhill_bonus=downhill_bonus,
                downhill_mode=downhill_mode,
            )

            self.physics.step(state, context, self.dt)
            if battle.stamina_multiplier > 1.0:
                base_drain = state.debug_log.get("stamina_drain", 0.0)
                extra = base_drain * (battle.stamina_multiplier - 1.0)
                if extra > 0.0:
                    state.current_hp = max(0.0, state.current_hp - extra)
                    state.debug_log.setdefault("events", []).append(f"battle_drain(+{extra:.2f})")
            self.spatial.step_lateral(state, self.dt)

            state.pos += state.current_speed * self.dt
            if state.pos >= self.track.distance:
                state.pos = self.track.distance
                state.status_flags["isFinished"] = True
                if state.profile.racer_id not in self.finished_order:
                    self.finished_order.append(state.profile.racer_id)

            state.section = self._section_for(state.pos)
            state.phase = racer_phase.name.lower()
            state.status_flags["segmentType"] = segment_type
            state.world_position = self._world_position(state.pos, state.lane_position)

        self.spatial.resolve_overlaps(self._states)
        for state in self._states:
            state.world_position = self._world_position(state.pos, state.lane_position)

        if collect:
            racer_frames: List[TelemetryRacerFrame] = []
            for state in ordered_states:
                prev_hp = previous_hp.get(state.profile.racer_id, state.current_hp)
                prev_pos = previous_positions.get(state.profile.racer_id, state.pos)
                distance_delta = state.pos - prev_pos
                prev_speed = previous_speeds.get(state.profile.racer_id, state.current_speed)
                speed_delta = state.current_speed - prev_speed
                stamina_drain = state.debug_log.get("stamina_drain", max(0.0, prev_hp - state.current_hp))
                state.debug_log["speed_delta"] = speed_delta
                racer_frames.append(
                    TelemetryRacerFrame(
                        racer_id=state.profile.racer_id,
                        name=state.profile.name,
                        lane_position=state.lane_position,
                        world_position=self._world_position(state.pos, state.lane_position),
                        pos=state.pos,
                        distance_delta=distance_delta,
                        speed_delta=speed_delta,
                        speed=state.current_speed,
                        target_speed=state.target_speed,
                        accel=state.accel,
                        hp=state.current_hp,
                        hp_delta=state.current_hp - prev_hp,
                        stamina_drain=stamina_drain,
                        ai_mode=state.ai_mode,
                        status_flags=dict(state.status_flags),
                        debug=dict(state.debug_log),
                    )
                )
                state.debug_log.clear()

            frame_time = self.time_elapsed + self.dt
            leading_segment = lead_state.status_flags.get("segmentType", "straight") if lead_state else "straight"
            self.telemetry.record_frame(
                TelemetryFrame(
                    tick=self.tick_index,
                    time=frame_time,
                    phase=global_phase.name.lower(),
                    leading_segment=leading_segment,
                    racers=racer_frames,
                )
            )
            self.tick_index += 1
            self.time_elapsed = frame_time
        else:
            for state in ordered_states:
                state.debug_log.clear()
            self.time_elapsed += self.dt

        return self._snapshot()

    def run_until_finished(
        self,
        on_tick: Optional[Callable[[TickSnapshot], None]] = None,
        max_time: float = 600.0,
    ) -> List[TickSnapshot]:
        snapshots: List[TickSnapshot] = []

        max_ticks = int(max_time / self.dt) if self.dt > 0 else 0
        for _ in range(max_ticks):
            snapshot = self.tick()
            snapshots.append(snapshot)
            if on_tick:
                on_tick(snapshot)
            if all(state.status_flags.get("isFinished") for state in self._states):
                break

        return snapshots

    def _snapshot(self) -> TickSnapshot:
        racers = [
            TickRacerSnapshot(
                racer_id=state.profile.racer_id,
                name=state.profile.name,
                pos=state.pos,
                speed=state.current_speed,
                hp=state.current_hp,
                lane=state.lane_position,
                section=state.section,
                phase=Phase.from_string(state.phase) if isinstance(state.phase, str) else Phase(state.phase),
                is_finished=bool(state.status_flags.get("isFinished")),
            )
            for state in self._states
        ]
        return TickSnapshot(time=self.time_elapsed, racers=racers)

    def _build_lane_polylines(self, guides: Sequence[LaneGuide]) -> Dict[str, Dict[int, Polyline]]:
        lane_map: Dict[str, Dict[int, Polyline]] = {}
        for guide in guides:
            side_map = lane_map.setdefault(guide.side, {})
            side_map[guide.index] = Polyline.from_points(guide.spline.points)
        return lane_map

    def _determine_normal_sign(self) -> float:
        if self.track.distance <= 0:
            return 1.0

        sample_positions = [
            max(0.0, min(self.track.distance * ratio, self.track.distance - 1e-6))
            for ratio in (0.1, 0.5, 0.9)
        ]

        sign = 1.0
        for pos in sample_positions:
            base = self._polyline.position_at(pos)
            normal = self._polyline.normal_at(pos)
            if self._outer_polyline is not None:
                ratio = pos / self.track.distance
                outer_point = self._sample_polyline(self._outer_polyline, ratio)
                vector = (outer_point[0] - base[0], outer_point[1] - base[1])
                dot = normal[0] * vector[0] + normal[1] * vector[1]
                if abs(dot) > 1e-6:
                    sign = 1.0 if dot > 0 else -1.0
                    return self._verify_normal_sign(sign)

            for side in ("right", "left"):
                side_map = self._lane_polylines.get(side)
                if not side_map:
                    continue
                lane_poly = side_map[min(side_map.keys())]
                ratio = pos / self.track.distance
                lane_point = self._sample_polyline(lane_poly, ratio)
                vector = (lane_point[0] - base[0], lane_point[1] - base[1])
                dot = normal[0] * vector[0] + normal[1] * vector[1]
                if abs(dot) > 1e-6:
                    if side == "right":
                        sign = 1.0 if dot > 0 else -1.0
                        return self._verify_normal_sign(sign)
                    else:
                        sign = -1.0 if dot > 0 else 1.0
                        return self._verify_normal_sign(sign)

        return self._verify_normal_sign(sign)

    def _verify_normal_sign(self, sign: float) -> float:
        boundary_inner = self.track.boundary_inner.points if self.track.boundary_inner else None
        boundary_outer = self.track.boundary_outer.points if self.track.boundary_outer else None
        if not boundary_inner and not boundary_outer:
            return sign

        sample_positions = [
            max(0.0, min(self.track.distance * ratio, self.track.distance - 1e-6))
            for ratio in (0.15, 0.45, 0.75)
        ]

        for pos in sample_positions:
            base = self._polyline.position_at(pos)
            normal = self._polyline.normal_at(pos)
            offset_point = (
                base[0] + normal[0] * HORSE_LANE_WIDTH * sign,
                base[1] + normal[1] * HORSE_LANE_WIDTH * sign,
            )

            if boundary_inner:
                base_dist = self._distance_to_boundary(base, boundary_inner)
                offset_dist = self._distance_to_boundary(offset_point, boundary_inner)
                if offset_dist + 1e-6 < base_dist:
                    return -sign
            elif boundary_outer:
                base_dist = self._distance_to_boundary(base, boundary_outer)
                offset_dist = self._distance_to_boundary(offset_point, boundary_outer)
                if offset_dist > base_dist + 1e-6:
                    return -sign

        return sign

    @staticmethod
    def _distance_to_boundary(point: Tuple[float, float], boundary: Sequence[Tuple[float, float]]) -> float:
        px, py = point
        best = float("inf")
        for idx in range(1, len(boundary)):
            ax, ay = boundary[idx - 1]
            bx, by = boundary[idx]
            apx = px - ax
            apy = py - ay
            abx = bx - ax
            aby = by - ay
            ab_len_sq = abx * abx + aby * aby
            if ab_len_sq == 0.0:
                best = min(best, math.hypot(apx, apy))
                continue
            t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len_sq))
            cx = ax + abx * t
            cy = ay + aby * t
            best = min(best, math.hypot(px - cx, py - cy))
        return best

    def _sample_polyline(self, polyline: Polyline, ratio: float) -> Tuple[float, float]:
        ratio_clamped = max(0.0, min(1.0, ratio))
        distance = polyline.length * ratio_clamped
        return polyline.position_at(distance)

    def _world_position(self, pos: float, lane_position: float) -> Tuple[float, float]:
        base = self._polyline.position_at(pos)
        offset = lane_position * HORSE_LANE_WIDTH
        if abs(offset) < 1e-6:
            return base

        normal = self._polyline.normal_at(pos)
        normal = (normal[0] * self._normal_sign, normal[1] * self._normal_sign)
        return (base[0] + normal[0] * offset, base[1] + normal[1] * offset)

    def _segment_type(self, pos: float) -> str:
        for corner in self.track.corners:
            if corner.start_pos <= pos <= corner.end_pos:
                return "corner"
        return "straight"

    def _compute_pacing_modifier(
        self,
        state: RacerState,
        lead_pos: float,
        second_pos: float,
        ordered_states: Sequence[RacerState],
        global_phase: Phase,
    ) -> float:
        macro_modifier = self._update_macro_pacing_state(state, lead_pos, second_pos, ordered_states, global_phase)
        self._update_position_keep_state(state, lead_pos, second_pos, ordered_states, global_phase)
        return macro_modifier

    def _update_macro_pacing_state(
        self,
        state: RacerState,
        lead_pos: float,
        second_pos: float,
        ordered_states: Sequence[RacerState],
        global_phase: Phase,
    ) -> float:
        state.pacing_timer = max(0.0, state.pacing_timer - self.dt)
        additive = 0.0

        if global_phase is Phase.MIDDLE and not state.status_flags.get("isStaminaKeep"):
            hp_ratio = state.current_hp / state.max_hp if state.max_hp > 0 else 0.0
            if hp_ratio < 0.4:
                cognition = max(state.profile.stats.cognition, 1.0)
                chance = 0.30 * (cognition / 1000.0 + math.pow(cognition, 0.03))
                chance = min(0.9, chance)
                rng = state.rng.ai_rng if state.rng else None
                roll = rng.random() if rng else 0.5
                if roll < chance:
                    state.status_flags["isStaminaKeep"] = True
                    state.ai_mode = "stamina_keep"
                    state.pacing_speed_factor = 0.92
                    state.debug_log.setdefault("events", []).append("stamina_keep")

        if state.status_flags.get("isStaminaKeep"):
            state.pacing_speed_factor = min(state.pacing_speed_factor, 0.92)
            additive = MACRO_PACING_BONUS.get("stamina_keep", -0.08)
            state.debug_log["pacing_state"] = {
                "mode": state.ai_mode,
                "factor": state.pacing_speed_factor,
                "macro_modifier": additive,
            }
            return additive

        if state.pacing_timer <= 0.0:
            self._evaluate_comfort_zone(state, lead_pos, second_pos, ordered_states, global_phase)
            state.pacing_timer = 2.0

        additive = MACRO_PACING_BONUS.get(state.ai_mode, 0.0)

        state.debug_log["pacing_state"] = {
            "mode": state.ai_mode,
            "factor": state.pacing_speed_factor,
            "macro_modifier": additive,
        }
        return additive

    def _update_position_keep_state(
        self,
        state: RacerState,
        lead_pos: float,
        second_pos: float,
        ordered_states: Sequence[RacerState],
        global_phase: Phase,
    ) -> None:
        if DISABLE_POSITION_KEEPING:
            state.pk_mode = "debug_disabled"
            state.pacing_speed_factor = 1.0
            state.debug_log["pk_mode"] = "debug_disabled"
            return
        state.pk_timer = max(0.0, state.pk_timer - self.dt)
        state.pk_cooldown = max(0.0, state.pk_cooldown - self.dt)
        state.pk_check_timer = max(0.0, state.pk_check_timer - self.dt)
        if state.pk_mode != "normal":
            if self._should_exit_pk_mode(state, lead_pos, second_pos, ordered_states):
                self._reset_pk_mode(state)
            elif state.pk_timer <= 0.0:
                self._reset_pk_mode(state)
        if state.pk_mode == "normal" and state.pk_cooldown <= 0.0 and state.pk_check_timer <= 0.0:
            self._try_enter_pk_mode(state, lead_pos, second_pos, ordered_states, global_phase)
            state.pk_check_timer = 2.0
        state.debug_log["pk_mode"] = state.pk_mode
        return

    def _try_enter_pk_mode(
        self,
        state: RacerState,
        lead_pos: float,
        second_pos: float,
        ordered_states: Sequence[RacerState],
        global_phase: Phase,
    ) -> None:
        if self._enter_pk_ex_mode(state, ordered_states):
            return
        if state.profile.strategy == Strategy.FRONT_RUNNER:
            if self._enter_front_speed_up(state, second_pos):
                return
            if self._enter_front_overtake(state, ordered_states):
                return
        else:
            if self._enter_pace_down(state, lead_pos, global_phase):
                return
            if self._enter_pace_up(state, lead_pos, global_phase):
                return

    def _enter_pk_mode(
        self,
        state: RacerState,
        mode: str,
        factor: float,
        threshold: float = 0.0,
        duration_scale: float = 1.0,
    ) -> None:
        state.pk_mode = mode
        state.pacing_speed_factor = factor
        base_duration = self._estimated_section_time(state) * duration_scale
        state.pk_timer = max(0.5, base_duration)
        state.pk_threshold = threshold

    def _reset_pk_mode(self, state: RacerState) -> None:
        state.pk_mode = "normal"
        state.pacing_speed_factor = 1.0
        state.pk_timer = 0.0
        state.pk_threshold = 0.0
        state.pk_cooldown = 1.5

    def _estimated_section_time(self, state: RacerState) -> float:
        speed = max(state.current_speed, self.physics.base_speed, 8.0)
        return (self._section_length or 1.0) / speed

    def _should_exit_pk_mode(
        self,
        state: RacerState,
        lead_pos: float,
        second_pos: float,
        ordered_states: Sequence[RacerState],
    ) -> bool:
        if state.pk_mode == "speed_up":
            gap = (state.pos - second_pos) if len(ordered_states) > 1 else FRONTRUNNER_GAP[1]
            return gap >= 12.5
        if state.pk_mode == "overtake":
            ahead_same = self._distance_to_next_same_strategy(state, ordered_states)
            return ahead_same is not None and ahead_same >= 10.0
        if state.pk_mode == "pace_up":
            gap = lead_pos - state.pos
            return gap <= state.pk_threshold
        if state.pk_mode == "pace_down":
            gap = lead_pos - state.pos
            return gap >= state.pk_threshold
        if state.pk_mode == "pace_up_ex":
            return not self._ex_condition(state, ordered_states)
        return False

    def _enter_front_speed_up(self, state: RacerState, second_pos: float) -> bool:
        if state.status_flags.get("order") != 1:
            return False
        gap = (state.pos - second_pos) if second_pos else 0.0
        if gap >= 12.5:
            return False
        if not self._wisdom_roll(state, 20.0):
            return False
        self._enter_pk_mode(
            state,
            "speed_up",
            PK_SPEED_FACTORS["speed_up"],
            duration_scale=PK_DURATION_SCALE["speed_up"],
        )
        return True

    def _enter_front_overtake(self, state: RacerState, ordered_states: Sequence[RacerState]) -> bool:
        same_ahead = any(
            other.profile.strategy == state.profile.strategy and other.status_flags.get("order", 0) < state.status_flags.get("order", 0)
            for other in ordered_states
        )
        if not same_ahead:
            return False
        if not self._wisdom_roll(state, 20.0):
            return False
        self._enter_pk_mode(
            state,
            "overtake",
            PK_SPEED_FACTORS["overtake"],
            duration_scale=PK_DURATION_SCALE["overtake"],
        )
        return True

    def _enter_pace_up(self, state: RacerState, lead_pos: float, global_phase: Phase) -> bool:
        if state.section > PK_SECTION_LIMIT:
            return False
        min_gap, max_gap = self._strategy_gaps(state.profile.strategy)
        if min_gap is None:
            return False
        gap = lead_pos - state.pos
        if gap <= max_gap:
            return False
        if not self._phase_weighted_roll(state, base_scale=15.0, phase=global_phase):
            return False
        threshold = self._random_between(min_gap, max_gap, state)
        self._enter_pk_mode(
            state,
            "pace_up",
            PK_SPEED_FACTORS["pace_up"],
            threshold,
            duration_scale=PK_DURATION_SCALE["pace_up"],
        )
        return True

    def _enter_pace_down(self, state: RacerState, lead_pos: float, global_phase: Phase) -> bool:
        if state.section > PK_SECTION_LIMIT:
            return False
        min_gap, max_gap = self._strategy_gaps(state.profile.strategy)
        if min_gap is None:
            return False
        gap = lead_pos - state.pos
        if gap >= min_gap:
            return False
        threshold_max = max_gap if global_phase is not Phase.MIDDLE else (min_gap + max_gap) / 2.0
        threshold = self._random_between(min_gap, threshold_max, state)
        factor = (
            PK_SPEED_FACTORS["pace_down_mid"] if global_phase is Phase.MIDDLE else PK_SPEED_FACTORS["pace_down_open"]
        )
        self._enter_pk_mode(
            state,
            "pace_down",
            factor,
            threshold,
            duration_scale=PK_DURATION_SCALE["pace_down"],
        )
        return True

    def _enter_pk_ex_mode(self, state: RacerState, ordered_states: Sequence[RacerState]) -> bool:
        if not self._ex_condition(state, ordered_states):
            return False
        self._enter_pk_mode(
            state,
            "pace_up_ex",
            PK_SPEED_FACTORS["pace_up_ex"],
            duration_scale=PK_DURATION_SCALE["pace_up_ex"],
        )
        return True

    def _ex_condition(self, state: RacerState, ordered_states: Sequence[RacerState]) -> bool:
        order_map = {
            Strategy.FRONT_RUNNER: 0,
            Strategy.PACE_CHASER: 1,
            Strategy.LATE_SURGER: 2,
            Strategy.END_CLOSER: 3,
        }
        state_rank = order_map.get(state.profile.strategy, 3)
        for other in ordered_states:
            if other is state:
                continue
            other_rank = order_map.get(other.profile.strategy, 3)
            if state_rank <= other_rank:
                continue
            if other.pos > state.pos:
                return True
        if state_rank > 0:
            pacemaker = ordered_states[0]
            pacer_rank = order_map.get(pacemaker.profile.strategy, 3)
            if pacer_rank < state_rank:
                return True
        return False

    def _strategy_gaps(self, strategy: Strategy) -> Tuple[Optional[float], Optional[float]]:
        base = PACING_COMFORT.get(strategy)
        if not base:
            return (None, None)
        min_gap = base[0] * self._course_factor
        max_gap = base[1] * self._course_factor
        return (min_gap, max_gap)

    def _distance_to_next_same_strategy(self, state: RacerState, ordered_states: Sequence[RacerState]) -> Optional[float]:
        same = [
            other for other in ordered_states if other.profile.strategy == state.profile.strategy and other is not state
        ]
        ahead = [other for other in same if other.pos > state.pos]
        if not ahead:
            return None
        return ahead[0].pos - state.pos

    def _wisdom_roll(self, state: RacerState, scale: float) -> bool:
        cognition = state.profile.stats.cognition
        value = max(cognition * 0.1, 1.0)
        chance = scale * math.log10(value)
        chance = max(0.0, min(100.0, chance))
        rng = state.rng.ai_rng if state.rng else None
        roll = rng.random() if rng else random.random()
        return roll * 100.0 < chance

    def _phase_weighted_roll(self, state: RacerState, base_scale: float, phase: Phase) -> bool:
        phase_bias = {
            Phase.OPENING: 1.0,
            Phase.MIDDLE: 0.65,
            Phase.FINAL: 0.0,
        }[phase]
        if phase_bias <= 0.0:
            return False
        return self._wisdom_roll(state, base_scale * phase_bias)

    def _random_between(self, low: float, high: float, state: RacerState) -> float:
        rng = state.rng.ai_rng if state.rng else None
        roll = rng.random() if rng else random.random()
        return low + (high - low) * roll

    def _evaluate_comfort_zone(
        self,
        state: RacerState,
        lead_pos: float,
        second_pos: float,
        ordered_states: Sequence[RacerState],
        global_phase: Phase,
    ) -> None:
        strategy = state.profile.strategy
        if strategy == Strategy.FRONT_RUNNER:
            if state.status_flags.get("order", 0) == 1 and len(ordered_states) > 1:
                gap_to_second = state.pos - second_pos
                if gap_to_second < 1.5:
                    state.ai_mode = "push_lead"
                    state.pacing_speed_factor = 1.02
                elif gap_to_second > 6.0:
                    state.ai_mode = "relax_lead"
                    state.pacing_speed_factor = 0.98
                else:
                    state.ai_mode = "normal"
                    state.pacing_speed_factor = 1.0
            else:
                state.ai_mode = "chase_lead"
                state.pacing_speed_factor = 1.02
            return

        if state.section > 10:
            state.ai_mode = "normal"
            state.pacing_speed_factor = 1.0
            return

        gaps = PACING_COMFORT.get(strategy)
        if not gaps:
            state.ai_mode = "normal"
            state.pacing_speed_factor = 1.0
            return

        min_gap = gaps[0] * self._course_factor
        max_gap = gaps[1] * self._course_factor
        gap_to_lead = lead_pos - state.pos

        if gap_to_lead < min_gap:
                state.ai_mode = "pace_down"
                multiplier = 0.93 if global_phase is Phase.MIDDLE else 0.90
                state.pacing_speed_factor = multiplier
                state.debug_log.setdefault("events", []).append("pace_down")
                return

        if gap_to_lead > max_gap:
            cognition = max(state.profile.stats.cognition, 1.0)
            chance = 0.15 * math.log10(cognition * 0.1)
            chance = max(0.0, min(0.9, chance))
            rng = state.rng.ai_rng if state.rng else None
            roll = rng.random() if rng else 0.5
            if roll < chance:
                state.ai_mode = "pace_up"
                state.pacing_speed_factor = 1.06
                state.debug_log.setdefault("events", []).append("pace_up")
                return

        state.ai_mode = "normal"
        state.pacing_speed_factor = 1.0

    def _apply_start_delay(self, state: RacerState) -> bool:
        if state.start_delay_remaining > 0.0:
            state.start_delay_remaining = max(0.0, state.start_delay_remaining - self.dt)
            state.current_speed = 0.0
            state.target_speed = 0.0
            state.accel = 0.0
            state.status_flags["isStartDelayed"] = True
            state.debug_log["start_delay_remaining"] = state.start_delay_remaining
            return True
        state.status_flags["isStartDelayed"] = False
        state.debug_log.pop("start_delay_remaining", None)
        return False

    def _decay_battle_cooldowns(self, states: Sequence[RacerState]) -> None:
        for state in states:
            if not state.battle_cooldowns:
                continue
            expired = []
            for key, value in state.battle_cooldowns.items():
                value -= self.dt
                if value <= 0.0:
                    expired.append(key)
                else:
                    state.battle_cooldowns[key] = value
            for key in expired:
                state.battle_cooldowns.pop(key, None)

    def _speed_noise(self, state: RacerState) -> float:
        focus_ratio = min(state.profile.stats.focus / MAX_STAT_VALUE, 1.0)
        cognition_ratio = min(state.profile.stats.cognition / MAX_STAT_VALUE, 1.0)
        state.speed_noise_timer -= self.dt
        if state.speed_noise_timer <= 0.0:
            amplitude = (1.0 - focus_ratio * 0.7) * (1.0 - cognition_ratio * 0.4)
            rng = state.rng.main_rng if state.rng else None
            roll = (rng.random() if rng else 0.5) * 2.0 - 1.0
            state.speed_noise_value = roll * amplitude
            base_interval = 0.45 + (1.0 - focus_ratio) * 0.4
            state.speed_noise_timer = base_interval
        return state.speed_noise_value

    def _ensure_spurt_plan(self, state: RacerState) -> None:
        if state.spurt_plan_committed:
            return
        hp_ratio = state.current_hp / state.max_hp if state.max_hp > 0 else 0.0
        cognition = max(state.profile.stats.cognition, 1.0)
        rng = state.rng.ai_rng if state.rng else None
        plan_specs = [
            {"multiplier": 1.10, "hp_req": 0.6, "label": "burst"},
            {"multiplier": 1.06, "hp_req": 0.45, "label": "push"},
            {"multiplier": 1.02, "hp_req": 0.25, "label": "steady"},
            {"multiplier": 0.98, "hp_req": 0.1, "label": "conserve"},
            {"multiplier": 0.92, "hp_req": 0.0, "label": "crawl"},
        ]
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        for spec in plan_specs:
            if hp_ratio < spec["hp_req"]:
                continue
            eta = self._estimate_spurt_eta(state, spec["multiplier"])
            candidates.append((eta, spec))
        if not candidates:
            fallback = plan_specs[-1]
            candidates.append((self._estimate_spurt_eta(state, fallback["multiplier"]), fallback))
        candidates.sort(key=lambda item: item[0])

        accept_chance = min(0.95, max(0.1, (15.0 + 0.05 * cognition) / 100.0))
        chosen_spec = candidates[-1][1]
        for _, spec in candidates:
            roll = rng.random() if rng else 0.5
            if roll <= accept_chance:
                chosen_spec = spec
                break
            accept_chance *= 0.82

        state.spurt_multiplier = chosen_spec["multiplier"]
        state.spurt_plan_committed = True
        state.status_flags["hasSpurtPlan"] = True
        state.debug_log.setdefault("events", []).append(
            f"spurt_plan:{chosen_spec['multiplier']:.3f}:{chosen_spec.get('label', 'plan')}"
        )

    def _section_random_bonus(self, state: RacerState, section: int) -> float:
        if state.section_random_section == section:
            return state.section_random_offset
        wiz = max(state.profile.stats.cognition, 1.0)
        min_pct, max_pct = self._wisdom_random_bounds(wiz)
        rng = state.rng.main_rng if state.rng else None
        roll = rng.random() if rng else 0.5
        pct = min_pct + (max_pct - min_pct) * roll
        bonus = self.physics.base_speed * (pct / 100.0)
        state.section_random_section = section
        state.section_random_offset = bonus
        return bonus

    @staticmethod
    def _wisdom_random_bounds(wisdom: float) -> Tuple[float, float]:
        ratio = max(0.0, min(1.0, wisdom / 1200.0))
        min_pct = -2.45 + 1.3 * ratio
        positive_bias = 0.15 + 5.2 * (ratio ** 1.4)
        max_pct = min_pct + positive_bias
        return min_pct, max_pct

    def _force_in_bonus(self, state: RacerState, phase: Phase, lane_position: float) -> float:
        if phase is not Phase.OPENING:
            return 0.0
        track_width = max(self.track.track_width, HORSE_LANE_WIDTH)
        offset_m = lane_position * HORSE_LANE_WIDTH
        if offset_m <= track_width * 0.12:
            return 0.0
        if state.status_flags.get("isBlockedSideLeft"):
            return 0.0
        scale = min(2.0, max(0.5, offset_m / (track_width * 0.12)))
        return state.force_in_roll * scale

    def _move_lane_bonus(self, state: RacerState) -> float:
        if not state.moved_last_tick:
            return 0.0
        power = max(state.profile.stats.acceleration, 1.0)
        return math.sqrt(0.0002 * power)

    def _update_downhill_state(self, state: RacerState, slope_gradient: float) -> Tuple[float, bool]:
        rng = state.rng.ai_rng if state.rng else None
        active = state.downhill_mode
        if slope_gradient < -0.1:
            wiz = max(state.profile.stats.cognition, 0.0)
            chance_per_sec = min(0.8, wiz * 0.0004)
            chance = chance_per_sec * self.dt
            if not active:
                roll = rng.random() if rng else 0.5
                if roll < chance:
                    active = True
            else:
                end_roll = rng.random() if rng else 0.5
                if end_roll < 0.2 * self.dt:
                    active = False
        else:
            active = False
        state.downhill_mode = active
        if active:
            bonus = max(0.0, 0.3 + abs(slope_gradient) / 10.0)
            state.debug_log.setdefault("events", []).append("downhill_accel")
            return bonus, True
        return 0.0, False

    def _update_battles(
        self,
        state: RacerState,
        ordered_states: Sequence[RacerState],
        segment_type: str,
        global_phase: Phase,
    ) -> BattleResult:
        if state.status_flags.get("isStaminaKeep"):
            return BattleResult()
        result = BattleResult()
        grit = max(state.profile.stats.grit, 1.0)
        width = self.track.track_width or (self.spatial.max_lane_position * HORSE_LANE_WIDTH)
        lane_threshold_lead = (0.165 * width) / HORSE_LANE_WIDTH
        lane_threshold_duel = (0.25 * width) / HORSE_LANE_WIDTH
        events = state.debug_log.setdefault("events", [])

        world_pos = state.world_position
        if state.profile.strategy is Strategy.FRONT_RUNNER and 1 <= state.section <= 6:
            for other in ordered_states:
                if other is state or other.profile.strategy is not Strategy.FRONT_RUNNER:
                    continue
                dist = self._world_gap(world_pos, other.world_position)
                if dist is None:
                    continue
                lane_gap = abs(other.lane_position - state.lane_position)
                if dist <= 3.75 and lane_gap <= lane_threshold_lead and self._battle_available(state, "lead"):
                    bonus = math.pow(500.0 * grit, 0.6) * 0.0001
                    result.speed_bonus += bonus
                    result.stamina_multiplier = max(result.stamina_multiplier, 1.4)
                    events.append("battle_lead_competition")
                    self._set_battle_cooldown(state, "lead", 2.0)
                    break

        if global_phase is Phase.FINAL and segment_type == "straight":
            for other in ordered_states:
                if other is state:
                    continue
                dist = self._world_gap(world_pos, other.world_position)
                if dist is None:
                    continue
                lane_gap = abs(other.lane_position - state.lane_position)
                if dist <= 4.0 and lane_gap <= lane_threshold_duel and self._battle_available(state, "duel"):
                    speed_bonus = math.pow(200.0 * grit, 0.708) * 0.0001
                    accel_bonus = math.pow(160.0 * grit, 0.59) * 0.0001
                    result.speed_bonus += speed_bonus
                    result.accel_bonus += accel_bonus
                    result.stamina_multiplier = max(result.stamina_multiplier, 1.25)
                    events.append("battle_duel")
                    self._set_battle_cooldown(state, "duel", 3.0)
                    break

        if 11 <= state.section <= 15 and not state.status_flags.get("isLastSpurt"):
            cognition = max(state.profile.stats.cognition, 1.0)
            accel = max(state.profile.stats.acceleration, 1.0)
            rng = state.rng.ai_rng if state.rng else None
            chance = min(0.9, 0.25 + cognition / 2000.0)
            roll = rng.random() if rng else 0.5
            if roll < chance and self._battle_available(state, "compete"):
                accel_term = math.sqrt(accel / 1500.0)
                grit_term = math.pow(grit / 3000.0, 0.2)
                coef = STRATEGY_COMPETE_COEF.get(state.profile.strategy, 1.0)
                bonus = (accel_term * 2.0 + grit_term) * 0.1 * coef
                result.speed_bonus += bonus
                result.stamina_multiplier = max(result.stamina_multiplier, 1.15)
                events.append("battle_compete")
                self._set_battle_cooldown(state, "compete", 2.5)

        if 11 <= state.section <= 15 and (state.status_flags.get("order", 0) == 1):
            nearby_threat = next(
                (
                    other
                    for other in ordered_states
                    if other is not state and self._world_gap(world_pos, other.world_position, default=999) < 4.0
                ),
                None,
            )
            if nearby_threat and self._battle_available(state, "secure"):
                coef = STRATEGY_COMPETE_COEF.get(state.profile.strategy, 1.0)
                bonus = math.pow(grit / 2000.0, 0.5) * 0.3 * coef
                result.speed_bonus += bonus
                result.stamina_multiplier = max(result.stamina_multiplier, 1.1)
                events.append("battle_secure_lead")
                self._set_battle_cooldown(state, "secure", 3.0)

        return result

    def _battle_available(self, state: RacerState, key: str) -> bool:
        return state.battle_cooldowns.get(key, 0.0) <= 0.0

    def _set_battle_cooldown(self, state: RacerState, key: str, duration: float) -> None:
        state.battle_cooldowns[key] = duration

    @staticmethod
    def _world_gap(
        a: Optional[Tuple[float, float]],
        b: Optional[Tuple[float, float]],
        default: Optional[float] = None,
    ) -> Optional[float]:
        if a is None or b is None:
            return default
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.hypot(dx, dy)

    def _phase_from_ratio(self, ratio: float) -> Phase:
        if ratio >= 0.75:
            return Phase.FINAL
        if ratio >= 0.35:
            return Phase.MIDDLE
        return Phase.OPENING

    def _section_for(self, pos: float) -> int:
        if self._section_length <= 0:
            return 1
        section = int(pos / self._section_length) + 1
        return max(1, min(24, section))

    def _curvature_at(self, pos: float) -> float:
        try:
            return self._polyline.curvature_at(pos)
        except Exception:
            return 0.0

    def _estimate_spurt_eta(self, state: RacerState, multiplier: float) -> float:
        distance_left = max(self.track.distance - state.pos, 1.0)
        base_speed = max(5.0, state.current_speed if state.current_speed > 1.0 else state.target_speed)
        hp_ratio = state.current_hp / state.max_hp if state.max_hp > 0 else 0.0
        stamina_factor = 0.6 + hp_ratio * 0.6
        effective_speed = max(3.0, base_speed * multiplier * stamina_factor)
        return distance_left / effective_speed
