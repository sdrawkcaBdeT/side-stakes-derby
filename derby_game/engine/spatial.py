from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .constants import HORSE_LANE_WIDTH, MAX_STAT_VALUE
from .data_models import RacerState, Strategy, Track
from .physics import Phase

FRONT_BLOCK_LENGTH = 2.0
FRONT_BLOCK_HALF_WIDTH = 0.75 / 2.0
SIDE_BLOCK_HALF_LENGTH = 1.05 / 2.0
SIDE_BLOCK_HALF_WIDTH = 1.0
OVERLAP_DISTANCE = 0.4
OVERLAP_LANE = 0.4
CORNER_OUTWARD_PENALTY = 0.75
BLOCKED_REPLAN_STEP = 0.5
HORSE_LENGTH = 3.0
HORSE_WIDTH = 1.5

LANE_PREFERENCES: Dict[Strategy, Dict[str, float]] = {
    Strategy.FRONT_RUNNER: {"opening": 0.5, "middle": 0.7, "final": 0.6},
    Strategy.PACE_CHASER: {"opening": 1.2, "middle": 1.0, "final": 0.8},
    Strategy.LATE_SURGER: {"opening": 2.0, "middle": 1.5, "final": 0.8},
    Strategy.END_CLOSER: {"opening": 2.5, "middle": 2.0, "final": 1.0},
}
LANE_PREF_WEIGHT = {"opening": 25.0, "middle": 8.0, "final": 5.0}

LANE_WEIGHTS: Dict[Strategy, Dict[str, Tuple[float, float]]] = {
    Strategy.FRONT_RUNNER: {
        "opening": (0.6, 80.0),
        "middle": (0.8, 1.4),
        "final": (0.7, 1.3),
    },
    Strategy.PACE_CHASER: {
        "opening": (0.8, 90.0),
        "middle": (0.9, 1.1),
        "final": (0.8, 1.25),
    },
    Strategy.LATE_SURGER: {
        "opening": (0.9, 95.0),
        "middle": (0.9, 1.0),
        "final": (0.8, 0.9),
    },
    Strategy.END_CLOSER: {
        "opening": (1.0, 95.0),
        "middle": (0.95, 0.95),
        "final": (0.8, 0.85),
    },
}

PHASE_LANE_WEIGHTS = {
    "opening": (1.0, 100.0),
    "middle": (1.0, 1.0),
    "final": (1.0, 1.15),
}


@dataclass(frozen=True)
class FeelerSpec:
    angle_deg: float
    lane_delta_hint: float


class SpatialEngine:
    """Handles perception, lane planning, blocking, and lateral motion."""

    def __init__(self, track: Track) -> None:
        self.track = track
        self.max_lane_position = max(track.track_width / HORSE_LANE_WIDTH, 1.0)
        self.section_types = self._build_section_types(track)
        self.lane_penalties = self._build_lane_penalties(track)


    # ------------------------------------------------------------------ #
    # Perception

    def update_perception(self, racers: Sequence[RacerState]) -> None:
        for state in racers:
            self._reset_state(state)
            state.debug_log.clear()

        for i, state in enumerate(racers):
            phase = self._phase_for_position(state.pos)
            focus_range = self._focus_range(state)
            feelers = self._feeler_layout(state)
            visible_ids: List[int] = []
            feeler_blocked: Dict[float, bool] = {spec.angle_deg: False for spec in feelers}

            for other in racers:
                if other is state:
                    continue

                distance = other.pos - state.pos
                lane_gap = other.lane_position - state.lane_position

                # visibility
                if distance >= 0.0 and distance <= focus_range and abs(lane_gap) <= 6.0:
                    visible_ids.append(other.profile.racer_id)

                # feeler occlusion
                for spec in feelers:
                    if self._feeler_intersects(spec, distance, lane_gap, focus_range):
                        feeler_blocked[spec.angle_deg] = True

                # proximity boxes
                self._evaluate_front_box(state, other, distance, lane_gap)
                self._evaluate_side_boxes(state, other, distance, lane_gap)
                self._evaluate_overlap(state, other, distance, lane_gap)

            state.visible_racers = visible_ids
            overtake_candidates, overtake_meta = self._collect_overtake_targets(state, racers, focus_range)
            state.visible_overtake_targets = [item["racer_id"] for item in overtake_meta]
            state.overtake_targets_meta = overtake_meta
            state.debug_log["visible"] = visible_ids
            if overtake_meta:
                state.debug_log["overtake_targets"] = overtake_meta
            else:
                state.debug_log.pop("overtake_targets", None)
            if state.status_flags["isBlockedFront"]:
                state.blocking_timer = min(state.blocking_timer + 1, 120)
            else:
                state.blocking_timer = 0.0

            # Candidate lanes from feelers
            candidates = list(overtake_candidates)
            candidates.append(self._clamp_lane(state.lane_position))
            for spec in feelers:
                if not feeler_blocked[spec.angle_deg]:
                    candidate = state.lane_position + spec.lane_delta_hint
                    candidate = self._clamp_lane(candidate)
                    if not self._direction_blocked(state, candidate):
                        candidates.append(candidate)
            if not candidates and state.status_flags["isBlockedFront"]:
                candidates.extend(self._expand_blocking_candidates(state))
            if not state.status_flags["isBlockedSideLeft"]:
                if phase in (Phase.OPENING, Phase.MIDDLE):
                    desired = max(0.0, state.lane_position - 0.05)
                else:
                    inward_step = max(0.5, state.lane_position * 0.4)
                    desired = max(0.0, state.lane_position - inward_step)
                if desired < state.lane_position - 1e-3 and not self._direction_blocked(state, desired):
                    candidates.append(desired)
            if phase is Phase.MIDDLE:
                inside_gap = self._nearest_inside_gap(state, racers)
                if inside_gap is not None and inside_gap < 1.75:
                    target = min(self.max_lane_position, state.lane_position + 2.0)
                    if not self._direction_blocked(state, target):
                        candidates.append(target)
            if state.status_flags.get("isStaminaKeep"):
                if not self._direction_blocked(state, 0.0):
                    candidates.append(0.0)
            if state.status_flags["isBlockedFront"] and state.blocking_timer >= 8:
                left_lane = max(0.0, state.lane_position - 2.0)
                right_lane = min(self.max_lane_position, state.lane_position + 2.0)
                candidates.extend([left_lane, right_lane])
                state.debug_log.setdefault("events", []).append("panic_replan")
            state.candidate_lanes = self._dedupe_sorted(candidates)
            state.debug_log["blocking"] = {
                "front": state.status_flags["isBlockedFront"],
                "left": state.status_flags["isBlockedSideLeft"],
                "right": state.status_flags["isBlockedSideRight"],
                "timer": state.blocking_timer,
            }

    def _reset_state(self, state: RacerState) -> None:
        state.visible_racers.clear()
        state.visible_overtake_targets.clear()
        state.candidate_lanes.clear()
        state.status_flags["isBlockedFront"] = False
        state.status_flags["isBlockedSide"] = False
        state.status_flags["isBlockedSideLeft"] = False
        state.status_flags["isBlockedSideRight"] = False
        state.blocking_racer_id = None
        state.blocking_speed = None
        state.blocking_distance = 0.0
        if "hasOverlap" in state.status_flags:
            state.status_flags.pop("hasOverlap")
        state.blocking_timer = 0.0
        state.lane_mode = "normal"

    def _focus_range(self, state: RacerState) -> float:
        focus = state.profile.stats.focus
        return 20.0 * (1.0 + min(focus, MAX_STAT_VALUE) / MAX_STAT_VALUE * 0.25)

    def _feeler_layout(self, state: RacerState) -> List[FeelerSpec]:
        cognition = state.profile.stats.cognition
        if cognition < 350:
            angles = [-45, 0, 45]
        elif cognition < 650:
            angles = [-45, -15, 0, 15, 45]
        else:
            angles = [-60, -30, -10, 0, 10, 30, 60]

        specs: List[FeelerSpec] = []
        for angle in angles:
            lane_delta = self._lane_hint_from_angle(angle)
            specs.append(FeelerSpec(angle_deg=angle, lane_delta_hint=lane_delta))
        return specs

    def _lane_hint_from_angle(self, angle: float) -> float:
        angle_abs = abs(angle)
        if angle_abs <= 5:
            delta = 0.0
        elif angle_abs <= 12:
            delta = 0.5
        elif angle_abs <= 25:
            delta = 1.0
        elif angle_abs <= 40:
            delta = 1.5
        else:
            delta = 2.0
        return -delta if angle < 0 else delta

    def _nearest_inside_gap(self, state: RacerState, racers: Sequence[RacerState]) -> Optional[float]:
        best: Optional[float] = None
        for other in racers:
            if other is state:
                continue
            if other.lane_position >= state.lane_position:
                continue
            if abs(other.pos - state.pos) > HORSE_LENGTH:
                continue
            gap = state.lane_position - other.lane_position
            if best is None or gap < best:
                best = gap
        return best

    def _feeler_intersects(self, spec: FeelerSpec, distance: float, lane_gap: float, max_range: float) -> bool:
        if distance < 0.0 or distance > max_range:
            return False
        if abs(spec.lane_delta_hint) < 1e-6:
            return abs(lane_gap) <= 0.75 and distance >= 0
        projected_lane = spec.lane_delta_hint
        # Acceptable corridor width that scales with distance
        corridor = max(0.5, abs(projected_lane) * 0.5)
        # Determine if other lies between current lane and target lane
        if spec.lane_delta_hint > 0:
            in_corridor = 0 <= lane_gap <= projected_lane
        else:
            in_corridor = projected_lane <= lane_gap <= 0
        return in_corridor and abs(lane_gap) <= abs(projected_lane) + corridor

    def _collect_overtake_targets(
        self,
        state: RacerState,
        racers: Sequence[RacerState],
        focus_range: float,
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        metas: List[Dict[str, float]] = []
        candidate_lanes: List[float] = []
        max_distance = min(20.0, focus_range)
        for other in racers:
            if other is state:
                continue
            distance = other.pos - state.pos
            if distance < 1.0 or distance > max_distance:
                continue
            relative_speed = max(
                state.target_speed - other.target_speed,
                state.current_speed - other.current_speed,
                state.target_speed - other.current_speed,
            )
            if relative_speed <= 0.15:
                continue
            catch_time = distance / max(relative_speed, 0.15)
            if catch_time > 15.0:
                continue
            if not (
                state.target_speed > other.target_speed
                or (other.status_flags.get("isBlockedFront") and state.target_speed > other.current_speed)
            ):
                continue
            inner_bound, outer_bound = self._crowd_bounds(other, racers)
            lane_options: List[float] = []
            if inner_bound is not None:
                lane_options.append(max(0.0, inner_bound - 1.0))
            if outer_bound is not None:
                lane_options.append(min(self.max_lane_position, outer_bound + 1.0))
            allowed_lanes = []
            for lane in lane_options:
                lane = self._clamp_lane(lane)
                if not self._direction_blocked(state, lane):
                    allowed_lanes.append(lane)
                    candidate_lanes.append(lane)
            metas.append(
                {
                    "racer_id": other.profile.racer_id,
                    "distance": distance,
                    "catch_time": catch_time,
                    "lane": other.lane_position,
                    "inner_bound": inner_bound,
                    "outer_bound": outer_bound,
                    "lanes": allowed_lanes,
                }
            )
        return candidate_lanes, metas

    def _crowd_bounds(self, target: RacerState, racers: Sequence[RacerState]) -> Tuple[Optional[float], Optional[float]]:
        inner = target.lane_position
        outer = target.lane_position
        found = False
        for other in racers:
            if other is target:
                continue
            distance = other.pos - target.pos
            lane_gap = other.lane_position - target.lane_position
            if distance < 0.0 or distance > 3.0:
                continue
            if lane_gap < 0 and abs(lane_gap) < 2.0:
                found = True
                inner = min(inner, other.lane_position)
            elif lane_gap > 0 and lane_gap < 2.0:
                found = True
                outer = max(outer, other.lane_position)
        if not found:
            return target.lane_position, target.lane_position
        return inner, outer

    def _direction_blocked(self, state: RacerState, target_lane: float) -> bool:
        delta = target_lane - state.lane_position
        if delta < -1e-3 and state.status_flags["isBlockedSideLeft"]:
            return True
        if delta > 1e-3 and state.status_flags["isBlockedSideRight"]:
            return True
        return False

    def _evaluate_front_box(self, state: RacerState, other: RacerState, distance: float, lane_gap: float) -> None:
        if distance <= 0.0 or distance > FRONT_BLOCK_LENGTH:
            return
        if abs(lane_gap) > FRONT_BLOCK_HALF_WIDTH:
            return
        state.status_flags["isBlockedFront"] = True
        state.blocking_racer_id = other.profile.racer_id
        state.blocking_speed = other.current_speed
        state.blocking_distance = distance

    def _evaluate_side_boxes(self, state: RacerState, other: RacerState, distance: float, lane_gap: float) -> None:
        if abs(distance) > SIDE_BLOCK_HALF_LENGTH:
            return
        if lane_gap < 0 and abs(lane_gap) <= SIDE_BLOCK_HALF_WIDTH:
            state.status_flags["isBlockedSideLeft"] = True
            state.status_flags["isBlockedSide"] = True
        if lane_gap > 0 and abs(lane_gap) <= SIDE_BLOCK_HALF_WIDTH:
            state.status_flags["isBlockedSideRight"] = True
            state.status_flags["isBlockedSide"] = True

    def _evaluate_overlap(self, state: RacerState, other: RacerState, distance: float, lane_gap: float) -> None:
        if abs(distance) < OVERLAP_DISTANCE and abs(lane_gap) < OVERLAP_LANE:
            # mark overlap for later resolution
            state.status_flags.setdefault("hasOverlap", False)
            other.status_flags.setdefault("hasOverlap", False)
            state.status_flags["hasOverlap"] = True
            other.status_flags["hasOverlap"] = True

    # ------------------------------------------------------------------ #
    # AI lane selection & lateral movement

    def update_ai_targets(self, racers: Sequence[RacerState], dt: float) -> None:
        order_lookup = {state.profile.racer_id: state.status_flags.get("order") for state in racers}
        racer_lookup = {state.profile.racer_id: state for state in racers}
        for state in racers:
            state.candidate_lanes.clear()

            phase = self._phase_for_position(state.pos)
            candidates = self._determine_lane_candidates(state, racers, racer_lookup, phase, dt)
            state.candidate_lanes = self._dedupe_sorted(candidates)

            if state.candidate_lanes:
                chosen, score_meta = self._score_candidates(state, state.candidate_lanes, order_lookup, phase)
                state.debug_log["candidate_scores"] = score_meta
                state.target_lane = chosen
            else:
                state.target_lane = self._clamp_lane(state.target_lane)
                state.debug_log.setdefault("events", []).append("no_candidates")

    def step_lateral(self, state: RacerState, dt: float) -> None:
        delta = state.target_lane - state.lane_position
        if abs(delta) < 1e-3:
            state.lateral_speed = 0.0
            bounced = state.lane_position < 0.0 or state.lane_position > self.max_lane_position
            if bounced:
                state.lane_position = self._clamp_lane(state.lane_position)
            self._apply_rail_constraints(state, bounced=bounced)
            state.moved_last_tick = False
            return

        stats = state.profile.stats
        order = state.status_flags.get("order", 0) or 0
        order_modifier = 1.0 + max(0, 4 - order) * 0.08

        base_lat_speed = (0.45 + 0.0008 * stats.acceleration) * order_modifier  # lanes / second
        lat_target_speed = base_lat_speed * min(abs(delta), 3.0) / 3.0
        lat_accel = base_lat_speed * 1.5

        direction = 1.0 if delta > 0 else -1.0
        target_speed = lat_target_speed
        if direction < 0:
            target_speed *= 1.25 + 0.05 * abs(state.lane_position)
        target_speed *= direction

        if state.lateral_speed < target_speed:
            state.lateral_speed = min(target_speed, state.lateral_speed + lat_accel * dt)
        else:
            state.lateral_speed = max(target_speed, state.lateral_speed - lat_accel * dt)

        state.lane_position += state.lateral_speed * dt
        bounced = False
        if state.lane_position < 0.0 or state.lane_position > self.max_lane_position:
            bounced = True
        state.lane_position = self._clamp_lane(state.lane_position)
        state.moved_last_tick = abs(state.lateral_speed) > 1e-3
        self._apply_rail_constraints(state, bounced=bounced)

    def resolve_overlaps(self, racers: Sequence[RacerState]) -> None:
        count = len(racers)
        for i in range(count):
            for j in range(i + 1, count):
                a = racers[i]
                b = racers[j]
                longitudinal_gap = abs(b.pos - a.pos)
                lateral_gap_m = abs(b.lane_position - a.lane_position) * HORSE_LANE_WIDTH
                if longitudinal_gap < HORSE_LENGTH and lateral_gap_m < HORSE_WIDTH:
                    push = (HORSE_WIDTH - lateral_gap_m) / HORSE_LANE_WIDTH / 2.0
                    if a.lane_position <= b.lane_position:
                        a.lane_position = self._clamp_lane(a.lane_position - push)
                        b.lane_position = self._clamp_lane(b.lane_position + push)
                    else:
                        a.lane_position = self._clamp_lane(a.lane_position + push)
                        b.lane_position = self._clamp_lane(b.lane_position - push)
                    a.current_speed *= 0.97
                    b.current_speed *= 0.97
                    a.status_flags["isBlockedSide"] = True
                    b.status_flags["isBlockedSide"] = True

    # ------------------------------------------------------------------ #
    # Blocking & speed penalties

    def speed_cap(self, state: RacerState) -> Optional[float]:
        lateral_cap = self._lateral_speed_cap(state)
        if lateral_cap is not None:
            return lateral_cap
        if not state.status_flags["isBlockedFront"]:
            return None
        if state.blocking_speed is None:
            return None
        escape_lane = next(
            (
                lane
                for lane in getattr(state, "candidate_lanes", [])
                if abs(lane - state.lane_position) > 0.2 and not self._direction_blocked(state, lane)
            ),
            None,
        )
        if escape_lane is not None:
            return None

        blocker_speed = state.blocking_speed
        distance = max(min(state.blocking_distance, FRONT_BLOCK_LENGTH), 0.0)
        cap = (0.988 + 0.012 * (distance / FRONT_BLOCK_LENGTH)) * blocker_speed

        # Grit push check
        if self._passes_grit_push(state):
            stamina_cost = 5.0 + (1.0 - min(state.profile.stats.grit / MAX_STAT_VALUE, 1.0)) * 10.0
            state.current_hp = max(0.0, state.current_hp - stamina_cost)
            state.status_flags["gritPush"] = True
            state.debug_log.setdefault("events", []).append(
                f"grit_push(-{stamina_cost:.2f}hp)"
            )
            return None

        cognition_ratio = min(state.profile.stats.cognition / MAX_STAT_VALUE, 1.0)
        adjusted = state.current_speed - (state.current_speed - cap) * (0.35 + 0.4 * cognition_ratio)
        adjusted = max(0.0, adjusted)
        state.debug_log["speed_cap"] = {
            "raw": cap,
            "adjusted": adjusted,
            "distance_gap": distance,
        }
        return adjusted

    def _passes_grit_push(self, state: RacerState) -> bool:
        blocker_id = state.blocking_racer_id
        if blocker_id is None or state.rng is None:
            return False
        grit = state.profile.stats.grit
        # Reference grit ratio probability
        chance = min(max(grit / MAX_STAT_VALUE, 0.0), 1.0) * 0.5
        roll = state.rng.battle_rng.random()
        state.debug_log["grit_roll"] = roll
        return roll < chance

    def _lateral_speed_cap(self, state: RacerState) -> Optional[float]:
        if state.status_flags["isBlockedSideLeft"] and state.status_flags["isBlockedSideRight"]:
            return max(0.0, state.current_speed * 0.97)
        if state.status_flags["isBlockedSide"]:
            return max(0.0, state.current_speed * 0.985)
        return None

    # ------------------------------------------------------------------ #

    def _score_candidates(
        self,
        state: RacerState,
        candidates: List[float],
        order_lookup: Dict[int, Optional[int]],
        phase: Phase,
    ) -> Tuple[float, List[Dict[str, float]]]:
        base_lane = state.lane_position
        best_lane = base_lane
        best_score = float("inf")

        strategy = state.profile.strategy
        weights = LANE_WEIGHTS.get(strategy, LANE_WEIGHTS[Strategy.PACE_CHASER])
        strat_inward, strat_outward = weights.get(phase, (1.0, 1.0))
        phase_key = self._phase_key(phase)
        phase_inward, phase_outward = PHASE_LANE_WEIGHTS.get(phase_key, (1.0, 1.0))
        pref_center = LANE_PREFERENCES.get(strategy, {}).get(phase_key, base_lane)
        pref_weight = LANE_PREF_WEIGHT.get(phase_key, 5.0)
        inward_weight = strat_inward * phase_inward
        outward_weight = strat_outward * phase_outward
        section_type = self._section_type(state.section)
        order = order_lookup.get(state.profile.racer_id) or 0

        score_meta: List[Dict[str, float]] = []
        base_penalty = self._lane_distance_penalty(base_lane)
        better_found = False
        for candidate in candidates:
            offset = candidate - base_lane
            if offset < 0:
                score = abs(offset) * inward_weight
            else:
                score = abs(offset) * outward_weight

            if section_type == "corner" and offset > 0:
                score += CORNER_OUTWARD_PENALTY * abs(offset)

            if state.status_flags["isBlockedFront"]:
                score -= abs(offset) * 0.1

            if order == 1 and offset > 0:
                score += 0.5 * abs(offset)

            candidate_penalty = self._lane_distance_penalty(candidate)
            penalty_delta = candidate_penalty - base_penalty
            if penalty_delta > 0:
                score += penalty_delta * 120.0
            else:
                score += penalty_delta * 30.0
            score += candidate * 0.35
            if offset < 0:
                score -= 2.0 * abs(offset)
            if state.status_flags.get("isStaminaKeep") and offset > 0:
                score += 5.0 * abs(offset)
            score += abs(candidate - pref_center) * pref_weight

            score_meta.append({"lane": candidate, "score": score})

            if score < best_score:
                best_score = score
                best_lane = candidate
                if offset < 0:
                    better_found = True

        if not better_found and base_lane > 0.25 and not state.status_flags.get("isBlockedSideLeft"):
            best_lane = max(0.0, base_lane - 0.5)

        return best_lane, score_meta

    def _expand_blocking_candidates(self, state: RacerState) -> List[float]:
        cognition_ratio = min(state.profile.stats.cognition / MAX_STAT_VALUE, 1.0)
        max_steps = 1 + int(2 + cognition_ratio * 4)
        results: List[float] = []
        for step in range(1, max_steps + 1):
            delta = step * BLOCKED_REPLAN_STEP
            for sign in (-1, 1):
                lane = self._clamp_lane(state.lane_position + sign * delta)
                results.append(lane)
        state.debug_log.setdefault("events", []).append("blocking_replan")
        return results

    def _build_section_types(self, track: Track) -> Dict[int, str]:
        section_types: Dict[int, str] = {idx: "straight" for idx in range(1, 25)}
        if track.distance <= 0:
            return section_types
        section_length = track.distance / 24.0
        for corner in track.corners:
            start = int(corner.start_pos / section_length) + 1
            end = int(corner.end_pos / section_length) + 1
            for section in range(start, min(end + 1, 25)):
                section_types[section] = "corner"
        return section_types

    def _section_type(self, section: int) -> str:
        if section <= 0:
            return "straight"
        return self.section_types.get(section, "straight")

    def _build_lane_penalties(self, track: Track) -> Dict[int, float]:
        penalties: Dict[int, float] = {}
        right_map = track.lane_length_deltas.get("right", {})
        for index, delta in right_map.items():
            if index < 0:
                continue
            penalties[index] = max(0.0, delta / max(track.distance, 1.0))
        return penalties

    def _lane_distance_penalty(self, lane_position: float) -> float:
        if lane_position <= 0.0:
            return 0.0
        idx = int(round(lane_position))
        if idx <= 0:
            return 0.0
        penalty = self.lane_penalties.get(idx)
        if penalty is not None:
            return penalty
        # fallback linear growth
        return max(0.0, lane_position * 0.002)

    def _dedupe_sorted(self, values: Iterable[float], epsilon: float = 1e-3) -> List[float]:
        unique: List[float] = []
        for value in sorted(values):
            if not unique or abs(unique[-1] - value) > epsilon:
                unique.append(value)
        return unique

    def _clamp_lane(self, lane: float) -> float:
        return max(0.0, min(self.max_lane_position, lane))

    def _apply_rail_constraints(self, state: RacerState, bounced: bool) -> None:
        inner_limit = max(state.collider_radius / HORSE_LANE_WIDTH, 0.05)
        outer_limit = max(inner_limit, self.max_lane_position - inner_limit)
        near_inner = state.lane_position <= inner_limit
        near_outer = state.lane_position >= outer_limit
        previous_contact = bool(state.status_flags.get("isRailContact"))
        event_log = state.debug_log.setdefault("events", [])

        contact_side: Optional[str] = None
        if bounced:
            if near_inner:
                contact_side = "inner"
                state.lane_position = inner_limit
                state.current_speed *= 0.985
                state.lateral_speed = max(0.0, state.lateral_speed)
                event_log.append("rail_bounce_in")
            elif near_outer:
                contact_side = "outer"
                state.lane_position = outer_limit
                state.current_speed *= 0.985
                state.lateral_speed = min(0.0, state.lateral_speed)
                event_log.append("rail_bounce_out")
        else:
            inward_press = near_inner and state.lateral_speed < -1e-3
            outward_press = near_outer and state.lateral_speed > 1e-3
            if inward_press:
                contact_side = "inner"
                state.lane_position = max(state.lane_position, inner_limit)
            elif outward_press:
                contact_side = "outer"
                state.lane_position = min(state.lane_position, outer_limit)

            if contact_side and not previous_contact:
                event_log.append(f"rail_scrape_{'in' if contact_side == 'inner' else 'out'}")

        if contact_side:
            state.debug_log["rail_contact"] = {"side": contact_side, "bounced": bounced}
        else:
            state.debug_log.pop("rail_contact", None)

        state.status_flags["isRailContact"] = contact_side is not None

    def _phase_for_position(self, pos: float) -> Phase:
        if self.track.distance <= 0:
            return Phase.OPENING
        ratio = pos / self.track.distance
        if ratio >= 0.75:
            return Phase.FINAL
        if ratio >= 0.35:
            return Phase.MIDDLE
        return Phase.OPENING

    @staticmethod
    def _phase_key(phase: Phase) -> str:
        if phase is Phase.MIDDLE:
            return "middle"
        if phase is Phase.FINAL:
            return "final"
        return "opening"

    def _determine_lane_candidates(
        self,
        state: RacerState,
        racers: Sequence[RacerState],
        lookup: Dict[int, RacerState],
        phase: Phase,
        dt: float,
    ) -> List[float]:
        candidates: List[float] = []
        if state.status_flags.get("isFinished"):
            return [state.lane_position]

        if state.status_flags["isBlockedFront"]:
            state.blocking_timer = min(4.0, state.blocking_timer + dt)
        else:
            state.blocking_timer = max(0.0, state.blocking_timer - dt)

        targets = self._find_overtake_targets(state, lookup)
        overtake_meta = getattr(state, "overtake_targets_meta", [])
        if targets:
            state.lane_mode = "overtake"
            state.overtake_timer = 1.5
        else:
            state.overtake_timer = max(0.0, state.overtake_timer - dt)
            if state.overtake_timer <= 0.0 and state.lane_mode == "overtake":
                state.lane_mode = "normal"
        state.debug_log["lane_mode"] = state.lane_mode

        if state.lane_mode == "overtake" or state.overtake_timer > 0.0:
            candidates.extend(self._overtake_lane_candidates(state, targets, overtake_meta, phase, lookup))
        else:
            candidates.extend(self._normal_lane_candidates(state, phase))

        if not candidates:
            candidates.append(state.lane_position)
        return candidates

    def _find_overtake_targets(
        self,
        state: RacerState,
        lookup: Dict[int, RacerState],
    ) -> List[RacerState]:
        targets: List[RacerState] = []
        meta_targets = getattr(state, "overtake_targets_meta", None)
        if meta_targets:
            for meta in meta_targets:
                rid = meta.get("racer_id")
                if rid is None:
                    continue
                other = lookup.get(rid)
                if other and other not in targets:
                    targets.append(other)
            return targets
        for rid in state.visible_racers:
            other = lookup.get(rid)
            if not other or other is state:
                continue
            distance = other.pos - state.pos
            if distance < 1.0 or distance > 20.0:
                continue
            speed_gap = max(state.target_speed - other.current_speed, 0.1)
            catch_time = distance / speed_gap
            if catch_time <= 15.0 or (
                other.status_flags.get("isBlockedFront") and other.current_speed < state.target_speed
            ):
                targets.append(other)
        if state.status_flags.get("isBlockedFront"):
            rid = state.blocking_racer_id
            if rid:
                blocker = lookup.get(rid)
                if blocker and blocker not in targets:
                    targets.append(blocker)
        return targets

    def _normal_lane_candidates(self, state: RacerState, phase: Phase) -> List[float]:
        candidates: List[float] = []
        if state.status_flags.get("isExhausted"):
            return [state.lane_position]
        track_width = max(self.track.track_width, HORSE_LANE_WIDTH)
        course_width_lanes = track_width / HORSE_LANE_WIDTH
        lanes_from_fence = state.lane_position

        if state.ai_mode == "pace_down":
            target_lane = (0.18 * track_width) / HORSE_LANE_WIDTH
            candidates.append(self._clamp_lane(target_lane))

        if state.status_flags.get("isStaminaKeep"):
            candidates.append(0.0)

        if phase in (Phase.OPENING, Phase.MIDDLE) and lanes_from_fence > 0.2:
            candidates.append(self._clamp_lane(state.lane_position - 0.5))
        else:
            candidates.append(self._clamp_lane(state.lane_position))

        if state.blocking_timer >= 1.0:
            candidates.append(self._clamp_lane(state.lane_position + 0.6))

        return candidates

    def _overtake_lane_candidates(
        self,
        state: RacerState,
        targets: Sequence[RacerState],
        phase: Phase,
        lookup: Dict[int, RacerState],
    ) -> List[float]:
        candidates: List[float] = []
        if not targets:
            return [self._clamp_lane(state.lane_position)]

        for target in targets:
            inner = self._clamp_lane(target.lane_position - 1.0)
            outer = self._clamp_lane(target.lane_position + 1.0)
            if self._lane_has_space(state, inner, lookup):
                candidates.append(inner)
            if self._lane_has_space(state, outer, lookup):
                candidates.append(outer)

        if phase in (Phase.OPENING, Phase.MIDDLE) and state.lane_position >= 1.0:
            candidates.append(self._clamp_lane(state.lane_position - 1.0))
        else:
            candidates.append(self._clamp_lane(state.lane_position))

        if state.blocking_timer >= 0.8:
            candidates.append(self._clamp_lane(state.lane_position + 1.0))
            candidates.append(self._clamp_lane(state.lane_position - 1.0))

        return candidates

    def _lane_has_space(
        self,
        state: RacerState,
        candidate: float,
        lookup: Dict[int, RacerState],
        buffer: float = 0.8,
    ) -> bool:
        if candidate < state.lane_position and state.status_flags["isBlockedSideLeft"]:
            return False
        if candidate > state.lane_position and state.status_flags["isBlockedSideRight"]:
            return False
        lower = candidate - buffer
        upper = candidate + buffer
        for other_id in state.visible_racers:
            other = lookup.get(other_id)
            if not other or other is state:
                continue
            if lower <= other.lane_position <= upper and abs(other.pos - state.pos) < 3.0:
                return False
        return True
