#!/usr/bin/env python3
"""
Utility for probing hybrid race gaps by running repeatable simulations with
optional physics overrides and exporting per-tick telemetry.

Example usage:
    python scripts/hybrid_gap_probe.py --race-label baseline \
        --dump-summary analysis/baseline_summary.json

To dump full per-tick traces for four strategies:
    python scripts/hybrid_gap_probe.py --race-label instrumented \
        --dump-json analysis/instrumented_ticks.json \
        --dump-summary analysis/instrumented_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _preparse_env(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Consume env-affecting flags before importing engine modules."""

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--disable-position-keeping", action="store_true")
    return pre_parser.parse_known_args(argv)[0]


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

ENV_ARGS = _preparse_env()
if getattr(ENV_ARGS, "disable_position_keeping", False):
    os.environ["DERBY_DISABLE_POSITION_KEEP"] = "1"

from derby_game.engine import (  # noqa: E402
    Aptitudes,
    AptitudeGrade,
    HybridRaceLoop,
    RacerProfile,
    RacerStats,
    Strategy,
    TelemetryCollector,
    load_svg_track,
)
from derby_game.engine import physics as physics_mod  # noqa: E402
from derby_game.engine.track_registry import DEFAULT_TRACK_REGISTRY  # noqa: E402

Phase = physics_mod.Phase


DEFAULT_LINEUP = [
    {
        "name": "Blazing Comet",
        "strategy": "front_runner",
        "stats": {
            "speed": 1010,
            "acceleration": 980,
            "stamina": 880,
            "grit": 720,
            "cognition": 640,
            "focus": 650,
            "luck": 420,
        },
    },
    {
        "name": "Measured Tempo",
        "strategy": "pace_chaser",
        "stats": {
            "speed": 960,
            "acceleration": 910,
            "stamina": 940,
            "grit": 690,
            "cognition": 660,
            "focus": 600,
            "luck": 410,
        },
    },
    {
        "name": "Late Surge",
        "strategy": "late_surger",
        "stats": {
            "speed": 930,
            "acceleration": 900,
            "stamina": 990,
            "grit": 780,
            "cognition": 700,
            "focus": 570,
            "luck": 405,
        },
    },
    {
        "name": "Closing Bell",
        "strategy": "end_closer",
        "stats": {
            "speed": 910,
            "acceleration": 880,
            "stamina": 1020,
            "grit": 820,
            "cognition": 740,
            "focus": 560,
            "luck": 400,
        },
    },
]


def _normalize_strategy(label: str) -> Strategy:
    key = label.replace("-", "_").replace(" ", "_").upper()
    return Strategy[key]


def _load_lineup(path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path:
        return DEFAULT_LINEUP
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("Lineup file must contain a list of horse specs.")
    return data


def _build_profiles(lineup: List[Dict[str, Any]]) -> List[RacerProfile]:
    profiles: List[RacerProfile] = []
    for idx, spec in enumerate(lineup, start=1):
        stats = RacerStats(**spec["stats"])
        strategy = _normalize_strategy(spec["strategy"])
        aptitudes = Aptitudes(distance=AptitudeGrade.A, surface=AptitudeGrade.A)
        profiles.append(
            RacerProfile(
                racer_id=spec.get("racer_id", idx),
                name=spec.get("name", f"Horse {idx}"),
                stats=stats,
                strategy=strategy,
                aptitudes=aptitudes,
            )
        )
    return profiles


def _strategy_override_from_arg(arg: str) -> Tuple[Strategy, Phase, float]:
    try:
        strat_label, phase_label, value_str = arg.split(":")
    except ValueError as exc:
        raise ValueError(f"Invalid strategy modifier '{arg}'. Expected STRATEGY:PHASE:VALUE.") from exc
    strategy = _normalize_strategy(strat_label)
    phase = Phase[phase_label.strip().upper()]
    return strategy, phase, float(value_str)


def _apply_strategy_overrides(overrides: Sequence[str]) -> List[Tuple[Strategy, Phase, float]]:
    applied: List[Tuple[Strategy, Phase, float]] = []
    for override in overrides:
        strategy, phase, value = _strategy_override_from_arg(override)
        physics_mod.STRATEGY_SPEED_MOD[strategy][phase] = value
        applied.append((strategy, phase, value))
    return applied


def _patch_start_dash(target_factor: Optional[float], accel: Optional[float]) -> Tuple[float, float]:
    original = (physics_mod.START_DASH_TARGET_FACTOR, physics_mod.START_DASH_ACCEL_BONUS)
    if target_factor is not None:
        physics_mod.START_DASH_TARGET_FACTOR = target_factor
    if accel is not None:
        physics_mod.START_DASH_ACCEL_BONUS = accel
    return original


def _patch_spurt_scale(scale: float):
    if math.isclose(scale, 1.0):
        return lambda: None

    original_method = HybridRaceLoop._ensure_spurt_plan

    def scaled(self: HybridRaceLoop, state):
        original_method(self, state)
        if state.spurt_plan_committed:
            state.spurt_multiplier *= scale
            state.debug_log.setdefault("events", []).append(f"spurt_scale({scale:.3f})")

    HybridRaceLoop._ensure_spurt_plan = scaled  # type: ignore[assignment]

    def restore():
        HybridRaceLoop._ensure_spurt_plan = original_method  # type: ignore[assignment]

    return restore


def _load_track(identifier: str):
    path = Path(identifier)
    if not path.exists():
        alt = Path("racetracks") / "svg" / identifier
        if alt.exists():
            path = alt
    if path.exists():
        track = load_svg_track(path)
        return track, path.stem
    track = DEFAULT_TRACK_REGISTRY.load(identifier)
    return track, identifier


def _build_tick_records(
    frames: Sequence[Any],
    snapshots: Sequence[Any],
    strategy_lookup: Dict[int, Strategy],
) -> List[Dict[str, Any]]:
    ticks: List[Dict[str, Any]] = []
    for idx, (frame, snap) in enumerate(zip(frames, snapshots)):
        snap_map = {r.racer_id: r for r in snap.racers}
        racers_payload = []
        for racer in frame.racers:
            snap_racer = snap_map.get(racer.racer_id)
            pacing_state = racer.debug.get("pacing_state") or {}
            racers_payload.append(
                {
                    "racer_id": racer.racer_id,
                    "name": racer.name,
                    "strategy": strategy_lookup.get(racer.racer_id, "").name if racer.racer_id in strategy_lookup else "",
                    "pos": racer.pos,
                    "section": getattr(snap_racer, "section", None),
                    "phase": getattr(getattr(snap_racer, "phase", None), "name", "").lower(),
                    "speed": racer.speed,
                    "target_speed": racer.target_speed,
                    "accel": racer.accel,
                    "pacing_speed_factor": pacing_state.get("factor", 1.0),
                    "ai_mode": racer.ai_mode,
                    "status_flags": racer.status_flags,
                    "target_components": racer.debug.get("target_components"),
                    "accel_components": racer.debug.get("accel_components"),
                    "pacing_state": pacing_state,
                    "pk_mode": racer.debug.get("pk_mode"),
                    "speed_cap": racer.debug.get("speed_cap"),
                    "hp": racer.hp,
                    "hp_delta": racer.hp_delta,
                    "stamina_drain": racer.stamina_drain,
                }
            )
        ticks.append(
            {
                "tick": idx,
                "time": frame.time,
                "global_phase": frame.phase,
                "leading_segment": frame.leading_segment,
                "racers": racers_payload,
            }
        )
    return ticks


def _summarize_samples(
    frames: Sequence[Any],
    sample_points: Sequence[Tuple[str, float]],
    track_distance: float,
    strategy_lookup: Dict[int, Strategy],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for label, target_time in sample_points:
        frame = next((f for f in frames if f.time >= target_time), frames[-1])
        positions = sorted(frame.racers, key=lambda r: r.pos, reverse=True)
        leader = positions[0]
        trailer = positions[-1]
        sample = {
            "time": frame.time,
            "global_phase": frame.phase,
            "leader": {
                "name": leader.name,
                "strategy": strategy_lookup.get(leader.racer_id, "").name if leader.racer_id in strategy_lookup else "",
                "pos": leader.pos,
                "speed": leader.speed,
                "target_speed": leader.target_speed,
            },
            "trailer": {
                "name": trailer.name,
                "strategy": strategy_lookup.get(trailer.racer_id, "").name if trailer.racer_id in strategy_lookup else "",
                "pos": trailer.pos,
                "speed": trailer.speed,
                "target_speed": trailer.target_speed,
            },
            "gap_m": leader.pos - trailer.pos,
            "racers": [
                {
                    "name": r.name,
                    "strategy": strategy_lookup.get(r.racer_id, "").name if r.racer_id in strategy_lookup else "",
                    "pos": r.pos,
                    "speed": r.speed,
                    "target_speed": r.target_speed,
                    "pacing_speed_factor": (r.debug.get("pacing_state") or {}).get("factor", 1.0),
                }
                for r in positions
            ],
        }
        summary[label] = sample
    finish_frame = frames[-1]
    finish_positions = sorted(finish_frame.racers, key=lambda r: r.pos, reverse=True)
    summary["finish"] = {
        "time": finish_frame.time,
        "gap_m": finish_positions[0].pos - min(f.pos for f in finish_positions),
        "completed": [r.name for r in finish_positions if math.isclose(r.pos, track_distance, rel_tol=1e-3, abs_tol=1e-3)],
    }
    return summary


def _aggregate_horse_metrics(frames: Sequence[Any], strategy_lookup: Dict[int, Strategy]) -> Dict[int, Dict[str, Any]]:
    metrics: Dict[int, Dict[str, Any]] = {}
    for frame in frames:
        for racer in frame.racers:
            stat = metrics.setdefault(
                racer.racer_id,
                {
                    "name": racer.name,
                    "strategy": strategy_lookup.get(racer.racer_id, "").name if racer.racer_id in strategy_lookup else "",
                    "samples": 0,
                    "avg_speed": 0.0,
                    "avg_target_speed": 0.0,
                    "pacing_factor_min": float("inf"),
                    "pacing_factor_max": 0.0,
                    "speed_cap_ticks": 0,
                    "blocked_front_ticks": 0,
                    "blocked_side_ticks": 0,
                },
            )
            stat["samples"] += 1
            stat["avg_speed"] += racer.speed
            stat["avg_target_speed"] += racer.target_speed
            factor = (racer.debug.get("pacing_state") or {}).get("factor", 1.0)
            stat["pacing_factor_min"] = min(stat["pacing_factor_min"], factor)
            stat["pacing_factor_max"] = max(stat["pacing_factor_max"], factor)
            if racer.debug.get("speed_cap"):
                stat["speed_cap_ticks"] += 1
            if racer.status_flags.get("isBlockedFront"):
                stat["blocked_front_ticks"] += 1
            if racer.status_flags.get("isBlockedSide") or racer.status_flags.get("isBlockedSideLeft") or racer.status_flags.get("isBlockedSideRight"):
                stat["blocked_side_ticks"] += 1
    for stat in metrics.values():
        samples = max(1, stat.pop("samples"))
        stat["avg_speed"] /= samples
        stat["avg_target_speed"] /= samples
        if stat["pacing_factor_min"] == float("inf"):
            stat["pacing_factor_min"] = 1.0
        stat["pacing_factor_max"] = max(stat["pacing_factor_max"], stat["pacing_factor_min"])
    return metrics


def _max_gap(frames: Sequence[Any]) -> Dict[str, Any]:
    max_gap = -1.0
    at_time = 0.0
    pair: Tuple[str, str] = ("", "")
    for frame in frames:
        positions = sorted(frame.racers, key=lambda r: r.pos)
        gap = positions[-1].pos - positions[0].pos
        if gap > max_gap:
            max_gap = gap
            at_time = frame.time
            pair = (positions[-1].name, positions[0].name)
    return {"max_gap_m": max_gap, "at_time": at_time, "leader": pair[0], "trailer": pair[1]}


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    lineup_specs = _load_lineup(Path(args.lineup_file)) if args.lineup_file else [*DEFAULT_LINEUP]
    profiles = _build_profiles(lineup_specs)
    strategy_lookup = {profile.racer_id: profile.strategy for profile in profiles}
    track, track_label = _load_track(args.track)
    telemetry = TelemetryCollector()
    strategy_overrides = _apply_strategy_overrides(args.strategy_mod or [])
    original_dash = _patch_start_dash(args.start_dash_target, args.start_dash_accel)
    restore_spurt = _patch_spurt_scale(args.spurt_scale)

    loop = HybridRaceLoop(track, profiles, telemetry=telemetry, rng_seed=args.seed)
    snapshots = loop.run_until_finished(max_time=args.max_time)
    frames = telemetry.export()
    if len(frames) != len(snapshots):
        raise RuntimeError(f"Telemetry frames ({len(frames)}) did not match snapshots ({len(snapshots)}).")

    restore_spurt()
    physics_mod.START_DASH_TARGET_FACTOR, physics_mod.START_DASH_ACCEL_BONUS = original_dash

    if not frames:
        raise RuntimeError("Telemetry collector returned no frames.")
    ticks = _build_tick_records(frames, snapshots, strategy_lookup)
    sample_points = [(label, value) for label, value in zip(args.sample_labels, args.sample_times)]
    # Append first final-phase tick
    final_idx = next((i for i, f in enumerate(frames) if f.phase == "final"), len(frames) - 1)
    sample_points.append(("final_phase_start", frames[final_idx].time))

    summary = {
        "meta": {
            "label": args.race_label,
            "track": track_label,
            "distance": track.distance,
            "seed": args.seed,
            "dt": loop.dt,
            "overrides": {
                "strategy_mods": [(s.name, p.name, v) for s, p, v in strategy_overrides],
                "start_dash_target": args.start_dash_target,
                "start_dash_accel": args.start_dash_accel,
                "spurt_scale": args.spurt_scale,
                "disable_position_keep": bool(getattr(args, "disable_position_keeping", False)),
            },
        },
        "samples": _summarize_samples(frames, sample_points, track.distance, strategy_lookup),
        "horse_metrics": _aggregate_horse_metrics(frames, strategy_lookup),
        "max_gap": _max_gap(frames),
        "finish_order": [state.name for state in sorted(frames[-1].racers, key=lambda r: r.pos, reverse=True)],
    }

    payload = {"summary": summary, "ticks": ticks}
    return payload


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic hybrid race and export gap telemetry.",
        parents=[argparse.ArgumentParser(add_help=False)],
    )
    parser.add_argument("--track", default="1710m - track001 - Exeter Racecourse.svg", help="Track ID or SVG path.")
    parser.add_argument("--race-label", default="baseline", help="Label stored in summary metadata.")
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic RNG seed.")
    parser.add_argument("--max-time", type=float, default=140.0, help="Failsafe max simulation time in seconds.")
    parser.add_argument("--sample-times", default="10,30,60", help="Comma-separated seconds to snapshot.")
    parser.add_argument("--sample-labels", default="t10,t30,t60", help="Labels matching --sample-times.")
    parser.add_argument("--dump-json", type=Path, help="Path to write full per-tick telemetry JSON.")
    parser.add_argument("--dump-summary", type=Path, help="Path to write summary-only JSON.")
    parser.add_argument("--lineup-file", type=Path, help="Optional JSON file describing horses.")
    parser.add_argument("--strategy-mod", action="append", help="Override strategy phase multiplier (e.g., 'front_runner:opening:1.08').")
    parser.add_argument("--start-dash-target", type=float, help="Override START_DASH_TARGET_FACTOR.")
    parser.add_argument("--start-dash-accel", type=float, help="Override START_DASH_ACCEL_BONUS.")
    parser.add_argument("--spurt-scale", type=float, default=1.0, help="Scale factor applied to committed spurt multipliers.")
    parser.add_argument("--disable-position-keeping", action="store_true", help="Set DERBY_DISABLE_POSITION_KEEP=1 before importing engine.")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    args.sample_times = [float(item.strip()) for item in args.sample_times.split(",") if item.strip()]
    args.sample_labels = [label.strip() for label in args.sample_labels.split(",") if label.strip()]
    if len(args.sample_labels) != len(args.sample_times):
        raise ValueError("--sample-labels must match --sample-times count.")
    result = run_probe(args)
    summary = result["summary"]
    max_gap = summary["max_gap"]
    print(
        f"[{summary['meta']['label']}] track={summary['meta']['track']} seed={summary['meta']['seed']} "
        f"max_gap={max_gap['max_gap_m']:.2f}m at {max_gap['at_time']:.2f}s"
    )
    for label, sample in summary["samples"].items():
        if label == "finish":
            continue
        print(
            f"  {label}: t={sample['time']:.2f}s phase={sample['global_phase']} "
            f"gap={sample['gap_m']:.2f}m leader={sample['leader']['name']} trailer={sample['trailer']['name']}"
        )
    finish = summary["samples"].get("finish")
    if finish:
        print(
            f"  finish: t={finish['time']:.2f}s max_gap={finish['gap_m']:.2f}m "
            f"completed={', '.join(finish.get('completed', []))}"
        )

    if args.dump_json:
        _write_json(Path(args.dump_json), result)
        print(f"Wrote per-tick telemetry to {args.dump_json}")
    if args.dump_summary:
        _write_json(Path(args.dump_summary), summary)
        print(f"Wrote summary to {args.dump_summary}")


if __name__ == "__main__":
    main()
