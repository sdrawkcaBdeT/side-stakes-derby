"""
Generate telemetry dumps or lightweight visual replays for hybrid races.

Examples:
    # Save the per-tick JSON for a race
    python scripts/replay_race.py --race-id 12 --dump replays/race_12.json

    # Render a simple MP4 animation (requires matplotlib + ffmpeg/pillow)
    python scripts/replay_race.py --race-id 12 --animate replays/race_12.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from derby_game.database.connection import get_db_connection
from derby_game.engine.geometry import Polyline
from derby_game.engine.track_registry import DEFAULT_TRACK_REGISTRY

COLLIDER_LENGTH_M = 3.0
COLLIDER_WIDTH_M = 1.5


Frame = Dict[str, object]


def _fetch_race_meta(race_id: int) -> Dict[str, object]:
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT race_id, distance, tier FROM races WHERE race_id = %s", (race_id,))
            record = cur.fetchone()
            if not record:
                raise ValueError(f"Race {race_id} not found.")
            return {"race_id": record[0], "distance": float(record[1]), "tier": record[2]}
    finally:
        if conn:
            conn.close()


def _fetch_telemetry_rows(race_id: int):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT round_number, horse_id, round_events
                FROM race_rounds
                WHERE race_id = %s
                ORDER BY round_number ASC, horse_id ASC;
                """,
                (race_id,),
            )
            return cur.fetchall()
    finally:
        if conn:
            conn.close()


def _fetch_horse_names(race_id: int, horse_ids: Iterable[int]) -> Dict[int, str]:
    ids = tuple(sorted(set(horse_ids)))
    if not ids:
        return {}
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT h.horse_id, h.name
                FROM race_entries re
                JOIN horses h ON h.horse_id = re.horse_id
                WHERE re.race_id = %s AND h.horse_id = ANY(%s);
                """,
                (race_id, list(ids)),
            )
            return {row[0]: row[1] for row in cur.fetchall()}
    finally:
        if conn:
            conn.close()


def _group_frames(rows) -> Tuple[List[Frame], Sequence[int]]:
    frames: Dict[int, Frame] = {}
    horse_ids: List[int] = []
    for round_number, horse_id, payload in rows:
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            event = json.loads(payload)
        elif isinstance(payload, dict):
            event = payload
        else:
            raise TypeError(f"Unsupported payload type: {type(payload)}")
        tick = int(event.get("tick", round_number - 1))
        frame = frames.setdefault(
            tick,
            {
                "tick": tick,
                "time": event.get("time", 0.0),
                "phase": event.get("phase", "opening"),
                "leading_segment": event.get("leading_segment", "straight"),
                "horses": [],
            },
        )
        world_pos = event.get("world_position") or {}
        frame["time"] = event.get("time", frame["time"])
        frame["phase"] = event.get("phase", frame["phase"])
        frame["leading_segment"] = event.get("leading_segment", frame["leading_segment"])
        horse_entry = {
            "horse_id": horse_id,
            "name": None,
            "lane_position": event.get("lane_position", 0.0),
            "world_position": (world_pos.get("x", 0.0), world_pos.get("y", 0.0)),
            "speed": event.get("speed", 0.0),
            "target_speed": event.get("target_speed", 0.0),
            "hp": event.get("hp", 0.0),
            "status_flags": event.get("status_flags", {}),
            "pos": event.get("pos", 0.0),
        }
        frame["horses"].append(horse_entry)
        horse_ids.append(horse_id)

    ordered_ticks = sorted(frames.keys())
    ordered_frames = [frames[tick] for tick in ordered_ticks]
    return ordered_frames, ordered_ticks and sorted(set(horse_ids)) or []


def _attach_names(frames: List[Frame], names: Dict[int, str]) -> None:
    if not names:
        return
    for frame in frames:
        for horse in frame["horses"]:
            horse["name"] = names.get(horse["horse_id"])


def dump_frames(frames: Sequence[Frame], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(frames, fh, indent=2)
    print(f"[replay] wrote {len(frames)} frames to {output_path}")


def _load_track(distance: float):
    try:
        return DEFAULT_TRACK_REGISTRY.find_by_distance(distance)
    except Exception as exc:
        raise RuntimeError(f"Unable to load track for {distance}m: {exc}") from exc


def _collect_track_lines(track) -> List[Tuple[Sequence[float], Sequence[float], str]]:
    lines: List[Tuple[Sequence[float], Sequence[float], str]] = []
    if track.boundary_inner:
        xs, ys = zip(*track.boundary_inner.points)
        lines.append((xs, ys, "Inner Rail"))
    if track.boundary_outer:
        xs, ys = zip(*track.boundary_outer.points)
        lines.append((xs, ys, "Outer Rail"))
    if track.spline.points:
        xs, ys = zip(*track.spline.points)
        lines.append((xs, ys, "Racing Line"))
    return lines


def _calc_bounds(lines: Sequence[Tuple[Sequence[float], Sequence[float], str]], frames: Sequence[Frame]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for line in lines:
        xs.extend(line[0])
        ys.extend(line[1])
    for frame in frames:
        for horse in frame["horses"]:
            wx, wy = horse["world_position"]
            xs.append(wx)
            ys.append(wy)
    if not xs or not ys:
        return (-10.0, 10.0, -10.0, 10.0)
    margin = max(10.0, 0.05 * max(max(xs) - min(xs), max(ys) - min(ys)))
    return (
        min(xs) - margin,
        max(xs) + margin,
        min(ys) - margin,
        max(ys) + margin,
    )


def animate_frames(
    frames: Sequence[Frame],
    horse_names: Dict[int, str],
    track_distance: float,
    output_path: Path,
    fps: int = 15,
    follow_lead: bool = False,
    follow_padding: float = 60.0,
    marker_scale: float = 1.0,
) -> None:
    if not frames:
        raise RuntimeError("No telemetry frames to render.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Polygon

    track = _load_track(track_distance)
    lines = _collect_track_lines(track)
    min_x, max_x, min_y, max_y = _calc_bounds(lines, frames)
    racing_line = None
    if track.spline.points:
        try:
            racing_line = Polyline.from_points(track.spline.points)
        except ValueError:
            racing_line = None

    horse_ids = sorted({horse["horse_id"] for frame in frames for horse in frame["horses"]})
    cmap = plt.get_cmap("tab20", len(horse_ids) or 1)
    color_map = {horse_id: cmap(idx % cmap.N) for idx, horse_id in enumerate(horse_ids)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    if not follow_lead:
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
    ax.set_title(f"Race Replay")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    for xs, ys, label in lines:
        ax.plot(xs, ys, linewidth=1.0, label=label)
    if lines:
        ax.legend(loc="upper right")

    name_annotations = {horse_id: ax.text(0, 0, "", fontsize=6, ha="right", va="bottom") for horse_id in horse_ids}
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10)
    collider_scale = max(marker_scale, 0.05)
    half_length = 0.5 * COLLIDER_LENGTH_M * collider_scale
    half_width = 0.5 * COLLIDER_WIDTH_M * collider_scale
    horse_patches: Dict[int, Polygon] = {}
    for horse_id in horse_ids:
        polygon = Polygon(
            [[0.0, 0.0]] * 4,
            closed=True,
            facecolor=color_map.get(horse_id, "tab:blue"),
            edgecolor="black",
            alpha=0.65,
            linewidth=0.5,
            zorder=3,
        )
        polygon.set_visible(False)
        ax.add_patch(polygon)
        horse_patches[horse_id] = polygon

    def _tangent(pos_value: Optional[float]) -> Tuple[float, float]:
        if racing_line and racing_line.length > 0.0 and pos_value is not None:
            clamped = max(0.0, min(racing_line.length, float(pos_value)))
            return racing_line.tangent_at(clamped)
        return (1.0, 0.0)

    def _horse_polygon(horse: Dict[str, object]) -> List[Tuple[float, float]]:
        wx, wy = horse["world_position"]
        tx, ty = _tangent(horse.get("pos"))
        magnitude = math.hypot(tx, ty) or 1.0
        tx /= magnitude
        ty /= magnitude
        nx, ny = -ty, tx
        fx, fy = tx * half_length, ty * half_length
        sx, sy = nx * half_width, ny * half_width
        return [
            (wx + fx + sx, wy + fy + sy),
            (wx + fx - sx, wy + fy - sy),
            (wx - fx - sx, wy - fy - sy),
            (wx - fx + sx, wy - fy + sy),
        ]

    def init():
        time_text.set_text("")
        for patch in horse_patches.values():
            patch.set_visible(False)
        for artist in name_annotations.values():
            artist.set_text("")
        return (*horse_patches.values(), time_text, *name_annotations.values())

    def update(frame: Frame):
        horses = frame["horses"]
        time_text.set_text(f"t={frame['time']:.2f}s  phase={frame['phase']}  tick={frame['tick']}")
        if follow_lead and horses:
            leader = min(
                horses,
                key=lambda h: h.get("status_flags", {}).get("order", float("inf")),
            )
            cx, cy = leader["world_position"]
            ax.set_xlim(cx - follow_padding, cx + follow_padding)
            ax.set_ylim(cy - follow_padding, cy + follow_padding)
        visible: set[int] = set()
        for horse in horses:
            horse_id = horse["horse_id"]
            patch = horse_patches[horse_id]
            patch.set_xy(_horse_polygon(horse))
            patch.set_facecolor(color_map.get(horse_id, "tab:blue"))
            patch.set_visible(True)
            visible.add(horse_id)
            artist = name_annotations[horse_id]
            artist.set_position(horse["world_position"])
            label = horse_names.get(horse_id) or f"H{horse_id}"
            artist.set_text(label)
        for horse_id, patch in horse_patches.items():
            if horse_id not in visible:
                patch.set_visible(False)
                name_annotations[horse_id].set_text("")
        return (*horse_patches.values(), time_text, *name_annotations.values())

    animation = FuncAnimation(fig, update, frames=frames, init_func=init, interval=1000 / fps, blit=False)

    writer: Optional[object] = None
    if output_path.suffix.lower() in {".gif"}:
        writer = PillowWriter(fps=fps)
    kwargs = {"writer": writer} if writer else {}
    animation.save(str(output_path), fps=fps, **kwargs)
    plt.close(fig)
    print(f"[replay] saved animation to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export or replay hybrid race telemetry.")
    parser.add_argument("--race-id", type=int, required=True, help="Race ID to load.")
    parser.add_argument("--dump", type=Path, help="Optional JSON file to dump frames.")
    parser.add_argument("--animate", type=Path, help="Optional MP4/GIF path for a simple track replay (requires matplotlib).")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for animation output (default: 15).")
    parser.add_argument(
        "--follow-lead",
        action="store_true",
        help="Dynamically center the camera on the current leader for a zoomed-in view.",
    )
    parser.add_argument(
        "--follow-padding",
        type=float,
        default=60.0,
        help="Padding (meters) around the leader when --follow-lead is enabled (default: 60).",
    )
    parser.add_argument(
        "--marker-scale",
        type=float,
        default=1.0,
        help="Multiplier for the physical 3m x 1.5m collider rectangles (1.0 = real size).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = _fetch_race_meta(args.race_id)
    rows = _fetch_telemetry_rows(args.race_id)
    if not rows:
        raise RuntimeError(f"No telemetry found for race {args.race_id}. Did you run the hybrid engine?")

    frames, horse_ids = _group_frames(rows)
    horse_names = _fetch_horse_names(args.race_id, horse_ids)
    _attach_names(frames, horse_names)

    if args.dump:
        dump_frames(frames, args.dump)

    if args.animate:
        animate_frames(
            frames,
            horse_names,
            meta["distance"],
            args.animate,
            fps=args.fps,
            follow_lead=args.follow_lead,
            follow_padding=args.follow_padding,
            marker_scale=args.marker_scale,
        )
    elif not args.dump:
        default_path = Path("replays") / f"race_{args.race_id}_telemetry.json"
        dump_frames(frames, default_path)


if __name__ == "__main__":
    main()
