from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

from .data_models import LaneGuide, Track, TrackBoundary, TrackSpline
from .geometry import Polyline, _distance, _lerp

SVG_POINT_TO_METER = 10.0 / 28.34645
COMMANDS = set("MmLlHhVvCcSsQqTtAaZz")
TOKEN_RE = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# Sampling density controls
LINE_SAMPLES = 1
CUBIC_SAMPLES = 32
QUADRATIC_SAMPLES = 24
LANE_ID_PATTERN = re.compile(r"^Lane_(Left|Right)_(\d+)$", re.IGNORECASE)


class UnsupportedSVGError(RuntimeError):
    pass


def load_svg_track(path: Path | str, *, track_id: Optional[str] = None, name: Optional[str] = None) -> Track:
    """Loads a racetrack SVG file, returning a Track dataclass."""

    svg_path = Path(path)
    if not svg_path.exists():
        raise FileNotFoundError(svg_path)

    tree = ET.parse(svg_path)
    root = tree.getroot()

    ns = _extract_namespace(root)

    racing_line_element = _find_element(root, "path", "RacingLine", ns)
    if racing_line_element is None:
        raise UnsupportedSVGError("SVG racetrack missing RacingLine path")

    racing_points = _convert_path_to_points(racing_line_element.attrib.get("d", ""))
    racing_polyline = Polyline.from_points(racing_points)

    inner_points = None
    outer_points = None

    inner_element = _find_element(root, "path", "InnerRail", ns)
    if inner_element is not None:
        inner_points = _convert_path_to_points(inner_element.attrib.get("d", ""))

    outer_element = _find_element(root, "path", "OuterRail", ns)
    if outer_element is not None:
        outer_points = _convert_path_to_points(outer_element.attrib.get("d", ""))

    inner_boundary = Polyline.from_points(inner_points) if inner_points else None
    outer_boundary = Polyline.from_points(outer_points) if outer_points else None

    track_width = _estimate_track_width(inner_boundary, outer_boundary)
    lane_guides = _parse_lane_guides(root, ns)
    lane_length_deltas = _build_lane_length_deltas(lane_guides, racing_polyline.length)

    derived_name = name or svg_path.stem
    derived_id = track_id or derived_name.lower().replace(" ", "_")

    track = Track(
        track_id=derived_id,
        name=derived_name,
        distance=racing_polyline.length,
        spline=TrackSpline(points=racing_polyline.points, length=racing_polyline.length),
        track_width=track_width,
        boundary_inner=TrackBoundary(name="InnerRail", points=inner_boundary.points) if inner_boundary else None,
        boundary_outer=TrackBoundary(name="OuterRail", points=outer_boundary.points) if outer_boundary else None,
        lane_guides=tuple(lane_guides),
        lane_length_deltas=lane_length_deltas,
    )

    return track


def _extract_namespace(root: ET.Element) -> str:
    m = re.match(r"\{(.*)\}", root.tag)
    if m:
        return m.group(1)
    return ""


def _find_element(root: ET.Element, tag: str, element_id: str, ns: str) -> Optional[ET.Element]:
    if ns:
        xpath = f".//{{{ns}}}{tag}[@id='{element_id}']"
    else:
        xpath = f".//{tag}[@id='{element_id}']"
    return root.find(xpath)


def _parse_lane_guides(root: ET.Element, ns: str) -> List[LaneGuide]:
    path_tag = f"{{{ns}}}path" if ns else "path"
    guides: List[LaneGuide] = []

    for element in root.findall(f".//{path_tag}"):
        raw_id = element.get("id")
        if not raw_id:
            continue

        normalized = raw_id.replace("_x5F_", "_")
        match = LANE_ID_PATTERN.match(normalized)
        if not match:
            continue

        side = match.group(1).lower()
        index = int(match.group(2))
        points = _convert_path_to_points(element.attrib.get("d", ""))
        polyline = Polyline.from_points(points)
        spline = TrackSpline(points=polyline.points, length=polyline.length)
        guides.append(LaneGuide(side=side, index=index, spline=spline))

    guides.sort(key=lambda guide: (guide.side, guide.index))
    return guides


def _build_lane_length_deltas(lane_guides: Sequence[LaneGuide], base_length: float) -> Dict[str, Dict[int, float]]:
    deltas: Dict[str, Dict[int, float]] = {}
    for guide in lane_guides:
        delta = guide.spline.length - base_length
        side_map = deltas.setdefault(guide.side, {})
        side_map[guide.index] = delta
    return deltas

def _convert_path_to_points(d: str) -> List[Tuple[float, float]]:
    raw_points = list(_parse_path(d))
    if len(raw_points) < 2:
        raise UnsupportedSVGError("Path did not yield enough points")

    scaled_points = [
        (x * SVG_POINT_TO_METER, y * SVG_POINT_TO_METER) for x, y in _dedupe_points(raw_points)
    ]
    return scaled_points


def _parse_path(d: str) -> Iterator[Tuple[float, float]]:
    tokens = TOKEN_RE.findall(d)
    if not tokens:
        return iter([])

    idx = 0
    current_command: Optional[str] = None
    current_point = (0.0, 0.0)
    start_point = (0.0, 0.0)
    prev_cubic_ctrl: Optional[Tuple[float, float]] = None
    prev_quad_ctrl: Optional[Tuple[float, float]] = None

    def _has_more() -> bool:
        return idx < len(tokens)

    def _peek() -> str:
        return tokens[idx]

    def _next_token() -> str:
        nonlocal idx
        token = tokens[idx]
        idx += 1
        return token

    def _read_number() -> float:
        return float(_next_token())

    output: List[Tuple[float, float]] = []

    while _has_more():
        token = _next_token()
        if token in COMMANDS:
            current_command = token
        else:
            # token is a number; reuse previous command
            if current_command is None:
                raise UnsupportedSVGError("Path data missing initial command")
            idx -= 1  # put the number back
        if current_command is None:
            continue

        cmd = current_command

        if cmd in "Mm":
            is_relative = cmd == "m"
            points: List[Tuple[float, float]] = []
            while _has_more() and _peek() not in COMMANDS:
                x = _read_number()
                y = _read_number()
                if is_relative:
                    x += current_point[0]
                    y += current_point[1]
                points.append((x, y))
            for idx_point, point in enumerate(points):
                current_point = point
                if not output:
                    start_point = current_point
                    output.append(current_point)
                else:
                    output.extend(_sample_line(output[-1], current_point, LINE_SAMPLES))
                if idx_point == 0:
                    # Subsequent coordinate pairs after initial moveto are treated as lineto
                    current_command = "l" if is_relative else "L"
        elif cmd in "Ll":
            is_relative = cmd == "l"
            while _has_more() and _peek() not in COMMANDS:
                x = _read_number()
                y = _read_number()
                if is_relative:
                    x += current_point[0]
                    y += current_point[1]
                new_point = (x, y)
                output.extend(_sample_line(current_point, new_point, LINE_SAMPLES))
                current_point = new_point
            prev_cubic_ctrl = None
            prev_quad_ctrl = None
        elif cmd in "Hh":
            is_relative = cmd == "h"
            while _has_more() and _peek() not in COMMANDS:
                dx = _read_number()
                x = current_point[0] + dx if is_relative else dx
                new_point = (x, current_point[1])
                output.extend(_sample_line(current_point, new_point, LINE_SAMPLES))
                current_point = new_point
            prev_cubic_ctrl = None
            prev_quad_ctrl = None
        elif cmd in "Vv":
            is_relative = cmd == "v"
            while _has_more() and _peek() not in COMMANDS:
                dy = _read_number()
                y = current_point[1] + dy if is_relative else dy
                new_point = (current_point[0], y)
                output.extend(_sample_line(current_point, new_point, LINE_SAMPLES))
                current_point = new_point
            prev_cubic_ctrl = None
            prev_quad_ctrl = None
        elif cmd in "Cc":
            is_relative = cmd == "c"
            while _has_more() and _peek() not in COMMANDS:
                x1 = _read_number()
                y1 = _read_number()
                x2 = _read_number()
                y2 = _read_number()
                x = _read_number()
                y = _read_number()
                if is_relative:
                    control1 = (current_point[0] + x1, current_point[1] + y1)
                    control2 = (current_point[0] + x2, current_point[1] + y2)
                    end_point = (current_point[0] + x, current_point[1] + y)
                else:
                    control1 = (x1, y1)
                    control2 = (x2, y2)
                    end_point = (x, y)
                output.extend(_sample_cubic(current_point, control1, control2, end_point, CUBIC_SAMPLES))
                current_point = end_point
                prev_cubic_ctrl = control2
                prev_quad_ctrl = None
        elif cmd in "Ss":
            is_relative = cmd == "s"
            while _has_more() and _peek() not in COMMANDS:
                x2 = _read_number()
                y2 = _read_number()
                x = _read_number()
                y = _read_number()
                if prev_cubic_ctrl is not None:
                    control1 = (2 * current_point[0] - prev_cubic_ctrl[0], 2 * current_point[1] - prev_cubic_ctrl[1])
                else:
                    control1 = current_point
                if is_relative:
                    control2 = (current_point[0] + x2, current_point[1] + y2)
                    end_point = (current_point[0] + x, current_point[1] + y)
                else:
                    control2 = (x2, y2)
                    end_point = (x, y)
                output.extend(_sample_cubic(current_point, control1, control2, end_point, CUBIC_SAMPLES))
                current_point = end_point
                prev_cubic_ctrl = control2
                prev_quad_ctrl = None
        elif cmd in "Qq":
            is_relative = cmd == "q"
            while _has_more() and _peek() not in COMMANDS:
                x1 = _read_number()
                y1 = _read_number()
                x = _read_number()
                y = _read_number()
                if is_relative:
                    control = (current_point[0] + x1, current_point[1] + y1)
                    end_point = (current_point[0] + x, current_point[1] + y)
                else:
                    control = (x1, y1)
                    end_point = (x, y)
                output.extend(_sample_quadratic(current_point, control, end_point, QUADRATIC_SAMPLES))
                current_point = end_point
                prev_quad_ctrl = control
                prev_cubic_ctrl = None
        elif cmd in "Tt":
            is_relative = cmd == "t"
            while _has_more() and _peek() not in COMMANDS:
                x = _read_number()
                y = _read_number()
                if prev_quad_ctrl is not None:
                    control = (2 * current_point[0] - prev_quad_ctrl[0], 2 * current_point[1] - prev_quad_ctrl[1])
                else:
                    control = current_point
                if is_relative:
                    end_point = (current_point[0] + x, current_point[1] + y)
                else:
                    end_point = (x, y)
                output.extend(_sample_quadratic(current_point, control, end_point, QUADRATIC_SAMPLES))
                current_point = end_point
                prev_quad_ctrl = control
                prev_cubic_ctrl = None
        elif cmd in "Zz":
            if output and (current_point != start_point):
                output.extend(_sample_line(current_point, start_point, LINE_SAMPLES))
            current_point = start_point
            prev_cubic_ctrl = None
            prev_quad_ctrl = None
        else:
            raise UnsupportedSVGError(f"Path command '{cmd}' is not supported")

    return iter(output)


def _sample_line(p0: Tuple[float, float], p1: Tuple[float, float], samples: int) -> List[Tuple[float, float]]:
    if samples <= 1:
        return [p1]
    step = 1.0 / samples
    return [_lerp(p0, p1, step * i) for i in range(1, samples + 1)]


def _sample_cubic(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    samples: int,
) -> List[Tuple[float, float]]:
    result: List[Tuple[float, float]] = []
    for i in range(1, samples + 1):
        t = i / samples
        inv = 1 - t
        x = inv ** 3 * p0[0] + 3 * inv * inv * t * p1[0] + 3 * inv * t * t * p2[0] + t ** 3 * p3[0]
        y = inv ** 3 * p0[1] + 3 * inv * inv * t * p1[1] + 3 * inv * t * t * p2[1] + t ** 3 * p3[1]
        result.append((x, y))
    return result


def _sample_quadratic(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    samples: int,
) -> List[Tuple[float, float]]:
    result: List[Tuple[float, float]] = []
    for i in range(1, samples + 1):
        t = i / samples
        inv = 1 - t
        x = inv * inv * p0[0] + 2 * inv * t * p1[0] + t * t * p2[0]
        y = inv * inv * p0[1] + 2 * inv * t * p1[1] + t * t * p2[1]
        result.append((x, y))
    return result


def _dedupe_points(points: Sequence[Tuple[float, float]], epsilon: float = 1e-6) -> List[Tuple[float, float]]:
    if not points:
        return []
    deduped: List[Tuple[float, float]] = [points[0]]
    for point in points[1:]:
        if _distance(point, deduped[-1]) > epsilon:
            deduped.append(point)
    return deduped


def _estimate_track_width(
    inner: Optional[Polyline],
    outer: Optional[Polyline],
    sample_count: int = 256,
) -> float:
    if not inner or not outer:
        return 0.0

    inner_points = inner.sample_evenly(sample_count)
    outer_points = outer.sample_evenly(sample_count)

    if not inner_points or not outer_points:
        return 0.0

    distances: List[float] = []

    for inner_pt in inner_points:
        distances.append(min(_distance(inner_pt, outer_pt) for outer_pt in outer_points))

    for outer_pt in outer_points:
        distances.append(min(_distance(outer_pt, inner_pt) for inner_pt in inner_points))

    return sum(distances) / len(distances) if distances else 0.0
