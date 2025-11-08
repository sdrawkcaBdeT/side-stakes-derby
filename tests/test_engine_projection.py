import math
from pathlib import Path

from derby_game.engine import (
    Aptitudes,
    AptitudeGrade,
    RacerProfile,
    RacerStats,
    Strategy,
    load_svg_track,
)
from derby_game.engine.constants import HORSE_LANE_WIDTH
from derby_game.engine.race_loop import HybridRaceLoop


def _profile() -> RacerProfile:
    stats = RacerStats(
        speed=900,
        acceleration=850,
        stamina=1000,
        grit=700,
        cognition=600,
        focus=550,
        luck=400,
    )
    aptitudes = Aptitudes(distance=AptitudeGrade.A, surface=AptitudeGrade.A)
    return RacerProfile(
        racer_id=1,
        name="Lane Tester",
        stats=stats,
        strategy=Strategy.FRONT_RUNNER,
        aptitudes=aptitudes,
    )


def test_lane_guides_influence_world_projection():
    track = load_svg_track(Path("racetracks/svg/826m - track_003 - Canterbury Park.svg"))
    assert track.lane_guides, "Expected lane guides to be parsed from SVG"

    loop = HybridRaceLoop(track, [_profile()])
    state = loop.states[0]

    sample_pos = track.distance * 0.3
    base = loop._world_position(sample_pos, 0.0)
    outer = loop._world_position(sample_pos, 2.0)
    inner = loop._world_position(sample_pos, -2.0)

    outer_dist = math.hypot(outer[0] - base[0], outer[1] - base[1])
    inner_dist = math.hypot(inner[0] - base[0], inner[1] - base[1])

    expected_offset = 2.0 * HORSE_LANE_WIDTH
    assert abs(outer_dist - expected_offset) < 0.25
    assert abs(inner_dist - expected_offset) < 0.25

    dot = (outer[0] - base[0]) * (inner[0] - base[0]) + (outer[1] - base[1]) * (inner[1] - base[1])
    assert dot < 0.0
