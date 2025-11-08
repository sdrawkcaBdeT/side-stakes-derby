from pathlib import Path

from derby_game.engine import (
    Aptitudes,
    AptitudeGrade,
    RacerProfile,
    RacerStats,
    Strategy,
    load_svg_track,
)
from derby_game.engine.physics import Phase
from derby_game.engine.data_models import RacerState
from derby_game.engine.spatial import SpatialEngine


def _profile(racer_id: int, strategy: Strategy = Strategy.PACE_CHASER) -> RacerProfile:
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
        racer_id=racer_id,
        name=f"Horse {racer_id}",
        stats=stats,
        strategy=strategy,
        aptitudes=aptitudes,
    )


def test_lane_penalty_prefers_inner_candidate():
    track = load_svg_track(Path("racetracks/svg/1710m - track001 - Exeter Racecourse.svg"))
    spatial = SpatialEngine(track)

    state = RacerState(profile=_profile(1, Strategy.PACE_CHASER))
    state.lane_position = 6.0
    state.phase = "middle"
    state.section = 5
    state.status_flags["isBlockedFront"] = False
    order_lookup = {state.profile.racer_id: 3}

    candidates = [6.5, 5.0]
    _, scores = spatial._score_candidates(state, candidates, order_lookup, Phase.MIDDLE)
    score_map = {entry["lane"]: entry["score"] for entry in scores}
    assert score_map[5.0] < score_map[6.5]
