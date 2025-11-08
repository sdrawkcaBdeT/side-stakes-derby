from pathlib import Path

from derby_game.engine import (
    Aptitudes,
    AptitudeGrade,
    RacerProfile,
    RacerStats,
    Strategy,
    load_svg_track,
)
from derby_game.engine.data_models import RacerState, RNGContainer
from derby_game.engine.spatial import SpatialEngine


def _profile(
    racer_id: int,
    cognition: float = 600,
    focus: float = 600,
    grit: float = 700,
) -> RacerProfile:
    stats = RacerStats(
        speed=900,
        acceleration=850,
        stamina=1000,
        grit=grit,
        cognition=cognition,
        focus=focus,
        luck=400,
    )
    aptitudes = Aptitudes(distance=AptitudeGrade.A, surface=AptitudeGrade.A)
    return RacerProfile(
        racer_id=racer_id,
        name=f"Horse {racer_id}",
        stats=stats,
        strategy=Strategy.FRONT_RUNNER,
        aptitudes=aptitudes,
    )


def test_feelers_generate_candidates_when_unblocked():
    track = load_svg_track(Path("racetracks/svg/826m - track_003 - Canterbury Park.svg"))
    spatial = SpatialEngine(track)

    a_state = RacerState(profile=_profile(1))
    b_state = RacerState(profile=_profile(2))
    a_state.pos = 10.0
    b_state.pos = 13.0
    b_state.lane_position = 2.0

    spatial.update_perception([a_state, b_state])

    assert a_state.status_flags["isBlockedFront"] is False
    assert a_state.candidate_lanes  # should have at least one option


def test_front_block_sets_speed_cap_information():
    track = load_svg_track(Path("racetracks/svg/826m - track_003 - Canterbury Park.svg"))
    spatial = SpatialEngine(track)

    a_state = RacerState(profile=_profile(1))
    b_state = RacerState(profile=_profile(2))
    a_state.pos = 10.0
    b_state.pos = 11.5  # within 2m
    a_state.current_speed = 15.0
    b_state.current_speed = 12.0

    spatial.update_perception([a_state, b_state])

    assert a_state.status_flags["isBlockedFront"] is True
    cap = spatial.speed_cap(a_state)
    assert cap is None or cap <= b_state.current_speed * 1.2


def test_grit_push_consumes_stamina_and_skips_cap():
    track = load_svg_track(Path("racetracks/svg/826m - track_003 - Canterbury Park.svg"))
    spatial = SpatialEngine(track)

    state = RacerState(profile=_profile(3, grit=1200))
    state.status_flags["isBlockedFront"] = True
    state.blocking_speed = 12.0
    state.blocking_distance = 1.0
    state.blocking_racer_id = 2
    state.current_speed = 14.0
    state.max_hp = 200.0
    state.current_hp = 200.0
    state.rng = RNGContainer(main_seed=1, ai_seed=2, battle_seed=3)
    state.rng.battle_rng.random = lambda: 0.0  # force success

    cap = spatial.speed_cap(state)
    assert cap is None
    assert state.current_hp < 200.0
    assert state.status_flags["gritPush"] is True


def test_rail_contact_clamps_lane_and_penalizes_speed():
    track = load_svg_track(Path("racetracks/svg/826m - track_003 - Canterbury Park.svg"))
    spatial = SpatialEngine(track)

    state = RacerState(profile=_profile(4))
    state.lane_position = spatial.max_lane_position + 1.0
    state.target_lane = spatial.max_lane_position + 1.0
    state.current_speed = 15.0

    spatial.step_lateral(state, dt=1 / 15.0)

    assert 0.0 <= state.lane_position <= spatial.max_lane_position
    assert state.status_flags["isRailContact"] is True
    assert state.current_speed < 15.0


def test_lane_center_without_pressure_does_not_flag_contact():
    track = load_svg_track(Path("racetracks/svg/826m - track_003 - Canterbury Park.svg"))
    spatial = SpatialEngine(track)

    state = RacerState(profile=_profile(6))
    state.lane_position = 0.0
    state.target_lane = 0.0

    spatial.step_lateral(state, dt=1 / 15.0)

    assert state.status_flags["isRailContact"] is False
    assert "rail_contact" not in state.debug_log


def test_rail_scrape_sets_contact_flag_without_bounce():
    track = load_svg_track(Path("racetracks/svg/826m - track_003 - Canterbury Park.svg"))
    spatial = SpatialEngine(track)

    state = RacerState(profile=_profile(5))
    state.lane_position = 0.04
    state.target_lane = -0.2

    spatial.step_lateral(state, dt=1 / 15.0)

    assert state.status_flags["isRailContact"] is True
    assert state.debug_log["rail_contact"]["side"] == "inner"


def test_unblocked_racer_generates_inward_candidate():
    track = load_svg_track(Path("racetracks/svg/1710m - track001 - Exeter Racecourse.svg"))
    spatial = SpatialEngine(track)

    state = RacerState(profile=_profile(7))
    state.lane_position = 4.0
    state.phase = "middle"
    state.section = 4

    spatial.update_perception([state])
    assert any(candidate < 4.0 for candidate in state.candidate_lanes)


def test_step_lateral_moves_toward_target():
    track = load_svg_track(Path("racetracks/svg/826m - track_003 - Canterbury Park.svg"))
    spatial = SpatialEngine(track)
    state = RacerState(profile=_profile(8))
    state.lane_position = 6.0
    state.target_lane = 4.0
    before = state.lane_position
    for _ in range(30):
        spatial.step_lateral(state, dt=1 / 15.0)
    assert state.lane_position < before
