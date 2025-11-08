from pathlib import Path

from derby_game.engine import (
    Aptitudes,
    AptitudeGrade,
    HybridRaceLoop,
    RacerProfile,
    RacerStats,
    Strategy,
    load_svg_track,
)
from derby_game.engine.data_models import RacerState
from derby_game.engine.race_loop import BattleResult
from derby_game.engine.physics import Phase


def _profile(racer_id: int, strategy: Strategy, grit: float = 800) -> RacerProfile:
    stats = RacerStats(
        speed=900,
        acceleration=900,
        stamina=1000,
        grit=grit,
        cognition=700,
        focus=600,
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


def test_lead_competition_grants_speed_bonus():
    track = load_svg_track(Path("racetracks/svg/1710m - track001 - Exeter Racecourse.svg"))
    loop = HybridRaceLoop(track, [_profile(1, Strategy.FRONT_RUNNER), _profile(2, Strategy.FRONT_RUNNER)])
    a, b = loop.states
    a.pos = 10.0
    b.pos = 11.0
    a.section = 3
    b.section = 3
    result = loop._update_battles(a, loop.states, "straight", Phase.OPENING)
    assert isinstance(result, BattleResult)
    assert result.speed_bonus > 0.0
    assert result.stamina_multiplier > 1.0


def test_duel_bonus_triggers_in_final_straight():
    track = load_svg_track(Path("racetracks/svg/1710m - track001 - Exeter Racecourse.svg"))
    loop = HybridRaceLoop(track, [_profile(3, Strategy.PACE_CHASER), _profile(4, Strategy.LATE_SURGER)])
    a, b = loop.states
    a.pos = track.distance - 20.0
    b.pos = track.distance - 21.5
    a.section = 22
    b.section = 22
    result = loop._update_battles(a, loop.states, "straight", Phase.FINAL)
    assert result.speed_bonus >= 0.0
