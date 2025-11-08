from types import SimpleNamespace
from unittest.mock import call, patch

from derby_game.engine import TelemetryCollector, TelemetryFrame, TelemetryRacerFrame, Strategy, AptitudeGrade
from derby_game.simulation import Race


def _stub_race(distance: int = 1600) -> Race:
    race = Race.__new__(Race)  # bypass __init__
    race.distance = distance
    return race


def test_map_strategy_accepts_string_variants():
    race = _stub_race()
    assert race._map_strategy("Front Runner") is Strategy.FRONT_RUNNER
    assert race._map_strategy("late_surfer") is Strategy.LATE_SURGER
    assert race._map_strategy(4) is Strategy.END_CLOSER
    assert race._map_strategy(None) is Strategy.PACE_CHASER


def test_infer_distance_aptitude_grades():
    race = _stub_race(distance=1700)
    horse = SimpleNamespace(min_preferred_distance=1680, max_preferred_distance=1720)
    assert race._infer_distance_aptitude(horse) is AptitudeGrade.S

    horse = SimpleNamespace(min_preferred_distance=1500, max_preferred_distance=1600)
    assert race._infer_distance_aptitude(horse) is AptitudeGrade.A

    horse = SimpleNamespace(min_preferred_distance=1200, max_preferred_distance=1400)
    assert race._infer_distance_aptitude(horse) is AptitudeGrade.C


def test_telemetry_collector_records_frames():
    collector = TelemetryCollector()
    frame = TelemetryFrame(
        tick=0,
        time=0.0,
        phase="opening",
        leading_segment="straight",
        racers=[
            TelemetryRacerFrame(
                racer_id=1,
                name="Test Horse",
                lane_position=0.0,
                world_position=(0.0, 0.0),
                pos=0.0,
                distance_delta=0.0,
                speed_delta=0.0,
                speed=0.0,
                target_speed=0.0,
                accel=0.0,
                hp=100.0,
                hp_delta=0.0,
                stamina_drain=0.0,
                ai_mode="normal",
                status_flags={"isFinished": False},
            )
        ],
    )
    collector.record_frame(frame)
    exported = collector.export()
    assert len(exported) == 1
    assert exported[0].racers[0].name == "Test Horse"


def test_persist_results_records_once():
    race = _stub_race()
    race.race_id = 321
    race._results_persisted = False
    finishers = [10, 11, 12]

    with patch("derby_game.simulation.derby_queries") as mock_queries:
        race._persist_results(finishers)
        mock_queries.clear_race_results.assert_called_once_with(race.race_id)
        mock_queries.record_race_result.assert_has_calls(
            [
                call(race.race_id, 10, 1),
                call(race.race_id, 11, 2),
                call(race.race_id, 12, 3),
            ]
        )
        mock_queries.set_race_winner.assert_called_once_with(race.race_id, 10)

        race._persist_results([99, 100])
        # No additional writes after first persistence
        assert mock_queries.record_race_result.call_count == 3
