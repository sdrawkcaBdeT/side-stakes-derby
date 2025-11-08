import json
from unittest.mock import MagicMock, patch

from derby_game.engine import TelemetryCollector, TelemetryFrame, TelemetryRacerFrame
from derby_game.simulation import Race


def _make_race() -> Race:
    race = Race.__new__(Race)
    race.race_id = 999
    return race


def test_write_hybrid_telemetry_persists_frames_to_db():
    race = _make_race()
    collector = TelemetryCollector()
    frame = TelemetryFrame(
        tick=0,
        time=0.0,
        phase="opening",
        leading_segment="straight",
        racers=[
            TelemetryRacerFrame(
                racer_id=1,
                name="Test",
                lane_position=0.5,
                world_position=(10.0, 5.0),
                pos=12.0,
                distance_delta=1.2,
                speed_delta=0.5,
                speed=15.0,
                target_speed=15.5,
                accel=0.2,
                hp=900.0,
                hp_delta=-5.0,
                stamina_drain=5.0,
                ai_mode="normal",
                status_flags={"isFinished": False},
            )
        ],
    )
    collector.record_frame(frame)

    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor

    with patch("derby_game.simulation.get_db_connection", return_value=conn), patch(
        "psycopg2.extras.execute_values"
    ) as mock_execute_values:
        race._write_hybrid_telemetry(collector, silent=True)

    cursor.execute.assert_called_with("DELETE FROM race_rounds WHERE race_id = %s", (race.race_id,))
    assert mock_execute_values.called
    args_list = mock_execute_values.call_args[0][2]
    payload = json.loads(args_list[0][6])
    assert payload["speed_delta"] == 0.5
    assert payload["distance_delta"] == 1.2


def test_write_hybrid_telemetry_no_frames_no_db():
    race = _make_race()
    collector = TelemetryCollector()
    with patch("derby_game.simulation.get_db_connection") as mock_conn:
        race._write_hybrid_telemetry(collector, silent=True)
    mock_conn.assert_not_called()
