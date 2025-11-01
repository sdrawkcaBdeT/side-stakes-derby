import unittest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

from derby_game import betting_service


class GetRecentPlayerBetsTests(unittest.TestCase):
    def test_returns_empty_when_service_unavailable(self):
        with patch.object(betting_service, "market_db", None):
            self.assertEqual(betting_service.get_recent_player_bets("123"), [])

    def test_returns_normalised_rows(self):
        fake_cursor = MagicMock()
        naive_timestamp = datetime(2025, 1, 1, 12, 0, 0)
        fake_cursor.__enter__.return_value = fake_cursor
        fake_cursor.fetchall.return_value = [
            (7, 99, "Iron Fury", Decimal("150"), Decimal("2.50"), naive_timestamp)
        ]

        fake_conn = MagicMock()
        fake_conn.cursor.return_value = fake_cursor

        market_mock = MagicMock()
        market_mock.get_connection.return_value = fake_conn

        with patch.object(betting_service, "market_db", market_mock):
            results = betting_service.get_recent_player_bets("123", limit=3)

        self.assertEqual(len(results), 1)
        bet = results[0]
        self.assertEqual(bet["bet_id"], 7)
        self.assertEqual(bet["race_id"], 99)
        self.assertEqual(bet["horse_name"], "Iron Fury")
        self.assertEqual(bet["amount"], Decimal("150"))
        self.assertEqual(bet["locked_in_odds"], Decimal("2.50"))
        self.assertEqual(bet["placed_at"].tzinfo, timezone.utc)

        fake_conn.close.assert_called_once()
        fake_cursor.fetchall.assert_called_once()

    def test_market_pool_total(self):
        fake_cursor = MagicMock()
        fake_cursor.__enter__.return_value = fake_cursor
        fake_cursor.fetchall.return_value = [
            ("Iron Fury", Decimal("150")),
            ("Midnight Relay", Decimal("100")),
            ("Iron Fury", Decimal("50")),
        ]

        fake_conn = MagicMock()
        fake_conn.cursor.return_value = fake_cursor

        market_mock = MagicMock()
        market_mock.get_connection.return_value = fake_conn

        with patch.object(betting_service, "market_db", market_mock):
            total = betting_service.get_market_pool_total(42)

        self.assertEqual(total, Decimal("300"))
        fake_conn.close.assert_called_once()
        fake_cursor.fetchall.assert_called_once()


if __name__ == "__main__":
    unittest.main()
