"""
One-time helper script to populate the new `acc` stat for legacy horses.

Usage:
    python scripts/backfill_acc.py

The script:
1. Selects every horse whose `acc` is zero.
2. Generates a random acceleration score using the same distribution as SPD/STA.
3. Recomputes HG with the new value and persists the update.
"""

from __future__ import annotations

import numpy as np

from derby_game.config import BALANCE_CONFIG
from derby_game.database.connection import get_db_connection
from derby_game.simulation import Horse


def _rand_stat() -> int:
    cfg = BALANCE_CONFIG["horse_generation"]["base_stats"]
    return int(
        np.clip(
            np.random.normal(cfg["mean"], cfg["std_dev"]),
            cfg["min"],
            cfg["max"],
        )
    )


def backfill_acc() -> None:
    conn = get_db_connection()
    updated = 0
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT horse_id, spd, sta, acc, fcs, grt, cog, lck
                FROM horses
                WHERE COALESCE(acc, 0) = 0
                FOR UPDATE
                """
            )
            rows = cur.fetchall()

        if not rows:
            print("All horses already have acceleration values.")
            return

        with conn.cursor() as cur:
            for horse_id, spd, sta, _acc, fcs, grt, cog, lck in rows:
                acc = _rand_stat()
                hg_score = Horse._calculate_hg(spd, sta, acc, fcs, grt, cog, lck)
                cur.execute(
                    "UPDATE horses SET acc = %s, hg_score = %s WHERE horse_id = %s",
                    (acc, hg_score, horse_id),
                )
                updated += 1

        conn.commit()
        print(f"Updated acceleration for {updated} horses.")
    except Exception as exc:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    backfill_acc()
