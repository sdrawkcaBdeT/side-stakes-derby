"""
Utility script to run a single race via Race.run_simulation().

Usage:
    python scripts/run_race.py --race-id 123 --silent

If the hybrid engine flag is enabled (see configs/game_balance.json or
DERBY_USE_HYBRID_ENGINE), this will automatically use the new 2.5D loop
and persist telemetry into derby.race_rounds.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure repo root is on sys.path when script is executed from anywhere
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from derby_game.simulation import Race  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Derby race simulation.")
    parser.add_argument("--race-id", type=int, required=True, help="Race ID to simulate.")
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Suppress console output (telemetry still saves when hybrid flag is on).",
    )
    args = parser.parse_args()

    race = Race(args.race_id, verbose=not args.silent)
    results = race.run_simulation(silent=args.silent)

    if args.silent:
        print(f"Race {args.race_id} completed.")
    else:
        print("\nFinish Order:")
        for idx, horse in enumerate(results, start=1):
            print(f"{idx}. {horse.name} (ID {horse.horse_id})")


if __name__ == "__main__":
    main()
