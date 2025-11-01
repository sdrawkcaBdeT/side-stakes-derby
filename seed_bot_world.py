"""
Utility script to seed the Derby world with bot trainers, bettors, wallets, and horses.

Usage:
    python seed_bot_world.py --tier G --horses-per-bot 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Dict, Any, List

from derby_game.database import queries as derby_queries
from derby_game.simulation import Horse

# Ensure we can import the shared market database helpers
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OTHER_REPO_PATH = os.path.join(os.path.dirname(PROJECT_ROOT), "prettyDerbyClubAnalysis")
if OTHER_REPO_PATH not in sys.path:
    sys.path.append(OTHER_REPO_PATH)

try:
    from market import database as market_db
except ImportError:  # pragma: no cover
    market_db = None

PERSONALITIES_PATH = os.path.join(PROJECT_ROOT, "configs", "bot_personalities.json")


@dataclass
class WalletSeed:
    discord_id: str
    ingamename: str
    balance: int


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "bot"


def load_betting_profiles() -> List[Dict[str, Any]]:
    try:
        with open(PERSONALITIES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("WARNING: bot_personalities.json not found; betting wallets will not be seeded.")
    except json.JSONDecodeError as e:
        print(f"WARNING: Failed to parse bot_personalities.json ({e}); skipping betting wallets.")
    return []


def upsert_market_wallet(wallet: WalletSeed):
    if not market_db:
        raise RuntimeError("market.database module not available; cannot seed wallets.")

    conn = market_db.get_connection()
    if not conn:
        raise RuntimeError("Failed to obtain market DB connection.")

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO balances (discord_id, ingamename, balance)
                VALUES (%s, %s, %s)
                ON CONFLICT (discord_id) DO UPDATE
                SET ingamename = EXCLUDED.ingamename,
                    balance     = EXCLUDED.balance;
                """,
                (wallet.discord_id, wallet.ingamename, wallet.balance),
            )
        conn.commit()
    finally:
        conn.close()


def ensure_trainer_profile(profile: Dict[str, Any], tier: str, horses_per_bot: int):
    user_id = int(profile["trainer_id"])
    economy_id = str(profile["discord_id"])
    horse_target = profile.get("stable_targets", {}).get(tier, horses_per_bot)
    if horse_target <= 0:
        return {"created_horses": 0, "existing_horses": 0}

    derby_queries.ensure_trainer_record(user_id, is_bot=True, economy_id=economy_id)
    upsert_market_wallet(
        WalletSeed(
            discord_id=economy_id,
            ingamename=profile["ingamename"],
            balance=profile.get("starting_balance", 0),
        )
    )

    existing_horses = derby_queries.get_trainer_horses(user_id, include_retired=False)
    current_count = len(existing_horses)
    horses_to_create = max(horse_target - current_count, 0)

    created_ids = []
    for _ in range(horses_to_create):
        horse_id = Horse.generate_new_horse(owner_id=user_id, is_bot=True)
        if horse_id:
            created_ids.append(horse_id)

    if created_ids:
        stats = list(derby_queries.TRAINABLE_STATS)
        for horse_id in created_ids:
            derby_queries.set_training_plan(horse_id, random.choice(stats))

    if existing_horses:
        stats = list(derby_queries.TRAINABLE_STATS)
        for horse in existing_horses:
            horse_id = horse["horse_id"]
            if not derby_queries.has_training_plan(horse_id):
                derby_queries.set_training_plan(horse_id, random.choice(stats))

    return {
        "created_horses": len(created_ids),
        "existing_horses": current_count,
        "horse_ids": created_ids,
    }


def ensure_bettor_wallets():
    profiles = load_betting_profiles()
    if not profiles or not market_db:
        return

    for profile in profiles:
        discord_id = str(profile.get("discord_id") or f"derby_bot_{slugify(profile['name'])}")
        balance = int(profile.get("starting_balance", profile.get("bankroll", 0)))
        upsert_market_wallet(
            WalletSeed(
                discord_id=discord_id,
                ingamename=profile.get("ingamename", profile["name"]),
                balance=balance,
            )
        )


def main():
    parser = argparse.ArgumentParser(description="Seed Derby bot trainers, bettors, and horses.")
    parser.add_argument("--tier", default="G", help="Tier to seed (default: G).")
    parser.add_argument(
        "--horses-per-bot",
        type=int,
        default=8,
        help="Target number of active horses per trainer bot when tier-specific target is absent.",
    )
    args = parser.parse_args()

    tier = args.tier.upper()
    print(f"--- Seeding Derby bot world (tier {tier}) ---")
    print(f"Target horses per trainer bot (fallback): {args.horses_per_bot}\n")

    profiles = load_betting_profiles()
    if not profiles:
        print("No bot personalities found. Nothing to seed.")
        return

    for profile in profiles:
        result = ensure_trainer_profile(profile, tier, args.horses_per_bot)
        print(
            f"[{profile['name']}] Existing horses: {result['existing_horses']} | "
            f"Created: {result['created_horses']}"
        )
        if result.get("horse_ids"):
            print(f"    -> New horse IDs: {', '.join(str(hid) for hid in result['horse_ids'])}")

    print("\nSeeding bettor wallets...")
    ensure_bettor_wallets()
    print("Seeding complete.")


if __name__ == "__main__":  # pragma: no cover
    main()
