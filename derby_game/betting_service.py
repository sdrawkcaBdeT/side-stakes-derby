from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Optional

from derby_game.config import BALANCE_CONFIG
from derby_game.database import queries as derby_queries
from derby_game.simulation import Bookie, Race

try:
    from market import database as market_db
except ImportError:  # pragma: no cover
    market_db = None

MARKET_RACES_TABLE = "derby.market_races"
MARKET_BETS_TABLE = "derby.market_bets"
MARKET_RACE_HORSES_TABLE = "derby.market_race_horses"

HOUSE_VIG = Decimal(str(BALANCE_CONFIG["economy"].get("house_vig", 0.08)))
PROB_WEIGHT = Decimal("0.6")
POOL_WEIGHT = Decimal("0.4")
MIN_ODDS = Decimal("0.10")
MONTE_CARLO_SIMULATIONS = 500

_probability_cache: Dict[int, Dict[int, Decimal]] = {}


@dataclass
class BetResult:
    success: bool
    message: str
    locked_odds: Optional[float] = None


def clear_live_odds_cache(race_id: int | None = None):
    """
    Removes cached Monte Carlo probabilities.
    """
    if race_id is None:
        _probability_cache.clear()
    else:
        _probability_cache.pop(race_id, None)


def _ensure_market_race(race_obj: Race):
    if not market_db:
        return
    conn = market_db.get_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT 1 FROM {MARKET_RACES_TABLE} WHERE race_id = %s;",
                (race_obj.race_id,)
            )
            exists = cur.fetchone() is not None
    finally:
        conn.close()

    if exists:
        return

    horses = []
    for horse in race_obj.horses:
        horses.append(
            type("MarketHorse", (), {
                "name": horse.name,
                "strategy_name": horse.strategy,
                "stats": {
                    "spd": horse.spd,
                    "sta": horse.sta,
                    "acc": getattr(horse, "acc", horse.spd),
                    "fcs": horse.fcs,
                    "grt": horse.grt,
                    "cog": horse.cog,
                    "lck": horse.lck,
                }
            })()
        )

    try:
        market_db.create_race(race_obj.race_id, race_obj.distance, horses)
    except Exception as err:
        print(f"  -> Failed to register race {race_obj.race_id} with market DB: {err}")


def _fetch_market_bets(race_id: int):
    if not market_db:
        return []
    conn = market_db.get_connection()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT horse_name, amount FROM {MARKET_BETS_TABLE} WHERE race_id = %s;",
                (race_id,)
            )
            return cur.fetchall()
    finally:
        conn.close()

def _insert_market_bet(race_id: int, bettor_id: str, horse_name: str, amount: float, odds: float) -> bool:
    if not market_db:
        return False
    conn = market_db.get_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {MARKET_BETS_TABLE} (race_id, bettor_id, horse_name, amount, locked_in_odds)
                VALUES (%s, %s, %s, %s, %s);
                """,
                (race_id, bettor_id, horse_name, amount, odds)
            )
        conn.commit()
        return True
    except Exception as err:
        conn.rollback()
        print(f"  -> Failed to record bet for race {race_id}: {err}")
        return False
    finally:
        conn.close()

def summarize_market_pool(race_id: int):
    """
    Aggregates total wager amounts per horse and overall pool for a race.
    Returns (totals_by_horse_name, total_pool).
    """
    totals_by_name: Dict[str, Decimal] = {}
    total_pool = Decimal("0")
    for horse_name, amount in _fetch_market_bets(race_id):
        amount_dec = Decimal(str(amount))
        totals_by_name[horse_name] = totals_by_name.get(horse_name, Decimal("0")) + amount_dec
        total_pool += amount_dec
    return totals_by_name, total_pool


def calculate_live_odds(race_id: int) -> Dict[int, Dict[str, Decimal]]:
    race_obj = Race(race_id, verbose=False)
    if not race_obj.horses:
        return {}

    base_probabilities = _probability_cache.get(race_id)
    if base_probabilities is None:
        bookie = Bookie(race_obj)
        bookie.run_monte_carlo(simulations=MONTE_CARLO_SIMULATIONS)
        base_probabilities = {
            horse_id: Decimal(str(prob))
            for horse_id, prob in bookie.win_probabilities.items()
        }
        _probability_cache[race_id] = base_probabilities

    totals_by_name, total_pool = summarize_market_pool(race_id)

    live_odds = {}
    for horse in race_obj.horses:
        prob = base_probabilities.get(horse.horse_id, Decimal("0.05"))
        share = Decimal("0")
        if total_pool > 0:
            share = totals_by_name.get(horse.name, Decimal("0")) / total_pool
        blended = (prob * PROB_WEIGHT) + (share * POOL_WEIGHT)
        if blended <= Decimal("0"):
            blended = Decimal("0.02")
        fair_odds = (Decimal("1") / blended) - Decimal("1")
        priced = (fair_odds * (Decimal("1") - HOUSE_VIG)).quantize(Decimal("0.01"))
        if priced < MIN_ODDS:
            priced = MIN_ODDS
        live_odds[horse.horse_id] = {
            "name": horse.name,
            "prob": blended,
            "odds": priced,
        }
    return live_odds


def place_player_bet(discord_id: str, race_id: int, horse_id: int, amount: int) -> BetResult:
    if amount <= 0:
        return BetResult(False, "Bet amount must be greater than zero.")
    if not market_db:
        return BetResult(False, "Betting system unavailable.")

    race = derby_queries.get_race_details(race_id)
    if not race:
        return BetResult(False, "Race not found.")

    status = (race["status"] or "").lower()
    if status != "open":
        return BetResult(False, "Betting is closed for this race.")

    start_time = race.get("start_time")
    if start_time:
        start_time = start_time.replace(tzinfo=timezone.utc) if start_time.tzinfo is None else start_time
        lock_lead = timedelta(minutes=BALANCE_CONFIG["racing"].get("lock_lead_minutes", 1))
        if datetime.now(timezone.utc) >= start_time - lock_lead:
            return BetResult(False, "Betting window has closed.")

    race_obj = Race(race_id, verbose=False)
    if not race_obj.horses:
        return BetResult(False, "Race has no active horses.")

    entries = {horse.horse_id: horse for horse in race_obj.horses}
    if horse_id not in entries:
        return BetResult(False, "Selected horse is not entered in this race.")

    _ensure_market_race(race_obj)
    odds_map = calculate_live_odds(race_id)
    odds_entry = odds_map.get(horse_id)
    if not odds_entry:
        return BetResult(False, "Failed to calculate odds for selected horse.")

    locked_odds = float(odds_entry["odds"])
    bettor_id_str = str(discord_id)
    details = {
        "race_id": race_id,
        "horse_id": horse_id,
        "horse_name": odds_entry["name"],
        "bet_type": "WIN",
        "locked_odds": locked_odds,
        "source": "player",
    }

    new_balance = market_db.execute_gambling_transaction(
        actor_id=bettor_id_str,
        game_name=f"Derby Race #{race_id}",
        bet_amount=float(amount),
        winnings=0.0,
        details=details,
    )
    if new_balance is None:
        return BetResult(False, "Bet failed. Check your balance and try again.")

    if not _insert_market_bet(race_id, bettor_id_str, odds_entry["name"], float(amount), locked_odds):
        market_db.execute_gambling_transaction(
            actor_id=bettor_id_str,
            game_name=f"Derby Race #{race_id}",
            bet_amount=0.0,
            winnings=float(amount),
            details={**details, "note": "bet_record_failed"},
        )
        return BetResult(False, "Bet failed to record. Your stake was refunded.")

    try:
        numeric_id = int(discord_id)
    except (TypeError, ValueError):
        numeric_id = None
    if numeric_id:
        derby_queries.ensure_trainer_record(numeric_id, is_bot=False)
        derby_queries.add_notification(
            numeric_id,
            f"[Bet] Bet placed on {odds_entry['name']} in Race #{race_id}: {amount:,} CC at {locked_odds:.2f} odds."
        )
    return BetResult(True, "Bet placed successfully.", locked_odds)

def place_bot_bet(discord_id: str, race_id: int, horse_id: int, amount: int) -> bool:
    if amount <= 0 or not market_db:
        return False

    race = derby_queries.get_race_details(race_id)
    if not race or (race["status"] or "").lower() != "open":
        return False

    race_obj = Race(race_id, verbose=False)
    if not race_obj.horses:
        return False

    entries = {horse.horse_id: horse for horse in race_obj.horses}
    if horse_id not in entries:
        return False

    odds_map = calculate_live_odds(race_id)
    odds_entry = odds_map.get(horse_id)
    if not odds_entry:
        return False

    locked_odds = float(odds_entry["odds"])
    bettor_id = str(discord_id)
    details = {
        "race_id": race_id,
        "horse_id": horse_id,
        "horse_name": odds_entry["name"],
        "bet_type": "WIN",
        "locked_odds": locked_odds,
        "source": "bot",
    }

    new_balance = market_db.execute_gambling_transaction(
        actor_id=bettor_id,
        game_name=f"Derby Race #{race_id}",
        bet_amount=float(amount),
        winnings=0.0,
        details=details,
    )
    if new_balance is None:
        return False

    if not _insert_market_bet(race_id, bettor_id, odds_entry["name"], float(amount), locked_odds):
        market_db.execute_gambling_transaction(
            actor_id=bettor_id,
            game_name=f"Derby Race #{race_id}",
            bet_amount=0.0,
            winnings=float(amount),
            details={**details, "note": "bet_record_failed"},
        )
        return False

    return True

def get_market_pool_total(race_id: int) -> Decimal:
    """
    Returns the total wagered CC for the given race.
    """
    _, total_pool = summarize_market_pool(race_id)
    return total_pool


def get_recent_player_bets(discord_id: str, limit: int = 10):
    """
    Returns a list of recent bets for the supplied Discord ID ordered by most recent first.
    Each entry includes amount, odds, and timestamp metadata.
    """
    if not market_db or limit <= 0:
        return []

    conn = market_db.get_connection()
    if not conn:
        return []

    rows = []
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT bet_id, race_id, horse_name, amount, locked_in_odds, placed_at
                FROM {MARKET_BETS_TABLE}
                WHERE bettor_id = %s
                ORDER BY placed_at DESC
                LIMIT %s;
                """,
                (discord_id, limit),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    recent = []
    for bet_id, race_id, horse_name, amount, odds, placed_at in rows:
        amount_dec = Decimal(str(amount))
        odds_dec = Decimal(str(odds))
        timestamp = placed_at
        if isinstance(timestamp, datetime) and (timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None):
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        recent.append(
            {
                "bet_id": bet_id,
                "race_id": race_id,
                "horse_name": horse_name,
                "amount": amount_dec,
                "locked_in_odds": odds_dec,
                "placed_at": timestamp,
            }
        )
    return recent
