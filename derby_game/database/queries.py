from derby_game.database.connection import get_db_connection
from datetime import datetime, timedelta, timezone
from derby_game.config import BALANCE_CONFIG
from typing import Optional, Sequence, Dict, Any, List
import traceback
import json
import os
import sys
import psycopg2.extras as pg_extras

# Get the path to the current script's directory (side-stakes-derby/derby_game/database/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Step up to the side-stakes-derby root
project_root = os.path.dirname(os.path.dirname(current_dir))
# The sibling repository lives beside side-stakes-derby
other_project_path = os.path.join(os.path.dirname(project_root), 'prettyDerbyClubAnalysis')
if other_project_path not in sys.path:
    sys.path.append(other_project_path)

# --- Race Queries ---
try:
    import market.database as market_db
except ImportError:
    print("FATAL ERROR: Could not import 'market.database'. Make sure 'prettyDerbyClubAnalysis' project is accessible.")
    market_db = None

MARKET_BETS_TABLE = "derby.market_bets"
TRAINABLE_STATS = {"spd", "sta", "acc", "fcs", "grt", "cog"}
TRAINING_DURATION_HOURS = BALANCE_CONFIG['training'].get('session_duration_hours', 16)
TRAINING_FEE = BALANCE_CONFIG['economy']['training_fee']
MARKET_ADMIN_ID = "derby_system"

def get_available_races(tier_filter: str = None):
    """
    Fetches races from the database that are 'pending' or 'open'.
    Optionally filters by tier.

    Returns:
        list: A list of dictionaries, where each dict represents a race.
              Returns empty list on error or if no races found.
    """
    conn = None
    races = []
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            sql = """
                SELECT race_id, tier, distance, entry_fee, status, start_time, purse
                FROM races
                WHERE status IN ('pending', 'open')
            """
            params = []
            if tier_filter:
                sql += " AND tier = %s"
                params.append(tier_filter.upper()) # Ensure tier is uppercase

            sql += " ORDER BY start_time NULLS FIRST, race_id;" # Show scheduled first

            cur.execute(sql, params)
            results = cur.fetchall()

            # Format results as dictionaries for easier use
            for row in results:
                races.append({
                    "race_id": row[0],
                    "tier": row[1],
                    "distance": row[2],
                    "entry_fee": row[3],
                    "status": row[4],
                    "start_time": row[5], # Will be datetime object or None
                    "purse": row[6]
                })
    except Exception as e:
        print(f"Error in get_available_races: {e}")
        # Optionally rollback if it was a write error, though this is read-only
        # if conn: conn.rollback()
    finally:
        if conn:
            conn.close()
    return races

# --- Race Queries ---

def get_race_details(race_id: int):
    """
    Fetches details for a single race by its ID.

    Returns:
        dict: A dictionary representing the race, or None if not found/error.
    """
    conn = None
    race_details = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT race_id, tier, distance, entry_fee, status, start_time, purse, winner_horse_id "
                "FROM races WHERE race_id = %s",
                (race_id,)
            )
            result = cur.fetchone()
            if result:
                race_details = {
                    "race_id": result[0],
                    "tier": result[1],
                    "distance": result[2],
                    "entry_fee": result[3],
                    "status": result[4],
                    "start_time": result[5],
                    "purse": result[6],
                    "winner_horse_id": result[7]
                }
    except Exception as e:
        print(f"Error in get_race_details for race {race_id}: {e}")
    finally:
        if conn:
            conn.close()
    return race_details

def get_horses_in_race(race_id: int):
    """
    Fetches details of horses entered in a specific race.

    Returns:
        list: A list of dictionaries, each representing a horse entry, or empty list.
    """
    conn = None
    horses = []
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Join race_entries with horses to get horse details
            cur.execute(
                """
                SELECT
                    h.horse_id, h.name, h.strategy,
                    h.min_preferred_distance, h.max_preferred_distance,
                    h.spd, h.sta, h.acc, h.fcs, h.grt, h.cog, h.lck, h.hg_score
                FROM race_entries re
                JOIN horses h ON re.horse_id = h.horse_id
                WHERE re.race_id = %s
                ORDER BY h.name; -- Or order by post position if you add that later
                """,
                (race_id,)
            )
            results = cur.fetchall()
            for row in results:
                horses.append({
                    "horse_id": row[0],
                    "name": row[1],
                    "strategy": row[2],
                    "min_pref_dist": row[3],
                    "max_pref_dist": row[4],
                    "spd": row[5],
                    "sta": row[6],
                    "acc": row[7],
                    "fcs": row[8],
                    "grt": row[9],
                    "cog": row[10],
                    "lck": row[11],
                    "hg_score": row[12]
                })
    except Exception as e:
        print(f"Error in get_horses_in_race for race {race_id}: {e}")
    finally:
        if conn:
            conn.close()
    return horses

def list_race_entries(race_id: int):
    """
    Returns raw race_entries rows for a race, including entry metadata.
    """
    conn = None
    entries = []
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT entry_id, race_id, horse_id, entry_fee, is_bot_entry, entered_at
                FROM race_entries
                WHERE race_id = %s
                ORDER BY entered_at, entry_id;
                """,
                (race_id,)
            )
            for row in cur.fetchall():
                entries.append({
                    "entry_id": row[0],
                    "race_id": row[1],
                    "horse_id": row[2],
                    "entry_fee": row[3],
                    "is_bot_entry": row[4],
                    "entered_at": row[5]
                })
    except Exception as e:
        print(f"Error listing race entries for race {race_id}: {e}")
    finally:
        if conn:
            conn.close()
    return entries

def ensure_race_broadcast_record(race_id: int):
    """
    Ensures the race_broadcasts table has a row for the given race.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO derby.race_broadcasts (race_id)
                VALUES (%s)
                ON CONFLICT (race_id) DO NOTHING;
                """,
                (race_id,)
            )
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error ensuring broadcast record for race {race_id}: {e}")
    finally:
        if conn:
            conn.close()

def get_race_broadcast(race_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieves broadcast metadata for a race if available.
    """
    conn = None
    record = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT race_id, lobby_channel_id, lobby_message_id,
                       live_message_id, summary_message_id,
                       bet_thread_id, last_logged_bet_id,
                       broadcast_status, last_odds,
                       created_at, updated_at
                FROM derby.race_broadcasts
                WHERE race_id = %s;
                """,
                (race_id,)
            )
            row = cur.fetchone()
            if row:
                record = {
                    "race_id": row[0],
                    "lobby_channel_id": row[1],
                    "lobby_message_id": row[2],
                    "live_message_id": row[3],
                    "summary_message_id": row[4],
                    "bet_thread_id": row[5],
                    "last_logged_bet_id": row[6],
                    "broadcast_status": row[7],
                    "last_odds": row[8],
                    "created_at": row[9],
                    "updated_at": row[10],
                }
    except Exception as e:
        print(f"Error retrieving race broadcast for race {race_id}: {e}")
    finally:
        if conn:
            conn.close()
    return record

def update_race_broadcast(race_id: int, **fields) -> bool:
    """
    Updates the broadcast metadata table for a race.
    """
    if not fields:
        return True

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            assignments = []
            params = []
            for column, value in fields.items():
                if column == "last_odds" and value is not None:
                    assignments.append(f"{column} = %s")
                    params.append(pg_extras.Json(value))
                else:
                    assignments.append(f"{column} = %s")
                    params.append(value)
            assignments.append("updated_at = NOW()")
            params.append(race_id)
            sql = f"""
                UPDATE derby.race_broadcasts
                SET {', '.join(assignments)}
                WHERE race_id = %s;
            """
            cur.execute(sql, params)
        conn.commit()
        return True
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error updating race broadcast for race {race_id}: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_race_rounds(race_id: int):
    """
    Returns ordered round logs for a race.
    """
    conn = None
    rounds = []
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT round_number, horse_id, movement_roll,
                       stamina_multiplier, final_position, round_events
                FROM derby.race_rounds
                WHERE race_id = %s
                ORDER BY round_number, horse_id;
                """,
                (race_id,)
            )
            for row in cur.fetchall():
                rounds.append({
                    "round_number": row[0],
                    "horse_id": row[1],
                    "movement_roll": row[2],
                    "stamina_multiplier": row[3],
                    "final_position": row[4],
                    "round_events": row[5],
                })
    except Exception as e:
        print(f"Error fetching race rounds for race {race_id}: {e}")
    finally:
        if conn:
            conn.close()
    return rounds

def get_race_results_with_horses(race_id: int):
    """
    Returns finished race results with horse metadata.
    """
    conn = None
    results = []
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT rr.horse_id, rr.finish_position, rr.payout,
                       h.name, h.owner_id
                FROM derby.race_results rr
                JOIN derby.horses h ON h.horse_id = rr.horse_id
                WHERE rr.race_id = %s
                ORDER BY rr.finish_position, rr.horse_id;
                """,
                (race_id,)
            )
            for row in cur.fetchall():
                results.append({
                    "horse_id": row[0],
                    "finish_position": row[1],
                    "payout": float(row[2]) if row[2] is not None else 0.0,
                    "name": row[3],
                    "owner_id": row[4],
                })
    except Exception as e:
        print(f"Error fetching race results for race {race_id}: {e}")
    finally:
        if conn:
            conn.close()
    return results

def get_market_bets_since(race_id: int, last_bet_id: int = 0):
    """
    Fetches market bets for a race placed after the specified bet_id.
    """
    if not market_db:
        return []
    conn = market_db.get_connection()
    if not conn:
        return []
    bets = []
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT bet_id, bettor_id, horse_name, amount, locked_in_odds, placed_at
                FROM {MARKET_BETS_TABLE}
                WHERE race_id = %s AND bet_id > %s
                ORDER BY bet_id;
                """,
                (race_id, last_bet_id)
            )
            for row in cur.fetchall():
                bets.append({
                    "bet_id": row[0],
                    "bettor_id": row[1],
                    "horse_name": row[2],
                    "amount": row[3],
                    "locked_in_odds": row[4],
                    "placed_at": row[5],
                })
    except Exception as e:
        print(f"Error fetching market bets for race {race_id}: {e}")
    finally:
        conn.close()
    return bets

def count_race_entries(race_id: int):
    """
    Returns total number of entries for a race.
    """
    conn = None
    count = 0
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM race_entries WHERE race_id = %s;",
                (race_id,)
            )
            result = cur.fetchone()
            if result:
                count = result[0]
    except Exception as e:
        print(f"Error counting race entries for race {race_id}: {e}")
    finally:
        if conn:
            conn.close()
    return count

def clear_race_entries(race_id: int):
    """
    Deletes all entries for a race.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM race_entries WHERE race_id = %s;", (race_id,))
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error clearing race entries for race {race_id}: {e}")
    finally:
        if conn:
            conn.close()

def clear_race_results(race_id: int):
    """
    Deletes existing race results for a race.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM race_results WHERE race_id = %s;", (race_id,))
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error clearing race results for race {race_id}: {e}")
    finally:
        if conn:
            conn.close()

def record_race_result(race_id: int, horse_id: int, position: int, payout: int = 0):
    """
    Records an individual race result row (upsert).
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO race_results (race_id, horse_id, finish_position, payout)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (race_id, horse_id)
                DO UPDATE SET finish_position = EXCLUDED.finish_position,
                              payout = EXCLUDED.payout;
                """,
                (race_id, horse_id, position, payout)
            )
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error recording race result for race {race_id}, horse {horse_id}: {e}")
    finally:
        if conn:
            conn.close()

def set_race_winner(race_id: int, horse_id: int):
    """
    Stores the winning horse for a race.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE races SET winner_horse_id = %s WHERE race_id = %s;",
                (horse_id, race_id)
            )
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error setting winner for race {race_id}: {e}")
    finally:
        if conn:
            conn.close()

def is_bet_settled(market_bet_id: int) -> bool:
    """
    Checks whether a market bet has already been settled in Derby.
    """
    conn = None
    settled = False
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM race_bets WHERE market_bet_id = %s;",
                (market_bet_id,)
            )
            settled = cur.fetchone() is not None
    except Exception as e:
        print(f"Error checking bet settlement for bet {market_bet_id}: {e}")
    finally:
        if conn:
            conn.close()
    return settled

def record_bet_settlement(market_bet_id: int, race_id: int, bettor_id: str,
                          horse_id: Optional[int], amount: float, odds: float, winnings: float):
    """
    Records the settlement of a market bet in Derby's tracking table.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO race_bets (market_bet_id, race_id, bettor_id, horse_id, amount, odds, winnings, settled_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (market_bet_id)
                DO UPDATE SET winnings = EXCLUDED.winnings,
                              settled_at = EXCLUDED.settled_at;
                """,
                (market_bet_id, race_id, bettor_id, horse_id, amount, odds, winnings)
            )
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error recording bet settlement for bet {market_bet_id}: {e}")
    finally:
        if conn:
            conn.close()

def get_race_bet_summary(race_id: int):
    """
    Returns aggregated betting outcomes for the given race, sorted by net winnings.
    """
    conn = None
    results = []
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT bettor_id,
                       COALESCE(SUM(amount), 0) AS total_staked,
                       COALESCE(SUM(winnings), 0) AS total_won,
                       COALESCE(SUM(winnings) - SUM(amount), 0) AS net_result
                FROM derby.race_bets
                WHERE race_id = %s
                GROUP BY bettor_id
                ORDER BY net_result DESC, bettor_id;
                """,
                (race_id,)
            )
            for row in cur.fetchall():
                results.append(
                    {
                        "bettor_id": row[0],
                        "staked": float(row[1]),
                        "won": float(row[2]),
                        "net": float(row[3]),
                    }
                )
    except Exception as e:
        print(f"Error summarizing bets for race {race_id}: {e}")
    finally:
        if conn:
            conn.close()
    return results

def add_notification(user_id: int, message: str):
    """
    Inserts a notification for a trainer/player.
    """
    if not user_id:
        return
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO notifications (user_id, message) VALUES (%s, %s);",
                (user_id, message)
            )
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error adding notification for user {user_id}: {e}")
    finally:
        if conn:
            conn.close()

def create_race(tier: str, distance: int, entry_fee: int = 0, purse: int = 0,
                start_time: Optional[datetime] = None, status: str = "pending"):
    """
    Creates a new race record and returns the race metadata.
    """
    conn = None
    race = None
    tier = tier.upper()
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO races (tier, distance, entry_fee, purse, status, start_time)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING race_id, tier, distance, entry_fee, status, start_time, purse, winner_horse_id;
                """,
                (tier, distance, entry_fee, purse, status, start_time)
            )
            row = cur.fetchone()
            conn.commit()
            race = {
                "race_id": row[0],
                "tier": row[1],
                "distance": row[2],
                "entry_fee": row[3],
                "status": row[4],
                "start_time": row[5],
                "purse": row[6],
                "winner_horse_id": row[7]
            }
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error creating race (tier={tier}, distance={distance}): {e}")
    finally:
        if conn:
            conn.close()
    return race

def add_race_entry(race_id: int, horse_id: int, entry_fee: int = 0, *, is_bot_entry: bool = True):
    """
    Adds a horse to a race and returns the inserted entry.
    """
    conn = None
    entry = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO race_entries (race_id, horse_id, entry_fee, is_bot_entry)
                VALUES (%s, %s, %s, %s)
                RETURNING entry_id, entered_at;
                """,
                (race_id, horse_id, entry_fee, is_bot_entry)
            )
            row = cur.fetchone()
            conn.commit()
            entry = {
                "entry_id": row[0],
                "race_id": race_id,
                "horse_id": horse_id,
                "entry_fee": entry_fee,
                "is_bot_entry": is_bot_entry,
                "entered_at": row[1]
            }
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error adding race entry (race {race_id}, horse {horse_id}): {e}")
    finally:
        if conn:
            conn.close()
    return entry

def get_races_by_status(statuses: Sequence[str]):
    """
    Returns a list of races matching any of the provided statuses.
    """
    conn = None
    races = []
    status_list = list({s.lower() for s in statuses})
    if not status_list:
        return races
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT race_id, tier, distance, entry_fee, purse, status, start_time, winner_horse_id
                FROM races
                WHERE LOWER(status) = ANY(%s)
                ORDER BY start_time NULLS LAST, race_id;
                """,
                (status_list,)
            )
            for row in cur.fetchall():
                races.append({
                    "race_id": row[0],
                    "tier": row[1],
                    "distance": row[2],
                    "entry_fee": row[3],
                    "purse": row[4],
                    "status": row[5],
                    "start_time": row[6],
                    "winner_horse_id": row[7]
                })
    except Exception as e:
        print(f"Error fetching races by status {statuses}: {e}")
    finally:
        if conn:
            conn.close()
    return races

def remove_race_entry(race_id: int, horse_id: int):
    """
    Removes a specific horse from a race.
    Returns the deleted entry metadata (for refunds) or None.
    """
    conn = None
    entry = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM race_entries
                WHERE race_id = %s AND horse_id = %s
                RETURNING entry_id, entry_fee, is_bot_entry, entered_at;
                """,
                (race_id, horse_id)
            )
            row = cur.fetchone()
            if row:
                entry = {
                    "entry_id": row[0],
                    "race_id": race_id,
                    "horse_id": horse_id,
                    "entry_fee": row[1],
                    "is_bot_entry": row[2],
                    "entered_at": row[3]
                }
            conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error removing race entry (race {race_id}, horse {horse_id}): {e}")
    finally:
        if conn:
            conn.close()
    return entry

def get_race_entry(race_id: int, horse_id: int):
    """
    Returns a specific race entry if present.
    """
    conn = None
    entry = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT entry_id, entry_fee, is_bot_entry, entered_at
                FROM race_entries
                WHERE race_id = %s AND horse_id = %s;
                """,
                (race_id, horse_id)
            )
            row = cur.fetchone()
            if row:
                entry = {
                    "entry_id": row[0],
                    "entry_fee": row[1],
                    "is_bot_entry": row[2],
                    "entered_at": row[3]
                }
    except Exception as e:
        print(f"Error fetching race entry for race {race_id}, horse {horse_id}: {e}")
    finally:
        if conn:
            conn.close()
    return entry

def get_oldest_bot_entry(race_id: int):
    """
    Returns the oldest bot entry for the specified race.
    """
    conn = None
    entry = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT horse_id, entry_fee, entered_at
                FROM race_entries
                WHERE race_id = %s AND is_bot_entry = TRUE
                ORDER BY entered_at, entry_id
                LIMIT 1;
                """,
                (race_id,)
            )
            row = cur.fetchone()
            if row:
                entry = {
                    "horse_id": row[0],
                    "entry_fee": row[1],
                    "entered_at": row[2]
                }
    except Exception as e:
        print(f"Error fetching oldest bot entry for race {race_id}: {e}")
    finally:
        if conn:
            conn.close()
    return entry

def update_race_status(race_id: int, new_status: str):
    """
    Updates a race's status.
    """
    conn = None
    success = False
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE races SET status = %s WHERE race_id = %s",
                (new_status, race_id)
            )
            success = cur.rowcount > 0
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error updating race {race_id} to status {new_status}: {e}")
        success = False
    finally:
        if conn:
            conn.close()
    return success

def update_race_start_time(race_id: int, start_time: datetime):
    """
    Updates the start_time for a race.
    """
    conn = None
    success = False
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE races SET start_time = %s WHERE race_id = %s",
                (start_time, race_id)
            )
            success = cur.rowcount > 0
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error updating start_time for race {race_id}: {e}")
        success = False
    finally:
        if conn:
            conn.close()
    return success

def _player_entry_fee_transaction(trainer_id: int, cost: int):
    """
    Charges the player's CC balance using the shared market DB.
    Returns tuple(success, new_balance|message).
    """
    if not market_db:
        return False, "Economy system unavailable. Please try again later."

    actor_id = get_trainer_economy_id(trainer_id)

    try:
        new_balance = market_db.execute_purchase_transaction(
            actor_id=actor_id,
            item_name="Race Entry Fee",
            cost=cost,
            upgrade_tier=None
        )
    except Exception as e:
        print(f"Error charging entry fee for trainer {trainer_id}: {e}")
        new_balance = None

    if new_balance is None:
        return False, "Insufficient CC or billing system error."

    return True, new_balance

def _player_entry_fee_refund(trainer_id: int, amount: int):
    """
    Refunds CC to the player if an entry is cancelled before the race locks.
    """
    if not market_db:
        print("Warning: market_db unavailable. Cannot refund entry fee automatically.")
        return False

    target_id = get_trainer_economy_id(trainer_id)

    try:
        new_balance = market_db.execute_admin_award(
            admin_id="derby_system_refund",
            target_id=target_id,
            amount=amount
        )
    except Exception as e:
        print(f"Error refunding entry fee for trainer {trainer_id}: {e}")
        return False

    return True

def player_enter_race(trainer_id: int, horse_id: int, race_id: int):
    """
    Allows a player to enter a pending race, swapping out a bot if needed.
    Returns (success, message).
    """
    race = get_race_details(race_id)
    if not race:
        return False, "Race not found."

    if race['status'].lower() != 'pending':
        return False, "This race is no longer accepting new entries."

    now = datetime.now(timezone.utc)
    start_time = race.get('start_time')
    betting_window = timedelta(minutes=BALANCE_CONFIG['racing'].get('betting_window_minutes', 10))
    if start_time and now >= start_time - betting_window:
        return False, "Betting has opened for this race. Entries are closed."

    # Validate horse ownership and eligibility
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT owner_id, is_retired, in_training_until
                FROM horses
                WHERE horse_id = %s
                """,
                (horse_id,)
            )
            horse_row = cur.fetchone()
            if not horse_row:
                return False, "Horse not found."
            if horse_row[0] != trainer_id:
                return False, "You do not own this horse."
            if horse_row[1]:
                return False, "This horse is retired and cannot enter races."
            if horse_row[2] and horse_row[2] > now:
                return False, "This horse is currently in training."
    finally:
        if conn:
            conn.close()

    existing_entry = get_race_entry(race_id, horse_id)
    if existing_entry:
        return False, "This horse is already entered in the race."

    racing_cfg = BALANCE_CONFIG.get('racing', {})
    tier = race['tier']
    field_size = racing_cfg.get('field_size', {}).get(tier, 10)
    entry_fee = race.get('entry_fee', 0)

    current_entries = list_race_entries(race_id)

    slot_available = len(current_entries) < field_size
    bot_to_remove = None

    if not slot_available:
        bot_to_remove = get_oldest_bot_entry(race_id)
        if not bot_to_remove:
            return False, "The race is full and no bot slots are available to swap."

    # Charge entry fee
    if entry_fee > 0:
        success, charge_result = _player_entry_fee_transaction(trainer_id, entry_fee)
        if not success:
            return False, charge_result

    removed_bot = None
    if bot_to_remove:
        removed_bot = remove_race_entry(race_id, bot_to_remove['horse_id'])
        if not removed_bot:
            # Refund if charge already applied
            if entry_fee > 0:
                _player_entry_fee_refund(trainer_id, entry_fee)
            return False, "Failed to free a slot for your horse. Please try again."

    new_entry = add_race_entry(race_id, horse_id, entry_fee=entry_fee, is_bot_entry=False)
    if not new_entry:
        # Revert removal and refund
        if removed_bot:
            add_race_entry(race_id, bot_to_remove['horse_id'], entry_fee=removed_bot['entry_fee'], is_bot_entry=True)
        if entry_fee > 0:
            _player_entry_fee_refund(trainer_id, entry_fee)
        return False, "Failed to add your horse. Please try again."

    return True, f"Horse successfully entered into race {race_id}."

def player_withdraw_from_race(trainer_id: int, horse_id: int, race_id: int):
    """
    Allows a player to withdraw from a pending race. Refunds entry fee and optionally refills slot with a bot.
    Returns (success, message).
    """
    race = get_race_details(race_id)
    if not race:
        return False, "Race not found."

    if race['status'].lower() != 'pending':
        return False, "This race is locked for entries. You can no longer withdraw."

    entry = get_race_entry(race_id, horse_id)
    if not entry or entry['is_bot_entry']:
        return False, "Your horse is not entered in this race."

    if not ensure_trainer_record(trainer_id):
        return False, "Trainer not found."

    if not remove_race_entry(race_id, horse_id):
        return False, "Failed to withdraw your horse. Please try again."

    entry_fee = race.get('entry_fee', 0)
    if entry_fee > 0:
        _player_entry_fee_refund(trainer_id, entry_fee)

    racing_cfg = BALANCE_CONFIG.get('racing', {})
    tier = race['tier']
    field_size = racing_cfg.get('field_size', {}).get(tier, 10)
    current_entries = list_race_entries(race_id)
    if len(current_entries) < field_size:
        available = get_available_bot_horses(tier, limit=1)
        if available:
            add_race_entry(race_id, available[0]['horse_id'], entry_fee=entry_fee, is_bot_entry=True)

    return True, "Your horse has been withdrawn from the race."

def get_available_bot_horses(tier: str, limit: int = 10, *,
                             exclude_race_ids: Optional[Sequence[int]] = None):
    """
    Returns a list of bot-owned horses eligible for the requested tier.
    Currently tier gating is simple (future: use HG ranges per tier).
    """
    conn = None
    horses = []
    tier = tier.upper()
    exclude_race_ids = list(exclude_race_ids) if exclude_race_ids else []

    # Determine HG ceiling per tier (placeholder: derived from current config)
    base_cfg = BALANCE_CONFIG['horse_generation']['base_stats']
    lck_cfg = BALANCE_CONFIG['horse_generation']['lck_stats']
    weights = BALANCE_CONFIG['hg_formula_weights']

    base_max = base_cfg['max']
    lck_max = lck_cfg['max']

    hg_cap_by_tier = {
        "G": int(
            base_max * (
                weights['spd'] +
                weights.get('acc', 0) +
                weights['sta'] +
                weights['fcs'] +
                weights['grt'] +
                weights['cog']
            ) + lck_max * weights['lck']
        ),
    }
    hg_cap = hg_cap_by_tier.get(tier)

    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            sql = """
                SELECT h.horse_id, h.owner_id, h.name, h.hg_score
                FROM horses h
                WHERE h.is_bot = TRUE
                  AND h.is_retired = FALSE
                  AND (h.in_training_until IS NULL OR h.in_training_until <= NOW())
                  AND NOT EXISTS (
                        SELECT 1
                        FROM race_entries re
                        JOIN races r ON re.race_id = r.race_id
                        WHERE re.horse_id = h.horse_id
                          AND r.status IN ('pending', 'open', 'locked', 'running')
                          {exclude_clause}
                  )
                  {hg_clause}
                ORDER BY h.hg_score DESC
                LIMIT %s;
            """
            exclude_clause = ""
            params = []
            if exclude_race_ids:
                placeholders = ", ".join(["%s"] * len(exclude_race_ids))
                exclude_clause = f"AND r.race_id NOT IN ({placeholders})"
                params.extend(exclude_race_ids)

            hg_clause = ""
            if hg_cap:
                hg_clause = "AND h.hg_score <= %s"
                params.append(hg_cap)

            final_sql = sql.format(exclude_clause=exclude_clause, hg_clause=hg_clause)
            params.append(limit)
            cur.execute(final_sql, params)
            for row in cur.fetchall():
                horses.append({
                    "horse_id": row[0],
                    "owner_id": row[1],
                    "name": row[2],
                    "hg_score": row[3]
                })
    except Exception as e:
        print(f"Error fetching available bot horses for tier {tier}: {e}")
    finally:
        if conn:
            conn.close()
    return horses

# --- Trainer Helpers ---

def ensure_trainer_record(user_id: int, *, is_bot: Optional[bool] = None, economy_id: Optional[str] = None):
    """
    Ensures there is a trainer row for the supplied user ID.
    Returns a dict with trainer data or None if something goes wrong.
    """
    conn = None
    trainer = None
    desired_economy_id = economy_id or str(user_id)
    try:
        dirty = False
        conn = get_db_connection()
        with conn.cursor() as cur:
            insert_is_bot = is_bot if is_bot is not None else False
            cur.execute(
                """
                INSERT INTO trainers (user_id, is_bot, economy_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO NOTHING;
                """,
                (user_id, insert_is_bot, desired_economy_id)
            )
            dirty = cur.rowcount > 0
            cur.execute(
                "SELECT user_id, is_bot, prestige, stable_slots, economy_id FROM trainers WHERE user_id = %s",
                (user_id,)
            )
            row = cur.fetchone()
            if row:
                current_is_bot = row[1]
                current_economy_id = row[4]
                if is_bot is not None and current_is_bot != is_bot:
                    cur.execute(
                        "UPDATE trainers SET is_bot = %s WHERE user_id = %s",
                        (is_bot, user_id)
                    )
                    dirty = True
                    current_is_bot = is_bot
                if desired_economy_id and current_economy_id != desired_economy_id:
                    cur.execute(
                        "UPDATE trainers SET economy_id = %s WHERE user_id = %s",
                        (desired_economy_id, user_id)
                    )
                    dirty = True
                    current_economy_id = desired_economy_id
                trainer = {
                    "user_id": row[0],
                    "is_bot": current_is_bot,
                    "prestige": row[2],
                    "stable_slots": row[3],
                    "economy_id": current_economy_id,
                }
        if dirty:
            conn.commit()
    except Exception as e:
        print(f"Error ensuring trainer record for {user_id}: {e}")
    finally:
        if conn:
            conn.close()
    return trainer


def get_trainer_economy_id(user_id: int) -> str:
    """Returns the economy identifier used for shared wallet transactions."""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT economy_id FROM trainers WHERE user_id = %s;",
                (user_id,)
            )
            row = cur.fetchone()
            if row and row[0]:
                return row[0]
    except Exception as e:
        print(f"Error fetching economy ID for trainer {user_id}: {e}")
    finally:
        if conn:
            conn.close()
    return str(user_id)

def get_trainer_horses(user_id: int, *, include_retired: bool = False):
    """
    Fetches all horses owned by a specific trainer.
    """
    conn = None
    horses = []
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            sql = """
                SELECT h.horse_id, h.name, h.strategy,
                       h.min_preferred_distance, h.max_preferred_distance,
                       h.birth_timestamp,
                       h.spd, h.sta, h.acc, h.fcs, h.grt, h.cog, h.lck,
                       h.hg_score, h.is_retired, h.in_training_until,
                       plan.stat_code, plan.is_active
                FROM horses h
                LEFT JOIN horse_training_plans plan ON plan.horse_id = h.horse_id
                WHERE h.owner_id = %s
            """
            params = [user_id]
            if not include_retired:
                sql += " AND is_retired = FALSE"
            sql += " ORDER BY horse_id"

            cur.execute(sql, params)
            rows = cur.fetchall()
            for row in rows:
                horses.append({
                    "horse_id": row[0],
                    "name": row[1],
                    "strategy": row[2],
                    "min_pref_distance": row[3],
                    "max_pref_distance": row[4],
                    "birth_timestamp": row[5],
                    "spd": row[6],
                    "sta": row[7],
                    "acc": row[8],
                    "fcs": row[9],
                    "grt": row[10],
                    "cog": row[11],
                    "lck": row[12],
                    "hg_score": row[13],
                    "is_retired": row[14],
                    "in_training_until": row[15],
                    "training_plan_stat": row[16],
                    "training_plan_active": row[17],
                })
    except Exception as e:
        print(f"Error getting horses for trainer {user_id}: {e}")
    finally:
        if conn:
            conn.close()
    return horses


def mark_training_plan_attempt(horse_id: int):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE horse_training_plans SET updated_at = NOW() WHERE horse_id = %s;",
                (horse_id,)
            )
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error marking training plan attempt for horse {horse_id}: {e}")
    finally:
        if conn:
            conn.close()


def get_auto_training_candidates(now: datetime) -> List[Dict[str, Any]]:
    conn = None
    candidates: List[Dict[str, Any]] = []
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT h.horse_id,
                       h.owner_id,
                       plan.stat_code,
                       t.is_bot
                FROM horse_training_plans plan
                JOIN horses h ON plan.horse_id = h.horse_id
                JOIN trainers t ON h.owner_id = t.user_id
                WHERE plan.is_active = TRUE
                  AND h.is_retired = FALSE
                  AND (h.in_training_until IS NULL OR h.in_training_until <= %s)
                  AND NOT EXISTS (
                        SELECT 1 FROM training_queue q WHERE q.horse_id = h.horse_id
                  )
                  AND (plan.updated_at IS NULL OR plan.updated_at <= %s)
                ORDER BY plan.updated_at NULLS FIRST;
                """,
                (now, now - timedelta(minutes=5))
            )
            for row in cur.fetchall():
                candidates.append(
                    {
                        "horse_id": row[0],
                        "owner_id": row[1],
                        "stat_code": row[2],
                        "is_bot": bool(row[3]),
                    }
                )
    except Exception as e:
        print(f"Error fetching auto-training candidates: {e}")
    finally:
        if conn:
            conn.close()
    return candidates


def set_training_plan(horse_id: int, stat_code: str, active: bool = True):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO horse_training_plans (horse_id, stat_code, is_active, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (horse_id)
                DO UPDATE SET stat_code = EXCLUDED.stat_code,
                              is_active = EXCLUDED.is_active,
                              updated_at = NOW();
                """,
                (horse_id, stat_code.lower(), active)
            )
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error setting training plan for horse {horse_id}: {e}")
    finally:
        if conn:
            conn.close()


def deactivate_training_plan(horse_id: int):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE horse_training_plans SET is_active = FALSE, updated_at = NOW() WHERE horse_id = %s;",
                (horse_id,)
            )
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error deactivating training plan for horse {horse_id}: {e}")
    finally:
        if conn:
            conn.close()


def has_training_plan(horse_id: int) -> bool:
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM horse_training_plans WHERE horse_id = %s;",
                (horse_id,)
            )
            return cur.fetchone() is not None
    except Exception as e:
        print(f"Error checking training plan for horse {horse_id}: {e}")
    finally:
        if conn:
            conn.close()
    return False

def start_training_session(trainer_id: int, horse_id: int, stat: str):
    """
    Attempts to queue a training session for the requested stat.
    Returns (success: bool, message: str, finish_time: Optional[datetime]).
    """
    stat_code = (stat or "").lower()
    if stat_code not in TRAINABLE_STATS:
        return False, "Invalid stat. Choose from SPD, STA, ACC, FCS, GRT, or COG.", None

    conn = None
    now_utc = datetime.now(timezone.utc)
    finish_time = now_utc + timedelta(hours=TRAINING_DURATION_HOURS)
    training_cost = TRAINING_FEE

    try:
        conn = get_db_connection()
        conn.autocommit = False
        with conn.cursor() as cur:
            # Lock the horse row to avoid concurrent training starts
            cur.execute(
                """
                SELECT owner_id, is_retired, in_training_until
                FROM horses
                WHERE horse_id = %s
                FOR UPDATE;
                """,
                (horse_id,)
            )
            horse_row = cur.fetchone()
            if not horse_row:
                return False, "Horse not found.", None

            horse_owner, is_retired, in_training_until = horse_row
            if horse_owner != trainer_id:
                return False, "You do not own this horse.", None
            if is_retired:
                return False, "Retired horses cannot be trained.", None
            if in_training_until and in_training_until > now_utc:
                return False, "This horse is already in training.", None

            trainer_record = ensure_trainer_record(trainer_id)
            if not trainer_record:
                return False, "Trainer record could not be found.", None

            economy_actor = trainer_record.get("economy_id") or str(trainer_id)

            # Deduct CC using shared market database
            new_remote_balance = None
            if market_db:
                item_name = f"Training Session: {stat_code.upper()}"
                try:
                    new_remote_balance = market_db.execute_purchase_transaction(
                        actor_id=economy_actor,
                        item_name=item_name,
                        cost=float(training_cost),
                        upgrade_tier=None
                    )
                except Exception as err:
                    print(f"Error executing shared purchase transaction: {err}")
                    new_remote_balance = None

            if new_remote_balance is None and trainer_record.get("is_bot") and market_db:
                subsidy_amount = max(int(training_cost) * 5, int(training_cost))
                try:
                    result = market_db.execute_admin_award(
                        admin_id=MARKET_ADMIN_ID,
                        target_id=economy_actor,
                        amount=subsidy_amount
                    )
                except Exception as err:
                    print(f"Error subsidizing bot trainer {trainer_id}: {err}")
                    result = None
                if result is not None:
                    try:
                        new_remote_balance = market_db.execute_purchase_transaction(
                            actor_id=economy_actor,
                            item_name=item_name,
                            cost=float(training_cost),
                            upgrade_tier=None
                        )
                    except Exception as err:
                        print(f"Error executing purchase after subsidy: {err}")
                        new_remote_balance = None

            if new_remote_balance is None:
                if not trainer_record.get("is_bot"):
                    deactivate_training_plan(horse_id)
                return False, "Training failed. Insufficient CC or economy system unavailable.", None

            # Prevent duplicate queue entries
            cur.execute(
                "SELECT queue_id FROM training_queue WHERE horse_id = %s",
                (horse_id,)
            )
            if cur.fetchone():
                return False, "This horse already has a training session queued.", None

            cur.execute(
                """
                INSERT INTO training_queue (horse_id, stat_to_train, finish_time)
                VALUES (%s, %s, %s)
                RETURNING queue_id;
                """,
                (horse_id, stat_code, finish_time)
            )

            cur.execute(
                "UPDATE horses SET in_training_until = %s WHERE horse_id = %s",
                (finish_time, horse_id)
            )

            cur.execute(
                """
                INSERT INTO horse_training_plans (horse_id, stat_code, is_active, updated_at)
                VALUES (%s, %s, TRUE, NOW())
                ON CONFLICT (horse_id)
                DO UPDATE SET stat_code = EXCLUDED.stat_code,
                              is_active = TRUE,
                              updated_at = NOW();
                """,
                (horse_id, stat_code)
            )

        conn.commit()
        return True, "Training queued!", finish_time

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error starting training session for trainer {trainer_id}, horse {horse_id}: {e}")
        return False, "An unexpected error occurred while starting training. Please try again.", None
    finally:
        if conn:
            conn.close()

# --- Claiming Functions ---

def _get_claimable_horse_details(cur, horse_id: int, race_id: int):
    """
    Checks if a horse is eligible to be claimed from a specific finished G race.
    Returns horse details (owner_id, name) if claimable, None otherwise.
    Uses the provided cursor within an existing transaction.
    """
    cur.execute(
        """
        SELECT h.owner_id, h.name
        FROM horses h
        JOIN race_results rr ON rr.horse_id = h.horse_id
        JOIN races r ON rr.race_id = r.race_id
        WHERE h.horse_id = %s
          AND rr.race_id = %s
          AND r.status = 'finished'
          AND r.tier = 'G'
          AND h.owner_id IN (SELECT user_id FROM trainers WHERE is_bot = TRUE)
        LIMIT 1;
        """,
        (horse_id, race_id)
    )
    result = cur.fetchone()
    if result:
        return {"owner_id": result[0], "name": result[1]}
    return None

def execute_claim_horse(player_user_id: str, horse_id: int, race_id: int) -> tuple[bool, str]:
    """
    Executes the horse claiming process.

    1. Verifies horse claim eligibility (in Derby DB).
    2. Deducts CC and logs transaction (via market_db).
    3. Updates horse ownership (in Derby DB).

    Returns:
        tuple[bool, str]: (success_status, message)
    """
    if not market_db:
        return False, "Database integration error. Please contact admin."

    claimer_price = BALANCE_CONFIG['economy']['claimer_fixed_price']
    
    # --- Part 1: Verify Claim Eligibility (Read-only check first) ---
    conn = None
    horse_details = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            horse_details = _get_claimable_horse_details(cur, horse_id, race_id)
        
        if not horse_details:
            return False, "This horse is not eligible to be claimed (Must be a finished G-Tier race, horse must be bot-owned and have participated)."
    except Exception as e:
        print(f"Error in claim pre-check: {e}")
        return False, "An error occurred checking horse eligibility."
    finally:
        if conn: conn.close()

    bot_owner_id_str = get_trainer_economy_id(horse_details['owner_id'])
    horse_name = horse_details['name']

    # --- Part 2: Execute Financial Transaction (using your existing function) ---
    item_name_log = f"Claimed Horse: {horse_name} (ID: {horse_id})"
    # We will need to modify execute_purchase_transaction to accept 'details'
    # For now, we'll assume it doesn't.
    
    print(f"Attempting CC deduction for {player_user_id} for {item_name_log}...")
    new_balance = market_db.execute_purchase_transaction(
        actor_id=player_user_id,
        item_name=item_name_log,
        cost=claimer_price,
        upgrade_tier=None 
    )

    if new_balance is None:
        # Financial transaction failed (e.g., insufficient funds)
        current_balance = market_db.get_user_balance_by_discord_id(player_user_id)
        return False, f"Claim failed. You need {claimer_price:,.0f} CC, but your balance is only {current_balance:,.0f} CC."

    # --- Part 3: Execute Game State Change (Update Horse Ownership) ---
    print(f"CC deduction successful. New balance: {new_balance}. Updating horse ownership...")
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE horses SET owner_id = %s WHERE horse_id = %s AND owner_id = %s",
                (player_user_id, horse_id, bot_owner_id_str) # Final check that bot still owns it
            )
            if cur.rowcount == 0:
                # This is a rare edge case (e.g., two players claim at same time)
                # The money *was* spent, but they didn't get the horse.
                # We must manually refund the user!
                print(f"CRITICAL ERROR: Claim failed after payment! Horse {horse_id} owner was not {bot_owner_id_str}. Refunding {player_user_id}...")
                market_db.execute_admin_award(
                    admin_id="derby_system_refund", 
                    target_id=player_user_id, 
                    amount=claimer_price
                )
                conn.rollback()
                return False, "Claim failed! Someone else may have claimed this horse just before you. Your CC has been refunded."

            conn.commit()
            print("Horse ownership update successful.")
            return True, f"Successfully claimed **{horse_name}** (ID: {horse_id}) for {claimer_price:,.0f} CC! Your new balance is {new_balance:,.0f} CC."

    except Exception as e:
        if conn: conn.rollback()
        # This is the "manual fix" scenario. The player PAID but the horse DB update FAILED.
        print(f"CRITICAL ERROR: CC was deducted but horse ownership update FAILED for player {player_user_id}, horse {horse_id}. MANUAL FIX REQUIRED.")
        print(traceback.format_exc())
        # We must refund the player
        market_db.execute_admin_award(
            admin_id="derby_system_refund", 
            target_id=player_user_id, 
            amount=claimer_price
        )
        return False, "A critical error occurred while transferring horse ownership *after* payment. Your CC has been refunded. Please contact an admin."
    finally:
        if conn: conn.close()
            
# --- Horse Queries (Examples - we'll build these as needed) ---

# def get_trainer_horses(user_id):
#     # Fetches horses owned by a specific trainer
#     pass

# def get_race_entries(race_id):
#     # Fetches horses entered in a specific race
#     pass

# --- Trainer/Economy Queries (Examples - using PUBLIC schema) ---

# def check_trainer_prestige(user_id, required_prestige):
#     # Checks if trainer meets prestige requirement from public.prestige
#     pass

# --- Notification Queries ---

# def get_unread_notifications(user_id):
#     # Fetches unread messages for a user
#     pass

# def mark_notifications_read(user_id, notification_ids):
#     # Marks specific notifications as read
#     pass
