from derby_game.database.connection import get_db_connection
from datetime import datetime, timedelta, timezone
from derby_game.config import BALANCE_CONFIG
import traceback
import json
import os
import sys

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
TRAINABLE_STATS = {"spd", "sta", "fcs", "grt", "cog"}
TRAINING_DURATION_HOURS = BALANCE_CONFIG['training'].get('session_duration_hours', 16)
TRAINING_FEE = BALANCE_CONFIG['economy']['training_fee']

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
                SELECT race_id, tier, distance, status, start_time, purse
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
                    "status": row[3],
                    "start_time": row[4], # Will be datetime object or None
                    "purse": row[5]
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
                "SELECT race_id, tier, distance, status, start_time, purse "
                "FROM races WHERE race_id = %s",
                (race_id,)
            )
            result = cur.fetchone()
            if result:
                race_details = {
                    "race_id": result[0],
                    "tier": result[1],
                    "distance": result[2],
                    "status": result[3],
                    "start_time": result[4],
                    "purse": result[5]
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
                    h.spd, h.sta, h.fcs, h.grt, h.cog, h.lck, h.hg_score
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
                    "fcs": row[7],
                    "grt": row[8],
                    "cog": row[9],
                    "lck": row[10],
                    "hg_score": row[11]
                })
    except Exception as e:
        print(f"Error in get_horses_in_race for race {race_id}: {e}")
    finally:
        if conn:
            conn.close()
    return horses

# --- Trainer Helpers ---

def ensure_trainer_record(user_id: int, *, is_bot: bool = False):
    """
    Ensures there is a trainer row for the supplied user ID.
    Returns a dict with trainer data or None if something goes wrong.
    """
    conn = None
    trainer = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO trainers (user_id, is_bot)
                VALUES (%s, %s)
                ON CONFLICT (user_id) DO NOTHING;
                """,
                (user_id, is_bot)
            )
            cur.execute(
                "SELECT user_id, is_bot, cc_balance, prestige, stable_slots FROM trainers WHERE user_id = %s",
                (user_id,)
            )
            row = cur.fetchone()
            if row:
                trainer = {
                    "user_id": row[0],
                    "is_bot": row[1],
                    "cc_balance": row[2],
                    "prestige": row[3],
                    "stable_slots": row[4]
                }
    except Exception as e:
        print(f"Error ensuring trainer record for {user_id}: {e}")
    finally:
        if conn:
            conn.close()
    return trainer

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
                SELECT horse_id, name, strategy, spd, sta, fcs, grt, cog, lck,
                       hg_score, is_retired, in_training_until
                FROM horses
                WHERE owner_id = %s
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
                    "spd": row[3],
                    "sta": row[4],
                    "fcs": row[5],
                    "grt": row[6],
                    "cog": row[7],
                    "lck": row[8],
                    "hg_score": row[9],
                    "is_retired": row[10],
                    "in_training_until": row[11]
                })
    except Exception as e:
        print(f"Error getting horses for trainer {user_id}: {e}")
    finally:
        if conn:
            conn.close()
    return horses

def start_training_session(trainer_id: int, horse_id: int, stat: str):
    """
    Attempts to queue a training session for the requested stat.
    Returns (success: bool, message: str).
    """
    stat_code = (stat or "").lower()
    if stat_code not in TRAINABLE_STATS:
        return False, "Invalid stat. Choose from SPD, STA, FCS, GRT, or COG."

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
                return False, "Horse not found."

            horse_owner, is_retired, in_training_until = horse_row
            if horse_owner != trainer_id:
                return False, "You do not own this horse."
            if is_retired:
                return False, "Retired horses cannot be trained."
            if in_training_until and in_training_until > now_utc:
                return False, "This horse is already in training."

            # Ensure trainer exists and lock the row
            cur.execute(
                """
                SELECT user_id, is_bot, cc_balance
                FROM trainers
                WHERE user_id = %s
                FOR UPDATE;
                """,
                (trainer_id,)
            )
            trainer_row = cur.fetchone()
            if not trainer_row:
                cur.execute(
                    "INSERT INTO trainers (user_id, is_bot) VALUES (%s, FALSE)",
                    (trainer_id,)
                )
                cur.execute(
                    """
                    SELECT user_id, is_bot, cc_balance
                    FROM trainers
                    WHERE user_id = %s
                    FOR UPDATE;
                    """,
                    (trainer_id,)
                )
                trainer_row = cur.fetchone()

            is_bot_owner = trainer_row[1]
            local_balance = trainer_row[2] or 0

            # Deduct CC using shared market database when available
            new_remote_balance = None
            if not is_bot_owner and market_db:
                item_name = f"Training Session: {stat_code.upper()}"
                try:
                    new_remote_balance = market_db.execute_purchase_transaction(
                        actor_id=str(trainer_id),
                        item_name=item_name,
                        cost=float(training_cost),
                        upgrade_tier=None
                    )
                except Exception as err:
                    print(f"Error executing shared purchase transaction: {err}")
                    new_remote_balance = None

                if new_remote_balance is None:
                    return False, "Training failed. Insufficient CC or market system unavailable."
            else:
                if local_balance < training_cost:
                    return False, "Training failed. Insufficient CC balance."

            # Update local trainer balance
            if new_remote_balance is not None:
                try:
                    remote_balance_int = int(round(float(new_remote_balance)))
                except (TypeError, ValueError):
                    remote_balance_int = local_balance - training_cost
                cur.execute(
                    "UPDATE trainers SET cc_balance = %s WHERE user_id = %s",
                    (remote_balance_int, trainer_id)
                )
            else:
                cur.execute(
                    "UPDATE trainers SET cc_balance = cc_balance - %s WHERE user_id = %s",
                    (training_cost, trainer_id)
                )

            # Prevent duplicate queue entries
            cur.execute(
                "SELECT queue_id FROM training_queue WHERE horse_id = %s",
                (horse_id,)
            )
            if cur.fetchone():
                return False, "This horse already has a training session queued."

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

        conn.commit()
        readable_finish = finish_time.strftime("%Y-%m-%d %H:%M UTC")
        return True, f"Training queued! {stat_code.upper()} session completes at {readable_finish}."

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error starting training session for trainer {trainer_id}, horse {horse_id}: {e}")
        return False, "An unexpected error occurred while starting training. Please try again."
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
        JOIN race_entries re ON h.horse_id = re.horse_id
        JOIN races r ON re.race_id = r.race_id
        WHERE h.horse_id = %s
          AND r.race_id = %s
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

    bot_owner_id_str = str(horse_details['owner_id']) # Ensure it's a string
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

# def get_trainer_cc_balance(user_id):
#     # Fetches CC balance from public.balances
#     conn = None
#     try:
#         conn = get_db_connection()
#         with conn.cursor() as cur:
#             # IMPORTANT: Use the public schema explicitly
#             cur.execute("SELECT balance FROM public.balances WHERE user_id = %s", (user_id,))
#             result = cur.fetchone()
#             return result[0] if result else 0
#     # ... (error handling) ...
#     finally:
#         if conn: conn.close()

# def update_trainer_cc_balance(user_id, amount_change):
#     # Updates CC balance in public.balances (use negative amount for deduction)
#     pass

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
