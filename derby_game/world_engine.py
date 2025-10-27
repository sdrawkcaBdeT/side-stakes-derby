import asyncio
import random
import numpy as np
import math
from datetime import datetime, timezone, timedelta
from derby_game.database.connection import get_db_connection
from derby_game.simulation import Horse
from derby_game.config import BALANCE_CONFIG

# --- Configuration ---
DAILY_STABLE_FEE_PER_HORSE = BALANCE_CONFIG['economy']['daily_stable_fee']
TASKS_RUN_HOUR = 4 # 4:00 AM UTC 
SECONDS_PER_DAY = 86400

def _get_age_training_modifier(age_in_years):
    """
    Returns the training modifier % based on the horse's age, from config.
    """
    # Uses .get() to safely find the modifier, defaulting to the Age 7+ modifier
    age_key = str(age_in_years)
    default_mod = BALANCE_CONFIG['training']['age_modifiers']['7']
    return BALANCE_CONFIG['training']['age_modifiers'].get(age_key, default_mod)

def _calculate_training_roll(lck_stat):
    """
    Runs the complete training roll logic from the config.
    Returns: A dictionary with all roll details.
    """
    cfg_rolls = BALANCE_CONFIG['training']['roll_chances']
    cfg_gains = BALANCE_CONFIG['training']['roll_gains']
    
    # 1. The Outcome Roll
    roll = random.random()
    details = {
        "lck_triggered": False,
        "roll_1": 0,
        "roll_2": 0
    }

    if roll < cfg_rolls['crit_success']:
        details["outcome"] = "Critical Success"
        gain_range = cfg_gains['crit_success']
    elif roll < (cfg_rolls['crit_success'] + cfg_rolls['standard_success']):
        details["outcome"] = "Standard Success"
        gain_range = cfg_gains['standard_success']
    elif roll < (cfg_rolls['crit_success'] + cfg_rolls['standard_success'] + cfg_rolls['failure']):
        details["outcome"] = "Failure"
        details["final_gain"] = 0
        return details
    else:
        details["outcome"] = "Setback"
        details["final_gain"] = cfg_gains['setback']
        return details

    # 2. The Magnitude Roll & LCK "Advantage"
    cfg_lck = BALANCE_CONFIG['training']['lck_advantage_map']
    
    lck_range = cfg_lck['max_lck'] - cfg_lck['min_lck']
    prob_range = cfg_lck['max_prob'] - cfg_lck['min_prob']
    clamped_lck = np.clip(lck_stat, cfg_lck['min_lck'], cfg_lck['max_lck'])
    adv_chance = cfg_lck['min_prob'] + ((clamped_lck - cfg_lck['min_lck']) / lck_range) * prob_range
    
    details["roll_1"] = random.randint(gain_range[0], gain_range[1])
    
    if random.random() < adv_chance: # Advantage triggers
        details["lck_triggered"] = True
        details["roll_2"] = random.randint(gain_range[0], gain_range[1])
        details["final_gain"] = max(details["roll_1"], details["roll_2"])
    else: # No Advantage
        details["final_gain"] = details["roll_1"]
        
    return details

def _apply_stable_fees(cur):
    """
    Applies the daily stable fee sink.
    
    1. Counts horses for each trainer.
    2. Calculates total fee.
    3. Deducts fee from 'trainers' table.
    """
    print(f"  -> Applying daily stable fee of {DAILY_STABLE_FEE_PER_HORSE} CC per horse...")
    
    # This query groups by trainer, counts their horses, and calculates the total fee
    cur.execute("""
        UPDATE trainers
        SET cc_balance = cc_balance - (horse_counts.total_fee)
        FROM (
            SELECT 
                owner_id, 
                COUNT(*) as num_horses,
                (COUNT(*) * %s) as total_fee
            FROM horses
            WHERE is_retired = FALSE
            GROUP BY owner_id
        ) AS horse_counts
        WHERE trainers.user_id = horse_counts.owner_id;
    """, (DAILY_STABLE_FEE_PER_HORSE,))
    
    print(f"     ...Applied fees to {cur.rowcount} trainers.")
    
def _process_training_queue(cur):
    """
    Processes all completed training jobs in the queue.
    Saves a detailed notification to the user's inbox.
    """
    print("  -> Processing training queue...")
    
    cur.execute(
        "SELECT queue_id, horse_id, stat_to_train "
        "FROM training_queue WHERE finish_time <= NOW()"
    )
    completed_jobs = cur.fetchall()

    if not completed_jobs:
        print("     ...No training jobs to process.")
        return

    processed_count = 0
    for job in completed_jobs:
        queue_id, horse_id, stat_to_train = job
        notification_message = "" # We will build this
        
        try:
            horse = Horse(horse_id)
            
            # 1. Get Training Roll
            # We refactor _calculate_training_roll to return a dict
            roll_details = _calculate_training_roll(horse.lck)
            base_gain = roll_details['final_gain']
            
            # 2. Get Age Modifier
            age_mod = _get_age_training_modifier(horse.age_in_years)
            
            # 3. Calculate Final Gain
            if base_gain > 0:
                final_gain = int(base_gain * (1 + age_mod))
            else:
                final_gain = base_gain # Setbacks are not modified

            # 4. Update the Horse Object
            current_stat = getattr(horse, stat_to_train)
            new_stat = max(1, current_stat + final_gain)
            setattr(horse, stat_to_train, new_stat)
            
            # 5. Recalculate HG Score
            new_hg_score = horse._calculate_hg(
                horse.spd, horse.sta, horse.fcs, 
                horse.grt, horse.cog, horse.lck
            )
            
            # 6. Update Database
            cur.execute(
                f"UPDATE horses SET {stat_to_train} = %s, hg_score = %s, in_training_until = NULL "
                "WHERE horse_id = %s",
                (new_stat, new_hg_score, horse_id)
            )
            
            # 7. Build Notification Message
            stat_name = stat_to_train.upper()
            notification_message = (
                f"**Training Complete for {horse.name} ({stat_name})!**\n"
                f"> **Outcome:** {roll_details['outcome']}\n"
            )
            if roll_details['lck_triggered']:
                notification_message += f"> **LUCK triggered!** (Rolls: {roll_details['roll_1']}, {roll_details['roll_2']})\n"
            else:
                notification_message += f"> **Roll:** {roll_details['roll_1']}\n"
            
            if base_gain > 0:
                notification_message += f"> **Age Modifier (Age {horse.age_in_years}):** {age_mod*100:+.0f}%\n"
            
            if final_gain > 0:
                notification_message += f"> **Final Gain:** `+{final_gain} {stat_name}` (New Total: {new_stat})\n"
            elif final_gain < 0:
                notification_message += f"> **Final Loss:** `{final_gain} {stat_name}` (New Total: {new_stat})\n"
            else:
                notification_message += "> **No Stat Change.**\n"
            notification_message += f"> **New HG Score:** {new_hg_score}"

            # 8. Save Notification to Inbox
            cur.execute(
                "INSERT INTO notifications (user_id, message) VALUES (%s, %s)",
                (horse.owner_id, notification_message)
            )
            
            # 9. Delete from Queue
            cur.execute("DELETE FROM training_queue WHERE queue_id = %s", (queue_id,))
            
            processed_count += 1
        
        except Exception as e:
            print(f"!!! ERROR processing queue job {queue_id} for horse {horse_id}: {e}")
            if notification_message: # If we failed after building the message
                cur.execute("INSERT INTO notifications (user_id, message) VALUES (%s, %s)",
                    (horse.owner_id, f"An error occurred during {stat_to_train} training for {horse.name}. Please contact admin."))
            # Delete the broken job so it doesn't run forever
            cur.execute("DELETE FROM training_queue WHERE queue_id = %s", (queue_id,))

    print(f"     ...Processed {processed_count} training jobs.")

def run_frequent_tasks():
    """
    Master function for all tasks that run frequently (e.g., every minute).
    """
    print(f"\n--- Running Frequent Tasks ({datetime.now(timezone.utc)}) ---")
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            _process_training_queue(cur)
            # We will add race creation/betting here later
        
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"!!! ERROR in frequent tasks: {e}")
    finally:
        if conn:
            conn.close()

def _apply_stat_decay(cur):
    """
    Applies stat decay for all veteran horses (Age 5+).
    """
    print("  -> Applying stat decay for veteran horses (Age 5+)...")
    
    cfg_decay = BALANCE_CONFIG['lifecycle']['stat_decay_chances']
    cfg_loss = BALANCE_CONFIG['lifecycle']['stat_decay_loss']
    
    cur.execute("SELECT horse_id FROM horses WHERE is_retired = FALSE")
    all_horse_ids = [record[0] for record in cur.fetchall()]
    
    horses_decayed = 0
    stats_to_decay = ['spd', 'sta', 'fcs', 'grt', 'cog']

    for horse_id in all_horse_ids:
        horse = Horse(horse_id)
        
        # Get decay chance from config using the same .get() trick
        age_key = str(horse.age_in_years)
        decay_chance = cfg_decay.get(age_key, cfg_decay.get("7", 0.90)) # Default to 7+

        # If age is 5+ and the RNG roll passes...
        if horse.age_in_years >= 5 and random.random() < decay_chance:
            stat_to_lose = random.choice(stats_to_decay)
            loss_amount = random.randint(cfg_loss[0], cfg_loss[1])
            
            # ... (rest of the function is the same, recalculating HG score) ...
            current_value = getattr(horse, stat_to_lose)
            new_value = max(1, current_value - loss_amount)
            setattr(horse, stat_to_lose, new_value)
            
            new_hg_score = horse._calculate_hg(
                horse.spd, horse.sta, horse.fcs, 
                horse.grt, horse.cog, horse.lck
            )
            
            cur.execute(
                f"UPDATE horses SET {stat_to_lose} = %s, hg_score = %s WHERE horse_id = %s",
                (new_value, new_hg_score, horse.horse_id)
            )
            horses_decayed += 1

    print(f"     ...{horses_decayed} horses had stats decay.")

def _create_test_veterans(cur):
    """
    Helper function to create aged horses for testing stat decay.
    We'll set their birth_timestamp manually in the past.
    """
    print("  -> Creating test veteran horses (Age 5, 6, 7)...")
    
    # We use 'NOW() - interval' to create past timestamps
    # Per design doc: Age 5=37 days, Age 6=49 days, Age 7=61 days
    veteran_horses = [
        # (owner_id, name, birth_timestamp, spd, sta, fcs, grt, cog, lck, hg_score)
        (1, 'Old Timer (Age 5)', "NOW() - interval '37 days'", 100, 100, 100, 100, 100, 300, 1200),
        (1, 'Senior Stallion (Age 6)', "NOW() - interval '49 days'", 100, 100, 100, 100, 100, 300, 1200),
        (2, 'The Geezer (Age 7)', "NOW() - interval '61 days'", 100, 100, 100, 100, 100, 300, 1200),
        (2, 'Mostly Retired (Age 7)', "NOW() - interval '70 days'", 100, 100, 100, 100, 100, 300, 1200),
    ]
    
    # Clear any previously generated test veterans
    cur.execute("DELETE FROM horses WHERE name LIKE 'Old Timer%' OR name LIKE 'Senior Stallion%' OR name LIKE 'The Geezer%' OR name LIKE 'Mostly Retired%';")

    # Insert the new ones
    for horse in veteran_horses:
        sql = f"""
        INSERT INTO horses (owner_id, name, birth_timestamp, spd, sta, fcs, grt, cog, lck, hg_score)
        VALUES (%s, %s, {horse[2]}, %s, %s, %s, %s, %s, %s, %s);
        """
        # Note: We are formatting horse[2] directly into the SQL
        # This is safe ONLY because we defined the string ourselves.
        cur.execute(sql, (
            horse[0], horse[1], horse[3], horse[4], horse[5], horse[6], horse[7], horse[8], horse[9]
        ))
    
    print(f"     ...Created {len(veteran_horses)} veteran horses.")

def run_daily_tasks():
    """
    The master function for all daily scheduled tasks.
    """
    print(f"\n--- Running Daily Tasks ({datetime.now(timezone.utc)}) ---")
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            _apply_stable_fees(cur)
            _apply_stat_decay(cur)
        
        conn.commit()
        print("--- Daily Tasks Complete ---")
        
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"!!! ERROR in daily tasks: {e}")
    finally:
        if conn:
            conn.close()

async def daily_task_loop():
    """The async loop for running daily tasks."""
    while True:
        now = datetime.now(timezone.utc)
        
        # Calculate time until 4:00 AM UTC
        next_run = now.replace(hour=TASKS_RUN_HOUR, minute=0, second=0, microsecond=0)
        if now >= next_run:
            next_run = next_run + timedelta(days=1)
            
        wait_seconds = (next_run - now).total_seconds()
        print(f"[DailyEngine] Next run at {next_run} (in {wait_seconds / 3600:.2f} hours)")
        
        await asyncio.sleep(wait_seconds)
        
        run_daily_tasks()
        
        await asyncio.sleep(60) # Sleep 1 min to avoid double-runs

async def frequent_task_loop():
    """The async loop for running frequent tasks (every 60s)."""
    while True:
        print("[FrequentEngine] Running tasks...")
        run_frequent_tasks()
        
        print("[FrequentEngine] Sleeping for 60 seconds...")
        await asyncio.sleep(60)

async def run_world_engine():
    """
    Gathers and runs all concurrent engine loops.
    """
    print("--- World Engine Started ---")
    await asyncio.gather(
        daily_task_loop(),
        frequent_task_loop()
    )

def _create_test_training_job(cur, horse_id):
    """Helper to create a finished training job for a specific horse."""
    print(f"  -> Creating a finished 'spd' training job for horse {horse_id}...")
    
    # Clear any old jobs for this horse
    cur.execute("DELETE FROM training_queue WHERE horse_id = %s;", (horse_id,))
    
    # Insert a new job that finished 1 second ago
    cur.execute(
        "INSERT INTO training_queue (horse_id, stat_to_train, finish_time) "
        "VALUES (%s, 'spd', NOW() - interval '1 second');",
        (horse_id,)
    )

if __name__ == "__main__":
    # --- MANUAL TEST ---
    print("--- Running One-Time Manual Test for Training Processor ---")
    
    conn = None
    TEST_HORSE_ID = 1 # We'll use the bot horse
    
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # 1. Create a test veteran (Age 5, so modifier is -0.25)
            # This horse has 100 SPD, 1200 HG
            cur.execute("DELETE FROM horses WHERE horse_id = 1;")
            cur.execute(
                "INSERT INTO horses (horse_id, owner_id, name, birth_timestamp, spd, sta, fcs, grt, cog, lck, hg_score, is_bot) "
                "VALUES (1, 1, 'Old Timer (Age 5)', NOW() - interval '37 days', 100, 100, 100, 100, 100, 300, 1200, TRUE);"
            )
            
            # 2. Create a finished training job for this horse
            _create_test_training_job(cur, TEST_HORSE_ID)
        
        conn.commit()
        print("\nTest horse and training job created.")
        print("Horse 1 has 100 SPD and 1200 HG.")
        print("Training with Age 5 modifier (-0.25). Expecting a small gain or setback.\n")
        
        # 3. Manually run the frequent task processor
        run_frequent_tasks()
        
        print("\n--- TEST COMPLETE ---")
        print(f"Check the 'horses' table for horse_id {TEST_HORSE_ID}.")
        print("Its 'spd' and 'hg_score' should be different from 100/1200.")
        print("Also check 'training_queue' table; it should be empty.")

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"\n!!! TEST FAILED: {e}")
    finally:
        if conn:
            conn.close()

    # We are not starting the full engine loop in this test
    print("\n(Not starting the full engine loop. Test finished.)")