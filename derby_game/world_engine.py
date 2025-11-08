import asyncio
import random
import numpy as np
import math
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from derby_game.database.connection import get_db_connection
from derby_game.simulation import Horse, Race, Bookie
from derby_game.config import BALANCE_CONFIG
from derby_game.database import queries as derby_queries
from derby_game.bot_bettors import BotBettingManager
from derby_game import betting_service
import os
import sys

# --- Configuration ---
# We will use this to generate names.
# Note the path change to the new 'configs/' directory.
NAME_CONFIG_PATH = 'configs/horse_names.json'

# Get the path to the current script's directory (side-stakes-derby/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path to the parent directory (the one containing both projects)
parent_dir = os.path.dirname(current_dir)
# Add the path to the 'prettyDerbyClubAnalysis' project to sys.path
other_project_path = os.path.join(parent_dir, 'prettyDerbyClubAnalysis')
sys.path.append(other_project_path)

# --- Configuration ---
DAILY_STABLE_FEE_PER_HORSE = BALANCE_CONFIG['economy']['daily_stable_fee']
TASKS_RUN_HOUR = 4 # 4:00 AM UTC 
SECONDS_PER_DAY = 86400
MARKET_DB = derby_queries.market_db
MARKET_ADMIN_ID = "derby_system"
BOT_BETTING_MANAGER = BotBettingManager()
MARKET_RACES_TABLE = "derby.market_races"
MARKET_RACE_HORSES_TABLE = "derby.market_race_horses"

def _set_market_race_status(race_id: int, status: str):
    if not MARKET_DB:
        return
    try:
        MARKET_DB.update_race_status(race_id, status)
    except Exception as err:
        print(f"  -> Warning: failed to set market race {race_id} status to '{status}': {err}")

def _to_decimal(value):
    if value is None:
        return Decimal(0)
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))

def _decimal_to_int(value):
    return int(_to_decimal(value).quantize(Decimal('1'), rounding=ROUND_HALF_UP))

def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _as_aware(dt):
    if dt is None:
        return None
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

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
    
    cur.execute("""
        SELECT h.owner_id, t.is_bot, t.economy_id, COUNT(*) as num_horses
        FROM horses h
        JOIN trainers t ON h.owner_id = t.user_id
        WHERE h.is_retired = FALSE AND h.owner_id IS NOT NULL
        GROUP BY h.owner_id, t.is_bot, t.economy_id;
    """)
    trainer_rows = cur.fetchall()

    if not trainer_rows:
        print("     ...No trainers currently own active horses.")
        return

    trainers_charged = 0
    for owner_id, is_bot, economy_id, num_horses in trainer_rows:
        if not owner_id:
            continue

        total_fee = int(num_horses * DAILY_STABLE_FEE_PER_HORSE)

        # Ensure trainer record exists
        cur.execute(
            "INSERT INTO trainers (user_id, is_bot) VALUES (%s, %s) ON CONFLICT (user_id) DO NOTHING;",
            (owner_id, is_bot)
        )

        if MARKET_DB:
            try:
                target_id = economy_id or str(owner_id)
                result = MARKET_DB.execute_admin_removal(
                    admin_id=MARKET_ADMIN_ID,
                    target_id=target_id,
                    amount=total_fee
                )
                if result is None:
                    if is_bot:
                        subsidy_amount = max(total_fee * 5, total_fee)
                        try:
                            MARKET_DB.execute_admin_award(
                                admin_id=MARKET_ADMIN_ID,
                                target_id=target_id,
                                amount=subsidy_amount
                            )
                            result = MARKET_DB.execute_admin_removal(
                                admin_id=MARKET_ADMIN_ID,
                                target_id=target_id,
                                amount=total_fee
                            )
                        except Exception as err:
                            print(f"     !!! Failed to subsidize trainer {owner_id}: {err}")
                            result = None
                    if result is None:
                        print(f"     !!! Failed to charge stable fee for trainer {owner_id}: insufficient funds or error.")
                        continue
            except Exception as err:
                print(f"     !!! Failed to charge stable fee for trainer {owner_id}: {err}")
                continue
        else:
            print("     !!! Stable fee charge skipped (market DB unavailable).")
            continue

        trainers_charged += 1

    print(f"     ...Applied fees to {trainers_charged} trainers.")
    
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
                horse.spd, horse.sta, horse.acc, horse.fcs, 
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
            gain_ranges = BALANCE_CONFIG['training']['roll_gains']
            range_lookup = {
                "Critical Success": gain_ranges['crit_success'],
                "Standard Success": gain_ranges['standard_success']
            }
            roll_range = range_lookup.get(roll_details['outcome'])
            range_text = f" (range {roll_range[0]}-{roll_range[1]})" if roll_range else ""

            if roll_details['lck_triggered']:
                notification_message += (
                    f"> **LUCK triggered!** Rolls: {roll_details['roll_1']}, {roll_details['roll_2']}{range_text}\n"
                )
            else:
                notification_message += f"> **Roll:** {roll_details['roll_1']}{range_text}\n"

            if base_gain > 0:
                age_bonus = final_gain - base_gain
                notification_message += (
                    f"> **Age Modifier (Age {horse.age_in_years}):** {age_bonus:+d} ({age_mod*100:+.0f}%)\n"
                )
            
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


def _auto_schedule_training():
    now = datetime.now(timezone.utc)
    candidates = derby_queries.get_auto_training_candidates(now)
    if not candidates:
        return

    print(f"  -> Auto-scheduling training for {len(candidates)} horses.")
    for candidate in candidates:
        trainer_id = candidate["owner_id"]
        horse_id = candidate["horse_id"]
        stat_code = candidate["stat_code"]
        if candidate.get("is_bot"):
            stat_code = random.choice(list(derby_queries.TRAINABLE_STATS))
            derby_queries.set_training_plan(horse_id, stat_code)
        success, message, finish_time = derby_queries.start_training_session(trainer_id, horse_id, stat_code)
        if not success:
            derby_queries.mark_training_plan_attempt(horse_id)
            print(f"     !! Failed to auto-train horse {horse_id} (trainer {trainer_id}): {message}")
        else:
            derby_queries.mark_training_plan_attempt(horse_id)
            print(f"     -> Queued auto-training for horse {horse_id} ({stat_code.upper()}) finishing at {finish_time}.")

def _get_racing_config():
    return BALANCE_CONFIG.get('racing', {})

def _fill_race_with_bots(race_info, field_size):
    """
    Ensures the pending race has enough bot horses entered.
    """
    race_id = race_info['race_id']
    current_entries = derby_queries.list_race_entries(race_id)
    missing = field_size - len(current_entries)
    if missing <= 0:
        return 0

    available = derby_queries.get_available_bot_horses(race_info['tier'], limit=missing)
    added = 0
    for horse in available:
        entry = derby_queries.add_race_entry(
            race_id,
            horse['horse_id'],
            entry_fee=race_info.get('entry_fee', 0)
        )
        if entry:
            added += 1
    if added < missing:
        print(f"  -> Warning: race {race_id} still short by {missing - added} entries (Tier {race_info['tier']}).")
    elif added:
        print(f"  -> Filled race {race_id} with {added} bot entries.")
    return added


class _MarketHorseAdapter:
    __slots__ = ("name", "strategy_name", "stats")

    def __init__(self, horse):
        self.name = horse.name
        self.strategy_name = horse.strategy
        self.stats = {
            "spd": horse.spd,
            "sta": horse.sta,
            "acc": horse.acc,
            "fcs": horse.fcs,
            "grt": horse.grt,
            "cog": horse.cog,
            "lck": horse.lck,
        }


def _register_market_race(race_obj):
    if not MARKET_DB:
        return
    conn = MARKET_DB.get_connection()
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
    horses = [_MarketHorseAdapter(h) for h in race_obj.horses]
    try:
        MARKET_DB.create_race(race_obj.race_id, race_obj.distance, horses)
    except Exception as err:
        print(f"  -> Failed to register race {race_obj.race_id} with market DB: {err}")

def _race_scheduler_tick():
    """
    Maintains the pipeline of scheduled races and transitions them through lifecycle.
    """
    racing_cfg = _get_racing_config()
    if not racing_cfg:
        return

    now = datetime.now(timezone.utc)
    pending_lead = timedelta(hours=racing_cfg.get('pending_lead_hours', 24))
    betting_window = timedelta(minutes=racing_cfg.get('betting_window_minutes', 10))
    race_spacing = timedelta(minutes=racing_cfg.get('race_spacing_minutes', 60))
    lock_lead = timedelta(minutes=racing_cfg.get('lock_lead_minutes', 1))
    min_pending = racing_cfg.get('min_pending_races', 3)
    field_size_map = racing_cfg.get('field_size', {})
    entry_fee_map = racing_cfg.get('entry_fee', {})
    purse_map = racing_cfg.get('purse', {})
    distance_map = racing_cfg.get('distance_options', {})

    races = derby_queries.get_races_by_status(['pending', 'open'])
    pending_races = [r for r in races if r['status'] == 'pending']

    # Ensure existing pending races are populated and transition when ready
    for race in pending_races:
        tier = race['tier']
        field_size = field_size_map.get(tier, 10)

        start_time = _as_aware(race.get('start_time'))
        if start_time is None:
            start_time = now + pending_lead
            derby_queries.update_race_start_time(race['race_id'], start_time)
            race['start_time'] = start_time
        else:
            race['start_time'] = start_time

        _fill_race_with_bots(race, field_size)

        start_time = _as_aware(race['start_time'])
        if start_time and start_time - betting_window <= now:
            if derby_queries.update_race_status(race['race_id'], 'open'):
                race['status'] = 'open'
                print(f"  -> Race {race['race_id']} (Tier {tier}) is now OPEN.")
                race_obj = Race(race['race_id'], verbose=False)
                _register_market_race(race_obj)
                if race_obj.horses:
                    bookie = Bookie(race_obj)
                    bookie.run_monte_carlo(simulations=betting_service.MONTE_CARLO_SIMULATIONS)
                    lock_time = start_time - lock_lead if start_time else datetime.now(timezone.utc)
                    if lock_time <= now:
                        lock_time = now + timedelta(seconds=5)
                    open_time = start_time - betting_window if start_time else None
                    BOT_BETTING_MANAGER.schedule_betting_session(
                        race_obj,
                        bookie,
                        lock_time,
                        open_time=open_time,
                        betting_window=betting_window,
                    )
                _set_market_race_status(race['race_id'], 'open')

    active_pending = [r for r in pending_races if r['status'] == 'pending']

    # Create new pending races if pipeline is below target
    if min_pending > 0:
        existing_starts = sorted(
            [_as_aware(r['start_time']) for r in active_pending if r['start_time']]
        )
        next_start = existing_starts[-1] + race_spacing if existing_starts else now + pending_lead

        while len(active_pending) < min_pending:
            tier = 'G'  # Tier rollout starts with Claimers
            field_size = field_size_map.get(tier, 10)
            entry_fee = entry_fee_map.get(tier, 0)
            purse = purse_map.get(tier, 0)
            distance_options = distance_map.get(tier, [1600])
            distance = random.choice(distance_options)

            new_race = derby_queries.create_race(
                tier,
                distance,
                entry_fee=entry_fee,
                purse=purse,
                start_time=next_start,
                status='pending'
            )
            if not new_race:
                print("!!! Failed to create a new pending race. Scheduler will retry next tick.")
                break

            print(f"  -> Scheduled new Tier {tier} race #{new_race['race_id']} for {next_start}.")
            _fill_race_with_bots(new_race, field_size)
            active_pending.append(new_race)
            next_start = next_start + race_spacing

    # Handle open races approaching lock time
    lock_lead = timedelta(minutes=racing_cfg.get('lock_lead_minutes', 1))
    open_races = derby_queries.get_races_by_status(['open'])
    for race in open_races:
        start_time = _as_aware(race.get('start_time'))
        if start_time and start_time - lock_lead <= now < start_time:
            if derby_queries.update_race_status(race['race_id'], 'locked'):
                print(f"  -> Race {race['race_id']} locked for betting (Tier {race['tier']}).")
                BOT_BETTING_MANAGER.clear_session(race['race_id'])
                _set_market_race_status(race['race_id'], 'locked')

    # Run races whose start time has arrived
    ready_ids = []
    for race in open_races:
        start_time = _as_aware(race.get('start_time'))
        if start_time and start_time <= now:
            ready_ids.append(race['race_id'])

    locked_races = derby_queries.get_races_by_status(['locked'])
    for race in locked_races:
        start_time = _as_aware(race.get('start_time'))
        if start_time and start_time <= now:
            ready_ids.append(race['race_id'])

    for race_id in dict.fromkeys(ready_ids):
        _run_race_lifecycle(race_id, racing_cfg)

def _run_race_lifecycle(race_id: int, racing_cfg: dict):
    race = derby_queries.get_race_details(race_id)
    if not race:
        BOT_BETTING_MANAGER.clear_session(race_id)
        return

    status = (race['status'] or '').lower()
    if status == 'finished':
        BOT_BETTING_MANAGER.clear_session(race_id)
        return

    entries = derby_queries.list_race_entries(race_id)
    if not entries:
        print(f"  -> Race {race_id} cancelled: no entries available.")
        BOT_BETTING_MANAGER.clear_session(race_id)
        derby_queries.update_race_status(race_id, 'finished')
        _set_market_race_status(race_id, 'cancelled')
        return

    derby_queries.update_race_status(race_id, 'running')
    _set_market_race_status(race_id, 'running')
    print(f"\n=== Running Race #{race_id} ({race['tier']} Tier, {race['distance']}m) ===")
    race_sim = Race(race_id)
    if not race_sim.horses:
        print(f"  -> Race {race_id} cancelled: failed to load horses.")
        derby_queries.clear_race_entries(race_id)
        BOT_BETTING_MANAGER.clear_session(race_id)
        derby_queries.update_race_status(race_id, 'finished')
        _set_market_race_status(race_id, 'cancelled')
        return

    finishers = race_sim.run_simulation(silent=False)
    if not finishers:
        print(f"  -> Race {race_id} produced no results. Marking as finished.")
        derby_queries.clear_race_entries(race_id)
        BOT_BETTING_MANAGER.clear_session(race_id)
        derby_queries.update_race_status(race_id, 'finished')
        _set_market_race_status(race_id, 'finished')
        return

    payout_map = _calculate_purse_payouts(race, finishers, racing_cfg)
    results_persisted = getattr(race_sim, "_results_persisted", False)
    if not results_persisted:
        derby_queries.clear_race_results(race_id)
    for idx, horse in enumerate(finishers, start=1):
        derby_queries.record_race_result(
            race_id,
            horse.horse_id,
            idx,
            payout_map.get(horse.horse_id, 0)
        )
    derby_queries.set_race_winner(race_id, finishers[0].horse_id)
    _distribute_purse_to_owners(race, finishers, payout_map)
    derby_queries.clear_race_entries(race_id)

    _settle_race_bets(race, race_sim.horses, finishers, payout_map)
    _notify_race_results(race, finishers, payout_map)

    derby_queries.update_race_status(race_id, 'finished')
    BOT_BETTING_MANAGER.clear_session(race_id)
    betting_service.clear_live_odds_cache(race_id)
    _set_market_race_status(race_id, 'finished')
    print(f"=== Race #{race_id} complete ===\n")

def _calculate_purse_payouts(race: dict, finishers, racing_cfg: dict):
    purse = int(race.get('purse') or 0)
    payouts = {}
    if purse <= 0 or not finishers:
        return payouts

    default_shares = [Decimal('0.60'), Decimal('0.20'), Decimal('0.10'), Decimal('0.07'), Decimal('0.03')]
    config_shares = racing_cfg.get('purse_shares')
    if config_shares:
        shares = [Decimal(str(share)) for share in config_shares]
    else:
        shares = default_shares

    total_decimal = Decimal(purse)
    allocated = 0
    limit = min(len(finishers), len(shares))

    for idx in range(limit):
        horse = finishers[idx]
        if idx == limit - 1:
            payout = purse - allocated
        else:
            payout = _decimal_to_int(total_decimal * shares[idx])
            payout = max(0, min(payout, purse - allocated))
        payouts[horse.horse_id] = payout
        allocated += payout

    # If there are more finishers than shares, ensure they are recorded with zero payout.
    for horse in finishers:
        payouts.setdefault(horse.horse_id, 0)

    return payouts

def _distribute_purse_to_owners(race: dict, finishers, payout_map):
    total_paid = 0
    for horse in finishers:
        payout = int(payout_map.get(horse.horse_id, 0) or 0)
        if payout <= 0:
            continue
        owner_id = horse.owner_id
        if not owner_id or not MARKET_DB:
            continue
        try:
            target_id = derby_queries.get_trainer_economy_id(owner_id)
            MARKET_DB.execute_admin_award(
                admin_id=MARKET_ADMIN_ID,
                target_id=target_id,
                amount=payout
            )
            total_paid += payout
        except Exception as err:
            print(f"  -> Failed to distribute purse ({payout} CC) to trainer {owner_id}: {err}")
    if total_paid:
        print(f"  -> Distributed {total_paid:,} CC in purse payouts.")

def _settle_race_bets(race: dict, horses, finishers, payout_map):
    if not MARKET_DB:
        print("  -> Market database unavailable. Skipping bet settlement.")
        return

    bets = derby_queries.get_market_bets_since(race['race_id'], last_bet_id=0)
    if not bets:
        print("  -> No wagers to settle for this race.")
        return

    horse_lookup = {horse.name: horse for horse in horses}
    placement_map = {horse.horse_id: idx + 1 for idx, horse in enumerate(finishers)}

    for bet in bets:
        bet_id = bet["bet_id"]
        bettor_id = str(bet["bettor_id"])
        horse_name = bet["horse_name"]
        amount = bet["amount"]
        odds = bet["locked_in_odds"]
        if derby_queries.is_bet_settled(bet_id):
            continue

        horse = horse_lookup.get(horse_name)
        if not horse:
            target_lower = (horse_name or "").lower()
            for name, candidate in horse_lookup.items():
                if name.lower() == target_lower:
                    horse = candidate
                    break

        horse_id = horse.horse_id if horse else None
        stake_dec = _to_decimal(amount)
        stake_value = _decimal_to_int(stake_dec)
        odds_dec = _to_decimal(odds)
        payout_total = 0
        is_winner = horse is not None and placement_map.get(horse_id) == 1

        if is_winner and stake_value > 0:
            try:
                payout_total = _decimal_to_int(Decimal(stake_value) * (Decimal(1) + odds_dec))
                details = {
                    "race_id": race["race_id"],
                    "horse_id": horse_id,
                    "horse_name": horse_name,
                    "bet_type": "WIN",
                    "locked_odds": float(odds_dec),
                    "source": "bot" if bettor_id.startswith("derby_bot_") else "player",
                    "settlement": True,
                }
                result_balance = MARKET_DB.execute_gambling_transaction(
                    actor_id=bettor_id,
                    game_name=f"Derby Race #{race['race_id']}",
                    bet_amount=0.0,
                    winnings=float(payout_total),
                    details=details,
                )
                if result_balance is None:
                    print(f"  -> Failed to award winnings for bet {bet_id} ({bettor_id}).")
                    payout_total = 0
                    is_winner = False
            except Exception as err:
                print(f"  -> Failed to award winnings for bet {bet_id} ({bettor_id}): {err}")
                payout_total = 0
                is_winner = False

        derby_queries.record_bet_settlement(
            bet_id,
            race['race_id'],
            bettor_id,
            horse_id,
            stake_value,
            float(odds_dec),
            payout_total
        )

        bettor_user_id = _safe_int(bettor_id)
        display_name = horse.name if horse else horse_name
        if bettor_user_id and display_name:
            if is_winner and payout_total > 0:
                derby_queries.add_notification(
                    bettor_user_id,
                    f"💰 Your bet on {display_name} won Race #{race['race_id']}! Payout: {payout_total:,} CC."
                )
            elif not is_winner:
                derby_queries.add_notification(
                    bettor_user_id,
                    f"💸 Your bet on {display_name} lost Race #{race['race_id']}."
                )

def _notify_race_results(race: dict, finishers, payout_map):
    if not finishers:
        return

    summary_lines = []
    for idx, horse in enumerate(finishers[:5], start=1):
        payout = payout_map.get(horse.horse_id, 0)
        line = f"{_ordinal(idx)} - {horse.name}"
        if payout > 0:
            line += f" ({payout:,} CC)"
        summary_lines.append(line)

    summary_text = "\n".join(summary_lines)
    print(f"  -> Results:\n{summary_text}")

    for idx, horse in enumerate(finishers, start=1):
        owner_id = horse.owner_id
        if not owner_id:
            continue
        payout = payout_map.get(horse.horse_id, 0)
        message = (
            f"🏁 Your horse **{horse.name}** finished {_ordinal(idx)} in Race #{race['race_id']} "
            f"({race['tier']} Tier, {race['distance']}m)."
        )
        if payout > 0:
            message += f" Payout: {payout:,} CC."
        derby_queries.add_notification(owner_id, message)

def _ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(value % 10, 'th')
    return f"{value}{suffix}"

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
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"!!! ERROR in frequent tasks: {e}")
    finally:
        if conn:
            conn.close()

    try:
        _auto_schedule_training()
    except Exception as auto_err:
        print(f"!!! ERROR during auto-training scheduling: {auto_err}")

    try:
        _race_scheduler_tick()
    except Exception as scheduler_error:
        print(f"!!! ERROR in race scheduler: {scheduler_error}")

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
    stats_to_decay = ['spd', 'sta', 'acc', 'fcs', 'grt', 'cog']

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
                horse.spd, horse.sta, horse.acc, horse.fcs, 
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
        # (owner_id, name, birth_timestamp, spd, sta, acc, fcs, grt, cog, lck, hg_score)
        (1, 'Old Timer (Age 5)', "NOW() - interval '37 days'", 100, 100, 100, 100, 100, 100, 300, 1200),
        (1, 'Senior Stallion (Age 6)', "NOW() - interval '49 days'", 100, 100, 100, 100, 100, 100, 300, 1200),
        (2, 'The Geezer (Age 7)', "NOW() - interval '61 days'", 100, 100, 100, 100, 100, 100, 300, 1200),
        (2, 'Mostly Retired (Age 7)', "NOW() - interval '70 days'", 100, 100, 100, 100, 100, 100, 300, 1200),
    ]
    
    # Clear any previously generated test veterans
    cur.execute("DELETE FROM horses WHERE name LIKE 'Old Timer%' OR name LIKE 'Senior Stallion%' OR name LIKE 'The Geezer%' OR name LIKE 'Mostly Retired%';")

    # Insert the new ones
    for horse in veteran_horses:
        sql = f"""
        INSERT INTO horses (owner_id, name, birth_timestamp, spd, sta, acc, fcs, grt, cog, lck, hg_score)
        VALUES (%s, %s, {horse[2]}, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        # Note: We are formatting horse[2] directly into the SQL
        # This is safe ONLY because we defined the string ourselves.
        cur.execute(sql, (
            horse[0], horse[1], horse[3], horse[4], horse[5], horse[6], horse[7], horse[8], horse[9], horse[10]
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
                "INSERT INTO horses (horse_id, owner_id, name, birth_timestamp, spd, sta, acc, fcs, grt, cog, lck, hg_score, is_bot) "
                "VALUES (1, 1, 'Old Timer (Age 5)', NOW() - interval '37 days', 100, 100, 100, 100, 100, 100, 300, 1200, TRUE);"
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
