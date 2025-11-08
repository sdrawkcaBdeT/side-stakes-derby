import numpy as np
import psycopg2.extras
from derby_game.database.connection import get_db_connection
from datetime import datetime, timezone
from copy import deepcopy
from derby_game.config import BALANCE_CONFIG
from typing import List
from derby_game.horse_name_generator import NameGenerator
from derby_game.engine.track_registry import DEFAULT_TRACK_REGISTRY
from derby_game.engine import (
    Aptitudes,
    AptitudeGrade,
    HybridRaceLoop,
    RacerProfile,
    RacerStats,
    Strategy,
    TelemetryCollector,
)
from derby_game.database import queries as derby_queries
import random
import json
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

def _hybrid_engine_enabled() -> bool:
    config_flag = False
    if isinstance(BALANCE_CONFIG, dict):
        config_flag = BALANCE_CONFIG.get("race_engine", {}).get("use_hybrid_engine", False)
    env_value = os.getenv("DERBY_USE_HYBRID_ENGINE")
    if env_value is not None:
        config_flag = env_value.lower() in ("1", "true", "yes", "on")
    return config_flag

STRATEGY_STRING_MAP = {
    "front runner": Strategy.FRONT_RUNNER,
    "front_runner": Strategy.FRONT_RUNNER,
    "nige": Strategy.FRONT_RUNNER,
    "pace chaser": Strategy.PACE_CHASER,
    "pace_chaser": Strategy.PACE_CHASER,
    "senkou": Strategy.PACE_CHASER,
    "late surger": Strategy.LATE_SURGER,
    "late_surser": Strategy.LATE_SURGER,
    "late_surfer": Strategy.LATE_SURGER,
    "sasi": Strategy.LATE_SURGER,
    "end closer": Strategy.END_CLOSER,
    "end_closer": Strategy.END_CLOSER,
    "oikomi": Strategy.END_CLOSER,
}
STRATEGY_INT_MAP = {
    1: Strategy.FRONT_RUNNER,
    2: Strategy.PACE_CHASER,
    3: Strategy.LATE_SURGER,
    4: Strategy.END_CLOSER,
}
DEFAULT_SURFACE_APTITUDE = AptitudeGrade.A


# --- The new Horse Class ---

class Horse:
    """
    Represents a single horse in the database, using the new 7-stat system.
    """
    def __init__(self, horse_id):
        """
        Initializes a Horse object by loading its data from the database.
        """
        self.horse_id = horse_id
        
        # All stats will be populated by load_from_db()
        self.name = ""
        self.owner_id = 0
        self.is_bot = False
        self.birth_timestamp = None
        self.spd = 0
        self.sta = 0
        self.acc = 0
        self.fcs = 0
        self.grt = 0
        self.cog = 0
        self.lck = 0
        self.hg_score = 0
        self.is_retired = False
        self.in_training_until = None
        
        # Call the loader method
        self.load_from_db()

    def __repr__(self):
        return f"<Horse ID: {self.horse_id} | Name: {self.name} | HG: {self.hg_score}>"

    def load_from_db(self):
        """
        Populates the object's attributes with data from the 'horses' table.
        """
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM horses WHERE horse_id = %s", (self.horse_id,))
                record = cur.fetchone()
            
            if record:
                # Adjust indices based on your *actual* final schema order
                # (horse_id, owner_id, is_bot, name, strategy, min_pref, max_pref, birth_ts, spd, sta, ...)
                self.owner_id = record[1]
                self.is_bot = record[2]
                self.name = record[3]
                self.strategy = record[4]
                self.min_preferred_distance = record[5]
                self.max_preferred_distance = record[6]
                self.birth_timestamp = record[7]
                self.spd = record[8]
                self.sta = record[9]
                self.acc = record[10]
                self.fcs = record[11]
                self.grt = record[12]
                self.cog = record[13]
                self.lck = record[14]
                self.hg_score = record[15]
                self.is_retired = record[16]
                self.in_training_until = record[17]
            else:
                print(f"Warning: No horse found with ID {self.horse_id}")
                
        except Exception as e:
            print(f"Error loading horse {self.horse_id}: {e}")
        finally:
            if conn:
                conn.close()

    @property
    def age_in_days(self):
        """Calculates the horse's age in days."""
        if not self.birth_timestamp:
            return 0
        # We compare the (timezone-aware) birth_timestamp from the DB
        # with a (timezone-aware) current time in UTC.
        return (datetime.now(timezone.utc) - self.birth_timestamp).days

    @property
    def age_in_years(self):
        """Calculates the horse's 'year' based on the design doc."""
        # Age 2: 0-12 days
        # Age 3: 13-24 days
        # ...
        # We add 2 to the result of (days // 12)
        return (self.age_in_days // 12) + 2

    @staticmethod
    def _calculate_hg(spd, sta, acc, fcs, grt, cog, lck):
        """
        Calculates the HG score based on config formula.
        """
        cfg = BALANCE_CONFIG['hg_formula_weights']
        hg = (spd * cfg['spd']) + (acc * cfg.get('acc', cfg['spd'])) + (sta * cfg['sta']) + (fcs * cfg['fcs']) + \
             (grt * cfg['grt']) + (cog * cfg['cog']) + (lck * cfg['lck'])
        return int(hg)

    @staticmethod
    def generate_new_horse(owner_id, *, is_bot=False):
        """
        Generates a new G-Grade horse with random stats, saves it to the DB,
        and returns the new horse's ID.
        """
        
        # --- 1. Generate Stats (Based on Config) ---
        cfg_base = BALANCE_CONFIG['horse_generation']['base_stats']
        cfg_lck = BALANCE_CONFIG['horse_generation']['lck_stats']
        
        spd = int(np.clip(np.random.normal(cfg_base['mean'], cfg_base['std_dev']), cfg_base['min'], cfg_base['max']))
        sta = int(np.clip(np.random.normal(cfg_base['mean'], cfg_base['std_dev']), cfg_base['min'], cfg_base['max']))
        acc = int(np.clip(np.random.normal(cfg_base['mean'], cfg_base['std_dev']), cfg_base['min'], cfg_base['max']))
        fcs = int(np.clip(np.random.normal(cfg_base['mean'], cfg_base['std_dev']), cfg_base['min'], cfg_base['max']))
        grt = int(np.clip(np.random.normal(cfg_base['mean'], cfg_base['std_dev']), cfg_base['min'], cfg_base['max']))
        cog = int(np.clip(np.random.normal(cfg_base['mean'], cfg_base['std_dev']), cfg_base['min'], cfg_base['max']))
        
        lck = int(np.clip(np.random.normal(cfg_lck['mean'], cfg_lck['std_dev']), cfg_lck['min'], cfg_lck['max']))

        # --- 2. Calculate HG ---
        hg_score = Horse._calculate_hg(spd, sta, acc, fcs, grt, cog, lck)
        
        # --- 3. Generate Name ---
        try:
            gen = NameGenerator(config_path=NAME_CONFIG_PATH)  # pass a seed=123 for determinism 
            name = gen.generate()
        except Exception as e:
            print(f"Warning: Could not load horse names. Using default. Error: {e}")
            name = "Generic Horse"

        # --- 4. Assign Strategy ---
        strategies = list(BALANCE_CONFIG['race_strategies'].keys())
        strategy = random.choice(strategies)

        # --- 5. Generate Preferred Distance Range ---
        cfg_pref_dist = BALANCE_CONFIG['preferred_distance_generation']
        center_point = random.randint(cfg_pref_dist['center_point_min'], cfg_pref_dist['center_point_max'])
        range_size = random.choice(cfg_pref_dist['range_size_options'])
        min_pref = int(center_point - (range_size / 2))
        max_pref = int(center_point + (range_size / 2))
        # Ensure min isn't nonsensically low (e.g., negative)
        min_pref = max(1000, min_pref) # E.g., Min possible preference is 1000m

        # --- 6. Save to Database ---
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                sql = """
                INSERT INTO horses
                (owner_id, is_bot, name, strategy, min_preferred_distance, max_preferred_distance,
                 spd, sta, acc, fcs, grt, cog, lck, hg_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING horse_id;
                """
                cur.execute(sql, (
                    owner_id, is_bot, name, strategy, min_pref, max_pref,
                    spd, sta, acc, fcs, grt, cog, lck, hg_score
                ))

                new_horse_id = cur.fetchone()[0]
                conn.commit()

                print(f"Generated new horse: {name} (ID: {new_horse_id}), HG: {hg_score}, Strategy: {strategy}, PrefDist: {min_pref}-{max_pref}m, BotOwned: {is_bot}")
                return new_horse_id
                
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Error generating new horse: {e}")
            return None
        finally:
            if conn:
                conn.close()

# --- The new Race Class (The Simulator) ---

class Race:
    """
    Manages the simulation of a single race event based on the new stat system.
    """
    def __init__(self, race_id, verbose=True):
        self.race_id = race_id
        self.horses = []
        self.distance = 0
        self.tier = ""
        self.positions = {}
        self.results_log = [] # A simple log of the finish order
        self.verbose = verbose
        self.track = None
        self._results_persisted = False

        self._load_race_data()
        self._load_horse_entries()
        self._load_track_geometry()

    def _load_race_data(self):
        """Loads the race's core data (distance, tier) from the DB."""
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT distance, tier FROM races WHERE race_id = %s", (self.race_id,))
                record = cur.fetchone()
            if record:
                self.distance, self.tier = record
                if self.verbose:
                    print(f"Loaded race data: {self.tier} race, {self.distance}m")
            else:
                if self.verbose:
                    print(f"Warning: No race found with ID {self.race_id}")
        except Exception as e:
            if self.verbose:
                print(f"Error loading race data: {e}")
            if conn:
                conn.rollback() # <--- THIS IS THE FIX
        finally:
            if conn:
                conn.close()

    def _load_horse_entries(self):
        """Loads all participating Horse objects into self.horses."""
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                # Get all horse_ids for this race
                cur.execute("SELECT horse_id FROM race_entries WHERE race_id = %s", (self.race_id,))
                records = cur.fetchall()
            
            if not records:
                print(f"Warning: No horses entered in race {self.race_id}")
                return
            
            # Create a Horse object for each entry
            for record in records:
                horse_id = record[0]
                self.horses.append(Horse(horse_id))
            
            if self.verbose:
                print(f"Loaded {len(self.horses)} horses for race {self.race_id}.")
            
            # Initialize positions for all loaded horses
            self.positions = {h.horse_id: 0 for h in self.horses}
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading horse entries: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def _load_track_geometry(self):
        """Attempts to attach SVG-based track data for the race distance."""
        if self.distance <= 0:
            return

        try:
            track = DEFAULT_TRACK_REGISTRY.find_by_distance(float(self.distance))
        except Exception as exc:
            if self.verbose:
                print(f"Failed to match SVG track: {exc}")
            return

        if not track:
            if self.verbose:
                print(f"No SVG track registered for ~{self.distance}m races yet.")
            return

        self.track = track

        if self.verbose:
            delta = abs(track.distance - float(self.distance))
            print(f"Attached SVG track: {track.name} ({track.distance:.2f}m, Δ {delta:.2f}m)")

    def _get_current_phase(self, horse_position):
        """Determines the race phase based on percentage of distance covered."""
        # Avoid division by zero for safety, though distance should always be > 0
        if self.distance <= 0:
            return 'early_race'

        cfg_phases = BALANCE_CONFIG['race_simulation']['phase_thresholds']
        current_percent = horse_position / self.distance

        if current_percent <= cfg_phases['early_end_percent']:
            return 'early_race'
        elif current_percent <= cfg_phases['mid_end_percent']:
            return 'mid_race'
        else:
            return 'late_race'

    def _run_round(self, round_number):
        """
        Simulates one round of the race for all horses.
        Uses percentage phases, strategy mods, distance penalty.
        Returns a list of log dictionaries, one for each horse.
        """
        cfg_sim = BALANCE_CONFIG['race_simulation']
        cfg_strats = BALANCE_CONFIG['race_strategies']
        cfg_dist_penalty = BALANCE_CONFIG['preferred_distance_penalty']

        round_log_entries = []

        for horse in self.horses:
            # Skip if already finished
            current_pos = self.positions.get(horse.horse_id, 0) # Get current position
            if current_pos >= self.distance:
                continue

            # --- 1. Determine Current Phase ---
            phase = self._get_current_phase(current_pos)

            # --- 2. Base Speed + Strategy Speed Modifier ---
            strat_spd_multi = cfg_strats[horse.strategy]['spd_modifier'][phase]
            base_mean = horse.spd * strat_spd_multi

            # --- 3. Consistency (FCS + Strategy FCS Modifier) ---
            strat_fcs_mod = cfg_strats[horse.strategy]['fcs_modifier'][phase]
            effective_fcs = np.clip(horse.fcs + strat_fcs_mod, # Apply modifier
                                   cfg_sim['fcs_sigma_map']['min_fcs'], # Clip to valid range
                                   cfg_sim['fcs_sigma_map']['max_fcs'])

            cfg_fcs_map = cfg_sim['fcs_sigma_map']
            fcs_range = cfg_fcs_map['max_fcs'] - cfg_fcs_map['min_fcs']
            sigma_range = cfg_fcs_map['max_sigma'] - cfg_fcs_map['min_sigma']

            # Calculate sigma using the *effective* FCS
            sigma = cfg_fcs_map['max_sigma'] - ((effective_fcs - cfg_fcs_map['min_fcs']) / fcs_range) * sigma_range
            # Ensure sigma is positive
            sigma = max(1.0, sigma)

            # --- 4. Calculate Movement Roll ---
            movement = np.clip(np.random.normal(base_mean, sigma), cfg_sim['min_move_per_round'], None)

            # --- 5. Stamina (STA) ---
            cfg_sta = cfg_sim['sta_modifier']
            sta_difference = horse.sta - cfg_sta['base_sta']
            distance_factor = max(0, (self.distance - cfg_sta['base_distance']) / cfg_sta['distance_factor_per_m'])
            sta_modifier = sta_difference * distance_factor
            sta_multiplier = (cfg_sta['base_power'] + sta_modifier) / cfg_sta['base_power']
            movement *= np.clip(sta_multiplier, cfg_sta['clip_min'], cfg_sta['clip_max'])

            # --- 6. Preferred Distance Penalty ---
            distance_diff = 0
            if self.distance < horse.min_preferred_distance:
                distance_diff = horse.min_preferred_distance - self.distance
            elif self.distance > horse.max_preferred_distance:
                distance_diff = self.distance - horse.max_preferred_distance

            if distance_diff > 0:
                penalty_steps = distance_diff / cfg_dist_penalty['penalty_per_meter_diff']
                penalty_percent = penalty_steps * cfg_dist_penalty['penalty_percent_per_step']
                # Apply capped penalty
                final_penalty = min(cfg_dist_penalty['max_penalty_percent'], penalty_percent)
                movement *= (1.0 - final_penalty)

            events_this_round = [] # For logging Grit/Skills

            # --- 7. Grit (GRT + Strategy Grit Modifier) ---
            if phase == 'late_race':
                cfg_grt = cfg_sim['grt_boost']
                strat_grit_mod = cfg_strats[horse.strategy]['grit_chance_modifier']['late_race']

                grt_range = cfg_grt['max_grt'] - cfg_grt['min_grt']
                chance_range = cfg_grt['max_chance'] - cfg_grt['min_chance']

                # Calculate base chance from GRT stat
                base_grit_chance = cfg_grt['min_chance'] + ((horse.grt - cfg_grt['min_grt']) / grt_range) * chance_range
                # Add strategy modifier and clip (e.g., 0% to 100%)
                final_grit_chance = np.clip(base_grit_chance + strat_grit_mod, 0, 100)

                if np.random.uniform(0, 100) < final_grit_chance:
                    movement *= cfg_grt['boost_multiplier']
                    events_this_round.append({
                        "type": "grit_boost",
                        "multiplier": cfg_grt['boost_multiplier']
                    })

            # --- 8. Update Position ---
            # Ensure movement isn't negative after penalties
            movement = max(0, movement)
            new_pos = current_pos + movement
            self.positions[horse.horse_id] = new_pos

            # --- 9. Check if Finished This Round ---
            if new_pos >= self.distance and horse.horse_id not in self.results_log:
                 # Check if not already logged to prevent duplicates if loop continues
                self.results_log.append(horse.horse_id)

            # --- 10. Create Log Entry ---
            round_log_entries.append({
                "race_id": self.race_id,
                "round_number": round_number,
                "horse_id": horse.horse_id,
                "movement_roll": movement,
                "stamina_multiplier": sta_multiplier, # Log the STA multiplier applied
                "final_position": new_pos,
                "round_events": json.dumps(events_this_round) if events_this_round else None
            })

        return round_log_entries

    def run_simulation(self, silent=False):
        """
        Runs the full race simulation until all horses finish.
        Saves the detailed log to the database.
        """
        if self._should_use_hybrid_engine():
            return self._run_hybrid_simulation(silent=silent)
        return self._run_legacy_simulation(silent=silent)

    def _run_legacy_simulation(self, silent=False):
        if not silent:
            print(f"\n--- Simulating Race {self.race_id} ({self.distance}m) ---")

        if not self.horses:
            if not silent:
                print("Race simulation cancelled: No horses.")
            return []

        full_race_log = []
        round_number = 0
        max_rounds = 100  # Safety break to prevent infinite loops

        # Loop until all horses are in the results log
        while len(self.results_log) < len(self.horses) and round_number < max_rounds:
            round_number += 1
            if not silent:
                print(f"  Simulating Round {round_number}...")
            round_log = self._run_round(round_number)
            full_race_log.extend(round_log)

        if round_number >= max_rounds:
            print(f"!!! WARNING: Race {self.race_id} exceeded max rounds ({max_rounds}). Force finishing.")
            # Force add any remaining horses based on final position
            remaining_horses = sorted(
                [h for h in self.horses if h.horse_id not in self.results_log],
                key=lambda h: self.positions.get(h.horse_id, 0),
                reverse=True,
            )
            for h in remaining_horses:
                self.results_log.append(h.horse_id)

        # --- Save Log to Database ---
        if not silent:
            conn = None
            try:
                conn = get_db_connection()
                with conn.cursor() as cur:
                    # Clear any old logs for this race_id just in case
                    cur.execute("DELETE FROM race_rounds WHERE race_id = %s", (self.race_id,))

                    args_list = [
                        (
                            log["race_id"],
                            log["round_number"],
                            log["horse_id"],
                            log["movement_roll"],
                            log["stamina_multiplier"],
                            log["final_position"],
                            log["round_events"],
                        )
                        for log in full_race_log
                    ]
                    if args_list:  # Ensure list is not empty
                        psycopg2.extras.execute_values(
                            cur,
                            "INSERT INTO race_rounds (race_id, round_number, horse_id, movement_roll, stamina_multiplier, final_position, round_events) VALUES %s",
                            args_list,
                        )
                        conn.commit()
                        print(f"     ...Saved {len(full_race_log)} round logs to database.")
                    else:
                        print("     ...No round logs generated to save.")
            except Exception as e:
                if conn:
                    conn.rollback()
                print(f"!!! Error saving race log: {e}")
            finally:
                if conn:
                    conn.close()

        return self.get_results(silent=silent)

    def _persist_results(self, finisher_ids: List[int]) -> None:
        if self._results_persisted or not finisher_ids:
            return
        try:
            derby_queries.clear_race_results(self.race_id)
            for idx, horse_id in enumerate(finisher_ids, start=1):
                derby_queries.record_race_result(self.race_id, horse_id, idx)
            derby_queries.set_race_winner(self.race_id, finisher_ids[0])
            self._results_persisted = True
        except Exception as exc:
            print(f"Warning: failed to persist race results for race {self.race_id}: {exc}")

    def _run_hybrid_simulation(self, silent: bool = False):
        if not self.track:
            if not silent:
                print("Hybrid engine enabled but no track geometry found; falling back to legacy simulation.")
            return self._run_legacy_simulation(silent=silent)

        if not self.horses:
            if not silent:
                print("Race simulation cancelled: No horses.")
            return []

        if not silent:
            print(f"\n--- Simulating Race {self.race_id} ({self.distance}m) [Hybrid] ---")

        self.results_log = []
        self.positions = {horse.horse_id: 0.0 for horse in self.horses}

        collect = not silent
        telemetry = TelemetryCollector() if collect else None
        profiles = self._build_racer_profiles()

        loop = HybridRaceLoop(self.track, profiles, telemetry=telemetry)
        loop.run_until_finished(max_time=600.0)

        finished_ids = list(loop.finished_order)
        if len(finished_ids) < len(profiles):
            remaining = sorted(loop.states, key=lambda s: s.pos, reverse=True)
            for state in remaining:
                racer_id = state.profile.racer_id
                if racer_id not in finished_ids:
                    finished_ids.append(racer_id)

        self.results_log = finished_ids
        self._persist_results(finished_ids)
        for state in loop.states:
            self.positions[state.profile.racer_id] = state.pos

        if collect and telemetry:
            self._write_hybrid_telemetry(telemetry, silent=silent)

        return self.get_results(silent=silent)

    def _should_use_hybrid_engine(self) -> bool:
        return _hybrid_engine_enabled() and self.track is not None

    def _build_racer_profiles(self) -> List[RacerProfile]:
        profiles: List[RacerProfile] = []
        for horse in self.horses:
            strategy = self._map_strategy(horse.strategy)
            acc_value = getattr(horse, "acc", None)
            if not acc_value or acc_value <= 0:
                acc_value = horse.spd
            stats = RacerStats(
                speed=float(horse.spd),
                acceleration=float(max(acc_value, 1)),
                stamina=float(horse.sta),
                grit=float(horse.grt),
                cognition=float(horse.cog),
                focus=float(horse.fcs),
                luck=float(horse.lck),
            )
            distance_grade = self._infer_distance_aptitude(horse)
            aptitudes = Aptitudes(distance=distance_grade, surface=DEFAULT_SURFACE_APTITUDE)
            profiles.append(
                RacerProfile(
                    racer_id=horse.horse_id,
                    name=horse.name,
                    stats=stats,
                    strategy=strategy,
                    aptitudes=aptitudes,
                )
            )
        return profiles

    def _map_strategy(self, raw_strategy) -> Strategy:
        if isinstance(raw_strategy, Strategy):
            return raw_strategy
        if isinstance(raw_strategy, int):
            return STRATEGY_INT_MAP.get(raw_strategy, Strategy.PACE_CHASER)
        if raw_strategy is None:
            return Strategy.PACE_CHASER
        raw_key = str(raw_strategy).strip().lower()
        if raw_key in STRATEGY_STRING_MAP:
            return STRATEGY_STRING_MAP[raw_key]
        normalized_key = raw_key.replace('-', ' ').replace('_', ' ')
        return STRATEGY_STRING_MAP.get(normalized_key, Strategy.PACE_CHASER)

    def _infer_distance_aptitude(self, horse) -> AptitudeGrade:
        try:
            preferred_center = (horse.min_preferred_distance + horse.max_preferred_distance) / 2
            diff = abs(preferred_center - self.distance)
        except (TypeError, AttributeError):
            diff = None

        grade = 'A'
        if diff is not None:
            if diff <= 50:
                grade = 'S'
            elif diff <= 150:
                grade = 'A'
            elif diff <= 300:
                grade = 'B'
            else:
                grade = 'C'
        try:
            return AptitudeGrade.from_str(grade)
        except ValueError:
            return AptitudeGrade.A

    def _write_hybrid_telemetry(self, telemetry: TelemetryCollector, silent: bool = False) -> None:
        frames = telemetry.export()
        if not frames:
            return

        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute("DELETE FROM race_rounds WHERE race_id = %s", (self.race_id,))

                args_list = []
                for frame in frames:
                    for racer in frame.racers:
                        event_payload = {
                            "tick": frame.tick,
                            "time": frame.time,
                            "phase": frame.phase,
                            "leading_segment": frame.leading_segment,
                            "lane_position": racer.lane_position,
                            "world_position": {
                                "x": racer.world_position[0],
                                "y": racer.world_position[1],
                            },
                            "distance_delta": racer.distance_delta,
                            "speed_delta": racer.speed_delta,
                            "speed": racer.speed,
                            "target_speed": racer.target_speed,
                            "accel": racer.accel,
                            "hp": racer.hp,
                            "hp_delta": racer.hp_delta,
                            "stamina_drain": racer.stamina_drain,
                            "ai_mode": racer.ai_mode,
                            "status_flags": racer.status_flags,
                            "debug": racer.debug,
                        }
                        args_list.append(
                            (
                                self.race_id,
                                frame.tick + 1,
                                racer.racer_id,
                                racer.distance_delta,
                                racer.stamina_drain,
                                racer.pos,
                                json.dumps(event_payload),
                            )
                        )

                if args_list:
                    psycopg2.extras.execute_values(
                        cur,
                        "INSERT INTO race_rounds (race_id, round_number, horse_id, movement_roll, stamina_multiplier, final_position, round_events) VALUES %s",
                        args_list,
                    )
                    conn.commit()
                    if not silent:
                        print(f"     ...Saved {len(args_list)} telemetry rows to database.")
        except Exception as e:
            if conn:
                conn.rollback()
            if not silent:
                print(f"!!! Error saving telemetry log: {e}")
        finally:
            if conn:
                conn.close()

    def get_results(self, silent=False):
        """Returns the final, ordered list of horse objects based on finish log."""
        # Use the results_log (which stores IDs in finish order)
        finished_ids = self.results_log

        # Create a dict of {id: horse_obj} for easy lookup
        horse_map = {h.horse_id: h for h in self.horses}

        # Handle potential errors where a horse ID might be missing from the map
        ordered_horses = []
        for horse_id in finished_ids:
            horse = horse_map.get(horse_id)
            if horse:
                ordered_horses.append(horse)
            else:
                if not silent:
                     print(f"Warning: Horse ID {horse_id} from results log not found in loaded horses.")

        if not silent:
            print("\n--- Race Results ---")
            for i, horse in enumerate(ordered_horses):
                print(f"{i+1}. {horse.name} (HG: {horse.hg_score}, Strategy: {horse.strategy}, Pref: {horse.min_preferred_distance}-{horse.max_preferred_distance}m)")

        return ordered_horses

class Bookie:
    """
    Manages odds generation for a given race by running Monte Carlo simulations.
    """
    def __init__(self, race_object):
        """
        Initializes the Bookie.

        Args:
            race_object (Race): The specific Race object to manage.
        """
        self.race = race_object
        self.win_probabilities = {}
        self.opening_odds = {}

    def run_monte_carlo(self, simulations: int = 10000):
        """
        Simulates the race thousands of times to determine accurate win probabilities.
        This is migrated from your race_logic_v2.py.
        """
        print(f"\nBookie: Running {simulations} Monte Carlo simulations...")
        if not self.race.horses:
            print("Bookie: Simulation cancelled, no horses in race.")
            return

        # We use horse_id for tracking wins
        win_counts = {horse.horse_id: 0 for horse in self.race.horses}

        for _ in range(simulations):
            # Create a deep copy of the race to run a fresh simulation
            sim_race = deepcopy(self.race)
            
            # Run the simulation and get the winner
            # Our new run_simulation() is much simpler than the old v2 logic
            results = sim_race.run_simulation(silent=True)
            
            if results:
                winner_horse = results[0]
                win_counts[winner_horse.horse_id] += 1

        # Calculate probabilities
        for horse_id, wins in win_counts.items():
            self.win_probabilities[horse_id] = wins / simulations
            
        print("Monte Carlo complete. Calculating odds...")
        self._calculate_all_odds()
        return self.opening_odds

    def _calculate_odds_from_win_rate(self, win_rate: float):
        """
        Converts a win probability (0.0 to 1.0) into fractional odds.
        """
        house_vig = BALANCE_CONFIG['economy']['house_vig'] # <-- This is the change
        
        if win_rate == 0:
            return 999.0
        
        fair_odds = (1 / win_rate) - 1
        final_odds = fair_odds * (1 - house_vig)
        final_odds = np.clip(final_odds, 0.1, 1257)
        
        return final_odds

    def _calculate_all_odds(self):
        """
        Uses the win probabilities to generate odds for every horse.
        """
        horse_map = {h.horse_id: h for h in self.race.horses}
        
        for horse_id, probability in self.win_probabilities.items():
            odds = self._calculate_odds_from_win_rate(probability)
            horse_name = horse_map[horse_id].name
            
            self.opening_odds[horse_name] = {
                "horse_id": horse_id,
                "probability": probability,
                "odds": odds
            }
            print(f"  -> {horse_name}: {odds:.2f} to 1 (Prob: {probability*100:.1f}%)")

# --- Test Script ---
if __name__ == "__main__":
    print("--- Testing Phase 2 (Refactored): simulation.py ---")

    conn = None
    test_race_id = None
    horse1_id = None
    horse2_id = None

    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # --- Setup ---
            print("Clearing old test data...")
            cur.execute("DELETE FROM race_rounds;")
            cur.execute("DELETE FROM race_entries;")
            cur.execute("DELETE FROM races;")
            cur.execute("DELETE FROM horses WHERE owner_id = 1;") # Clear bot 1's horses

            print("Generating 2 random test horses...")
            # Generate two random horses using the updated generator
            horse1_id = Horse.generate_new_horse(owner_id=1)
            horse2_id = Horse.generate_new_horse(owner_id=1)

            if not horse1_id or not horse2_id:
                raise Exception("Failed to generate test horses.")

            print("Creating 2000m test race...")
            cur.execute(
                "INSERT INTO races (tier, distance, entry_fee, status, purse) "
                "VALUES ('G', 2000, 100, 'open', 1000) RETURNING race_id;"
            )
            test_race_id = cur.fetchone()[0]

            print("Entering horses into race...")
            cur.execute("INSERT INTO race_entries (race_id, horse_id) VALUES (%s, %s);", (test_race_id, horse1_id))
            cur.execute("INSERT INTO race_entries (race_id, horse_id) VALUES (%s, %s);", (test_race_id, horse2_id))

            conn.commit()

    except Exception as e:
        if conn: conn.rollback()
        print(f"\nAn error occurred during setup: {e}")
        test_race_id = None # Ensure tests are skipped
    finally:
        if conn: conn.close()

    # --- Run Simulation & Bookie ---
    if test_race_id:
        print("\n--- Running Simulation Test ---")
        race_sim = Race(test_race_id)
        results = race_sim.run_simulation() # Should save logs now

        print("\n--- Running Bookie Test ---")
        # Need to load horses again for the Bookie's race object
        bookie_race = Race(test_race_id)
        if bookie_race.horses: # Only run if horses loaded
             bookie = Bookie(bookie_race)
             odds = bookie.run_monte_carlo(simulations=1000) # Lower sims for faster test

             if odds:
                 print("\n--- BOOKIE TEST COMPLETE ---")
                 # We can't easily predict the winner now, just check odds generated
                 for name, data in odds.items():
                     print(f"  -> {name}: {data['odds']:.2f} to 1 (Prob: {data['probability']*100:.1f}%)")
             else:
                 print("\n--- BOOKIE TEST FAILED ---")
                 print("Odds generation failed.")
        else:
            print("\n--- BOOKIE TEST SKIPPED (No horses found) ---")

    else:
        print("\nSkipping simulation and bookie tests due to setup failure.")

    print("\n--- TEST COMPLETE ---")
    print("Check 'race_rounds' table for detailed logs.")
