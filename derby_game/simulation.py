import numpy as np
import json
import psycopg2.extras
from derby_game.database.connection import get_db_connection
from datetime import datetime, timezone
from copy import deepcopy
from derby_game.config import BALANCE_CONFIG

# --- Configuration ---
# We will use this to generate names.
# Note the path change to the new 'configs/' directory.
NAME_CONFIG_PATH = 'configs/horse_names.json'

# --- The new Horse Class ---

class Horse:
    """
    Represents a single horse in the database, using the new 6-stat system.
    """
    def __init__(self, horse_id):
        """
        Initializes a Horse object by loading its data from the database.
        """
        self.horse_id = horse_id
        
        # All stats will be populated by load_from_db()
        self.name = ""
        self.owner_id = 0
        self.birth_timestamp = None
        self.spd = 0
        self.sta = 0
        self.fcs = 0
        self.grt = 0
        self.cog = 0
        self.lck = 0
        self.hg_score = 0
        self.is_retired = False
        
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
                # This assumes column order from schema.sql
                # (horse_id, owner_id, name, birth_timestamp, spd, sta, fcs, grt, cog, lck, hg_score, is_retired, in_training_until)
                self.owner_id = record[1]
                self.name = record[2]
                self.birth_timestamp = record[3]
                self.spd = record[4]
                self.sta = record[5]
                self.fcs = record[6]
                self.grt = record[7]
                self.cog = record[8]
                self.lck = record[9]
                self.hg_score = record[10]
                self.is_retired = record[11]
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
    def _calculate_hg(spd, sta, fcs, grt, cog, lck):
        """
        Calculates the HG score based on config formula.
        """
        cfg = BALANCE_CONFIG['hg_formula_weights']
        hg = (spd * cfg['spd']) + (sta * cfg['sta']) + (fcs * cfg['fcs']) + \
             (grt * cfg['grt']) + (cog * cfg['cog']) + (lck * cfg['lck'])
        return int(hg)

    @staticmethod
    def generate_new_horse(owner_id):
        """
        Generates a new G-Grade horse with random stats, saves it to the DB,
        and returns the new horse's ID.
        """
        
        # --- 1. Generate Stats (Based on Config) ---
        cfg_base = BALANCE_CONFIG['horse_generation']['base_stats']
        cfg_lck = BALANCE_CONFIG['horse_generation']['lck_stats']
        
        spd = int(np.clip(np.random.normal(cfg_base['mean'], cfg_base['std_dev']), cfg_base['min'], cfg_base['max']))
        sta = int(np.clip(np.random.normal(cfg_base['mean'], cfg_base['std_dev']), cfg_base['min'], cfg_base['max']))
        fcs = int(np.clip(np.random.normal(cfg_base['mean'], cfg_base['std_dev']), cfg_base['min'], cfg_base['max']))
        grt = int(np.clip(np.random.normal(cfg_base['mean'], cfg_base['std_dev']), cfg_base['min'], cfg_base['max']))
        cog = int(np.clip(np.random.normal(cfg_base['mean'], cfg_base['std_dev']), cfg_base['min'], cfg_base['max']))
        
        lck = int(np.clip(np.random.normal(cfg_lck['mean'], cfg_lck['std_dev']), cfg_lck['min'], cfg_lck['max']))

        # --- 2. Calculate HG ---
        hg_score = Horse._calculate_hg(spd, sta, fcs, grt, cog, lck)
        
        # --- 3. Generate Name ---
        try:
            with open(NAME_CONFIG_PATH, 'r') as f:
                name_config = json.load(f)
            
            adjective = np.random.choice(name_config['adjectives'])
            noun = np.random.choice(name_config['nouns'])
            
            name = f"{adjective} {noun}"
        except Exception as e:
            print(f"Warning: Could not load horse names. Using default. Error: {e}")
            name = "Generic Horse"

        # --- 4. Save to Database ---
        # We no longer insert 'age'. We let 'birth_timestamp' use its default.
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                sql = """
                INSERT INTO horses 
                (owner_id, name, spd, sta, fcs, grt, cog, lck, hg_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING horse_id;
                """
                cur.execute(sql, (
                    owner_id, name,
                    spd, sta, fcs, grt, cog, lck, hg_score
                ))
                
                # Get the new horse_id that the database generated
                new_horse_id = cur.fetchone()[0]
                conn.commit()
                
                print(f"Generated new horse: {name} (ID: {new_horse_id}), HG: {hg_score}")
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
    def __init__(self, race_id):
        self.race_id = race_id
        self.horses = []
        self.distance = 0
        self.tier = ""
        self.positions = {}
        self.results_log = [] # A simple log of the finish order

        self._load_race_data()
        self._load_horse_entries()

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
                print(f"Loaded race data: {self.tier} race, {self.distance}m")
            else:
                print(f"Warning: No race found with ID {self.race_id}")
        except Exception as e:
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
            
            print(f"Loaded {len(self.horses)} horses for race {self.race_id}.")
            
            # Initialize positions for all loaded horses
            self.positions = {h.horse_id: 0 for h in self.horses}
            
        except Exception as e:
            print(f"Error loading horse entries: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def _get_current_phase(self, round_number):
        """Determines the race phase based on a 24-round system."""
        if round_number <= 4:
            return 'early_race'
        elif round_number <= 16:
            return 'mid_race'
        else:
            return 'late_race'

    def _run_round(self, round_number):
        """
        Simulates one round of the race for all horses.
        This is the core physics engine.
        Returns a list of log dictionaries, one for each horse.
        """
        phase = self._get_current_phase(round_number)
        cfg_sim = BALANCE_CONFIG['race_simulation']
        
        round_log_entries = []
        
        for horse in self.horses:
            if horse.horse_id in self.results_log:
                continue 

            base_mean = horse.spd
            
            # --- 2. Consistency (FCS) ---
            cfg_fcs = cfg_sim['fcs_sigma_map']
            fcs_range = cfg_fcs['max_fcs'] - cfg_fcs['min_fcs']
            sigma_range = cfg_fcs['max_sigma'] - cfg_fcs['min_sigma']
            sigma = cfg_fcs['max_sigma'] - ((horse.fcs - cfg_fcs['min_fcs']) / fcs_range) * sigma_range
            movement = np.clip(np.random.normal(base_mean, sigma), 10, None)

            # --- 3. Stamina (STA) ---
            cfg_sta = cfg_sim['sta_modifier']
            sta_difference = horse.sta - cfg_sta['base_sta']
            distance_factor = max(0, (self.distance - cfg_sta['base_distance']) / cfg_sta['distance_factor_per_m'])
            sta_modifier = sta_difference * distance_factor
            sta_multiplier = (cfg_sta['base_power'] + sta_modifier) / cfg_sta['base_power']
            movement *= np.clip(sta_multiplier, cfg_sta['clip_min'], cfg_sta['clip_max'])

            # This is our new event tracker
            events_this_round = []

            # --- 4. Grit (GRT) ---
            if phase == 'late_race':
                cfg_grt = cfg_sim['grt_boost']
                grt_range = cfg_grt['max_grt'] - cfg_grt['min_grt']
                chance_range = cfg_grt['max_chance'] - cfg_grt['min_chance']
                grit_chance = cfg_grt['min_chance'] + ((horse.grt - cfg_grt['min_grt']) / grt_range) * chance_range
                
                if np.random.uniform(0, 100) < grit_chance:
                    movement *= cfg_grt['boost_multiplier']
                    # We log the event
                    events_this_round.append({
                        "type": "grit_boost",
                        "multiplier": cfg_grt['boost_multiplier']
                    })
            
            # --- 5. COG / LCK (The Future) ---
            # When we implement skills, we'll add logic here.
            # LCK can influence the grit_chance roll.
            # COG can trigger a new skill.
            # Any triggers will just be added to the events_this_round list.
            
            # --- 6. Update Position & Check Finish ---
            self.positions[horse.horse_id] += movement
            if self.positions[horse.horse_id] >= self.distance:
                self.results_log.append(horse.horse_id)

            # --- 7. Create the Log Entry ---
            round_log_entries.append({
                "race_id": self.race_id,
                "round_number": round_number,
                "horse_id": horse.horse_id,
                "movement_roll": movement, # This is the final movement this round
                "stamina_multiplier": sta_multiplier,
                "final_position": self.positions[horse.horse_id],
                "round_events": json.dumps(events_this_round) if events_this_round else None
            })
            
        return round_log_entries

    def run_simulation(self, silent=False):
        """
        Runs the full race simulation and saves the detailed log to the database.
        """
        if not silent:
            print(f"\n--- Simulating Race {self.race_id} ({self.distance}m) ---")
            
        if not self.horses:
            if not silent:
                print("Race simulation cancelled: No horses.")
            return []
            
        full_race_log = [] # This will store all log entries
            
        # Run 24 rounds
        for i in range(1, 25):
            round_log = self._run_round(i)
            full_race_log.extend(round_log)
            if len(self.results_log) == len(self.horses):
                break
        
        # Add DNFs
        for horse in self.horses:
            if horse.horse_id not in self.results_log:
                self.results_log.append(horse.horse_id)
        
        # We only do this if it's NOT a silent Monte Carlo run
        if not silent:
            conn = None
            try:
                conn = get_db_connection()
                with conn.cursor() as cur:
                    # Use psycopg2's fast executemany to insert all rounds at once
                    args_list = [
                        (log['race_id'], log['round_number'], log['horse_id'], 
                         log['movement_roll'], log['stamina_multiplier'], 
                         log['final_position'], log['round_events'])
                        for log in full_race_log
                    ]
                    psycopg2.extras.execute_values(
                        cur,
                        "INSERT INTO race_rounds (race_id, round_number, horse_id, movement_roll, stamina_multiplier, final_position, round_events) VALUES %s",
                        args_list
                    )
                    conn.commit()
                    print(f"     ...Saved {len(full_race_log)} round logs to database.")
            except Exception as e:
                if conn: conn.rollback()
                print(f"!!! Error saving race log: {e}")
            finally:
                if conn: conn.close()
        
        return self.get_results(silent=silent)

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
    print("--- Testing Phase 2: simulation.py ---")
    
    # --- Test 1: Generate specialized horses for the test ---
    print("Generating test horses (Sprinter vs. Stayer)...")
    
    conn = None
    test_race_id = None
    stayer_id = None
    
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # First, clear any old test data
            cur.execute("DELETE FROM race_entries;")
            cur.execute("DELETE FROM races;")
            cur.execute("DELETE FROM horses WHERE owner_id = 1;")
            
            # Horse 1: "Speedy Sprinter" (High SPD, Low STA)
            hg_sprinter = Horse._calculate_hg(150, 50, 100, 100, 100, 300)
            cur.execute(
                "INSERT INTO horses (owner_id, name, spd, sta, fcs, grt, cog, lck, hg_score) "
                "VALUES (1, 'Speedy Sprinter', 150, 50, 100, 100, 100, 300, %s) RETURNING horse_id;",
                (hg_sprinter,)
            )
            sprinter_id = cur.fetchone()[0]

            # Horse 2: "Steady Stayer" (Low SPD, High STA)
            # SPD is 110 to give it the edge
            hg_stayer = Horse._calculate_hg(110, 150, 100, 100, 100, 300) 
            cur.execute(
                "INSERT INTO horses (owner_id, name, spd, sta, fcs, grt, cog, lck, hg_score) "
                "VALUES (1, 'Steady Stayer', 110, 150, 100, 100, 100, 300, %s) RETURNING horse_id;", 
                (hg_stayer,)
            )
            stayer_id = cur.fetchone()[0]
            
            # --- Test 2: Create a test race ---
            print("Creating 3000m test race...")
            cur.execute(
                "INSERT INTO races (tier, distance, entry_fee, status, purse) "
                "VALUES ('G', 3000, 100, 'open', 1000) RETURNING race_id;"
            )
            test_race_id = cur.fetchone()[0]
            
            # --- Test 3: Enter horses into the race ---
            print("Entering horses into race...")
            cur.execute("INSERT INTO race_entries (race_id, horse_id) VALUES (%s, %s);", (test_race_id, sprinter_id))
            cur.execute("INSERT INTO race_entries (race_id, horse_id) VALUES (%s, %s);", (test_race_id, stayer_id))
            
            conn.commit() # Commit all the setup

    except Exception as e:
        if conn:
            conn.rollback() # Rollback if setup failed
        print(f"\nAn error occurred during the setup: {e}")
    finally:
        if conn:
            conn.close() # Always close the connection

    # --- Test 4: Run the simulation & Bookie (outside the setup try/except) ---
    if test_race_id:
        print("\n--- Running Simulation Test ---")
        race_sim = Race(test_race_id)
        results = race_sim.run_simulation()
        
        if results and results[0].horse_id == stayer_id:
            print("\n--- SIMULATION TEST SUCCEEDED (or passed with high probability) ---")
            print("The 'Steady Stayer' won the long race, as expected.")
        else:
            print("\n--- SIMULATION TEST FAILED (or a rare upset) ---")
            print("The 'Speedy Sprinter' won.")

        print("\n--- Running Bookie Test ---")
        # Re-create a fresh Race object for the bookie
        bookie_race = Race(test_race_id)
        bookie = Bookie(bookie_race)
        odds = bookie.run_monte_carlo(simulations=5000) # 5k is faster for a test
        
        if odds and odds['Steady Stayer']['probability'] > odds['Speedy Sprinter']['probability']:
             print("\n--- BOOKIE TEST SUCCEEDED ---")
             print(f"The 'Steady Stayer' is the favorite ({odds['Steady Stayer']['probability']*100:.1f}%), as expected.")
        else:
            print("\n--- BOOKIE TEST FAILED ---")
            print("The 'Speedy Sprinter' is the favorite, which is wrong for this distance.")
    else:
        print("\nSkipping simulation and bookie tests due to setup failure.")