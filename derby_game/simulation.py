import numpy as np
import json
from derby_game.database.connection import get_db_connection
from datetime import datetime, timezone

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

    # ... (Keep the _calculate_hg and generate_new_horse static methods exactly as they are) ...
    @staticmethod
    def _calculate_hg(spd, sta, fcs, grt, cog, lck):
        """
        Calculates the HG score based on the design doc's formula.
        HG = (SPD * 3.5) + (STA * 3.5) + (FCS * 1.5) + (GRT * 1.0) + (COG * 1.0) + (LCK * 0.5)
        """
        hg = (spd * 3.5) + (sta * 3.5) + (fcs * 1.5) + (grt * 1.0) + (cog * 1.0) + (lck * 0.5)
        return int(hg)

    @staticmethod
    def generate_new_horse(owner_id):
        """
        Generates a new G-Grade horse with random stats, saves it to the DB,
        and returns the new horse's ID.
        
        This is Feature 2.1 from our plan, now with corrected stat ranges.
        """
        
        # --- 1. Generate Stats (Based on NEW 50-150 Range) ---
        # SPD/STA/FCS/GRT/COG: 50 - 150 (Normal Distribution)
        # LCK: 100 - 500 (Normal Distribution)
        
        # Stats with mean 100, std dev ~17, clipped to [50, 150]
        mean = 100
        std_dev = 17 
        spd = int(np.clip(np.random.normal(mean, std_dev), 50, 150))
        sta = int(np.clip(np.random.normal(mean, std_dev), 50, 150))
        fcs = int(np.clip(np.random.normal(mean, std_dev), 50, 150))
        grt = int(np.clip(np.random.normal(mean, std_dev), 50, 150))
        cog = int(np.clip(np.random.normal(mean, std_dev), 50, 150))
        
        # LCK with mean 300, std dev ~67, clipped to [100, 500]
        lck_mean = 300
        lck_std_dev = 67
        lck = int(np.clip(np.random.normal(lck_mean, lck_std_dev), 100, 500))

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
        """
        phase = self._get_current_phase(round_number)
        
        for horse in self.horses:
            if horse.horse_id in self.results_log:
                continue # This horse has already finished

            # 1. Base Movement (SPD)
            # Use SPD as the 'mean' for the normal distribution
            base_mean = horse.spd

            # 2. Consistency (FCS)
            # FCS determines the 'sigma' (standard deviation).
            # We want high FCS (150) -> low sigma
            # We want low FCS (50) -> high sigma
            # Let's map FCS [50, 150] to sigma [75, 25]
            max_sigma = 75
            min_sigma = 25
            fcs_range = 150 - 50 # 100
            sigma_range = max_sigma - min_sigma # 50
            
            sigma = max_sigma - ((horse.fcs - 50) / fcs_range) * sigma_range
            
            # Get the random movement, ensuring it's not negative
            movement = np.clip(np.random.normal(base_mean, sigma), 10, None)

            # 3. Stamina (STA)
            # This is a smooth function where every point of STA matters.
            # We define a 'base stamina' (100) and a 'base distance' (1000m).
            # A horse with 100 STA running at 1000m has a 1.0x multiplier.
            
            base_sta = 100.0
            base_distance = 1000.0
            
            # a. Calculate how different the horse is from the 'average'
            # This will be a value from -50 (worst) to +50 (best)
            sta_difference = horse.sta - base_sta
            
            # b. Calculate how much the distance amplifies the stamina effect.
            # The effect scales up every 500m over the base distance.
            distance_factor = max(0, (self.distance - base_distance) / 500.0)
            
            # c. Calculate the final modifier.
            # A -50 STA diff at 3000m (dist_factor=4) gives a -200 modifier.
            # A +50 STA diff at 3000m (dist_factor=4) gives a +200 modifier.
            sta_modifier = sta_difference * distance_factor
            
            # d. Apply this to a 'base power' to get a multiplier.
            # (1000 - 200) / 1000 = 0.8x multiplier
            # (1000 + 200) / 1000 = 1.2x multiplier
            base_power = 1000.0
            sta_multiplier = (base_power + sta_modifier) / base_power
            
            # e. Apply the multiplier to the horse's movement.
            # We'll clip it to prevent extreme results (e.g., 90% penalty or 50% bonus)
            movement *= np.clip(sta_multiplier, 0.1, 1.5)

            # 4. Grit (GRT)
            # Chance for a speed boost in the 'late_race' phase
            if phase == 'late_race':
                # Map GRT [50, 150] to a % chance [10%, 30%]
                grit_chance = 10 + ((horse.grt - 50) / fcs_range) * 20
                if np.random.uniform(0, 100) < grit_chance:
                    # Apply a significant boost (e.g., +25%)
                    movement *= 1.25 
            
            # 5. COG & LCK
            # (Not implemented in this core loop, as they affect
            # training and special skills, as per the design doc)

            # Update the horse's position
            self.positions[horse.horse_id] += movement
            
            # Check if finished
            if self.positions[horse.horse_id] >= self.distance:
                self.results_log.append(horse.horse_id)

    def run_simulation(self):
        """
        Runs the full race simulation from start to finish.
        """
        print(f"\n--- Simulating Race {self.race_id} ({self.distance}m) ---")
        if not self.horses:
            print("Race simulation cancelled: No horses.")
            return []
            
        # Run 24 rounds
        for i in range(1, 25):
            self._run_round(i)
            # Stop if all horses have finished
            if len(self.results_log) == len(self.horses):
                break
        
        # Add any horses that didn't finish (DNF)
        for horse in self.horses:
            if horse.horse_id not in self.results_log:
                self.results_log.append(horse.horse_id)
        
        return self.get_results()

    def get_results(self):
        """Returns the final, ordered list of horse objects."""
        # Use the results_log to get the finish order.
        # For DNFs (who were added last), sort them by their final position.
        
        finished_ids = self.results_log
        
        # Create a dict of {id: horse_obj} for easy lookup
        horse_map = {h.horse_id: h for h in self.horses}
        
        # Return the list of Horse objects in the order they finished
        ordered_horses = [horse_map[horse_id] for horse_id in finished_ids]
        
        print("\n--- Race Results ---")
        for i, horse in enumerate(ordered_horses):
            print(f"{i+1}. {horse.name} (HG: {horse.hg_score})")
            
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
            sim_race = deepcopy.deepcopy(self.race)
            
            # Run the simulation and get the winner
            # Our new run_simulation() is much simpler than the old v2 logic
            results = sim_race.run_simulation()
            
            if results:
                winner_horse = results[0]
                win_counts[winner_horse.horse_id] += 1

        # Calculate probabilities
        for horse_id, wins in win_counts.items():
            self.win_probabilities[horse_id] = wins / simulations
            
        print("Monte Carlo complete. Calculating odds...")
        self._calculate_all_odds()
        return self.opening_odds

    def _calculate_odds_from_win_rate(self, win_rate: float, house_vig: float = 0.08):
        """
        Converts a win probability (0.0 to 1.0) into fractional odds.
        This is migrated directly from your race_logic_v2.py.
        """
        if win_rate == 0:
            return 999.0  # Avoid division by zero, give it 999 to 1 odds
        
        # Fair odds (no house edge)
        fair_odds = (1 / win_rate) - 1
        
        # Add the house edge
        # We reduce the payout, which means we increase the "odds value"
        final_odds = fair_odds * (1 - house_vig)
        
        # Clamp to a minimum (e.g., 0.1 to 1) and a max (e.g., 999 to 1)
        final_odds = np.clip(final_odds, 0.1, 999)
        
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
            # Stats (spd, sta, fcs, grt, cog, lck) -> (150, 50, 100, 100, 100, 300)
            hg_sprinter = Horse._calculate_hg(150, 50, 100, 100, 100, 300)
            cur.execute(
                "INSERT INTO horses (owner_id, name, spd, sta, fcs, grt, cog, lck, hg_score) "
                "VALUES (1, 'Speedy Sprinter', 150, 50, 100, 100, 100, 300, %s) RETURNING horse_id;",
                (hg_sprinter,)
            )
            sprinter_id = cur.fetchone()[0]

            # Horse 2: "Steady Stayer" (Low SPD, High STA)
            # Stats (spd, sta, fcs, grt, cog, lck) -> (110, 150, 100, 100, 100, 300) <-- SPD changed to 110
            hg_stayer = Horse._calculate_hg(110, 150, 100, 100, 100, 300) # <--- THIS LINE IS CHANGED
            cur.execute(
                "INSERT INTO horses (owner_id, name, spd, sta, fcs, grt, cog, lck, hg_score) "
                "VALUES (1, 'Steady Stayer', 110, 150, 100, 100, 100, 300, %s) RETURNING horse_id;", # <--- THIS LINE IS CHANGED
                (hg_stayer,)
            )
            stayer_id = cur.fetchone()[0]
            
            print(f"Created Horse ID {sprinter_id} (Sprinter, HG: {hg_sprinter})")
            print(f"Created Horse ID {stayer_id} (Stayer, HG: {hg_stayer})")
            
            # --- Test 2: Create a test race ---
            # A long 3000m race to test stamina
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

            # --- Test 4: Run the simulation ---
            # Now we create the Race object, which will load all this new data
            race_sim = Race(test_race_id)
            
            # Run the simulation
            results = race_sim.run_simulation()
            
            if results and results[0].horse_id == stayer_id:
                print("\n--- TEST SUCCEEDED ---")
                print("The 'Steady Stayer' won the long race, as expected.")
            else:
                print("\n--- TEST FAILED ---")
                print("The 'Speedy Sprinter' won, or an error occurred.")

    except Exception as e:
        if conn:
            conn.rollback() # Rollback if setup failed
        print(f"\nAn error occurred during the test: {e}")
    finally:
        if conn:
            conn.close() # Always close the connection