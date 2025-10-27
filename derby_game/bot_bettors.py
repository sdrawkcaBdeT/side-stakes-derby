import json
import random
import asyncio
from market import database # We will call our new database functions from here

class BotManager:
    """
    Manages the AI bot population, their profiles, and their betting behavior
    during a race's betting window.
    """

    def __init__(self, race_id, bookie):
        """
        Initializes the BotManager for a specific race.

        Args:
            race_id (int): The unique ID of the race being managed.
            bookie (Bookie): The Bookie object managing the current race's odds.
        """
        self.race_id = race_id
        self.bookie = bookie
        self.bots = self._load_bot_profiles()
        self.participating_bots = self._get_participating_bots()
        self.bets_placed_this_race = {bot['name']: 0 for bot in self.bots}

    def _load_bot_profiles(self) -> list:
        """Loads the bot personality profiles from the JSON config."""
        try:
            with open('horse_racing_game/configs/bot_personalities.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("ERROR: bot_personalities.json not found.")
            return []

    def _get_participating_bots(self) -> list:
        """
        Determines which bots will be active in this race based on their
        bet_frequency profile.
        """
        participants = []
        for bot in self.bots:
            if random.random() < bot.get('bet_frequency', 0.5):
                participants.append(bot)
        print(f"BOTS: {len(participants)} bots are participating in race #{self.race_id}.")
        return participants

    def _get_target_horse(self, preference: str, current_odds: dict) -> str:
        """
        Selects a horse to bet on based on the bot's preference.

        Args:
            preference (str): The bot's target preference (e.g., "favorite").
            current_odds (dict): The current odds from the bookie.

        Returns:
            str: The name of the horse to bet on.
        """
        if not current_odds:
            return None

        # Sort horses by odds (lowest to highest)
        sorted_horses = sorted(current_odds.items(), key=lambda item: item[1]['odds'])
        
        if preference == "favorite":
            return sorted_horses[0][0]
        elif preference == "longshot":
            # Target a horse in the bottom 25% of odds
            longshot_index = int(len(sorted_horses) * 0.75)
            return sorted_horses[longshot_index][0]
        elif preference == "value":
            # A simple value calculation: find horse where win_rate is highest relative to odds
            best_value_horse = max(current_odds.items(), key=lambda item: item[1]['win_rate'] / (item[1]['odds'] + 1))[0]
            return best_value_horse
        else: # Default to random
            return random.choice(list(current_odds.keys()))


    async def run_betting_cycle(self, duration_seconds: int = 120, tick_interval: int = 10):
        """
        The main engine loop that simulates bot betting over time.

        Args:
            duration_seconds (int): The total length of the betting window.
            tick_interval (int): How often the loop should wake up to check for bets.
        """
        num_ticks = duration_seconds // tick_interval
        
        for i in range(num_ticks):
            current_time_ratio = (i + 1) / num_ticks # e.g., 0.1, 0.2, ... 1.0

            for bot in self.participating_bots:
                # Check if bot has already placed its max number of bets
                if self.bets_placed_this_race[bot['name']] >= bot.get('max_bets_per_race', 1):
                    continue

                # Determine if this is the right time for the bot to bet
                timing = bot.get('bet_timing', 'any')
                should_bet_now = False
                if (timing == 'early' and current_time_ratio <= 0.3) or \
                   (timing == 'mid' and 0.3 < current_time_ratio < 0.8) or \
                   (timing == 'late' and current_time_ratio >= 0.8) or \
                   (timing == 'any'):
                    
                    # This is the bot's window, give it a chance to bet
                    # The probability is spread out over its available window
                    if random.random() < (1.0 / (num_ticks / 2)): # Simple probability
                        should_bet_now = True

                if should_bet_now:
                    # --- IT'S TIME TO BET ---
                    # 1. Select Target
                    target_horse = self._get_target_horse(
                        bot.get('target_preference', 'random'),
                        self.bookie.morning_line_odds # For now, uses morning line
                    )
                    if not target_horse: continue

                    # 2. Calculate Wager
                    wager = 0
                    if bot.get('wager_strategy') == 'percentage':
                        wager = bot['bankroll'] * bot['wager_amount']
                    else: # flat
                        wager = bot['wager_amount']
                    
                    wager = int(wager) # Bet whole numbers
                    if wager <= 0: continue
                    
                    # 3. Place Bet via Database
                    print(f"BOT ACTION: {bot['name']} is placing a {wager} CC bet on {target_horse}.")
                    success = database.place_bet_transaction(
                        race_id=self.race_id,
                        bettor_id=bot['name'],
                        horse_name=target_horse,
                        amount=wager,
                        odds=self.bookie.morning_line_odds[target_horse]['odds']
                    )

                    if success:
                        self.bets_placed_this_race[bot['name']] += 1
                        # In a real scenario, we'd update the bot's bankroll in the DB
                        bot['bankroll'] -= wager 
            
            await asyncio.sleep(tick_interval)
        
        print(f"BOTS: Betting cycle complete for race #{self.race_id}.")