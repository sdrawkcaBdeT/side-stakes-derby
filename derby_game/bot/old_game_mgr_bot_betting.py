import discord
from discord.ext import commands, tasks
import os
import csv
from datetime import datetime, timedelta
import pytz
import pandas as pd
import re
import asyncio
import json
import random
from .race_logic import Race, Horse
from . import skills # We don't use it directly, but race_logic needs it
from .race_logic_v2 import Horse as HorseV2, Race as RaceV2, Bookie
from .race_bots import BotManager
from market import database
import numpy as np
import math

WINNERS_CIRCLE_CHANNEL_NAME = 'winners-circle-racing'

def format_cc(amount):
    """Formats a number as a string with commas and 'CC'."""
    if pd.isna(amount):
        amount = 0
    return f"{float(amount):,.0f} CC"

BOT_PERSONALITIES = {
    "StartingGateSally": {"type": "starter"},
    "PaddockPete": {"type": "starter"},
    "FirstTurnFrank": {"type": "starter"},
    "BackstretchBarry": {"type": "starter"},
    "HomestretchHarry": {"type": "starter"},
    "GrandstandGus": {"type": "starter"},
    "ClubhouseClara": {"type": "starter"},
    "BagginsTheBookie": {"type": "human_trigger"}, # The original bot
    "SniperSam": {"type": "sniper"},
    "DailyDoubleDoug": {"type": "sniper"},
    "PhotoFinishPhil": {"type": "sniper"},
}

active_races = {}  # In-memory dictionary to hold live Race objects

DEFINED_HORSES = ["Sakura Bakushin Oh", "Marzensty", "El", "Oguri Hat", "Gold Trip", "Earth Rut"]

HORSE_NAME_PARTS = {
    "adjectives": ["Galloping", "Midnight", "Dusty", "Iron", "Star", "Thunder", "Shadow", "Golden", "Baggins", "Cheating", "BOT", "Nice", "Mean", "Godly", "Inspector", "Dishonest", "Old School", "Broken", "Neon", "Lucky", "Turbo", "Sticky", "Captain", "Major", "Wild", "Furious", "Stoic", "Noble", "Cash"],
    "nouns": ["Bullet", "Fury", "Runner", "Chaser", "Stallion", "Dreamer", "Comet", "Baggins", "Cheater", "God", "Nature", "Gadget", "Wave", "Insomnia", "Twice", "Epidemic", "King", "Queen", "Jester", "Engine", "Glitch", "Rocket", "Mirror", "Major", "Pilgrim", "Mountain", "Crew"]
}
HORSE_STRATEGIES = ["Front Runner", "Pace Chaser", "Late Surger", "End Closer"]
GENERIC_SKILLS = ["Straightaway Adept", "Homestretch Haste", "Slipstream", "Late Start"]
UNIQUE_HORSES = {
    "Sakura Bakushin Oh": {"strategy": "Front Runner", "skills": ["Huge Lead"]},
    "Marzensty": {"strategy": "Front Runner", "skills": ["Early Lead"]},
    "El": {"strategy": "Pace Chaser", "skills": ["Victoria por plata"]},
    "Oguri Hat": {"strategy": "Pace Chaser", "skills": ["Trumpet Blast"]},
    "Gold Trip": {"strategy": "End Closer", "skills": ["Uma Stan"]},
    "Earth Rut": {"strategy": "Pace Chaser", "skills": ["Fiery Satisfaction"]},
}

def generate_race_field(num_horses_needed):
    """
    Generates a list of horse names for a race, prioritizing defined horses
    and filling the rest with unique, randomly generated names.
    """
    # 1. Start with the list of defined horses.
    available_horses = DEFINED_HORSES[:]

    # 2. Generate additional random horses if needed.
    while len(available_horses) < num_horses_needed:
        adj = random.choice(HORSE_NAME_PARTS["adjectives"])
        noun = random.choice(HORSE_NAME_PARTS["nouns"])
        new_name = f"{adj} {noun}"

        # 3. Ensure the new name is unique before adding it.
        if new_name not in available_horses:
            available_horses.append(new_name)

    # 4. Shuffle the list and return the exact number needed for the race.
    random.shuffle(available_horses)
    return available_horses[:num_horses_needed]

# --- Helper Functions for Racing ---
def get_or_create_csv(filepath, headers):
    """Checks if a CSV exists, creates it with headers if not."""
    if not os.path.exists(filepath):
        pd.DataFrame(columns=headers).to_csv(filepath, index=False)
        print(f"Created missing file: {filepath}")

def initialize_race_files():
    """Ensure all necessary CSV files for the racing game exist."""
    os.makedirs('horse_racing_game/data', exist_ok=True)
    # Define headers for all our files
    races_headers = ['race_id', 'message_id', 'channel_id', 'track_length', 'num_horses', 'status', 'winner', 'start_time']
    horses_headers = ['race_id', 'horse_number', 'horse_name', 'strategy', 'skills']
    bets_headers = ['race_id', 'bettor_id', 'horse_number', 'bet_amount', 'odds_at_bet', 'time_left_in_window', 'winnings']
    results_headers = ['race_id', 'horse_name', 'horse_number', 'strategy', 'skills', 'final_position', 'final_place']
    events_headers = ['race_id', 'round', 'horse_name', 'rolls_str', 'modifier', 'skill_bonus', 'total_movement', 'skill_roll', 'skill_chance', 'skills_activated', 'position_after']
    jackpot_headers = ['current_jackpot']

    # Create files if they don't exist
    get_or_create_csv('horse_racing_game/data/races.csv', races_headers)
    get_or_create_csv('horse_racing_game/data/race_horses.csv', horses_headers)
    get_or_create_csv('horse_racing_game/data/race_bets.csv', bets_headers)
    get_or_create_csv('horse_racing_game/data/race_results.csv', results_headers)
    get_or_create_csv('horse_racing_game/data/race_events.csv', events_headers)

    if not os.path.exists('horse_racing_game/data/jackpot_ledger.csv'):
        pd.DataFrame([{'current_jackpot': 0}]).to_csv('horse_racing_game/data/jackpot_ledger.csv', index=False)

    bot_ledger_file = 'horse_racing_game/data/bot_ledgers.csv'
    if not os.path.exists(bot_ledger_file):
        bot_data = [{'bot_name': name, 'bankroll': 10000, 'total_bets': 0, 'total_winnings': 0} for name in BOT_PERSONALITIES]
        pd.DataFrame(bot_data).to_csv(bot_ledger_file, index=False)

async def _record_bet(race, bettor_id, horse_number, bet_amount):
    """Calculates live odds and time, then records the bet to the CSV."""
    loop = asyncio.get_running_loop()

    races_df = await loop.run_in_executor(
        None,
        lambda: pd.read_csv('horse_racing_game/data/races.csv', parse_dates=['start_time'])
    )

    race_info = races_df[races_df['race_id'] == race.race_id].iloc[0]
    start_time = race_info['start_time']

    if start_time.tzinfo is None:
        start_time = start_time.tz_localize('UTC')

    duration = timedelta(seconds=120)
    time_elapsed = datetime.now(pytz.utc) - start_time
    time_left = (duration - time_elapsed).total_seconds()

    bets_df = await loop.run_in_executor(None, pd.read_csv, 'horse_racing_game/data/race_bets.csv')
    race_bets = bets_df[bets_df['race_id'] == race.race_id]
    total_pot = race_bets['bet_amount'].sum()
    bets_on_horse = race_bets[race_bets['horse_number'] == horse_number]['bet_amount'].sum()
    odds = (total_pot - bets_on_horse) / bets_on_horse if bets_on_horse > 0 else total_pot

    new_bet = pd.DataFrame([{
        'race_id': race.race_id,
        'bettor_id': bettor_id,
        'horse_number': horse_number,
        'bet_amount': bet_amount,
        'odds_at_bet': round(odds, 2),
        'time_left_in_window': round(time_left, 2),
        'winnings': 0
    }])
    await loop.run_in_executor(None, lambda: new_bet.to_csv('horse_racing_game/data/race_bets.csv', mode='a', header=False, index=False))

async def place_starting_bot_bets(race, channel):
    """Gets starter bots to bet on different horses."""
    print(f"Placing initial bot bets for Race ID: {race.race_id}")
    starter_bots = [name for name, props in BOT_PERSONALITIES.items() if props['type'] == 'starter']

    if len(race.horses) < len(starter_bots):
        return

    loop = asyncio.get_running_loop()
    bot_ledgers = await loop.run_in_executor(None, pd.read_csv, 'horse_racing_game/data/bot_ledgers.csv')
    horses_to_bet_on = random.sample(race.horses, len(starter_bots))

    for i, bot_name in enumerate(starter_bots):
        horse = horses_to_bet_on[i]
        bankroll = bot_ledgers.loc[bot_ledgers['bot_name'] == bot_name, 'bankroll'].iloc[0]
        bet_amount = random.randint(int(bankroll * 0.01), int(bankroll * 0.05))
        bet_amount = min(bet_amount, bankroll)

        if bet_amount > 0:
            bot_ledgers.loc[bot_ledgers['bot_name'] == bot_name, 'bankroll'] -= bet_amount
            bot_ledgers.loc[bot_ledgers['bot_name'] == bot_name, 'total_bets'] += bet_amount
            await _record_bet(race, bot_name, horse.number, bet_amount)

    await loop.run_in_executor(None, lambda: bot_ledgers.to_csv('horse_racing_game/data/bot_ledgers.csv', index=False))
    await channel.send("A flurry of early bets have come in from the regular crowd!")

async def run_complete_race(bot, channel, message, race):
    """
    A single, robust function that handles the entire race lifecycle.
    """
    try:
        print(f"Starting countdown for Race ID: {race.race_id}")
        start_time = datetime.now(pytz.utc)
        duration = timedelta(seconds=120)

        sniper_bots = [name for name, props in BOT_PERSONALITIES.items() if props['type'] == 'sniper']
        snipers_who_have_bet = []

        while True:
            time_elapsed = datetime.now(pytz.utc) - start_time
            time_remaining = duration - time_elapsed
            if time_remaining.total_seconds() <= 0:
                break

            try:
                loop = asyncio.get_running_loop()
                bets_df = await loop.run_in_executor(None, pd.read_csv, 'horse_racing_game/data/race_bets.csv')
                bets_df = bets_df.query(f"race_id == {race.race_id}")
                total_pot = bets_df['bet_amount'].sum()

                embed = message.embeds[0]
                field_text = ""
                for horse in race.horses:
                    bets_on_horse = bets_df[bets_df['horse_number'] == horse.number]['bet_amount'].sum()
                    odds = (total_pot - bets_on_horse) / bets_on_horse if bets_on_horse > 0 else 0
                    odds_str = f"{odds:.1f}:1" if bets_on_horse > 0 else "--:--"
                    field_text += f"`[{horse.number}]` **{horse.name}** ({horse.strategy}) - Odds: {odds_str}\n"

                embed.set_field_at(0, name="THE FIELD", value=field_text, inline=False)
                minutes, seconds = divmod(int(time_remaining.total_seconds()), 60)
                embed.set_field_at(1, name="Betting Window", value=f"Closing in {minutes}m {seconds}s")
                embed.description = f"The total pot is now **{format_cc(total_pot)}**!"
                await message.edit(embed=embed)
            except Exception as e:
                print(f"Error updating odds (expected during testing with no bets): {e}")

            if time_remaining.total_seconds() <= 10:
                for sniper_name in sniper_bots:
                    if sniper_name not in snipers_who_have_bet:
                        if random.randint(1, 50) == 1:
                            snipers_who_have_bet.append(sniper_name)
                            print(f"{sniper_name} is betting on Race ID: {race.race_id}!")

                            loop = asyncio.get_running_loop()
                            bot_ledgers = await loop.run_in_executor(None, pd.read_csv, 'horse_racing_game/data/bot_ledgers.csv')
                            sniper_row = bot_ledgers[bot_ledgers['bot_name'] == sniper_name]

                            if not sniper_row.empty:
                                bankroll = sniper_row['bankroll'].iloc[0]
                                bet_amount = random.randint(int(bankroll * 0.15), int(bankroll * 0.30))
                                bet_amount = min(bet_amount, bankroll)

                                if bet_amount > 0:
                                    chosen_horse = random.choice(race.horses)
                                    bot_ledgers.loc[bot_ledgers['bot_name'] == sniper_name, 'bankroll'] -= bet_amount
                                    bot_ledgers.loc[bot_ledgers['bot_name'] == sniper_name, 'total_bets'] += bet_amount

                                    bet_df = pd.DataFrame([{'race_id': race.race_id, 'bettor_id': sniper_name, 'horse_number': chosen_horse.number, 'bet_amount': bet_amount}])
                                    bet_df.to_csv('horse_racing_game/data/race_bets.csv', mode='a', header=False, index=False)
                                    await loop.run_in_executor(None, lambda: bot_ledgers.to_csv('horse_racing_game/data/bot_ledgers.csv', index=False))

                                    await channel.send(f"A huge last-minute bet has just come in! **{sniper_name}** places **{format_cc(bet_amount)}** on **{chosen_horse.name}**!")

            await asyncio.sleep(5)

        print(f"Countdown finished for Race ID: {race.race_id}. Checking for bets...")
        loop = asyncio.get_running_loop()
        bets_df = await loop.run_in_executor(None, pd.read_csv, 'horse_racing_game/data/race_bets.csv')
        race_bets = bets_df[bets_df['race_id'] == race.race_id]

        if race_bets.empty:
            print(f"No bets placed for Race ID: {race.race_id}. Cancelling.")
            races_df = await loop.run_in_executor(None, pd.read_csv, 'horse_racing_game/data/races.csv')
            races_df.loc[races_df['race_id'] == race.race_id, 'status'] = 'cancelled'
            await loop.run_in_executor(None, lambda: races_df.to_csv('horse_racing_game/data/races.csv', index=False))

            cancel_embed = discord.Embed(title="🏇 Race Cancelled 🏇", description="This race has been cancelled due to a lack of bets.", color=discord.Color.red())
            await message.edit(embed=cancel_embed)
            if channel.id in active_races: del active_races[channel.id]
            return

        print(f"Bets found! Starting race simulation for Race ID: {race.race_id}")
        races_df = await loop.run_in_executor(None, pd.read_csv, 'horse_racing_game/data/races.csv')
        races_df.loc[races_df['race_id'] == race.race_id, 'status'] = 'running'
        await loop.run_in_executor(None, lambda: races_df.to_csv('horse_racing_game/data/races.csv', index=False))

        start_embed = discord.Embed(title="🏁 THE PADDOCK DASH IS UNDERWAY! 🏁", description="The betting window has closed. And they're off!", color=discord.Color.blue())
        start_embed.add_field(name="THE FIELD", value=message.embeds[0].fields[0].value, inline=False)
        await message.edit(embed=start_embed)

        live_race_embed = (await channel.fetch_message(message.id)).embeds[0]

        while not race.is_finished():
            race.run_round()

            max_name_len = max(len(h.name) for h in race.horses)
            track_display = ""
            track_visual_len = 22

            for horse in sorted(race.horses, key=lambda h: h.number):
                name_str = f"#{horse.number} {horse.name}".ljust(max_name_len + 4)
                progress = int((horse.position / race.track_length) * track_visual_len)
                progress = min(track_visual_len, progress)
                track = '─' * progress + '🏇' + '─' * (track_visual_len - progress)
                track_display += f"`{name_str} |{track}| {horse.position}/{race.track_length}`\n"

            live_race_embed.set_field_at(0, name=f"LIVE RACE - Round {race.round_number}", value=track_display, inline=False)
            live_race_embed.description = "\n".join(race.log)
            await message.edit(embed=live_race_embed)
            await asyncio.sleep(6)

        print(f"Race finished for Race ID: {race.race_id}. Processing results...")
        finishers = [h for h in race.horses if h.position >= race.track_length]

        if not finishers:
            print(f"Race {race.race_id} ended with no finishers.")
            if channel.id in active_races: del active_races[channel.id]
            return

        top_position = max(h.position for h in finishers)
        tied_winners = [h for h in finishers if h.position == top_position]

        if len(tied_winners) > 1:
            tiebreaker_text = "📸 **IT'S A PHOTO FINISH!**\nA tiebreaker roll will determine the winner:\n"
            tiebreaker_results = []
            for horse in tied_winners:
                roll = random.randint(1, 100)
                tiebreaker_results.append((roll, horse))
                tiebreaker_text += f"**{horse.name}** rolls a **{roll}**!\n"

            await channel.send(tiebreaker_text)
            await asyncio.sleep(2)

            final_winner = max(tiebreaker_results, key=lambda item: item[0])[1]

        else:
            final_winner = tied_winners[0]

        loop = asyncio.get_running_loop()
        races_df = await loop.run_in_executor(None, pd.read_csv, 'horse_racing_game/data/races.csv')
        bets_df = await loop.run_in_executor(None, pd.read_csv, 'horse_racing_game/data/race_bets.csv')
        crew_coins_df = await loop.run_in_executor(None, lambda: pd.read_csv('horse_racing_game/data/crew_coins.csv', dtype={'discord_id': str}))
        bot_ledgers = await loop.run_in_executor(None, pd.read_csv, 'horse_racing_game/data/bot_ledgers.csv')
        jackpot_ledger = await loop.run_in_executor(None, pd.read_csv, 'horse_racing_game/data/jackpot_ledger.csv')

        race_bets = bets_df[bets_df['race_id'] == race.race_id]
        total_pot = race_bets['bet_amount'].sum()
        track_fee = total_pot * 0.0789
        winnings_pool = total_pot - track_fee
        jackpot_ledger.loc[0, 'current_jackpot'] += track_fee

        jackpot_hit = False
        if random.randint(1, 200) == 1 and jackpot_ledger['current_jackpot'].iloc[0] > 0:
            jackpot_hit = True
            current_jackpot = jackpot_ledger['current_jackpot'].iloc[0]
            winnings_pool += current_jackpot
            jackpot_ledger.loc[0, 'current_jackpot'] = 0
            await channel.send(f"🎉 **THE WINNER'S PURSE HAS BEEN HIT!** An extra **{format_cc(current_jackpot)}** has been added to the prize pool!")

        payout_summary = "No one bet on the winner."
        winner_horse_obj = next((h for h in race.horses if h.name == final_winner.name), None)
        if winner_horse_obj:
            winning_bets = race_bets[race_bets['horse_number'] == winner_horse_obj.number]
            total_winning_bets_amount = winning_bets['bet_amount'].sum()

            if total_winning_bets_amount > 0:
                payout_summary = ""
                for index, bet in winning_bets.iterrows():
                    bettor_id = str(bet['bettor_id'])
                    winnings = winnings_pool * (bet['bet_amount'] / total_winning_bets_amount)
                    bets_df.loc[index, 'winnings'] = winnings

                    if bettor_id in BOT_PERSONALITIES:
                        bot_ledgers.loc[bot_ledgers['bot_name'] == bettor_id, 'bankroll'] += winnings
                        bot_ledgers.loc[bot_ledgers['bot_name'] == bettor_id, 'total_winnings'] += winnings
                        payout_summary += f"💰 **{bettor_id}** won **{format_cc(winnings)}**!\n"
                    else:
                        if bettor_id in crew_coins_df['discord_id'].values:
                            crew_coins_df.loc[crew_coins_df['discord_id'] == bettor_id, 'balance'] += winnings
                            payout_summary += f"👑 <@{bettor_id}> won **{format_cc(winnings)}**!\n"

        podium_embed = discord.Embed(title=f"🏆 Winner's Circle: Race #{race.race_id} Results 🏆", color=discord.Color.gold())
        sorted_finishers = sorted(race.horses, key=lambda h: h.position, reverse=True)
        podium_text = ""
        if len(sorted_finishers) > 0: podium_text += f"🥇 **1st Place:** {sorted_finishers[0].name}\n"
        if len(sorted_finishers) > 1: podium_text += f"🥈 **2nd Place:** {sorted_finishers[1].name}\n"
        if len(sorted_finishers) > 2: podium_text += f"🥉 **3rd Place:** {sorted_finishers[2].name}\n"
        podium_embed.description = podium_text
        podium_embed.add_field(name="Payouts", value=payout_summary, inline=False)
        podium_embed.set_footer(text=f"Total Pot: {format_cc(total_pot)} | Winnings Pool: {format_cc(winnings_pool)}")
        if jackpot_hit:
            podium_embed.set_author(name="JACKPOT WIN!")

        results_data = []
        for i, horse in enumerate(sorted_finishers):
            results_data.append({
                'race_id': race.race_id, 'horse_name': horse.name, 'horse_number': horse.number,
                'strategy': horse.strategy, 'skills': ",".join(horse.skills),
                'final_position': horse.position, 'final_place': i + 1
            })
        pd.DataFrame(results_data).to_csv('horse_racing_game/data/race_results.csv', mode='a', header=False, index=False)

        log_df = pd.DataFrame(race.structured_log)
        if not log_df.empty:
            log_df['race_id'] = race.race_id
            log_df.to_csv('horse_racing_game/data/race_events.csv', mode='a', header=False, index=False)

        await loop.run_in_executor(None, lambda: bets_df.to_csv('horse_racing_game/data/race_bets.csv', index=False))
        await loop.run_in_executor(None, lambda: crew_coins_df.to_csv('horse_racing_game/data/crew_coins.csv', index=False))
        await loop.run_in_executor(None, lambda: bot_ledgers.to_csv('horse_racing_game/data/bot_ledgers.csv', index=False))
        await loop.run_in_executor(None, lambda: jackpot_ledger.to_csv('horse_racing_game/data/jackpot_ledger.csv', index=False))

        await channel.send(embed=podium_embed)
        if channel.id in active_races: del active_races[channel.id]
        print(f"Cleanup complete for Race ID: {race.race_id}")

    except Exception as e:
        print(f"A critical error occurred in the main race loop for Race ID {race.race_id}: {e}")
        if channel.id in active_races: del active_races[channel.id]


def setup(bot):
    initialize_race_files()

    @bot.group(invoke_without_command=True)
    async def race(ctx):
        """Parent command for Winner's Circle Racing."""
        await ctx.send("Invalid race command. Use `/race create` or `/race bet`.", ephemeral=True)

    @commands.cooldown(1, 300, commands.BucketType.user)
    @race.command(name="create")
    async def race_create(ctx, num_horses: int = 10, track_length: int = 60):
        """Creates a new horse race and starts the game loop. Defaults to 10 horses."""

        VALID_HORSE_COUNTS = [8, 9, 10, 16, 18]
        if num_horses not in VALID_HORSE_COUNTS:
            await ctx.send(f"Invalid number of horses. Please choose from: `{', '.join(map(str, VALID_HORSE_COUNTS))}`", ephemeral=True)
            return

        if ctx.channel.id in active_races:
            return await ctx.send("A race is already in progress in this channel!", ephemeral=True)

        race_id = int(datetime.now().timestamp())
        new_race = Race(race_id=race_id, track_length=track_length)

        horse_names_for_this_race = generate_race_field(num_horses)

        for i, horse_name in enumerate(horse_names_for_this_race, 1):
            if horse_name in UNIQUE_HORSES:
                horse_data = UNIQUE_HORSES[horse_name]
                h = Horse(number=i, name=horse_name, strategy=horse_data['strategy'], skills=horse_data['skills'][:])
            else:
                skills_list = random.sample(GENERIC_SKILLS, k=random.randint(0, 2))
                h = Horse(number=i, name=horse_name, strategy=random.choice(HORSE_STRATEGIES), skills=skills_list)
            new_race.add_horse(h)

        embed = discord.Embed(title=f"🏇 Race #{race_id} - Morning Line", description="Calculating odds... Betting is now open!", color=discord.Color.green())
        field_text = ""
        for horse in new_race.horses:
            field_text += f"`[{horse.number}]` **{horse.name}** ({horse.strategy}) - Odds: --:--\n"
        embed.add_field(name="THE FIELD", value=field_text, inline=False)
        embed.add_field(name="Betting Window", value="Closing in 2m 0s")
        race_message = await ctx.send(embed=embed)

        start_time = datetime.now(pytz.utc)
        races_df = pd.DataFrame([{'race_id': race_id, 'message_id': race_message.id, 'channel_id': race_message.channel.id, 'track_length': track_length, 'status': 'betting', 'winner': None, 'start_time': start_time.isoformat()}])
        races_df.to_csv('horse_racing_game/data/races.csv', mode='a', header=False, index=False)
        horses_data = [{'race_id': race_id, 'horse_number': h.number, 'horse_name': h.name, 'position': 0, 'strategy': h.strategy, 'skills': ",".join(h.skills)} for h in new_race.horses]
        pd.DataFrame(horses_data).to_csv('horse_racing_game/data/race_horses.csv', mode='a', header=False, index=False)

        active_races[ctx.channel.id] = new_race
        bot.loop.create_task(run_complete_race(bot, ctx.channel, race_message, new_race))
        await place_starting_bot_bets(new_race, ctx.channel)
        await ctx.send("Race created!", ephemeral=True)


    @race.command(name="bet")
    async def race_bet(ctx, horse_number: int, amount: int):
        """Places a bet on a horse in the current race."""
        if ctx.channel.id not in active_races:
            return await ctx.send("There is no race currently accepting bets in this channel.", ephemeral=True)

        race = active_races[ctx.channel.id]
        bettor_id = str(ctx.author.id)

        if race.round_number > 0:
            return await ctx.send("The betting window for this race has closed.", ephemeral=True)
        if amount <= 0:
            return await ctx.send("You must bet a positive amount.", ephemeral=True)

        lock_file = 'horse_racing_game/data/market.lock'
        if os.path.exists(lock_file):
            return await ctx.send("The betting windows are busy, please try again in a moment.", ephemeral=True)
        open(lock_file, 'w').close()

        try:
            loop = asyncio.get_running_loop()
            crew_coins_df = await loop.run_in_executor(None, lambda: pd.read_csv('horse_racing_game/data/crew_coins.csv', dtype={'discord_id': str}))
            user_row = crew_coins_df[crew_coins_df['discord_id'] == bettor_id]

            if user_row.empty:
                return await ctx.send("You do not have a Fan Exchange account to bet with.", ephemeral=True)

            user_balance = user_row['balance'].iloc[0]
            if user_balance < amount:
                return await ctx.send(f"You don't have enough CC. Your balance is {format_cc(user_balance)}.", ephemeral=True)

            crew_coins_df.loc[crew_coins_df['discord_id'] == bettor_id, 'balance'] -= amount
            await _record_bet(race, bettor_id, horse_number, amount)
            await ctx.send(f"✅ Your bet of **{format_cc(amount)}** on horse **#{horse_number}** has been placed!", ephemeral=True)

            all_bets_df = await loop.run_in_executor(None, pd.read_csv, 'horse_racing_game/data/race_bets.csv')
            race_bets = all_bets_df[all_bets_df['race_id'] == race.race_id]
            human_bets = race_bets[~race_bets['bettor_id'].isin(BOT_PERSONALITIES.keys())]

            if len(human_bets) == 1:
                print("First human bet detected, triggering BagginsTheBookie...")
                bot_ledgers = await loop.run_in_executor(None, pd.read_csv, 'horse_racing_game/data/bot_ledgers.csv')
                bookie_row = bot_ledgers[bot_ledgers['bot_name'] == 'BagginsTheBookie']

                if not bookie_row.empty:
                    baggins_bankroll = bookie_row['bankroll'].iloc[0]
                    baggins_bet_amount = random.randint(int(amount * 0.6), int(amount * 1.2))
                    baggins_bet_amount = min(baggins_bet_amount, baggins_bankroll)

                    possible_horses = [h.number for h in race.horses if h.number != horse_number]
                    if possible_horses and baggins_bet_amount > 0:
                        baggins_choice = random.choice(possible_horses)
                        baggins_horse_name = next((h.name for h in race.horses if h.number == baggins_choice), "Unknown Horse")

                        await _record_bet(race, 'BagginsTheBookie', baggins_choice, baggins_bet_amount)

                        await ctx.channel.send(f"**BagginsTheBookie** has entered the fray, placing a bet of **{format_cc(baggins_bet_amount)}** on **#{baggins_choice} {baggins_horse_name}**!")
                        await loop.run_in_executor(None, lambda: bot_ledgers.to_csv('horse_racing_game/data/bot_ledgers.csv', index=False))

            await loop.run_in_executor(None, lambda: all_bets_df.to_csv('horse_racing_game/data/race_bets.csv', index=False))
            await loop.run_in_executor(None, lambda: crew_coins_df.to_csv('horse_racing_game/data/crew_coins.csv', index=False))

        except Exception as e:
            print(f"An error occurred in /race bet: {e}")
            await ctx.send("An error occurred while placing your bet. Please contact an admin.", ephemeral=True)
        finally:
            os.remove(lock_file)

    @bot.command(name="refund_race")
    @commands.has_permissions(administrator=True)
    async def refund_race(ctx, race_id: int):
        """(Admin Only) Refunds all bets for a given race ID."""
        lock_file = 'horse_racing_game/data/market.lock'
        if os.path.exists(lock_file):
            await ctx.send("The market is busy. Please try again in a moment.", ephemeral=True)
            return
        open(lock_file, 'w').close()

        try:
            races_df = pd.read_csv('horse_racing_game/data/races.csv')
            bets_df = pd.read_csv('horse_racing_game/data/race_bets.csv')
            crew_coins_df = pd.read_csv('horse_racing_game/data/crew_coins.csv', dtype={'discord_id': str})
            bot_ledgers_df = pd.read_csv('horse_racing_game/data/bot_ledgers.csv')

            target_race = races_df[races_df['race_id'] == race_id]
            if target_race.empty:
                await ctx.send(f"Could not find a race with ID `{race_id}`.", ephemeral=True)
                return

            if target_race['status'].iloc[0] in ['refunded', 'cancelled']:
                await ctx.send(f"Race `{race_id}` has already been cancelled or refunded.", ephemeral=True)
                return

            bets_to_refund = bets_df[bets_df['race_id'] == race_id]
            if bets_to_refund.empty:
                await ctx.send(f"No bets were placed on race `{race_id}` to refund.", ephemeral=True)
                return

            refund_summary = ""
            total_refunded = 0

            for _, bet in bets_to_refund.iterrows():
                bettor_id = str(bet['bettor_id'])
                amount = bet['bet_amount']
                total_refunded += amount

                if bettor_id in bot_ledgers_df['bot_name'].values:
                    bot_ledgers_df.loc[bot_ledgers_df['bot_name'] == bettor_id, 'bankroll'] += amount
                    refund_summary += f"🤖 Refunded {format_cc(amount)} to bot **{bettor_id}**.\n"
                else:
                    user_index = crew_coins_df[crew_coins_df['discord_id'] == bettor_id].index
                    if not user_index.empty:
                        crew_coins_df.loc[user_index, 'balance'] += amount
                        refund_summary += f"👤 Refunded {format_cc(amount)} to <@{bettor_id}>.\n"

            races_df.loc[races_df['race_id'] == race_id, 'status'] = 'refunded'

            races_df.to_csv('horse_racing_game/data/races.csv', index=False)
            crew_coins_df.to_csv('horse_racing_game/data/crew_coins.csv', index=False)
            bot_ledgers_df.to_csv('horse_racing_game/data/bot_ledgers.csv', index=False)

            embed = discord.Embed(
                title="🏁 Race Refund Announcement 🏁",
                description=f"All bets for **Race #{race_id}** have been refunded by an administrator.",
                color=discord.Color.orange()
            )
            embed.add_field(name="Total CC Refunded", value=format_cc(total_refunded))

            winners_circle_channel = discord.utils.get(ctx.guild.channels, name=WINNERS_CIRCLE_CHANNEL_NAME)
            if winners_circle_channel:
                await winners_circle_channel.send(embed=embed)

            await ctx.send(f"✅ Successfully refunded all bets for race `{race_id}`.", embed=embed, ephemeral=True)

        finally:
            os.remove(lock_file)

    @refund_race.error
    async def refund_race_error(ctx, error):
        if isinstance(error, commands.MissingPermissions):
            await ctx.send("You do not have permission to use this command.", ephemeral=True)


    active_races_v2 = {}

    @bot.group(name="race_v2", invoke_without_command=True)
    async def race_v2(ctx):
        """Parent command for the new V2 Winner's Circle Racing."""
        await ctx.send("Invalid command. Use `/race_v2 create` or `/race_v2 bet`.", ephemeral=True)

    @race_v2.command(name="create")
    @commands.has_permissions(administrator=True)
    async def race_v2_create(ctx, distance: int = 2400, num_horses: int = 10):
        """(V2) Creates a new, statistically-driven horse race."""

        if ctx.channel.id in active_races_v2:
            return await ctx.send("A V2 race is already in progress in this channel.", ephemeral=True)

        await ctx.send("`🏁 V2 Race creation initiated...`\n`Generating horses and running Monte Carlo simulation (this may take a moment)...`")

        horse_list = generate_random_horse_field(num_horses)
        if not horse_list:
            return await ctx.send("Failed to generate horses. Check configuration.", ephemeral=True)

        race_obj = RaceV2(horses=horse_list, distance=distance)
        bookie_obj = Bookie(race_obj)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, bookie_obj.run_monte_carlo, 10000)

        race_id = int(datetime.now().timestamp())
        database.create_race(race_id, distance, horse_list)

        bot_manager = BotManager(race_id, bookie_obj)
        bot.loop.create_task(bot_manager.run_betting_cycle(duration_seconds=120))

        embed = discord.Embed(title=f"🏇 Race #{race_id} - Morning Line Odds", color=discord.Color.green())
        odds_text = "```\n"
        for name, data in bookie_obj.morning_line_odds.items():
            odds_text += f"{name:<20} | {data['odds']:.2f} to 1\n"
        odds_text += "```"
        embed.add_field(name="Betting is now OPEN for 2 minutes!", value=odds_text)

        race_message = await ctx.send(embed=embed)

        active_races_v2[ctx.channel.id] = {
            "race": race_obj, "bookie": bookie_obj, "bot_manager": bot_manager, "message": race_message
        }

        await ctx.send("V2 Race created successfully!", ephemeral=True)

def generate_random_horse_field(num_horses: int) -> list:
    """
    Reads the attributes config and generates a list of unique, random horses.
    """
    try:
        with open('horse_racing_game/configs/horse_attributes.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("ERROR: horse_attributes.json not found. Cannot generate horses.")
        return []

    adjectives = list(config['adjectives'].keys())
    nouns = list(config['nouns'].keys())
    strategies = list(config['strategies'].keys())

    generated_horses = []
    used_names = set()

    while len(generated_horses) < num_horses:
        adj = random.choice(adjectives)
        noun = random.choice(nouns)
        name = f"{adj} {noun}"

        if name not in used_names:
            used_names.add(name)
            strategy = random.choice(strategies)
            generated_horses.append(HorseV2(name, strategy))

    return generated_horses
