import os
import discord
from dotenv import load_dotenv
from derby_game.bot.manager import DerbyBotManager
import sys

# Get the path to the current script's directory (side-stakes-derby/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path to the parent directory (the one containing both projects)
parent_dir = os.path.dirname(current_dir)
# Add the path to the 'prettyDerbyClubAnalysis' project to sys.path
other_project_path = os.path.join(parent_dir, 'prettyDerbyClubAnalysis')
sys.path.append(other_project_path)

# Load environment variables from .env file
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
GUILD_ID = int(os.getenv('DISCORD_GUILD_ID')) # Your Server ID for testing commands

if not DISCORD_BOT_TOKEN or not GUILD_ID:
    print("FATAL ERROR: DISCORD_BOT_TOKEN or DISCORD_GUILD_ID not found in .env file.")
    exit()

def run_bot():
    """Initializes and runs the Discord bot."""
    intents = discord.Intents.default()
    # Add any specific intents your bot might need later (e.g., members, messages)
    # intents.members = True
    # intents.message_content = True

    bot = DerbyBotManager(command_prefix="!", intents=intents, guild_id=GUILD_ID)

    try:
        print("Starting Discord bot...")
        bot.run(DISCORD_BOT_TOKEN)
    except Exception as e:
        print(f"Error running bot: {e}")

if __name__ == "__main__":
    run_bot()