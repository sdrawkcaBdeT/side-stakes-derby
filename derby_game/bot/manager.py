import discord
from discord.ext import commands
import os

# Assuming your commands will be in a 'commands.py' file within this 'bot' directory
COMMANDS_COG_PATH = 'derby_game.bot.commands'

class DerbyBotManager(commands.Bot):
    """Custom Bot class for Derby Game management."""

    def __init__(self, command_prefix, intents, guild_id):
        super().__init__(command_prefix=command_prefix, intents=intents)
        self.guild_id = guild_id # Store guild ID for syncing commands

    async def setup_hook(self):
        """Loads extensions (cogs) and syncs commands."""
        print("Running setup_hook...")
        try:
            await self.load_extension(COMMANDS_COG_PATH)
            print(f"Successfully loaded cog: {COMMANDS_COG_PATH}")
        except Exception as e:
            print(f"Failed to load cog {COMMANDS_COG_PATH}: {e}")
            raise # Re-raise error to prevent bot from starting incorrectly

        # Sync commands for testing guild immediately
        # For production, you might sync globally or handle differently
        guild = discord.Object(id=self.guild_id)
        self.tree.copy_global_to(guild=guild)
        try:
            await self.tree.sync(guild=guild)
            print(f"Synced commands to guild {self.guild_id}")
        except Exception as e:
            print(f"Failed to sync commands to guild {self.guild_id}: {e}")


    async def on_ready(self):
        """Called when the bot is ready."""
        print(f'Logged in as {self.user.name} ({self.user.id})')
        print('------')