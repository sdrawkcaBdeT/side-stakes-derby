import discord
from discord import app_commands
from discord.ext import commands
from derby_game.database import queries # <-- Import queries
from datetime import timezone # For formatting time
from derby_game.simulation import Race, Bookie
import traceback
import market.database as market_db
import os
import sys

# Get the path to the current script's directory (side-stakes-derby/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path to the parent directory (the one containing both projects)
parent_dir = os.path.dirname(current_dir)
# Add the path to the 'prettyDerbyClubAnalysis' project to sys.path
other_project_path = os.path.join(parent_dir, 'prettyDerbyClubAnalysis')
sys.path.append(other_project_path)

# Helper function to format race times nicely
def format_time_optional(dt):
    if dt:
        # Convert timezone-aware datetime from DB (UTC) to a discord timestamp string
        unix_timestamp = int(dt.timestamp())
        # 'R' provides a relative timestamp like "in 5 minutes" or "2 hours ago"
        # 'f' provides a short date/time like "October 27, 2025 3:30 AM"
        return f"<t:{unix_timestamp}:f> (<t:{unix_timestamp}:R>)"
    return "Not scheduled"

class DerbyCommands(commands.Cog):
    """Cog containing all commands for the Derby game."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="view_races", description="View upcoming or open races.")
    @app_commands.describe(tier="Filter races by tier (e.g., G, D, C). Leave blank for all.")
    async def view_races(self, interaction: discord.Interaction, tier: str = None):
        """Displays a list of available races."""
        await interaction.response.defer(ephemeral=True)

        try:
            available_races = queries.get_available_races(tier_filter=tier)

            if not available_races:
                await interaction.followup.send("No available races found matching your criteria.", ephemeral=True)
                return

            embed = discord.Embed(
                title="Available Races" + (f" (Tier: {tier.upper()})" if tier else ""),
                color=discord.Color.blue()
            )

            description = ""
            for race in available_races:
                # Format start time using the helper
                start_time_str = format_time_optional(race['start_time'])

                description += (
                    f"**ID:** `{race['race_id']}` | **Tier:** {race['tier']} | "
                    f"**Dist:** {race['distance']}m | **Status:** {race['status'].capitalize()}\n"
                    f"   *Purse:* {race['purse']:,} CC | *Starts:* {start_time_str}\n\n" # Added purse and start time
                )
                # Limit embed description length
                if len(description) > 3500: # Discord limit is 4096
                    description += "**... (Too many races to display all) ...**"
                    break

            embed.description = description
            embed.set_footer(text=f"Found {len(available_races)} races.")

            await interaction.followup.send(embed=embed, ephemeral=True)

        except Exception as e:
            print(f"Error in /view_races command: {e}")
            await interaction.followup.send("An error occurred while fetching races.", ephemeral=True)
    
    @app_commands.command(name="race_info", description="View details and odds for a specific race.")
    @app_commands.describe(race_id="The ID of the race to view.")
    async def race_info(self, interaction: discord.Interaction, race_id: int):
        """Displays details, entries, and odds for a given race."""
        await interaction.response.defer(ephemeral=True)

        try:
            # 1. Get Race Details
            race_details = queries.get_race_details(race_id)
            if not race_details:
                await interaction.followup.send(f"Race with ID `{race_id}` not found or is no longer available.", ephemeral=True)
                return

            # 2. Get Horse Entries (We need these for the Bookie)
            # Note: The Race object itself loads horses, but we might want basic info first
            horse_entries = queries.get_horses_in_race(race_id)
            if not horse_entries:
                await interaction.followup.send(f"No horses seem to be entered in race `{race_id}` yet.", ephemeral=True)
                return

            # 3. Prepare Embed
            embed = discord.Embed(
                title=f"Race #{race_details['race_id']} Info - {race_details['tier']} Tier ({race_details['distance']}m)",
                color=discord.Color.green()
            )
            start_time_str = format_time_optional(race_details['start_time'])
            embed.add_field(name="Status", value=race_details['status'].capitalize(), inline=True)
            embed.add_field(name="Purse", value=f"{race_details['purse']:,} CC", inline=True)
            embed.add_field(name="Starts", value=start_time_str, inline=True)

            # 4. Generate Odds (if applicable)
            odds_display = "Odds generation pending or not applicable."
            # Only run bookie if race is open or pending (or maybe running?)
            if race_details['status'] in ['pending', 'open']:
                try:
                    # Instantiate Race object - this loads horses again internally
                    sim_race_obj = Race(race_id)
                    if sim_race_obj.horses: # Check if horses were loaded successfully
                        bookie = Bookie(sim_race_obj)
                        # Run Monte Carlo - potentially cache this later to avoid re-running frequently
                        odds_data = bookie.run_monte_carlo(simulations=1000) # Lower sims for faster command response

                        if odds_data:
                            odds_display = "```\n"
                            # Sort odds display maybe? By probability? Alphabetically?
                            sorted_odds = sorted(odds_data.items(), key=lambda item: item[1]['probability'], reverse=True)
                            for name, data in sorted_odds:
                                odds_val = data['odds']
                                prob_val = data['probability'] * 100
                                odds_display += f"{name:<20} | {odds_val:>5.2f} to 1 ({prob_val:4.1f}%)\n"
                            odds_display += "```"
                        else:
                             odds_display = "Error generating odds."
                    else:
                        odds_display = "Could not load horses for odds generation."

                except Exception as bookie_error:
                    print(f"Error running Bookie for race {race_id}: {bookie_error}")
                    traceback.print_exc() # Print full traceback for debugging
                    odds_display = "An error occurred during odds calculation."

            embed.add_field(name="Entries & Odds", value=odds_display, inline=False)

            # Add a simpler list of entries for reference (optional)
            entry_list = ""
            for horse in horse_entries:
                entry_list += f"- ID: `{horse['horse_id']}` {horse['name']} (HG: {horse['hg_score']}, Strat: {horse['strategy']}, Pref: {horse['min_pref_dist']}-{horse['max_pref_dist']}m)\n"
            if entry_list:
                 embed.add_field(name="Horse Details", value=entry_list[:1020] + ("..." if len(entry_list)>1020 else ""), inline=False) # Limit field length


            await interaction.followup.send(embed=embed, ephemeral=True)

        except Exception as e:
            print(f"Error in /race_info command for race {race_id}: {e}")
            traceback.print_exc() # Print full traceback for debugging
            await interaction.followup.send("An error occurred while fetching race info.", ephemeral=True)
    
    @app_commands.command(name="claim", description="Claim a horse from a recently finished Claimer (G-Tier) race.")
    @app_commands.describe(
        horse_id="The ID of the horse you want to claim.",
        race_id="The ID of the specific race the horse just ran in."
    )
    async def claim(self, interaction: discord.Interaction, horse_id: int, race_id: int):
        """Allows a player to purchase a bot-owned horse after a G race."""
        await interaction.response.defer(ephemeral=True)
        player_id_str = str(interaction.user.id) # Use string ID

        try:
            # Check if player is registered in the main system
            if not market_db.get_user_details(player_id_str):
                 await interaction.followup.send(f"❌ You must be registered with the main bot (`/register`) to have a CC balance.", ephemeral=True)
                 return

            success, message = queries.execute_claim_horse(
                player_user_id=player_id_str, # Pass the string ID
                horse_id=horse_id,
                race_id=race_id
            )

            if success:
                await interaction.followup.send(f"✅ {message}", ephemeral=True)
            else:
                await interaction.followup.send(f"❌ {message}", ephemeral=True)

        except Exception as e:
            print(f"Error in /claim command: {e}")
            traceback.print_exc()
            await interaction.followup.send("An unexpected server error occurred while trying to claim the horse.", ephemeral=True)

async def setup(bot: commands.Bot):
    await bot.add_cog(DerbyCommands(bot))
    print("DerbyCommands Cog loaded.")
    


# (Keep other commands and setup function)