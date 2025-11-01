import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from typing import Dict, List, Optional

import discord
from discord import app_commands
from discord.ext import commands

# Ensure the sibling project is importable before touching market modules.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
other_project_path = os.path.join(parent_dir, "prettyDerbyClubAnalysis")
if other_project_path not in sys.path:
    sys.path.append(other_project_path)

import traceback

from derby_game.database import queries
from derby_game import betting_service

try:
    import market.database as market_db
except ImportError:  # pragma: no cover
    market_db = None


def format_time_optional(dt):
    if not dt:
        return "Not scheduled"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    unix_timestamp = int(dt.timestamp())
    return f"<t:{unix_timestamp}:f> (<t:{unix_timestamp}:R>)"


def format_cc(value) -> str:
    if value is None:
        return "-"
    try:
        amount = Decimal(str(value))
    except Exception:
        return str(value)
    quantized = amount.quantize(Decimal("1")) if amount == amount.to_integral_value() else amount.quantize(Decimal("0.01"))
    if quantized == quantized.to_integral_value():
        return f"{int(quantized):,} CC"
    return f"{quantized:,.2f} CC"

def format_fractional_odds(decimal_odds: Optional[Decimal]) -> str:
    if decimal_odds is None:
        return "-"
    try:
        odds_value = Decimal(decimal_odds)
    except Exception:
        odds_value = Decimal(str(decimal_odds))

    if odds_value <= 1:
        return "EVS"

    implied = odds_value - Decimal("1")
    if implied >= Decimal("20"):
        whole = int(implied.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        whole = max(1, whole)
        return f"{whole}/1"

    fraction = Fraction(implied).limit_denominator(32)
    numerator, denominator = fraction.numerator, fraction.denominator

    if numerator <= 0:
        return "EVS"

    if numerator > 200 or denominator > 32:
        whole = int(implied.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        whole = max(1, whole)
        return f"{whole}/1"

    return f"{numerator}/{denominator}"



def summarize_entries(entries: List[Dict[str, object]], odds_map: Dict[int, Dict[str, Decimal]]) -> str:
    if not entries:
        return "No horses entered."
    lines: List[str] = []
    for entry in entries:
        odds_entry = odds_map.get(entry["horse_id"]) if odds_map else None
        odds_text = format_fractional_odds(Decimal(str(odds_entry['odds']))) if odds_entry else "-"
        line = (
            f"`#{entry['horse_id']}` {entry['name']} | HG {entry['hg_score']} | "
            f"Strategy {entry.get('strategy', '-')} | Odds {odds_text}"
        )
        lines.append(line)
    summary = "\n".join(lines)
    if len(summary) > 1024:
        summary = summary[:1000].rstrip() + "\n..."
    return summary


def format_odds_table(odds_map: Dict[int, Dict[str, Decimal]]) -> str:
    if not odds_map:
        return "Odds unavailable."
    sorted_entries = sorted(
        odds_map.values(),
        key=lambda item: (Decimal(str(item["odds"])), item["name"])
    )
    lines = []
    for data in sorted_entries:
        decimal_odds = Decimal(str(data["odds"]))
        fractional = format_fractional_odds(decimal_odds)
        lines.append(f"{data['name']:<20} {fractional:>8}")
    block = "\n".join(lines)
    return f"```\n{block}\n```"


class BetConfirmationView(discord.ui.View):
    def __init__(
        self,
        bettor_id: int,
        race_id: int,
        horse_id: int,
        horse_name: str,
        amount: int,
        displayed_odds: float,
    ):
        super().__init__(timeout=90)
        self.bettor_id = bettor_id
        self.race_id = race_id
        self.horse_id = horse_id
        self.horse_name = horse_name
        self.amount = amount
        self.displayed_odds = displayed_odds
        self.message: Optional[discord.Message] = None

    def disable_all_items(self):
        for child in self.children:
            child.disabled = True

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.bettor_id:
            await interaction.response.send_message(
                "Only the original bettor can respond to this confirmation.",
                ephemeral=True,
            )
            return False
        return True

    async def on_timeout(self) -> None:
        self.disable_all_items()
        if self.message:
            try:
                await self.message.edit(view=self)
            except discord.HTTPException:
                pass

    @discord.ui.button(label="Confirm Bet", style=discord.ButtonStyle.success)
    async def confirm_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True, thinking=True)

        if not market_db:
            embed = discord.Embed(
                title="Bet Failed",
                description="Betting service is currently unavailable.",
                color=discord.Color.red(),
            )
        else:
            result = betting_service.place_player_bet(
                str(self.bettor_id), self.race_id, self.horse_id, self.amount
            )
            if result.success:
                locked_decimal = Decimal(str(result.locked_odds if result.locked_odds is not None else self.displayed_odds))
                stake = Decimal(self.amount)
                potential = stake + (stake * locked_decimal)
                new_balance = market_db.get_user_balance_by_discord_id(str(self.bettor_id))

                embed = discord.Embed(
                    title="Bet Placed",
                    description=(
                        f"You wagered {format_cc(self.amount)} on **{self.horse_name}** "
                        f"in race #{self.race_id}."
                    ),
                    color=discord.Color.green(),
                )
                fraction_odds = format_fractional_odds(locked_decimal)
                embed.add_field(name="Locked Odds", value=f"{fraction_odds} ({locked_decimal:.2f}x)", inline=True)
                embed.add_field(name="Potential Return", value=format_cc(potential), inline=True)
                if new_balance is not None:
                    embed.add_field(name="New Balance", value=format_cc(new_balance), inline=True)
            else:
                embed = discord.Embed(
                    title="Bet Failed",
                    description=result.message,
                    color=discord.Color.red(),
                )

        self.disable_all_items()
        self.stop()
        target_message_id = self.message.id if self.message else None
        try:
            if target_message_id:
                await interaction.followup.edit_message(message_id=target_message_id, embed=embed, view=self)
            else:
                await interaction.followup.send(embed=embed, ephemeral=True)
        except discord.HTTPException:
            await interaction.followup.send(embed=embed, ephemeral=True)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True, thinking=False)

        embed = discord.Embed(
            title="Bet Cancelled",
            description="No wager was placed.",
            color=discord.Color.dark_grey(),
        )

        self.disable_all_items()
        self.stop()
        target_message_id = self.message.id if self.message else None
        try:
            if target_message_id:
                await interaction.followup.edit_message(message_id=target_message_id, embed=embed, view=self)
            else:
                await interaction.followup.send(embed=embed, ephemeral=True)
        except discord.HTTPException:
            await interaction.followup.send(embed=embed, ephemeral=True)


class RaceOddsView(discord.ui.View):
    def __init__(self, cog: "DerbyCommands", race_id: int, requester_id: int):
        super().__init__(timeout=180)
        self.cog = cog
        self.race_id = race_id
        self.requester_id = requester_id
        self.message: Optional[discord.Message] = None

    def disable_all_items(self):
        for child in self.children:
            child.disabled = True

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the original requester can refresh this view.",
                ephemeral=True,
            )
            return False
        return True

    async def on_timeout(self) -> None:
        self.disable_all_items()
        if self.message:
            try:
                await self.message.edit(view=self)
            except discord.HTTPException:
                pass

    @discord.ui.button(label="Refresh Odds", style=discord.ButtonStyle.primary)
    async def refresh_odds(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True, thinking=True)

        race_details, horse_entries, odds_map = self.cog._fetch_race_context(self.race_id)
        if not race_details:
            self.disable_all_items()
            self.stop()
            await interaction.followup.send("Race could not be found.", ephemeral=True)
            return

        embed = self.cog._build_race_embed(race_details, horse_entries, odds_map)
        try:
            await interaction.followup.edit_message(message_id=self.message.id, embed=embed, view=self)
        except discord.HTTPException:
            await interaction.followup.send(embed=embed, ephemeral=True)


class DerbyCommands(commands.Cog):
    """Cog containing all commands for the Derby game."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    def _safe_live_odds(self, race_id: int) -> Dict[int, Dict[str, Decimal]]:
        try:
            return betting_service.calculate_live_odds(race_id)
        except Exception as err:
            print(f"Error calculating odds for race {race_id}: {err}")
            traceback.print_exc()
            return {}

    def _fetch_race_context(self, race_id: int):
        race_details = queries.get_race_details(race_id)
        if not race_details:
            return None, None, None

        horse_entries = queries.get_horses_in_race(race_id)
        odds_map: Dict[int, Dict[str, Decimal]] = {}

        status = (race_details.get("status") or "").lower()
        if status in {"pending", "open", "locked"}:
            odds_map = self._safe_live_odds(race_id)

        return race_details, horse_entries, odds_map

    @staticmethod
    def _match_horse_identifier(identifier: str, candidates: List[Dict[str, object]]):
        if not identifier:
            return None
        identifier = identifier.strip()
        if not identifier:
            return None

        numeric_id = None
        if identifier.isdigit():
            numeric_id = identifier

        lowered = identifier.lower()

        for candidate in candidates:
            if numeric_id and str(candidate.get("horse_id")) == numeric_id:
                return candidate

        for candidate in candidates:
            name = str(candidate.get("name", "")).strip().lower()
            if name == lowered:
                return candidate

        return None

    def _resolve_race_horse(self, identifier: str, race_entries: List[Dict[str, object]]):
        return self._match_horse_identifier(identifier, race_entries)

    def _resolve_owned_horse(self, trainer_id: int, identifier: str):
        horses = queries.get_trainer_horses(trainer_id, include_retired=True)
        return self._match_horse_identifier(identifier, horses)

    @app_commands.command(name="stable", description="View the horses in your stable.")
    async def stable(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)

        trainer_id = int(interaction.user.id)
        trainer_record = queries.ensure_trainer_record(trainer_id, is_bot=False)
        horses = queries.get_trainer_horses(trainer_id, include_retired=True)

        if not horses:
            await interaction.followup.send("You do not own any horses yet.", ephemeral=True)
            return

        active_count = sum(1 for horse in horses if not horse.get("is_retired"))
        total_horses = len(horses)
        stable_slots = trainer_record.get("stable_slots") if trainer_record else None
        now_utc = datetime.now(timezone.utc)

        description_lines = [f"Horses owned: {total_horses} (Active: {active_count}, Retired: {total_horses - active_count})"]
        if stable_slots is not None:
            description_lines.append(f"Stable slots: {active_count}/{stable_slots}")

        embed = discord.Embed(
            title="Stable Overview",
            description="\n".join(description_lines),
            color=discord.Color.blurple(),
        )

        blocks: List[str] = []
        current_block = ""

        for horse in horses:
            status_parts = ["Retired" if horse.get("is_retired") else "Active"]

            training_until = horse.get("in_training_until")
            if training_until:
                if training_until.tzinfo is None or training_until.tzinfo.utcoffset(training_until) is None:
                    training_until = training_until.replace(tzinfo=timezone.utc)
                training_unix = int(training_until.timestamp())
                status_parts.append(f"Training until <t:{training_unix}:R>")

            status_text = " | ".join(status_parts)

            plan_stat = horse.get("training_plan_stat")
            plan_active = horse.get("training_plan_active")
            if plan_stat:
                plan_label = plan_stat.upper()
                plan_state = "On" if plan_active else "Paused"
                status_text += f" | Auto: {plan_label} ({plan_state})"

            birth_ts = horse.get("birth_timestamp")
            age_text = "-"
            if birth_ts:
                if birth_ts.tzinfo is None or birth_ts.tzinfo.utcoffset(birth_ts) is None:
                    birth_ts = birth_ts.replace(tzinfo=timezone.utc)
                age_days = max((now_utc - birth_ts).days, 0)
                age_months = age_days // 30
                age_years = 2 + (age_months // 12)
                remaining_months = age_months % 12
                age_text = f"{age_years}y {remaining_months}m"

            line = (
                f"#{horse['horse_id']} {horse['name']} — {horse['strategy']} | HG {horse['hg_score']}\n"
                f"Stats: SPD {horse['spd']} STA {horse['sta']} FCS {horse['fcs']} GRT {horse['grt']} COG {horse['cog']} LCK {horse['lck']}\n"
                f"Pref: {horse['min_pref_distance']}–{horse['max_pref_distance']}m | Age: {age_text} | {status_text}"
            )

            if not current_block:
                current_block = line
            elif len(current_block) + 2 + len(line) <= 1000:
                current_block += "\n\n" + line
            else:
                blocks.append(current_block)
                current_block = line

        if current_block:
            blocks.append(current_block)

        for idx, block in enumerate(blocks, start=1):
            embed.add_field(name=f"Horses ({idx}/{len(blocks)})", value=block, inline=False)

        embed.set_footer(text="All times shown local to you via Discord timestamps.")
        await interaction.followup.send(embed=embed, ephemeral=True)

    def _format_stat_grid(self, entries: List[Dict[str, object]], odds_map: Dict[int, Dict[str, Decimal]]) -> str:
        header = "Ln Horse               SPD STA FCS GRT COG LCK Odds"
        lines = [header, "-" * len(header)]
        for lane, entry in enumerate(entries, start=1):
            odds_entry = odds_map.get(entry['horse_id']) if odds_map else None
            if odds_entry:
                odds_text = format_fractional_odds(Decimal(str(odds_entry['odds'])))
            else:
                odds_text = "-"
            line = (
                f"{lane:>2} "
                f"{entry['name'][:18]:<18} "
                f"{entry['spd']:>3} "
                f"{entry['sta']:>3} "
                f"{entry['fcs']:>3} "
                f"{entry['grt']:>3} "
                f"{entry['cog']:>3} "
                f"{entry['lck']:>3} "
                f"{odds_text:>6}"
            )
            lines.append(line)
        return "\n".join(lines)

    def _build_race_embed(
        self,
        race_details: Dict[str, object],
        horse_entries: List[Dict[str, object]],
        odds_map: Dict[int, Dict[str, Decimal]],
    ) -> discord.Embed:
        title = f"Race #{race_details['race_id']} - Tier {race_details['tier']} ({race_details['distance']}m)"
        embed = discord.Embed(title=title, color=discord.Color.blurple())
        embed.add_field(name="Status", value=(race_details.get("status") or "Unknown").capitalize(), inline=True)
        embed.add_field(name="Purse", value=format_cc(race_details.get("purse")), inline=True)
        embed.add_field(name="Entry Fee", value=format_cc(race_details.get("entry_fee")), inline=True)
        embed.add_field(name="Starts", value=format_time_optional(race_details.get("start_time")), inline=True)
        embed.add_field(name="Field", value=summarize_entries(horse_entries, odds_map), inline=False)
        embed.add_field(name="Odds Snapshot", value=format_odds_table(odds_map), inline=False)
        stat_grid = self._format_stat_grid(horse_entries, odds_map)
        if stat_grid:
            block = f"```text\n{stat_grid}\n```"
            if len(block) > 1024:
                trimmed = stat_grid[:980].rsplit("\n", 1)[0]
                if not trimmed:
                    trimmed = stat_grid[:980]
                block = f"```text\n{trimmed}\n...```"
            embed.add_field(name="Horse Stat Grid", value=block, inline=False)
        embed.set_footer(text="Use /bet to place a wager. Odds refresh as the pool changes.")
        return embed

    @app_commands.command(name="races", description="List pending or open races.")
    @app_commands.describe(tier="Filter races by tier (e.g., G, D, C). Leave blank for all.")
    async def races(self, interaction: discord.Interaction, tier: Optional[str] = None):
        await interaction.response.defer(ephemeral=True, thinking=True)

        try:
            available_races = queries.get_available_races(tier_filter=tier)
        except Exception as err:
            print(f"Error fetching races: {err}")
            traceback.print_exc()
            await interaction.followup.send("An error occurred while retrieving races.", ephemeral=True)
            return

        if not available_races:
            await interaction.followup.send("No races are currently pending or open.", ephemeral=True)
            return

        embed = discord.Embed(
            title="Available Races" + (f" (Tier {tier.upper()})" if tier else ""),
            color=discord.Color.blue(),
        )

        open_odds_checks = 0
        for race in available_races[:10]:
            status = (race.get("status") or "").lower()
            odds_summary = "Odds pending."
            if status == "open" and open_odds_checks < 3:
                odds_map = self._safe_live_odds(race["race_id"])
                if odds_map:
                    top_entries = sorted(
                        odds_map.values(),
                        key=lambda item: Decimal(str(item["odds"]))
                    )[:3]
                    odds_summary = " | ".join(
                        f"{data['name']} {format_fractional_odds(Decimal(str(data['odds'])))}" for data in top_entries
                    )
                open_odds_checks += 1

            field_value = (
                f"Tier {race['tier']} - {race['distance']} m - Status: {(race.get('status') or '').capitalize()}\n"
                f"Purse: {format_cc(race.get('purse'))} - Entry Fee: {format_cc(race.get('entry_fee'))}\n"
                f"Starts: {format_time_optional(race.get('start_time'))}\n"
                f"Odds: {odds_summary}"
            )
            embed.add_field(name=f"Race #{race['race_id']}", value=field_value, inline=False)

        if len(available_races) > 10:
            embed.set_footer(text=f"Showing the first 10 of {len(available_races)} races. Use filters to narrow results.")
        else:
            embed.set_footer(text="Use /race_info <race_id> for detailed odds and entries.")

        await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="race_info", description="View details and live odds for a race.")
    @app_commands.describe(race_id="The ID of the race to view.")
    async def race_info(self, interaction: discord.Interaction, race_id: int):
        await interaction.response.defer(ephemeral=True, thinking=True)

        race_details, horse_entries, odds_map = self._fetch_race_context(race_id)
        if not race_details:
            await interaction.followup.send(f"Race with ID `{race_id}` was not found.", ephemeral=True)
            return

        embed = self._build_race_embed(race_details, horse_entries, odds_map)

        status = (race_details.get("status") or "").lower()
        view: Optional[RaceOddsView] = None
        if status in {"pending", "open"}:
            view = RaceOddsView(self, race_id, interaction.user.id)

        if view:
            message = await interaction.followup.send(embed=embed, view=view, ephemeral=True)
            view.message = message
        else:
            await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="bet", description="Place a wager on a horse in an open race.")
    @app_commands.describe(
        race_id="The race you want to bet on.",
        horse="The horse you want to back (ID or name).",
        amount="The amount of CC to wager.",
    )
    async def bet(self, interaction: discord.Interaction, race_id: int, horse: str, amount: int):
        await interaction.response.defer(ephemeral=True, thinking=True)

        if amount <= 0:
            await interaction.followup.send("Bet amount must be greater than zero.", ephemeral=True)
            return

        if not market_db:
            await interaction.followup.send("Betting service is currently offline.", ephemeral=True)
            return

        bettor_id_str = str(interaction.user.id)
        if not market_db.get_user_details(bettor_id_str):
            await interaction.followup.send("You need to register with the main bot before placing bets.", ephemeral=True)
            return

        race_details, horse_entries, odds_map = self._fetch_race_context(race_id)
        if not race_details:
            await interaction.followup.send("Race not found.", ephemeral=True)
            return

        status = (race_details.get("status") or "").lower()
        if status != "open":
            await interaction.followup.send("Betting is closed for this race.", ephemeral=True)
            return

        horse_entry = self._resolve_race_horse(horse, horse_entries)
        if not horse_entry:
            await interaction.followup.send("That horse is not entered in this race.", ephemeral=True)
            return

        horse_id = horse_entry["horse_id"]
        odds_entry = odds_map.get(horse_id)
        if not odds_entry:
            odds_map = self._safe_live_odds(race_id)
            odds_entry = odds_map.get(horse_id)
            if not odds_entry:
                await interaction.followup.send("Unable to calculate odds for the selected horse.", ephemeral=True)
                return

        balance = market_db.get_user_balance_by_discord_id(bettor_id_str)
        if balance is None:
            await interaction.followup.send("Could not determine your balance. Please try again later.", ephemeral=True)
            return

        try:
            balance_decimal = Decimal(str(balance))
        except Exception:
            balance_decimal = Decimal("0")

        if balance_decimal < Decimal(amount):
            await interaction.followup.send(
                f"Insufficient funds. Your balance is {format_cc(balance_decimal)}.",
                ephemeral=True,
            )
            return

        max_bet = None
        try:
            max_bet = market_db.get_player_betting_limit(bettor_id_str)
        except Exception as err:
            print(f"Warning: could not fetch betting limit for {bettor_id_str}: {err}")

        if max_bet is not None and amount > max_bet:
            await interaction.followup.send(
                f"Your bet exceeds your current limit of {format_cc(max_bet)}.",
                ephemeral=True,
            )
            return

        displayed_odds = float(odds_entry["odds"])
        potential = Decimal(amount) + (Decimal(amount) * Decimal(str(displayed_odds)))

        fractional = format_fractional_odds(Decimal(str(displayed_odds)))
        embed = discord.Embed(
            title="Confirm Bet",
            description=f"You are betting {format_cc(amount)} on **{horse_entry['name']}** in race #{race_id}.",
            color=discord.Color.gold(),
        )
        embed.add_field(name="Current Odds", value=f"{fractional} ({displayed_odds:.2f}x)", inline=True)
        embed.add_field(name="Potential Return", value=format_cc(potential), inline=True)
        embed.add_field(name="Balance", value=format_cc(balance_decimal), inline=True)
        embed.set_footer(text="Odds are locked when you confirm.")

        view = BetConfirmationView(
            bettor_id=interaction.user.id,
            race_id=race_id,
            horse_id=horse_id,
            horse_name=horse_entry["name"],
            amount=amount,
            displayed_odds=displayed_odds,
        )
        message = await interaction.followup.send(embed=embed, view=view, ephemeral=True)
        view.message = message

    @app_commands.command(name="my_bets", description="Show your recent race wagers.")
    async def my_bets(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)

        if not market_db:
            await interaction.followup.send("Betting service is currently offline.", ephemeral=True)
            return

        bettor_id_str = str(interaction.user.id)
        bets = betting_service.get_recent_player_bets(bettor_id_str, limit=10)
        if not bets:
            await interaction.followup.send("You have not placed any bets yet.", ephemeral=True)
            return

        embed = discord.Embed(
            title="Recent Bets",
            color=discord.Color.purple(),
        )

        for bet in bets:
            race_details = queries.get_race_details(bet["race_id"])
            status_text = "Unknown"
            result_note = None

            if race_details:
                status_text = (race_details.get("status") or "Unknown").capitalize()
                if race_details.get("status") == "finished" and race_details.get("winner_horse_id"):
                    winner_entries = queries.get_horses_in_race(bet["race_id"])
                    winner_name = next(
                        (horse["name"] for horse in winner_entries if horse["horse_id"] == race_details["winner_horse_id"]),
                        None,
                    )
                    if winner_name:
                        if winner_name == bet["horse_name"]:
                            result_note = "Result: WIN"
                        else:
                            result_note = f"Winner: {winner_name}"

            placed_at = format_time_optional(bet.get("placed_at"))

            value_lines = [
                f"Horse: {bet['horse_name']}",
                f"Amount: {format_cc(bet['amount'])} at {format_fractional_odds(Decimal(str(bet['locked_in_odds'])))}",
                f"Placed: {placed_at}",
                f"Status: {status_text}",
            ]
            if result_note:
                value_lines.append(result_note)

            embed.add_field(
                name=f"Race #{bet['race_id']}",
                value="\n".join(value_lines),
                inline=False,
            )

        await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="train", description="Queue a training session for one of your horses.")
    @app_commands.describe(
        horse="Horse ID or name.",
        stat="Which stat to train.",
    )
    @app_commands.choices(
        stat=[
            app_commands.Choice(name="Speed (SPD)", value="spd"),
            app_commands.Choice(name="Stamina (STA)", value="sta"),
            app_commands.Choice(name="Focus (FCS)", value="fcs"),
            app_commands.Choice(name="Grit (GRT)", value="grt"),
            app_commands.Choice(name="Cognition (COG)", value="cog"),
        ]
    )
    async def train(
        self,
        interaction: discord.Interaction,
        horse: str,
        stat: app_commands.Choice[str],
    ):
        await interaction.response.defer(ephemeral=True, thinking=True)

        trainer_id = int(interaction.user.id)
        horse_entry = self._resolve_owned_horse(trainer_id, horse)
        if not horse_entry:
            await interaction.followup.send(
                "Could not find a horse you own with that ID or name.",
                ephemeral=True,
            )
            return

        try:
            queries.ensure_trainer_record(trainer_id)
            success, message, finish_time = queries.start_training_session(
                trainer_id,
                horse_entry["horse_id"],
                stat.value,
            )
        except Exception as err:
            print(f"Error queuing training for horse {horse_entry['horse_id']}: {err}")
            traceback.print_exc()
            await interaction.followup.send(
                "An error occurred while starting training.",
                ephemeral=True,
            )
            return

        color = discord.Color.green() if success else discord.Color.red()
        title = "Training Queued" if success else "Training Failed"
        detail = f"Horse: {horse_entry['name']} (#{horse_entry['horse_id']})\nStat: {stat.name}"
        if success and finish_time:
            if finish_time.tzinfo is None or finish_time.tzinfo.utcoffset(finish_time) is None:
                finish_time = finish_time.replace(tzinfo=timezone.utc)
            finish_unix = int(finish_time.timestamp())
            detail += f"\nCompletes: <t:{finish_unix}:F> (<t:{finish_unix}:R>)"
        embed = discord.Embed(title=title, description=message, color=color)
        embed.add_field(name="Details", value=detail, inline=False)
        await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="enter_race", description="Enter one of your horses into a pending race.")
    @app_commands.describe(
        race_id="The pending race to join.",
        horse="Horse ID or name.",
    )
    async def enter_race(self, interaction: discord.Interaction, race_id: int, horse: str):
        await interaction.response.defer(ephemeral=True, thinking=True)

        trainer_id = int(interaction.user.id)
        horse_entry = self._resolve_owned_horse(trainer_id, horse)
        if not horse_entry:
            await interaction.followup.send(
                "Could not find a horse you own with that ID or name.",
                ephemeral=True,
            )
            return

        try:
            queries.ensure_trainer_record(trainer_id)
            success, message = queries.player_enter_race(trainer_id, horse_entry["horse_id"], race_id)
        except Exception as err:
            print(f"Error entering race {race_id}: {err}")
            traceback.print_exc()
            await interaction.followup.send("An error occurred while entering the race.", ephemeral=True)
            return

        color = discord.Color.green() if success else discord.Color.red()
        title = "Entry Confirmed" if success else "Entry Failed"
        embed = discord.Embed(title=title, description=message, color=color)
        await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="withdraw_race", description="Withdraw your horse from a pending race.")
    @app_commands.describe(
        race_id="The pending race to withdraw from.",
        horse="Horse ID or name.",
    )
    async def withdraw_race(self, interaction: discord.Interaction, race_id: int, horse: str):
        await interaction.response.defer(ephemeral=True, thinking=True)

        trainer_id = int(interaction.user.id)
        horse_entry = self._resolve_owned_horse(trainer_id, horse)
        if not horse_entry:
            await interaction.followup.send(
                "Could not find a horse you own with that ID or name.",
                ephemeral=True,
            )
            return

        try:
            success, message = queries.player_withdraw_from_race(trainer_id, horse_entry["horse_id"], race_id)
        except Exception as err:
            print(f"Error withdrawing from race {race_id}: {err}")
            traceback.print_exc()
            await interaction.followup.send("An error occurred while withdrawing from the race.", ephemeral=True)
            return

        color = discord.Color.green() if success else discord.Color.red()
        title = "Withdrawal Complete" if success else "Withdrawal Failed"
        embed = discord.Embed(title=title, description=message, color=color)
        await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="claim", description="Claim a horse from a recently finished claimer race.")
    @app_commands.describe(
        horse="The horse to claim (ID or name).",
        race_id="The race the horse ran in.",
    )
    async def claim(self, interaction: discord.Interaction, horse: str, race_id: int):
        await interaction.response.defer(ephemeral=True, thinking=True)
        player_id_str = str(interaction.user.id)

        if not market_db:
            await interaction.followup.send("Economy system unavailable. Please try again later.", ephemeral=True)
            return

        if not market_db.get_user_details(player_id_str):
            await interaction.followup.send("You must register with the main bot before claiming horses.", ephemeral=True)
            return

        race_entries = queries.get_horses_in_race(race_id)
        if not race_entries:
            # Race entries are cleared after completion; fall back to results snapshot.
            results = queries.get_race_results_with_horses(race_id)
            race_entries = [
                {
                    "horse_id": record["horse_id"],
                    "name": record["name"],
                    "strategy": None,
                    "spd": None,
                    "sta": None,
                    "fcs": None,
                    "grt": None,
                    "cog": None,
                    "lck": None,
                    "hg_score": None,
                }
                for record in results
            ]

        horse_entry = self._match_horse_identifier(horse, race_entries)
        if not horse_entry:
            await interaction.followup.send(
                "Could not find that horse in the specified race.",
                ephemeral=True,
            )
            return

        try:
            success, message = queries.execute_claim_horse(
                player_user_id=player_id_str,
                horse_id=horse_entry["horse_id"],
                race_id=race_id,
            )
        except Exception as err:
            print(f"Error during claim for horse {horse_entry['horse_id']}: {err}")
            traceback.print_exc()
            await interaction.followup.send("A server error occurred while processing your claim.", ephemeral=True)
            return

        color = discord.Color.green() if success else discord.Color.red()
        title = "Claim Successful" if success else "Claim Failed"
        embed = discord.Embed(title=title, description=message, color=color)
        await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="derby_help", description="Overview of Derby commands and gameplay loops.")
    async def derby_help(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)

        help_lines = [
            "`/races [tier]` - List pending and open races; odds preview for open pools.",
            "`/race_info <race_id>` - Full field sheet with live odds refresh.",
            "`/bet <race_id> <horse> <amount>` - Bet on a horse by ID or name; odds lock on confirm.",
            "`/my_bets` - Review your recent wagers and outcomes.",
            "`/train <horse> <stat>` - Queue a 16-hour training job (cost is charged immediately).",
            "`/stable` - View your stable roster and horse details.",
            "`/enter_race <race_id> <horse>` - Swap one of your horses into a pending race.",
            "`/withdraw_race <race_id> <horse>` - Pull your horse out before betting locks.",
            "`/claim <race_id> <horse>` - Purchase an eligible claimer horse after the race.",
        ]

        embed = discord.Embed(
            title="Derby Help",
            description="\n".join(help_lines),
            color=discord.Color.dark_blue(),
        )
        embed.set_footer(text="Tip: Horses can be referenced by numeric ID or their full name in all commands.")
        await interaction.followup.send(embed=embed, ephemeral=True)


async def setup(bot: commands.Bot):
    await bot.add_cog(DerbyCommands(bot))
    print("DerbyCommands cog loaded.")
