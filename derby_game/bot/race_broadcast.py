import asyncio
import json
import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from typing import Any, Dict, List, Optional

import discord
from discord.ext import commands
from discord.utils import get as discord_get

from derby_game import betting_service
from derby_game.config import BALANCE_CONFIG
from derby_game.database import queries


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



def to_timestamp(dt: Optional[datetime]) -> str:
    if not dt:
        return "Not scheduled"
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    unix_timestamp = int(dt.timestamp())
    return f"<t:{unix_timestamp}:f> (<t:{unix_timestamp}:R>)"


class RaceLobbyView(discord.ui.View):
    def __init__(self, cog: "RaceBroadcastCog", race_id: int):
        super().__init__(timeout=None)
        self.cog = cog
        self.race_id = race_id
        self._status = "pending"

    def set_status(self, status: str):
        self._status = status
        if hasattr(self, "refresh_button"):
            self.refresh_button.disabled = status not in {"open", "locked"}
        if hasattr(self, "bet_button"):
            self.bet_button.disabled = status != "open"

    @discord.ui.button(label="Refresh Odds", style=discord.ButtonStyle.primary, custom_id="race_refresh")
    async def refresh_button(self, interaction: discord.Interaction, _: discord.ui.Button):
        await interaction.response.defer(ephemeral=True, thinking=True)
        updated = await self.cog.request_manual_refresh(self.race_id)
        message = "Odds refreshed." if updated else "No changes since the last update."
        await interaction.followup.send(message, ephemeral=True)

    @discord.ui.button(label="Place Bet", style=discord.ButtonStyle.success, custom_id="race_place_bet")
    async def bet_button(self, interaction: discord.Interaction, _: discord.ui.Button):
        await interaction.response.send_message(
            f"Use `/bet {self.race_id} <horse> <amount>` to place a wager. "
            "You can reference horses by ID or their full name.",
            ephemeral=True,
        )


class RaceBroadcastCog(commands.Cog):
    POLL_INTERVAL = 10
    LOBBY_REFRESH_SECONDS = 30
    ROUND_DELAY_SECONDS = 1.8

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        config_channel = BALANCE_CONFIG.get("discord", {}).get("race_channel_id")
        config_name = BALANCE_CONFIG.get("discord", {}).get("race_channel_name")
        env_channel = os.getenv("DERBY_RACE_CHANNEL_ID")
        env_name = os.getenv("DERBY_RACE_CHANNEL_NAME")
        try:
            self.channel_id = int(config_channel or env_channel or 0)
        except (TypeError, ValueError):
            self.channel_id = 0
        self.channel_name = str(config_name or env_name or "").strip().lower()
        self.channel: Optional[discord.TextChannel] = None
        self.poll_task: Optional[asyncio.Task] = None
        self.lobby_state: Dict[int, Dict[str, float]] = defaultdict(lambda: {"last_refresh": 0.0, "last_hash": None})
        self.message_cache: Dict[int, discord.Message] = {}
        self.lobby_views: Dict[int, RaceLobbyView] = {}
        self.thread_cache: Dict[int, discord.Thread] = {}
        self.playback_tasks: Dict[int, asyncio.Task] = {}
        self.lock_lead = timedelta(minutes=BALANCE_CONFIG["racing"].get("lock_lead_minutes", 1))
        self.poll_task = self.bot.loop.create_task(self.poll_loop())

    async def cog_unload(self):
        if self.poll_task:
            self.poll_task.cancel()
        for task in list(self.playback_tasks.values()):
            task.cancel()

    async def poll_loop(self):
        await self.bot.wait_until_ready()
        if not self.channel_id and not self.channel_name:
            print("[RaceBroadcast] No race channel configured; broadcasts disabled.")
            return
        while True:
            try:
                if self.channel is None:
                    channel = None
                    if self.channel_id:
                        channel = self.bot.get_channel(self.channel_id)
                        if channel is None:
                            try:
                                channel = await self.bot.fetch_channel(self.channel_id)
                            except Exception as fetch_err:
                                print(f"[RaceBroadcast] Failed to fetch channel {self.channel_id}: {fetch_err}")
                                await asyncio.sleep(self.POLL_INTERVAL)
                                continue
                    elif self.channel_name:
                        for guild in self.bot.guilds:
                            candidate = discord_get(guild.text_channels, name=self.channel_name)
                            if candidate:
                                channel = candidate
                                self.channel_id = candidate.id
                                break
                        if channel is None:
                            print(f"[RaceBroadcast] Channel named '{self.channel_name}' not found yet.")
                            await asyncio.sleep(self.POLL_INTERVAL)
                            continue
                    if isinstance(channel, discord.TextChannel):
                        self.channel = channel
                    else:
                        identifier = self.channel_id or self.channel_name or "unknown"
                        print(f"[RaceBroadcast] Channel {identifier} is not a text channel.")
                        return

                await self.sync_races()
            except asyncio.CancelledError:
                break
            except Exception as sync_err:
                print(f"[RaceBroadcast] Error during sync: {sync_err}")
            await asyncio.sleep(self.POLL_INTERVAL)

    async def sync_races(self):
        statuses = ["open", "locked", "running", "finished"]
        races = await asyncio.to_thread(queries.get_races_by_status, statuses)
        for race in races:
            race_id = race["race_id"]
            status = race["status"].lower()
            await asyncio.to_thread(queries.ensure_race_broadcast_record, race_id)
            broadcast = await asyncio.to_thread(queries.get_race_broadcast, race_id)
            if broadcast is None:
                continue

            if status in {"open", "locked"}:
                await self.refresh_lobby_embed(race, broadcast, manual=False)
                if status == "locked" and not broadcast.get("last_odds"):
                    odds_map = await asyncio.to_thread(self._calculate_odds_payload, race_id)
                    await asyncio.to_thread(queries.update_race_broadcast, race_id, last_odds=odds_map)
            elif status == "running":
                await self.refresh_lobby_embed(race, broadcast, manual=False)
                await self.ensure_playback_task(race_id)
            elif status == "finished":
                await self.refresh_lobby_embed(race, broadcast, manual=False)
                await self.ensure_playback_task(race_id)
                await self.ensure_final_summary(race, broadcast)

    async def request_manual_refresh(self, race_id: int) -> bool:
        race = await asyncio.to_thread(queries.get_race_details, race_id)
        if not race:
            return False
        broadcast = await asyncio.to_thread(queries.get_race_broadcast, race_id)
        if not broadcast:
            return False
        return await self.refresh_lobby_embed(race, broadcast, manual=True)

    async def refresh_lobby_embed(self, race: Dict[str, Any], broadcast: Dict[str, Any], manual: bool) -> bool:
        race_id = race["race_id"]
        now = asyncio.get_running_loop().time()
        state = self.lobby_state[race_id]
        status = race["status"].lower()
        refresh_interval = 5 if status in {"open", "locked"} else self.LOBBY_REFRESH_SECONDS

        if not manual and now - state["last_refresh"] < refresh_interval:
            return False

        entries = await asyncio.to_thread(queries.get_horses_in_race, race_id)
        if not entries:
            return False

        if status == "open":
            odds_payload = await asyncio.to_thread(self._calculate_odds_payload, race_id)
        else:
            odds_payload = broadcast.get("last_odds") or await asyncio.to_thread(self._calculate_odds_payload, race_id)
            if status == "locked" and not broadcast.get("last_odds"):
                await asyncio.to_thread(queries.update_race_broadcast, race_id, last_odds=odds_payload)

        odds_map = {int(horse_id): data for horse_id, data in (odds_payload or {}).items()}
        total_pool = betting_service.get_market_pool_total(race_id)

        embed = self.build_lobby_embed(race, entries, odds_map, total_pool)
        message = await self.ensure_lobby_message(race_id, broadcast)
        view = self.get_lobby_view(race_id, status)

        hash_components = (
            embed.title,
            embed.description,
            tuple((field.name, field.value) for field in embed.fields),
            embed.footer.text if embed.footer else "",
        )
        new_hash = hash(hash_components)
        if new_hash != state["last_hash"] or manual:
            try:
                await message.edit(embed=embed, view=view)
                state["last_hash"] = new_hash
            except discord.HTTPException as err:
                print(f"[RaceBroadcast] Failed to edit lobby message for race {race_id}: {err}")
                return False

        await self.sync_bet_thread(race, broadcast, message)
        state["last_refresh"] = now
        return True

    async def ensure_lobby_message(self, race_id: int, broadcast: Dict[str, Any]) -> discord.Message:
        message_id = broadcast.get("lobby_message_id")
        if message_id in self.message_cache:
            return self.message_cache[message_id]

        if message_id:
            try:
                message = await self.channel.fetch_message(message_id)
                self.message_cache[message_id] = message
                return message
            except (discord.NotFound, discord.HTTPException):
                pass

        message = await self.channel.send(embed=discord.Embed(title=f"Race #{race_id}", description="Preparing race data..."))
        self.message_cache[message.id] = message
        await asyncio.to_thread(
            queries.update_race_broadcast,
            race_id,
            lobby_channel_id=self.channel.id,
            lobby_message_id=message.id,
        )

        return message

    async def ensure_bet_thread(self, race_id: int, broadcast: Dict[str, Any], lobby_message: discord.Message) -> Optional[discord.Thread]:
        thread_id = broadcast.get("bet_thread_id")
        thread: Optional[discord.Thread] = None
        if thread_id:
            thread = self.thread_cache.get(thread_id)
            if thread is None:
                try:
                    fetched = await self.bot.fetch_channel(thread_id)
                    if isinstance(fetched, discord.Thread):
                        thread = fetched
                except (discord.NotFound, discord.HTTPException):
                    thread = None
            if thread:
                self.thread_cache[thread.id] = thread
                return thread

        try:
            thread = await lobby_message.create_thread(
                name=f"Race #{race_id} Betting",
                auto_archive_duration=60,
            )
            self.thread_cache[thread.id] = thread
            await asyncio.to_thread(
                queries.update_race_broadcast,
                race_id,
                bet_thread_id=thread.id,
            )
            return thread
        except discord.HTTPException as err:
            print(f"[RaceBroadcast] Failed to create bet thread for race {race_id}: {err}")
            return None

    async def sync_bet_thread(self, race: Dict[str, Any], broadcast: Dict[str, Any], lobby_message: discord.Message):
        thread = await self.ensure_bet_thread(race["race_id"], broadcast, lobby_message)
        if not thread:
            return

        last_logged = broadcast.get("last_logged_bet_id") or 0
        new_bets = await asyncio.to_thread(queries.get_market_bets_since, race["race_id"], last_logged)
        if not new_bets:
            return

        for bet in new_bets:
            content = self._format_bet_message(bet)
            try:
                await thread.send(content)
            except discord.HTTPException as err:
                print(f"[RaceBroadcast] Failed to post bet #{bet['bet_id']} for race {race['race_id']}: {err}")
                continue
            last_logged = bet["bet_id"]

        await asyncio.to_thread(
            queries.update_race_broadcast,
            race["race_id"],
            last_logged_bet_id=last_logged,
        )
        broadcast["last_logged_bet_id"] = last_logged

    def _format_bet_message(self, bet: Dict[str, Any]) -> str:
        bettor_id = str(bet.get("bettor_id"))
        amount = format_cc(bet.get("amount"))
        horse_name = bet.get("horse_name", "?")
        locked_odds = bet.get("locked_in_odds")
        fractional = format_fractional_odds(Decimal(str(locked_odds))) if locked_odds is not None else "-"
        timestamp = bet.get("placed_at")
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            time_display = f"<t:{int(timestamp.timestamp())}:t>"
        else:
            time_display = "Just now"

        display_name = self._format_bettor_display(bettor_id)

        return f"{time_display} - {display_name} wagered {amount} on **{horse_name}** at {fractional}"

    def get_lobby_view(self, race_id: int, status: str) -> RaceLobbyView:
        view = self.lobby_views.get(race_id)
        if not view:
            view = RaceLobbyView(self, race_id)
            self.lobby_views[race_id] = view
        view.set_status(status)
        return view

    def build_lobby_embed(self, race: Dict[str, Any], entries: List[Dict[str, Any]], odds_map: Dict[int, Dict[str, Any]], total_pool: Decimal) -> discord.Embed:
        title = f"Race #{race['race_id']} - Tier {race['tier']} ({race['distance']}m)"
        start_time = race.get("start_time")
        embed = discord.Embed(title=title, color=self._status_color(race["status"]))

        lock_time = None
        if start_time:
            start_dt = start_time if start_time.tzinfo else start_time.replace(tzinfo=timezone.utc)
            lock_time = start_dt - self.lock_lead

        embed.add_field(name="Status", value=race["status"].capitalize(), inline=True)
        embed.add_field(name="Starts", value=to_timestamp(start_time), inline=True)
        embed.add_field(name="Locks", value=to_timestamp(lock_time), inline=True)
        embed.add_field(name="Purse", value=format_cc(race.get("purse")), inline=True)
        embed.add_field(name="Entry Fee", value=format_cc(race.get("entry_fee")), inline=True)
        embed.add_field(name="Pool", value=format_cc(total_pool), inline=True)

        leaderboard = self._build_leaderboard(entries, odds_map)
        embed.add_field(name="Odds Leaders", value=leaderboard, inline=False)

        stat_grid = self._build_stat_grid(entries, odds_map)
        if stat_grid:
            embed.add_field(name="Horse Stat Grid", value=f"```text\n{stat_grid}\n```", inline=False)

        embed.set_footer(text="Use /bet, /race_info, or /train for more options.")
        return embed

    def _build_leaderboard(self, entries: List[Dict[str, Any]], odds_map: Dict[int, Dict[str, Any]]) -> str:
        items = []
        for entry in entries:
            data = odds_map.get(entry["horse_id"])
            if data:
                decimal_odds = Decimal(str(data["odds"]))
                items.append((decimal_odds, entry["name"]))
        if not items:
            return "Odds pending..."
        items.sort(key=lambda item: item[0])
        top = [f"{idx+1}. {name} - {format_fractional_odds(odds)}" for idx, (odds, name) in enumerate(items[:3])]
        return "\n".join(top)

    def _build_stat_grid(self, entries: List[Dict[str, Any]], odds_map: Dict[int, Dict[str, Any]]) -> str:
        header = "Ln Horse               SPD STA FCS GRT COG LCK Odds"
        lines = [header, "-" * len(header)]
        for lane, entry in enumerate(entries, start=1):
            odds = odds_map.get(entry["horse_id"])
            if odds:
                decimal_odds = Decimal(str(odds["odds"]))
                odds_text = format_fractional_odds(decimal_odds)
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

    def _status_color(self, status: str) -> discord.Color:
        status = status.lower()
        if status == "open":
            return discord.Color.blue()
        if status == "locked":
            return discord.Color.orange()
        if status == "running":
            return discord.Color.red()
        if status == "finished":
            return discord.Color.green()
        return discord.Color.light_grey()

    def _calculate_odds_payload(self, race_id: int):
        odds = betting_service.calculate_live_odds(race_id)
        payload = {}
        for horse_id, data in odds.items():
            payload[str(horse_id)] = {
                "name": data["name"],
                "prob": float(data["prob"]),
                "odds": float(data["odds"]),
            }
        return payload

    async def ensure_playback_task(self, race_id: int):
        if race_id in self.playback_tasks and not self.playback_tasks[race_id].done():
            return
        task = self.bot.loop.create_task(self.playback_race(race_id))
        self.playback_tasks[race_id] = task

    async def playback_race(self, race_id: int):
        try:
            broadcast = await asyncio.to_thread(queries.get_race_broadcast, race_id)
            if not broadcast:
                return
            live_message = await self.ensure_live_message(race_id, broadcast)
            entries = await asyncio.to_thread(queries.get_horses_in_race, race_id)
            if not entries:
                return
            race_details = await asyncio.to_thread(queries.get_race_details, race_id)
            lane_order = {entry["horse_id"]: idx + 1 for idx, entry in enumerate(entries)}
            horse_map = {entry["horse_id"]: entry for entry in entries}

            rounds = []
            for attempt in range(10):
                rounds = await asyncio.to_thread(queries.get_race_rounds, race_id)
                if rounds:
                    break
                await asyncio.sleep(1)
            if not rounds:
                await live_message.edit(content=f"Race #{race_id} playback unavailable.")
                return

            grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            for record in rounds:
                grouped[int(record["round_number"])].append(record)

            positions = {horse_id: 0.0 for horse_id in horse_map.keys()}
            distance = race_details["distance"] if race_details and race_details.get("distance") else entries[0].get("max_pref_dist", 1000)
            distance = max(distance, 1)

            for round_number in sorted(grouped.keys()):
                logs = grouped[round_number]
                event_texts = []
                for log in logs:
                    horse_id = log["horse_id"]
                    positions[horse_id] = log["final_position"]
                    if log["round_events"]:
                        try:
                            events = json.loads(log["round_events"])
                        except Exception:
                            events = []
                        for event in events:
                            if event.get("type") == "grit_boost":
                                multiplier = event.get("multiplier", 1.0)
                                event_texts.append(
                                    f"{horse_map[horse_id]['name']} triggered Grit Surge! x{multiplier:.2f}"
                                )

                embed = self.render_race_snapshot(
                    race_id,
                    round_number,
                    lane_order,
                    horse_map,
                    positions,
                    distance,
                    event_texts,
                )
                try:
                    await live_message.edit(content=None, embed=embed)
                except discord.HTTPException:
                    pass
                await asyncio.sleep(self.ROUND_DELAY_SECONDS)

            await self.post_final_standings(race_id, live_message, horse_map, positions, distance, lane_order)
        except asyncio.CancelledError:
            pass
        except Exception as err:
            print(f"[RaceBroadcast] Playback error for race {race_id}: {err}")

    async def ensure_live_message(self, race_id: int, broadcast: Dict[str, Any]) -> discord.Message:
        message_id = broadcast.get("live_message_id")
        if message_id and message_id in self.message_cache:
            return self.message_cache[message_id]
        if message_id:
            try:
                message = await self.channel.fetch_message(message_id)
                self.message_cache[message_id] = message
                return message
            except (discord.NotFound, discord.HTTPException):
                pass
        message = await self.channel.send(f"Race #{race_id} is now underway...")
        self.message_cache[message.id] = message
        await asyncio.to_thread(
            queries.update_race_broadcast,
            race_id,
            live_message_id=message.id,
            broadcast_status="running",
        )
        return message

    def render_race_snapshot(
        self,
        race_id: int,
        round_number: int,
        lane_order: Dict[int, int],
        horse_map: Dict[int, Dict[str, Any]],
        positions: Dict[int, float],
        distance: int,
        events: List[str],
    ) -> discord.Embed:
        lines = []
        sorted_horses = sorted(horse_map.keys(), key=lambda hid: positions.get(hid, 0), reverse=True)
        for horse_id in sorted_horses:
            lane = lane_order.get(horse_id, 0)
            progress = min(max(positions.get(horse_id, 0) / distance, 0), 1)
            filled = int(progress * 20)
            bar = "#" * filled + "." * (20 - filled)
            lines.append(f"{lane:>2} {horse_map[horse_id]['name'][:18]:<18} [{bar}] {int(positions.get(horse_id, 0)):>4}m")
        board = "\n".join(lines) if lines else "No runners found."
        embed = discord.Embed(
            title=f"Race #{race_id} — Round {round_number}",
            color=discord.Color.gold(),
        )
        board_value = f"```text\n{board}\n```"
        if len(board_value) > 1024:
            board_value = board_value[:1010] + "\n...```"
        embed.add_field(name="Track Position", value=board_value, inline=False)
        if events:
            events_text = "\n".join(f"• {text}" for text in events)
        else:
            events_text = "No special events this round."
        if len(events_text) > 1024:
            events_text = events_text[:1021] + "..."
        embed.add_field(name="Highlights", value=events_text, inline=False)
        embed.set_footer(text="Watch the action unfold live.")
        return embed

    async def post_final_standings(
        self,
        race_id: int,
        live_message: discord.Message,
        horse_map: Dict[int, Dict[str, Any]],
        positions: Dict[int, float],
        distance: int,
        lane_order: Dict[int, int],
    ):
        results = await asyncio.to_thread(self._fetch_race_results, race_id)
        if not results:
            return
        lines = []
        for record in results:
            horse = horse_map.get(record["horse_id"])
            if not horse:
                continue
            lane = lane_order.get(record["horse_id"], 0)
            final_position = int(positions.get(record["horse_id"], 0))
            payout = format_cc(record.get("payout") or 0)
            lines.append(f"{record['finish_position']}. Lane {lane} — {horse['name']} ({final_position}m) • {payout}")
        board = "\n".join(lines) if lines else "No finishers recorded."
        embed = discord.Embed(
            title=f"Race #{race_id} — Final Standings",
            color=discord.Color.green(),
        )
        board_value = board
        if len(board_value) > 4096:
            board_value = board_value[:4093] + "..."
        embed.description = board_value
        try:
            await live_message.edit(content=None, embed=embed)
        except discord.HTTPException:
            pass

    async def ensure_final_summary(self, race: Dict[str, Any], broadcast: Dict[str, Any]):
        if broadcast.get("summary_message_id"):
            return
        results = await asyncio.to_thread(self._fetch_race_results, race["race_id"])
        if not results:
            return
        winners = []
        for record in results[:3]:
            winners.append(f"{record['finish_position']}. {record['name']} — {format_cc(record['payout'])}")
        odds_payload = broadcast.get("last_odds") or await asyncio.to_thread(self._calculate_odds_payload, race["race_id"])
        winner_name = winners[0] if winners else "N/A"
        embed = discord.Embed(
            title=f"Race #{race['race_id']} Results",
            description="\n".join(winners) if winners else "No finishers recorded.",
            color=discord.Color.green(),
        )
        if odds_payload:
            winning_odds = None
            for horse_id, data in odds_payload.items():
                if int(horse_id) == results[0]["horse_id"]:
                    winning_odds = data["odds"]
                    break
            if winning_odds:
                embed.add_field(name="Winning Odds", value=f"{winning_odds:.2f}x", inline=True)
        embed.add_field(name="Purse Paid", value=format_cc(sum(r["payout"] for r in results)), inline=True)
        embed.set_footer(text="Thanks for watching! Place your bets on the next race.")

        message = await self.channel.send(embed=embed)
        await asyncio.to_thread(
            queries.update_race_broadcast,
            race["race_id"],
            summary_message_id=message.id,
            broadcast_status="finished",
        )
        await self.post_betting_summary(race["race_id"])

    def _fetch_race_results(self, race_id: int):
        return queries.get_race_results_with_horses(race_id)

    def _format_bettor_display(self, bettor_id: str) -> str:
        if bettor_id.startswith("derby_bot_"):
            return bettor_id.replace("derby_bot_", "").replace("_", " ").title()
        if bettor_id.isdigit():
            return f"<@{bettor_id}>"
        return bettor_id

    async def post_betting_summary(self, race_id: int):
        summary = await asyncio.to_thread(queries.get_race_bet_summary, race_id)
        if not summary:
            await self.channel.send(f"Race #{race_id} betting summary: no wagers recorded.")
            return

        lines = []
        for record in summary:
            display = self._format_bettor_display(str(record["bettor_id"]))
            staked = format_cc(record["staked"])
            won = format_cc(record["won"])
            net = format_cc(record["net"])
            indicator = "🟢" if record["net"] > 0 else ("⚪" if record["net"] == 0 else "🔴")
            lines.append(f"{indicator} {display}: Net {net} | Won {won} | Staked {staked}")

        limit = 25
        display_lines = lines[:limit]
        if len(lines) > limit:
            display_lines.append(f"...and {len(lines) - limit} more bettors.")

        summary_text = "\n".join(display_lines)
        while len(summary_text) > 1024 and limit > 10:
            limit -= 5
            display_lines = lines[:limit]
            if len(lines) > limit:
                display_lines.append(f"...and {len(lines) - limit} more bettors.")
            summary_text = "\n".join(display_lines)

        embed = discord.Embed(
            title=f"Race #{race_id} Betting Outcomes",
            description="Net winners at the top, biggest losses at the bottom.",
            color=discord.Color.purple(),
        )
        embed.add_field(name="Leader Board", value=summary_text, inline=False)
        await self.channel.send(embed=embed)


async def setup(bot: commands.Bot):
    await bot.add_cog(RaceBroadcastCog(bot))

