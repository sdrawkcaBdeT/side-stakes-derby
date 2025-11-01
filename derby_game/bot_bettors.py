import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

from derby_game.config import BALANCE_CONFIG
from derby_game.database import queries as derby_queries
from derby_game import betting_service

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "bot_personalities.json")

try:
    from market import database as market_db
except ImportError:  # pragma: no cover
    market_db = None

HOUSE_VIG = Decimal(str(BALANCE_CONFIG["economy"].get("house_vig", 0.08)))


def load_bot_personalities() -> List[Dict[str, object]]:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("WARNING: bot_personalities.json not found. No bettor bots will participate.")
    except json.JSONDecodeError as err:
        print(f"WARNING: Failed to parse bot_personalities.json ({err}).")
    return []


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "bot"


@dataclass
class BotProfile:
    name: str
    trainer_id: int
    discord_id: str
    bankroll: Decimal
    bet_frequency: float
    bet_timing: str
    target_preference: str
    wager_strategy: str
    wager_amount: Decimal
    max_bets_per_race: int
    stable_targets: Dict[str, int]
    starting_balance: Decimal

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "BotProfile":
        return BotProfile(
            name=data["name"],
            trainer_id=int(data.get("trainer_id", 0)),
            discord_id=str(data.get("discord_id") or f"derby_bot_{slugify(data['name'])}"),
            bankroll=Decimal(str(data.get("bankroll", 0))),
            bet_frequency=float(data.get("bet_frequency", 0.5)),
            bet_timing=str(data.get("bet_timing", "any")).lower(),
            target_preference=str(data.get("target_preference", "random")).lower(),
            wager_strategy=str(data.get("wager_strategy", "percentage")).lower(),
            wager_amount=Decimal(str(data.get("wager_amount", 0))),
            max_bets_per_race=int(data.get("max_bets_per_race", 1)),
            stable_targets=dict(data.get("stable_targets", {})),
            starting_balance=Decimal(str(data.get("starting_balance", data.get("bankroll", 0)))),
        )


class BettingSession:
    TIMING_WINDOWS = {
        "early": (0.0, 0.35),
        "mid": (0.35, 0.7),
        "late": (0.7, 0.95),
        "momentum": (0.55, 0.95),
        "any": (0.05, 0.9),
    }
    MIN_WINDOW_SECONDS = 5.0

    def __init__(
        self,
        race,
        bookie,
        lock_time: datetime,
        rng: Optional[random.Random] = None,
        window_seconds: Optional[float] = None,
        elapsed_seconds: float = 0.0,
    ):
        self.race = race
        self.race_id = race.race_id
        self.bookie = bookie
        if lock_time.tzinfo is None or lock_time.tzinfo.utcoffset(lock_time) is None:
            self.lock_time = lock_time.replace(tzinfo=timezone.utc)
        else:
            self.lock_time = lock_time.astimezone(timezone.utc)
        self.rng = rng or random.Random()
        self.profiles = [BotProfile.from_dict(p) for p in load_bot_personalities()]
        self.horse_lookup = {horse.horse_id: horse for horse in race.horses}
        self.mc_probabilities = {horse_id: Decimal(str(prob)) for horse_id, prob in bookie.win_probabilities.items()}
        self.odds_map = self._build_initial_odds()
        self.bet_totals: Dict[int, Decimal] = {horse_id: Decimal("0") for horse_id in self.horse_lookup}
        self.events: List[tuple] = []
        self.placed_bets: List[Dict[str, object]] = []
        self.window_total = max(
            window_seconds if window_seconds is not None else self.MIN_WINDOW_SECONDS,
            self.MIN_WINDOW_SECONDS,
        )
        self.elapsed_window = min(max(elapsed_seconds, 0.0), self.window_total)
        self.remaining_window = max(self.window_total - self.elapsed_window, 0.0)
        self._prepare_schedule()

    def has_events(self) -> bool:
        return bool(self.events)

    def _prepare_schedule(self):
        if not self.profiles:
            return
        now = datetime.now(timezone.utc)
        available_window = max((self.lock_time - now).total_seconds(), 0.0)
        participants = self._select_participants()
        for profile in participants:
            bet_count = max(1, profile.max_bets_per_race)
            for _ in range(bet_count):
                target_point = self._sample_offset(profile.bet_timing, self.window_total)
                if target_point <= self.elapsed_window:
                    continue
                offset = target_point - self.elapsed_window
                if offset > available_window:
                    continue
                self.events.append((offset, profile))
        self.events.sort(key=lambda pair: pair[0])

    def _sample_offset(self, timing: str, window: float) -> float:
        start_ratio, end_ratio = self.TIMING_WINDOWS.get(timing, self.TIMING_WINDOWS["any"])
        base = self.rng.uniform(start_ratio, end_ratio)
        jitter = self.rng.uniform(-0.03, 0.03)
        ratio = max(0.0, min(base + jitter, 0.97))
        return window * ratio

    def _build_initial_odds(self) -> Dict[int, Dict[str, Decimal]]:
        odds_map = {}
        opening_odds = self.bookie.opening_odds or {}
        for horse in self.race.horses:
            data = opening_odds.get(horse.name)
            if data:
                prob = Decimal(str(data.get("probability", self.mc_probabilities.get(horse.horse_id, 0.1))))
                odds = Decimal(str(data.get("odds", 1.0)))
            else:
                prob = self.mc_probabilities.get(horse.horse_id, Decimal("0.1"))
                odds = (Decimal("1") / prob) - Decimal("1") if prob > 0 else Decimal("10")
            odds_map[horse.horse_id] = {
                "name": horse.name,
                "prob": prob,
                "odds": odds,
                "horse": horse,
            }
        return odds_map

    def _select_participants(self) -> List[BotProfile]:
        selected = []
        for profile in self.profiles:
            if self.rng.random() <= profile.bet_frequency:
                selected.append(profile)
        return selected

    async def run(self):
        if not self.events or not market_db:
            return
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        for offset, profile in self.events:
            wait = max(0.0, start_time + offset - loop.time())
            if wait > 0:
                await asyncio.sleep(wait)
            await self._attempt_bet(profile)

    async def _attempt_bet(self, profile: BotProfile):
        await self._refresh_odds()
        target_entry = self._select_target_entry(profile)
        if not target_entry:
            return

        wager = await self._determine_wager(profile)
        if wager <= 0:
            return

        success = await asyncio.to_thread(
            betting_service.place_bot_bet,
            profile.discord_id,
            self.race_id,
            target_entry["horse"].horse_id,
            wager,
        )
        if not success:
            return

        self.placed_bets.append(
            {
                "bettor": profile.name,
                "discord_id": profile.discord_id,
                "horse": target_entry["horse"].name,
                "horse_id": target_entry["horse"].horse_id,
                "amount": wager,
                "locked_odds": float(target_entry["odds"]),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        await self._refresh_odds()

    async def _refresh_odds(self):
        latest = await asyncio.to_thread(betting_service.calculate_live_odds, self.race_id)
        totals_by_name, _ = betting_service.summarize_market_pool(self.race_id)
        updated = {}
        for horse_id, horse in self.horse_lookup.items():
            data = latest.get(horse_id)
            if data:
                prob = Decimal(str(data["prob"]))
                odds = Decimal(str(data["odds"]))
            else:
                prob = self.mc_probabilities.get(horse_id, Decimal("0.1"))
                odds = (Decimal("1") / prob) - Decimal("1") if prob > 0 else Decimal("10")
            updated[horse_id] = {
                "name": horse.name,
                "prob": prob,
                "odds": odds,
                "horse": horse,
            }
            self.bet_totals[horse_id] = Decimal(str(totals_by_name.get(horse.name, 0)))
        self.odds_map = updated

    async def _determine_wager(self, profile: BotProfile) -> int:
        balance = None
        if market_db:
            balance = await asyncio.to_thread(market_db.get_user_balance_by_discord_id, profile.discord_id)
        if balance is None:
            balance = profile.bankroll
        balance_dec = Decimal(str(balance))
        if balance_dec <= 0:
            return 0

        if profile.wager_strategy == "flat":
            wager = profile.wager_amount
        else:
            wager = (balance_dec * profile.wager_amount).quantize(Decimal("1"))

        wager = max(Decimal("25"), wager)
        wager = min(wager, balance_dec)
        return int(wager)

    def _select_target_entry(self, profile: BotProfile):
        entries = list(self.odds_map.values())
        if not entries:
            return None
        entries.sort(key=lambda item: item["odds"])
        preference = profile.target_preference
        if preference == "favorite":
            return entries[0]
        if preference == "second_favorite" and len(entries) > 1:
            return entries[1]
        if preference == "longshot":
            return max(entries, key=lambda item: item["odds"])
        if preference == "value":
            return min(entries, key=lambda item: item["odds"] / max(item["prob"], Decimal("0.01")))
        if preference == "human_fade":
            return min(entries, key=lambda item: self.bet_totals.get(item["horse"].horse_id, Decimal("0")))
        if preference == "momentum":
            return max(entries, key=lambda item: self.bet_totals.get(item["horse"].horse_id, Decimal("0")))
        if preference == "statistical":
            return min(entries, key=lambda item: abs(item["prob"] - Decimal("0.2")))
        return self.rng.choice(entries)


class BotBettingManager:
    def __init__(self):
        self.rng = random.Random()
        self.sessions: Dict[int, asyncio.Task] = {}

    def schedule_betting_session(
        self,
        race,
        bookie,
        lock_time: datetime,
        open_time: Optional[datetime] = None,
        betting_window: Optional[timedelta] = None,
    ):
        if not market_db:
            return
        race_id = race.race_id
        if race_id in self.sessions:
            return
        if derby_queries.get_market_bets_since(race_id, last_bet_id=0):
            return
        now = datetime.now(timezone.utc)
        window_seconds = None
        elapsed_seconds = 0.0
        if open_time is not None:
            if open_time.tzinfo is None or open_time.tzinfo.utcoffset(open_time) is None:
                open_time = open_time.replace(tzinfo=timezone.utc)
            elapsed_seconds = max(0.0, (now - open_time).total_seconds())
            if betting_window is not None:
                window_span = (lock_time - open_time).total_seconds()
            else:
                window_span = (lock_time - open_time).total_seconds()
            window_seconds = max(window_span, BettingSession.MIN_WINDOW_SECONDS)
        elif betting_window is not None:
            window_seconds = max(betting_window.total_seconds(), BettingSession.MIN_WINDOW_SECONDS)
            remaining = max((lock_time - now).total_seconds(), 0.0)
            elapsed_seconds = max(0.0, window_seconds - remaining)
        else:
            remaining = max((lock_time - now).total_seconds(), 0.0)
            window_seconds = max(remaining, BettingSession.MIN_WINDOW_SECONDS)

        elapsed_seconds = min(elapsed_seconds, window_seconds)

        session = BettingSession(
            race,
            bookie,
            lock_time,
            rng=self.rng,
            window_seconds=window_seconds,
            elapsed_seconds=elapsed_seconds,
        )
        if not session.has_events():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        task = loop.create_task(session.run())
        self.sessions[race_id] = task
        task.add_done_callback(lambda _: self.sessions.pop(race_id, None))
        print(f"  -> Bot betting session scheduled for Race #{race_id}.")

    def clear_session(self, race_id: int):
        task = self.sessions.pop(race_id, None)
        if task:
            task.cancel()
