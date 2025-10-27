# horse_name_generator.py
# Standalone helper for generating race-horse names from JSON config.
# No external deps (stdlib only).

from __future__ import annotations
import json
import random
import re
from typing import Dict, Any, Optional, List

TOKEN_RE = re.compile(r"\[([A-Za-z]+)\]")
VOWELS = "aeiouyAEIOUY"

def _weighted_choice(d: Dict[str, Any], key="weight", rng: random.Random = random) -> str:
    items = list(d.items())
    labels = [k for k, _ in items]
    weights = [max(0.0, (v.get(key, 0.0) if isinstance(v, dict) else 0.0)) for _, v in items]
    total = sum(weights)
    if total <= 0:
        return rng.choice(labels)
    # normalize
    probs = [w / total for w in weights]
    return rng.choices(labels, weights=probs, k=1)[0]

def _fill_pattern(pattern: str, lex: Dict[str, List[str]], rng: random.Random = random) -> str:
    def repl(m: re.Match) -> str:
        token = m.group(1)
        pool = lex.get(token) or lex.get(token.capitalize()) or []
        return rng.choice(pool) if pool else token
    return TOKEN_RE.sub(repl, pattern)

def _blend_words(a: str, b: str) -> str:
    if not a or not b:
        return a or b
    cut_a = max((i for i, ch in enumerate(a) if ch in VOWELS), default=len(a)//2)
    cut_b = next((i for i, ch in enumerate(b) if ch in VOWELS), 0)
    fused = (a[:cut_a+1] + b[cut_b:]).strip()
    return re.sub(r"\s+", "", fused).title()

def _mutate_word(w: str, rng: random.Random = random) -> str:
    if len(w) <= 3:
        return w
    i = rng.randrange(1, max(2, len(w)-1))
    op = rng.choice(["drop","double","swap"])
    if op == "drop":
        w = w[:i] + w[i+1:]
    elif op == "double":
        w = w[:i] + w[i] + w[i:]
    else:
        if i+1 < len(w):
            w = w[:i] + w[i+1] + w[i] + w[i+2:]
    return w

def _apply_language_drift(name: str, cfg: Dict[str, Any], rng: random.Random = random) -> str:
    drift = cfg.get("language_drift", {})
    blend = drift.get("blend_chance", 0.0)
    mutate = drift.get("mutate_chance", 0.0)
    max_mut = drift.get("max_mutations", 1)

    # Blend first+last token into a fused proper-noun sometimes
    if rng.random() < blend:
        parts = name.split()
        if len(parts) >= 2:
            name = _blend_words(parts[0], parts[-1])

    # Mutate the longest token a little bit
    if rng.random() < mutate:
        parts = re.split(r"(\s+of\s+|\s+)", name)
        token_idxs = [i for i, p in enumerate(parts) if p.strip() and p.strip().lower() != "of"]
        if token_idxs:
            idx = max(token_idxs, key=lambda i: len(parts[i]))
            w = parts[idx]
            for _ in range(max_mut):
                w = _mutate_word(w, rng)
            parts[idx] = w
            name = "".join(parts)

    return name

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()

def _fits_rules(name: str, rules: Dict[str, Any], forbidden: List[str]) -> bool:
    max_len = rules.get("max_length", 32)
    if len(name) > max_len:
        return False
    forb = {_normalize(x) for x in forbidden}
    if _normalize(name) in forb:
        return False
    reserved = {x.lower() for x in rules.get("reserved_names", [])}
    if name.lower() in reserved:
        return False
    return True

class NameGenerator:
    def __init__(self, config_path: Optional[str] = None, *, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        if config_path is None and config is None:
            raise ValueError("Provide config_path or config dict.")
        if config is None:
            with open(config_path, "r", encoding="utf-8") as f:
                self.cfg = json.load(f)
        else:
            self.cfg = config

        # Convenience handles
        self.lex = self.cfg.get("lexicons", {})
        self.tiers = self.cfg.get("tiers", {})
        self.rules = self.cfg.get("rules", {})
        self.forbidden = self.cfg.get("forbidden_names", [])
        # Legacy support
        self.legacy_adjs = self.cfg.get("adjectives") or self.lex.get("Adjective") or []
        self.legacy_nouns = self.cfg.get("nouns") or self.lex.get("Noun") or []

    def _try_rare_singleton(self) -> Optional[str]:
        rarity = self.cfg.get("rarity", {})
        if self.rng.random() < rarity.get("rare_singleton_chance", 0.0):
            singles = self.lex.get("RareSingleton") or []
            if singles:
                cand = self.rng.choice(singles)
                if _fits_rules(cand, self.rules, self.forbidden):
                    return cand
        return None

    def _generate_tiered(self) -> Optional[str]:
        if not self.tiers:
            return None
        # Try some attempts to find a valid name
        for _ in range(12):
            tier = _weighted_choice(self.tiers, rng=self.rng)
            patterns = self.tiers.get(tier, {}).get("patterns", [])
            if not patterns:
                continue
            pattern = self.rng.choice(patterns)
            candidate = _fill_pattern(pattern, self.lex, rng=self.rng)
            # Apply drift only to generated candidates
            drifted = _apply_language_drift(candidate, self.cfg, rng=self.rng)
            for cand in (drifted, candidate):
                if _fits_rules(cand, self.rules, self.forbidden):
                    return cand
        return None

    def _fallback_legacy(self) -> str:
        if self.legacy_adjs and self.legacy_nouns:
            return f"{self.rng.choice(self.legacy_adjs)} {self.rng.choice(self.legacy_nouns)}"
        return "Generic Horse"

    def generate(self) -> str:
        # 1) Rare exact favorites (not the forbidden subset)
        rare = self._try_rare_singleton()
        if rare:
            return rare
        # 2) Tiered generation (canon/combo/hybrid/modern_phrase)
        tiered = self._generate_tiered()
        if tiered:
            return tiered
        # 3) Fallback to legacy adj+noun if anything goes sideways
        return self._fallback_legacy()

if __name__ == "__main__":
    NAME_CONFIG_PATH = 'configs/horse_names.json'
    gen = NameGenerator(config_path=NAME_CONFIG_PATH, seed=0)
    for _ in range(20):
        print(gen.generate())