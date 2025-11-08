# Hybrid Race Engine Overview

The hybrid engine combines the 1D physics kernel with the 2D spatial AI to drive
tick-based races. This document explains how to enable the engine, inspect the
telemetry, and diagnose behaviour.

## Feature Flag

| Scope | Setting | Description |
|-------|---------|-------------|
| `configs/game_balance.json` | `race_engine.use_hybrid_engine` | Default toggle for the world engine. |
| Environment | `DERBY_USE_HYBRID_ENGINE` | Overrides the config (`true/false/1/0`). |

When the flag is enabled `Race.run_simulation()` instantiates the new
`HybridRaceLoop`; otherwise the legacy simulation path executes untouched.

## Telemetry Payload

When telemetry is enabled (the default for non-silent races) each tick is stored in
`derby.race_rounds` as a JSON document under `round_events`.

```json
{
  "tick": 15,
  "time": 1.0,
  "phase": "opening",
  "leading_segment": "corner",
  "lane_position": 1.0,
  "world_position": { "x": 125.1, "y": 54.3 },
  "distance_delta": 1.15,
  "speed_delta": 0.32,
  "speed": 16.4,
  "target_speed": 17.2,
  "accel": 0.28,
  "hp": 820.0,
  "hp_delta": -4.5,
  "stamina_drain": 4.5,
  "ai_mode": "normal",
  "status_flags": {
    "isBlockedFront": false,
    "isRailContact": false,
    "...": "..."
  },
  "debug": {
    "target_components": {
      "base": 15.6,
      "pacing_bonus": 0.4,
      "corner_penalty": -0.2
    },
    "accel_components": {
      "base": 0.31,
      "skills": 0.0,
      "slope_penalty": -0.03
    },
    "candidate_scores": [
      { "lane": 0.5, "score": 0.3 }
    ],
    "grit_roll": 0.12,
    "events": ["blocking_replan"]
  }
}
```

Key highlights:

- `movement_roll` column stores the tick-to-tick distance delta (metres).
- `distance_delta` and `speed_delta` in the JSON capture the applied kinematics after clamps.
- `stamina_multiplier` stores the drain used for the legacy schema; in hybrid mode it
  equals the recorded `stamina_drain`.
- `round_events` contains the JSON above for downstream playback.

## Debug Signals

- **Candidate lane scores:** displayed whenever the AI evaluates options after
  perception. Higher scores indicate less favourable moves.
- **Grit rolls:** the random roll on grit pushes; presence of `gritPush` in
  `status_flags` indicates the horse fought through a block.
- **Rail contact:** `isRailContact` flag is raised when a horse scrapes a rail; speed
  is slightly reduced for the tick.
- **Start delay:** low-Focus horses keep `isStartDelayed=true` until their focus-based
  gate timer expires; `debug.start_delay_remaining` shows the seconds left.
- **Spurt plan:** once the final leg begins the AI rolls a cognition-based plan. When
  selected you’ll see `hasSpurtPlan=true` and an event like
  `spurt_plan:1.06:push` showing the chosen plan. The planner now simulates multiple
  finish ETAs (108% “burst”, 106% “push”, etc.) and the cognition roll determines how
  aggressively a horse commits.
- **Per-section randomness:** each section adds a cognition-driven offset to
  `target_speed`, visible as `section_random_offset` inside `target_components`. You’ll
  also see `force_in` events when a horse more than 12% of the course width off the rail
  gets the strategy-specific “force in” boost (reapplied until she hugs the rail),
  `downhill_accel` entries when wisdom procs the downhill mode (speed bonus + 60% less HP
  drain), and `move_lane_bonus` whenever a horse actually shifts lanes.
- **Lane AI:** telemetry now logs `lane_mode` (normal/overtake) and `candidate_scores`
  include the strategy/phase lane preference penalties. Strategies follow the doc’s
  preferred lane ranges (e.g., closers stay wide early), so the replay’s lateral spread
  matches expectations.
- **Position keeping:** `pk_mode` toggles through `speed_up`, `overtake`, `pace_up`,
  `pace_down`, and `pace_up_ex` according to the doc’s wisdom rolls and distance
  thresholds. Each mode lasts up to a section with the documented multipliers
  (e.g., front-runner speed up = 1.04×) before cooling down for a second.
- **Start delay:** gate reaction time is now up to 0.15 s, shrinking with higher Focus.
  Low-focus horses will visibly leave the gate later before their start dash kicks in.
- **Pacing state:** `debug.pacing_state` exposes the current macro AI mode
  (`pace_up`, `pace_down`, `stamina_keep`, etc.) along with the speed factor being
  applied. Stamina Keep sets `isStaminaKeep=true` when the horse decides to conserve
  HP mid-race.
- **Blocking telemetry:** `debug.blocking` now reports front/side block flags plus a
  `timer`, and you’ll see `panic_replan` events whenever a racer has been boxed in for
  several ticks and forcibly searches for rail paths.
- **Strategy bias:** `target_components.strategy_bias` shows the baseline speed push/pull
  coming from the horse’s running plan (e.g., Front Runners get bonus speed early,
  closers sacrifice opening pace).
- **Battle buffs:** grit-triggered events append entries such as `battle_lead_competition`,
  `battle_duel`, or `battle_compete` to the events log, and the applied bonuses show up
  in `target_components.spurt_multiplier`/`pacing_factor` plus the extra
  `battle_drain` HP deduction when stamina multipliers > 1. Geometry-aware lane scoring
  also means outer lanes accumulate higher `score` values in `candidate_scores`, so you
  can see why a horse dove toward the rail. Outer lanes now reduce `target_speed`
  (`target_components.lane_penalty`) and increase `stamina_drain`, and the penalty
  scales with actual track curvature so tight turns amplify the cost.
- **Exhaustion:** once `isExhausted` flips true (HP fully drained) the telemetry
  shows the applied `exhaustion_penalty` and `exhaustion` brake inside
  `target_components`/`accel_components`, so you can see why a closer stalled out.
- **Corner penalty:** `target_components.corner_penalty` shows how much speed the
  current lane on a corner is costing.

## Running a Hybrid Race Manually

1. Set `DERBY_USE_HYBRID_ENGINE=true` in your environment or adjust the balance config.
2. Ensure an SVG track exists for the race distance (see `racetracks/svg/`).
3. Trigger a race (e.g., via `Race.run_simulation()` or the world engine loop).
4. Inspect `derby.race_rounds` for the per-tick telemetry JSON to replay lane
   selection, collisions, and stamina behaviour.

Use the telemetry to build playback tools, troubleshoot AI decisions, or compare
legacy/hybrid results side-by-side.

### Replaying Telemetry

The `scripts/replay_race.py` helper can export a race's ticks or render a quick
2D animation laid over the SVG rails:

```bash
# Dump the JSON frames
python scripts/replay_race.py --race-id 123 --dump replays/race_123.json

# Create a GIF/MP4 (requires matplotlib + pillow/ffmpeg)
python scripts/replay_race.py --race-id 123 --animate replays/race_123.gif --fps 12
# Zoomed-in camera that follows the leader
python scripts/replay_race.py --race-id 123 --animate replays/race_123.mp4 --follow-lead --follow-padding 40
```

If no `--dump`/`--animate` path is provided the script writes
`replays/race_<id>_telemetry.json` by default.
