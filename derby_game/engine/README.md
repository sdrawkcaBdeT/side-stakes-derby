# 2.5D Race Engine Scaffold

This package hosts the new hybrid race simulation stack. It splits forward
physics, spatial logic, and asset parsing so the world engine can drive a
deterministic, stat-driven race loop.

## Current Modules

- `data_models.py` – canonical dataclasses for racer sheets, track geometry,
  and mutable per-tick state.
- `geometry.py` – arc-length aware polylines for sampling SVG splines.
- `svg_loader.py` – converts Illustrator-exported tracks to metre space using
  the required `0.35277…` scale factor.
- `physics.py` – the 1D physics kernel implementing speed, acceleration,
  stamina, and grit mechanics per the design doc (slope hooks default to 0
  when SVG metadata does not provide gradients).
- `spatial.py` – perception (feelers + proximity boxes), lane selection,
  blocking penalties, and lateral motion scaffolding.
- `race_loop.py` – preliminary orchestration binding physics to track splines,
  exposing a tick-based simulation loop.

## Minimal Usage

```python
from pathlib import Path
from derby_game.engine import (
    Aptitudes,
    AptitudeGrade,
    RacerProfile,
    RacerStats,
    Strategy,
    load_svg_track,
)
from derby_game.engine.race_loop import HybridRaceLoop

track = load_svg_track(Path("racetracks/svg/826m - track_003 - Canterbury Park.svg"))

profile = RacerProfile(
    racer_id=1,
    name="Sample Horse",
    stats=RacerStats(speed=900, acceleration=850, stamina=1000, grit=700, cognition=600, focus=550, luck=400),
    strategy=Strategy.FRONT_RUNNER,
    aptitudes=Aptitudes(distance=AptitudeGrade.A, surface=AptitudeGrade.A),
)

loop = HybridRaceLoop(track, [profile])
snapshots = loop.run_until_finished(max_time=120.0)

print(f"Finished in {len(snapshots)} ticks; final position {snapshots[-1].racers[0].pos:.2f}m")
```

The loop currently omits lateral AI/collision handling. Those hooks will be
added iteratively as the spatial engine comes online.
