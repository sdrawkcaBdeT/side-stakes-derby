import asyncio
import os
import sys

# Ensure sibling repo (prettyDerbyClubAnalysis) is on path for shared modules.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
other_project_path = os.path.join(parent_dir, "prettyDerbyClubAnalysis")
if other_project_path not in sys.path:
    sys.path.append(other_project_path)

from derby_game.world_engine import run_world_engine


def main():
    try:
        asyncio.run(run_world_engine())
    except KeyboardInterrupt:
        print("World engine stopped by user.")


if __name__ == "__main__":
    main()
