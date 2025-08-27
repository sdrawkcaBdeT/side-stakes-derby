Side-Stakes Derby

> Side-Stakes Derby is a persistent, automated horse racing simulation bot for Discord. It features a deep statistical engine where players can own, train, and race procedurally generated horses. The game includes a full economic model with betting, race purses, and a "living world" of bot-owned horses that creates a dynamic and ever-evolving competitive landscape.

This is a passion project inspired by and designed to be a side-game for the Umamusume: Pretty Derby community.

Core Features

    Train Your Stable: Acquire a G-Grade foal from the starting generation and guide its career. Through a deep, RNG-based training system, develop its unique stats to turn it into a legendary champion.

    Bet on Races: Analyze the odds, study the field, and bet against a dynamic market in automatically scheduled races that run 24/7.

    Become a Legend: Compete against other players and a world of bot-owned horses for prestige, glory, and massive CC purses in a persistent, simulated world.

The Genesis Event: The Race to Crown the First Champion

The world of Side-Stakes Derby begins at Day 0, Year 0. The entire initial population of horses are 2-year-old, G-Grade rookies—a "genesis generation" full of untapped potential. There are no established champions and no high-grade veterans to overcome.

Every trainer starts on a level playing field. The first horse to reach C-Grade, the first to win a Graded Stakes, the first to achieve the legendary S-Grade—these will become server-wide historical moments. The story of this world will be written by its first generation of players and their horses. The race is on.

Gameplay Overview

The core of the game is built on a detailed simulation of a horse's career, from its debut to its retirement.

The Horse

Every horse is unique, defined by six core statistics:

    Speed (SPD): Its raw pace and top-end speed. The primary driver of performance.

    Stamina (STA): Its ability to maintain speed over long distances. Crucial for winning prestigious, high-purse stakes races.

    Focus (FCS): Its mental consistency. High Focus leads to reliable, predictable performances (low sigma), while low Focus creates a volatile, high-risk/high-reward "challenger" (high sigma).

    Grit (GRT): Its fighting spirit. A high Grit stat gives a horse a chance to dig deep for a burst of speed in the final stretch of a race.

    Cognition (COG): Its intelligence and awareness. This will influence the trigger chance for special skills and abilities (Coming in a future update).

    Luck (LCK): An innate, un-trainable stat determined at birth that influences the outcomes of training.

The Trainer

As a trainer, your goal is to manage your horse's career to maximize its potential and winnings. The core gameplay loop is:

    Acquire: Claim a promising 2-year-old from the G-Grade Claimer races to start your stable.

    Train: Manage a 16-hour training cycle to improve your horse's stats. Every session has a chance for critical success, failure, or even a setback, with the outcome influenced by your horse's Luck.

    Age: Guide your horse through its career. A powerful Age Training Modifier creates a realistic lifecycle:

        Age 2: A "Hyper-Growth" phase with massive +150% training gains. This is the most critical year to build a champion.

        Age 3-4: The horse's prime, where it will compete for the biggest prizes.

        Age 5+: The veteran years, where natural stat decay begins and the challenge shifts from progression to preservation.

    Compete: Enter your horse into the appropriate race tier and watch the simulation unfold.

Technical Overview

This project is built with Python and the discord.py library. The simulation relies on numpy for its statistical calculations.

    race_logic_v2.py: The core simulation engine. This module contains the Horse, Race, and Bookie classes and handles the logic for a single, self-contained race event.

    world_engine.py: The persistence and automation layer. This script manages the game's clock, applying the aging and stat decay mechanics, triggering bot training sessions, and running the automated race schedule.

Getting Started

(This section will be updated with bot commands once they are finalized.)

    /stable - View the horses you own.

    /train [horse_id] [stat] - Begin a training regimen for one of your horses.

    /races - See the schedule of upcoming races.

Contributing

This is a community-driven project. If you have ideas, find a bug, or want to contribute to the code, please feel free to open an issue or submit a pull request on the GitHub repository.

License

This project is licensed under a Source-Available License. See the LICENSE file for details.
