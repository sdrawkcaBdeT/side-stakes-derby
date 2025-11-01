CREATE SCHEMA IF NOT EXISTS derby;

-- Market-facing race tables (shared gambling view)
CREATE TABLE derby.market_races (
    race_id BIGINT PRIMARY KEY,
    distance INT NOT NULL,
    status TEXT NOT NULL,
    winner_name TEXT,
    created_at TIMESTAMPTZ DEFAULT (NOW() AT TIME ZONE 'UTC')
);

CREATE TABLE derby.market_race_horses (
    horse_id SERIAL PRIMARY KEY,
    race_id BIGINT REFERENCES derby.market_races(race_id) ON DELETE CASCADE,
    horse_name TEXT NOT NULL,
    strategy TEXT NOT NULL,
    stats_json JSONB NOT NULL
);

CREATE TABLE derby.market_bets (
    bet_id SERIAL PRIMARY KEY,
    race_id BIGINT REFERENCES derby.market_races(race_id) ON DELETE CASCADE,
    bettor_id TEXT NOT NULL,
    horse_name TEXT NOT NULL,
    amount NUMERIC(15, 2) NOT NULL,
    locked_in_odds NUMERIC(10, 2) NOT NULL,
    placed_at TIMESTAMPTZ DEFAULT (NOW() AT TIME ZONE 'UTC')
);

-- Table to store player and bot "trainers"
CREATE TABLE derby.trainers (
    user_id BIGINT PRIMARY KEY, -- Using BIGINT for Discord IDs
    is_bot BOOLEAN DEFAULT FALSE,
    prestige INT DEFAULT 0,
    stable_slots INT DEFAULT 2,
    economy_id TEXT
);

-- The master table for all horses
CREATE TABLE derby.horses (
    horse_id SERIAL PRIMARY KEY,
    owner_id BIGINT REFERENCES derby.trainers(user_id),
    is_bot BOOLEAN DEFAULT FALSE,
    name VARCHAR(255) NOT NULL,
    strategy VARCHAR(50) NOT NULL,             -- <-- ADDED
    min_preferred_distance INT NOT NULL,     -- <-- ADDED
    max_preferred_distance INT NOT NULL,     -- <-- ADDED
    birth_timestamp TIMESTAMPTZ DEFAULT NOW(),

    -- Core Stats
    spd INT NOT NULL,
    sta INT NOT NULL,
    fcs INT NOT NULL,
    grt INT NOT NULL,
    cog INT NOT NULL,
    lck INT NOT NULL,

    hg_score INT NOT NULL,

    -- Status Fields
    is_retired BOOLEAN DEFAULT FALSE,
    in_training_until TIMESTAMP
);

-- Table for upcoming and past races
CREATE TABLE derby.races (
    race_id SERIAL PRIMARY KEY,
    tier VARCHAR(10) NOT NULL, -- G, D, C, B, A, S
    distance INT NOT NULL,
    entry_fee INT DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending', -- pending, open, running, finished
    start_time TIMESTAMP,
    purse INT DEFAULT 0,
    winner_horse_id INT REFERENCES derby.horses(horse_id)
);

-- Junction table for which horses are in which races
CREATE TABLE derby.race_entries (
    entry_id SERIAL PRIMARY KEY,
    race_id INT REFERENCES derby.races(race_id),
    horse_id INT REFERENCES derby.horses(horse_id),
    entry_fee INT DEFAULT 0,
    is_bot_entry BOOLEAN DEFAULT TRUE,
    entered_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table to log all bets
CREATE TABLE derby.race_bets (
    market_bet_id BIGINT PRIMARY KEY,
    race_id INT REFERENCES derby.races(race_id),
    bettor_id TEXT NOT NULL,
    horse_id INT REFERENCES derby.horses(horse_id),
    amount NUMERIC(15, 2) NOT NULL,
    odds FLOAT NOT NULL,
    winnings NUMERIC(15, 2) DEFAULT 0,
    settled_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE derby.race_results (
    race_id INT REFERENCES derby.races(race_id),
    horse_id INT REFERENCES derby.horses(horse_id),
    finish_position INT NOT NULL,
    payout NUMERIC(15, 2) DEFAULT 0,
    PRIMARY KEY (race_id, horse_id)
);

-- Table to manage the training queue
CREATE TABLE derby.training_queue (
    queue_id SERIAL PRIMARY KEY,
    horse_id INT REFERENCES derby.horses(horse_id) ON DELETE CASCADE,
    stat_to_train VARCHAR(5) NOT NULL, -- SPD, STA, etc.
    finish_time TIMESTAMP NOT NULL
);

CREATE TABLE derby.horse_training_plans (
    horse_id INT PRIMARY KEY REFERENCES derby.horses(horse_id) ON DELETE CASCADE,
    stat_code VARCHAR(5) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table for player-owned consumable items
CREATE TABLE derby.trainer_inventory (
    inventory_id SERIAL PRIMARY KEY,
    owner_id BIGINT REFERENCES derby.trainers(user_id),
    item_name VARCHAR(100) NOT NULL, -- e.g., 'Premium Feed'
    quantity INT DEFAULT 1
);

-- Stores the "playback" log for all races.
CREATE TABLE derby.race_rounds (
    round_id SERIAL PRIMARY KEY,
    race_id INT REFERENCES derby.races(race_id),
    round_number INT NOT NULL,
    horse_id INT REFERENCES derby.horses(horse_id),
    movement_roll FLOAT NOT NULL,           -- Final movement this round
    stamina_multiplier FLOAT NOT NULL,      -- STA effect applied
    final_position FLOAT NOT NULL,          -- Position *after* movement
    round_events JSONB                      -- JSON list of events (Grit, Skills)
);

CREATE TABLE derby.race_broadcasts (
    race_id INT PRIMARY KEY REFERENCES derby.races(race_id) ON DELETE CASCADE,
    lobby_channel_id BIGINT,
    lobby_message_id BIGINT,
    live_message_id BIGINT,
    summary_message_id BIGINT,
    bet_thread_id BIGINT,
    last_logged_bet_id BIGINT,
    broadcast_status VARCHAR(20) DEFAULT 'pending',
    last_odds JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Stores all asynchronous messages for players.
CREATE TABLE derby.notifications (
    notification_id SERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES derby.trainers(user_id),
    message TEXT NOT NULL,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
