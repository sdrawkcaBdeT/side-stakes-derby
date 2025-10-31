CREATE SCHEMA IF NOT EXISTS derby;

-- Table to store player and bot "trainers"
CREATE TABLE derby.trainers (
    user_id BIGINT PRIMARY KEY, -- Using BIGINT for Discord IDs
    is_bot BOOLEAN DEFAULT FALSE,
    cc_balance BIGINT DEFAULT 0,
    prestige INT DEFAULT 0,
    stable_slots INT DEFAULT 2
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
    purse INT DEFAULT 0
);

-- Junction table for which horses are in which races
CREATE TABLE derby.race_entries (
    entry_id SERIAL PRIMARY KEY,
    race_id INT REFERENCES derby.races(race_id),
    horse_id INT REFERENCES derby.horses(horse_id)
);

-- Table to log all bets
CREATE TABLE derby.race_bets (
    bet_id SERIAL PRIMARY KEY,
    race_id INT REFERENCES derby.races(race_id),
    bettor_id BIGINT NOT NULL, -- Can be a trainer user_id or a bot name
    horse_id INT REFERENCES derby.horses(horse_id),
    amount INT NOT NULL,
    odds FLOAT NOT NULL,
    winnings INT DEFAULT 0
);

-- Table to manage the training queue
CREATE TABLE derby.training_queue (
    queue_id SERIAL PRIMARY KEY,
    horse_id INT REFERENCES derby.horses(horse_id) ON DELETE CASCADE,
    stat_to_train VARCHAR(5) NOT NULL, -- SPD, STA, etc.
    finish_time TIMESTAMP NOT NULL
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

-- Stores all asynchronous messages for players.
CREATE TABLE derby.notifications (
    notification_id SERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES derby.trainers(user_id),
    message TEXT NOT NULL,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
