import json

CONFIG_FILE_PATH = 'configs/game_balance.json'

def load_config():
    """
    Loads the main game balance config file.
    """
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"FATAL ERROR: Could not find config file at {CONFIG_FILE_PATH}")
        return None
    except Exception as e:
        print(f"FATAL ERROR: Could not parse config file {CONFIG_FILE_PATH}: {e}")
        return None

# Load the config ONCE when the module is first imported
BALANCE_CONFIG = load_config()

# We can also add a simple helper here to get nested keys safely
def get_config(key_path, default=None):
    """
    Safely gets a value from the loaded config using a 'dot.path'.
    Example: get_config('economy.daily_stable_fee')
    """
    if not BALANCE_CONFIG:
        return default
        
    try:
        keys = key_path.split('.')
        value = BALANCE_CONFIG
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        print(f"Warning: Could not find config key: {key_path}")
        return default