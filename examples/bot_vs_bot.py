"""Play two bots against each other and compare results."""
import sys; sys.path.insert(0, '..')
from hexbot import Bot, Arena
from pathlib import Path

# Create bots
random_bot = Bot.random()
heuristic_bot = Bot.heuristic()

# If a trained model exists, load it too
trained_bot = None
checkpoints = sorted(Path('..').glob('hex_checkpoint_*.pt'))
if checkpoints:
    trained_bot = Bot.load(str(checkpoints[-1]))
    print(f"Loaded trained bot: {trained_bot}")

# Random vs Heuristic
print("\n--- Random vs Heuristic (20 games) ---")
result = Arena(random_bot, heuristic_bot, num_games=20).play()

# If trained model exists, test it
if trained_bot:
    print("\n--- Trained vs Heuristic (20 games) ---")
    result = Arena(trained_bot, heuristic_bot, num_games=20).play()

    print("\n--- Trained vs Random (20 games) ---")
    result = Arena(trained_bot, random_bot, num_games=20).play()
