"""Use the dashboard with your own custom bot.

Run:
    python examples/dashboard_custom_bot.py

Then open http://localhost:5001 in your browser.
Shows how to plug ANY function into the dashboard.
"""
import sys; sys.path.insert(0, '..')
import random
from hexbot import HexGame, evaluate_moves
from dashboard import Dashboard

# --- Your custom bot: just a function that takes a game and returns (q, r) ---

def my_bot(game):
    """Example custom bot: picks the highest-scored move 80% of the time,
    random legal move 20% of the time (for exploration)."""
    if random.random() < 0.2:
        moves = game.legal_moves()
        return random.choice(moves) if moves else (0, 0)
    top = evaluate_moves(game, top_n=5)
    if top:
        return top[0][0]  # best scored move
    moves = game.legal_moves()
    return random.choice(moves) if moves else (0, 0)

def random_bot(game):
    """Simple random baseline."""
    moves = game.legal_moves()
    return random.choice(moves) if moves else (0, 0)


# --- Dashboard setup ---

dash = Dashboard(port=5001)
dash.start()

print("Dashboard: http://localhost:5001")
print("Arena: my_bot vs random_bot (50 games)\n")

# Any function(game) -> (q, r) works
dash.run_arena(my_bot, random_bot, games=50, background=False)

print("\nNow training my_bot with auto-ELO (20 iterations)...\n")

# Train with auto-ELO - dashboard snapshots the bot and evaluates
# against past versions automatically
dash.train(my_bot, iterations=20, games_per_iter=10, background=False)

print("\nDone. Ctrl+C to exit.")
try:
    while True:
        import time; time.sleep(1)
except KeyboardInterrupt:
    pass
