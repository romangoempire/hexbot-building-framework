"""Train a bot with live dashboard visualization.

Run:
    python examples/dashboard_train.py

Then open http://localhost:5001 in your browser.
The dashboard auto-plays self-play games, snapshots the bot over time,
and computes ELO by playing current vs past versions.
"""
import sys; sys.path.insert(0, '..')
from hexbot import Bot
from dashboard import Dashboard

dash = Dashboard(port=5001)
dash.start()

print("Dashboard: http://localhost:5001")
print("Training: heuristic bot, 20 iterations\n")

# Dashboard handles everything: self-play, snapshots, ELO, charts
dash.train(
    bot=Bot.heuristic(),
    iterations=20,
    games_per_iter=10,
    eval_every=3,       # ELO eval every 3 iterations
    eval_games=6,       # 6 games per ELO eval
    snapshot_every=2,   # snapshot bot every 2 iterations
    background=False,
)

print("\nDone. Dashboard stays open. Ctrl+C to exit.")
try:
    while True:
        import time; time.sleep(1)
except KeyboardInterrupt:
    pass
