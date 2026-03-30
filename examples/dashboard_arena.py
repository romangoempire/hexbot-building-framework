"""Watch two bots play each other in the dashboard.

Run:
    python examples/dashboard_arena.py

Then open http://localhost:5001 in your browser.
The dashboard shows live game replays, auto-computed ELO, and win rate charts.
"""
import sys; sys.path.insert(0, '..')
from hexbot import Bot
from dashboard import Dashboard

dash = Dashboard(port=5001)
dash.start()

print("Dashboard: http://localhost:5001")
print("Playing: heuristic vs random (100 games)\n")

# One line - dashboard handles games, ELO, charts, everything
dash.run_arena(Bot.heuristic(), Bot.random(), games=100, background=False)

print("\nDone. Dashboard stays open. Ctrl+C to exit.")
try:
    while True:
        import time; time.sleep(1)
except KeyboardInterrupt:
    pass
