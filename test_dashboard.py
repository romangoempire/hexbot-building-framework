"""Test the dashboard with real hex bots - no training needed.

Run: python test_dashboard.py
Then open http://localhost:5002

Uses hexbot framework bots to play real games.
Auto-ELO computed from game results automatically.
"""

import sys
import time

# Check if hexbot is available
try:
    from hexbot import Bot
    HAS_HEXBOT = True
except ImportError:
    HAS_HEXBOT = False

from dashboard_clean import Dashboard

dash = Dashboard(port=5002)
dash.start()
time.sleep(1)

print("Test dashboard at http://localhost:5002")

if HAS_HEXBOT:
    print("Using hexbot framework bots")
    print("Running arena: heuristic vs random (100 games)")
    print("Press Ctrl+C to stop\n")

    bot1 = Bot.heuristic()
    bot2 = Bot.random()

    # This streams games to the dashboard with auto-ELO
    dash.run_arena(bot1, bot2, games=100, background=False)
    print("\nArena complete. Dashboard stays open. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
else:
    # Fallback: fake data if hexbot isn't available
    print("hexbot not available, using fake data")
    print("Press Ctrl+C to stop\n")
    import random

    iteration = 0
    try:
        while True:
            iteration += 1
            n_games = random.randint(3, 8)
            for g in range(n_games):
                n_moves = random.randint(12, 35)
                moves, occupied = [], set()
                cq, cr = 0, 0
                for m in range(n_moves):
                    for _ in range(20):
                        dq, dr = random.randint(-3, 3), random.randint(-3, 3)
                        q, r = cq + dq, cr + dr
                        if (q, r) not in occupied:
                            break
                    occupied.add((q, r))
                    moves.append([q, r])
                    if random.random() < 0.3:
                        cq += random.choice([-1, 0, 1])
                        cr += random.choice([-1, 0, 1])
                result = random.choice([1.0, -1.0])
                dash.add_game(moves, result)
                time.sleep(0.3)

            loss = max(0.3, 3.0 - iteration * 0.04 + random.gauss(0, 0.15))
            w0 = random.randint(1, n_games - 1)
            dash.add_metric(
                iteration=iteration,
                loss={'total': loss, 'value': loss * 0.3, 'policy': loss * 0.7},
                wins=[w0, n_games - w0, 0], games=n_games,
                avg_game_length=random.randint(15, 30),
                self_play_time=random.uniform(3, 10), workers=5,
            )
            print(f"  Iter {iteration}: {n_games} games, loss={loss:.3f}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped")
