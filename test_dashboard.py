"""Test the dashboard UI with fake data — no training needed.

Run: python test_dashboard.py
Then open http://localhost:5002

This feeds the dashboard with simulated games and metrics so you can
test the UI, settings, charts, and game replay without running training.
"""

from dashboard_clean import Dashboard
import time
import random
import math

dash = Dashboard(port=5002)
dash.start()
time.sleep(1)

print("Test dashboard at http://localhost:5002")
print("Press Ctrl+C to stop\n")

iteration = 0
game_idx = 0

try:
    while True:
        iteration += 1
        n_games = random.randint(3, 8)

        for g in range(n_games):
            game_idx += 1
            # Generate a realistic-ish game with no duplicate positions
            n_moves = random.randint(12, 35)
            moves = []
            occupied = set()
            cq, cr = 0, 0
            for m in range(n_moves):
                # Try to find an unoccupied cell nearby
                for _ in range(20):
                    dq = random.randint(-3, 3)
                    dr = random.randint(-3, 3)
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
            dash.update_progress(g + 1, n_games, phase='self-play')
            time.sleep(0.3)

        # Training phase
        steps = random.randint(100, 400)
        loss = max(0.3, 3.0 - iteration * 0.04 + random.gauss(0, 0.15))
        for s in range(0, steps, 20):
            step_loss = loss + random.gauss(0, 0.05)
            dash.update_progress(s + 20, steps, loss=step_loss, phase='training')
            time.sleep(0.05)

        # Push iteration metrics
        elo = 1000 + iteration * 2.5 + random.gauss(0, 8)
        w0 = random.randint(1, n_games - 1)
        w1 = n_games - w0
        dash.add_metric(
            iteration=iteration,
            loss={'total': loss, 'value': loss * 0.3, 'policy': loss * 0.7},
            elo=elo,
            wins=[w0, w1, 0],
            games=n_games,
            avg_game_length=random.randint(15, 30),
            self_play_time=random.uniform(3, 10),
            workers=5,
        )

        print(f"  Iter {iteration}: {n_games} games, loss={loss:.3f}, elo={elo:.0f}")
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nStopped")
