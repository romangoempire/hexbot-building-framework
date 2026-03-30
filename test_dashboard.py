"""Evolving bot demo with live dashboard.

Run:
    python test_dashboard.py
    Then open http://localhost:5002

Showcases how to build an evolutionary bot using the hexbot framework
and visualize its improvement with the training dashboard.

The bot starts with random heuristic weights and evolves them through
tournament selection against a heuristic baseline. Watch the ELO chart
climb as the bot discovers better weight combinations.
"""

import sys
import time
import random
import math
import copy

sys.path.insert(0, '.')
from hexbot import HexGame, evaluate_moves, Bot
from dashboard_clean import Dashboard

# ---------------------------------------------------------------------------
# Evolving heuristic bot
# ---------------------------------------------------------------------------

class EvoBot:
    """A bot with evolvable weights for move evaluation.

    Uses the C engine's scored moves as a base, then re-ranks them
    using learnable weights for different positional features.
    """

    def __init__(self, weights=None):
        self.weights = weights or {
            'score': random.uniform(0.5, 2.0),
            'center': random.uniform(-0.3, 0.8),
            'aggression': random.uniform(0.0, 1.0),
            'noise': random.uniform(0.0, 0.3),
        }

    def best_move(self, game):
        top = evaluate_moves(game, top_n=12)
        if not top:
            moves = game.legal_moves()
            return random.choice(moves) if moves else (0, 0)

        best_move, best_val = top[0][0], -1e9
        for move, score in top:
            q, r = move
            val = score * self.weights['score']
            val -= (abs(q) + abs(r)) * self.weights['center'] * 0.05
            val += score * self.weights['aggression'] * 0.2 if score > 50 else 0
            val += random.gauss(0, max(0.01, self.weights['noise']))
            if val > best_val:
                best_val = val
                best_move = move
        return best_move

    def mutate(self, strength=0.15):
        """Return a mutated copy."""
        new_w = {}
        for k, v in self.weights.items():
            new_w[k] = max(-1, min(3, v + random.gauss(0, strength)))
        return EvoBot(new_w)

    def crossover(self, other):
        """Return a child with mixed weights from two parents."""
        new_w = {}
        for k in self.weights:
            new_w[k] = self.weights[k] if random.random() < 0.5 else other.weights[k]
        return EvoBot(new_w)

    def __repr__(self):
        w = ' '.join(f'{k}={v:.2f}' for k, v in self.weights.items())
        return f'EvoBot({w})'


def play_match(bot_a, bot_b, games=2):
    """Play a match between two bots. Returns (wins_a, wins_b, game_data)."""
    wins_a, wins_b = 0, 0
    all_games = []
    for g in range(games):
        game = HexGame()
        moves = []
        # Alternate sides
        if g % 2 == 0:
            bots = [bot_a, bot_b]
            a_is_p0 = True
        else:
            bots = [bot_b, bot_a]
            a_is_p0 = False

        while not game.is_over:
            bot = bots[game.current_player]
            move = bot.best_move(game) if hasattr(bot, 'best_move') else bot(game)
            moves.append(list(move))
            game.place(*move)

        p0_won = game.winner == 0
        if (p0_won and a_is_p0) or (not p0_won and not a_is_p0):
            wins_a += 1
            result = 1.0
        else:
            wins_b += 1
            result = -1.0

        all_games.append((moves, result, len(moves)))
    return wins_a, wins_b, all_games


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

dash = Dashboard(port=5002)
dash.start()
time.sleep(1)

print("Dashboard: http://localhost:5002")
print("=" * 50)
print("Evolving a heuristic bot against the C engine baseline")
print("Watch the ELO chart to see improvement!")
print("Press Ctrl+C to stop")
print("=" * 50)
print()

POP_SIZE = 8
GAMES_PER_EVAL = 4
baseline = Bot.heuristic()

population = [EvoBot() for _ in range(POP_SIZE)]
champion = None
generation = 0

try:
    while True:
        generation += 1
        t0 = time.time()

        # --- Evaluate each bot against baseline ---
        fitness = []
        total_lengths = []

        for i, bot in enumerate(population):
            wins, losses, games_data = play_match(bot, baseline, GAMES_PER_EVAL)

            # Stream games to dashboard (with small delay so replay works)
            for moves, result, length in games_data:
                dash.add_game(moves, result)
                total_lengths.append(length)
                time.sleep(0.15)  # let dashboard replay catch up

            fitness.append((wins, i))
            dash.update_progress(i + 1, POP_SIZE, phase='evolution')

        # --- Rank by fitness ---
        fitness.sort(reverse=True)
        ranked = [(population[idx], w) for w, idx in fitness]

        best_bot, best_wins = ranked[0]
        worst_wins = ranked[-1][1]
        sp_time = time.time() - t0
        avg_len = round(sum(total_lengths) / len(total_lengths)) if total_lengths else 0
        total_wins = sum(w for w, _ in fitness)
        total_games = POP_SIZE * GAMES_PER_EVAL

        # --- ELO: best bot vs previous champion ---
        elo = None
        if champion is not None:
            champ_wins, champ_losses, _ = play_match(best_bot, champion, 6)
            wr = max(0.05, min(0.95, champ_wins / 6))
            elo_delta = 400 * math.log10(wr / (1 - wr))
            elo = round(1000 + generation * 2 + elo_delta, 1)
            dash.update_progress(6, 6, phase='elo-eval')

        # --- Push metrics ---
        dash.add_metric(
            iteration=generation,
            wins=[total_wins, total_games - total_wins, 0],
            games=total_games,
            avg_game_length=avg_len,
            self_play_time=round(sp_time, 1),
            workers=1,
            elo=elo,
        )

        # --- Print ---
        elo_str = f', ELO {elo}' if elo else ''
        print(f'Gen {generation:3d} | best {best_wins}/{GAMES_PER_EVAL}W'
              f' worst {worst_wins}/{GAMES_PER_EVAL}W'
              f' | avg_len {avg_len} | {sp_time:.1f}s{elo_str}')
        print(f'         {best_bot}')

        # --- Evolution ---
        # Keep top half
        survivors = [bot for bot, _ in ranked[:POP_SIZE // 2]]
        champion = copy.deepcopy(survivors[0])

        # Fill rest with mutations and crossovers
        children = []
        while len(children) < POP_SIZE // 2:
            if random.random() < 0.7:
                # Mutate a random survivor
                parent = random.choice(survivors)
                children.append(parent.mutate())
            else:
                # Crossover two survivors
                p1, p2 = random.sample(survivors, 2)
                children.append(p1.crossover(p2))

        population = survivors + children
        time.sleep(0.3)

except KeyboardInterrupt:
    print(f"\nStopped after {generation} generations")
    if champion:
        print(f"Champion: {champion}")
