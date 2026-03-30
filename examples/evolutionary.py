"""Evolve evaluation weights through tournament selection.

This demonstrates an evolutionary approach to bot development:
no neural network, no MCTS — just evolving weights for a simple
evaluation function through natural selection.
"""
import sys; sys.path.insert(0, '..')
import random
from hexbot import HexGame, Arena, evaluate_moves, find_winning_moves


class EvoBot:
    """Bot with evolvable weights for move scoring."""

    def __init__(self, weights=None):
        # Weights: how much to value different aspects
        if weights is None:
            self.weights = {
                'line_score': random.uniform(0.5, 2.0),
                'center_bonus': random.uniform(0.0, 1.0),
                'threat_bonus': random.uniform(1.0, 5.0),
                'block_bonus': random.uniform(1.0, 5.0),
            }
        else:
            self.weights = dict(weights)

    def best_move(self, game):
        player = game.current_player

        # Check for instant wins/blocks first
        wins = find_winning_moves(game, player)
        if wins:
            return wins[0]
        opp_wins = find_winning_moves(game, 1 - player)
        if opp_wins:
            return opp_wins[0]

        # Score each move with our evolved weights
        moves = evaluate_moves(game, top_n=15)
        if not moves:
            return game.legal_moves()[0]

        best_move = moves[0][0]
        best_score = -float('inf')

        for (q, r), base_score in moves:
            score = base_score * self.weights['line_score']

            # Center bonus
            dist = abs(q) + abs(r)
            score -= dist * self.weights['center_bonus']

            # Threat bonus: does this move create a 4+ line?
            ml = game.max_line(q, r, player)
            if ml >= 4:
                score += self.weights['threat_bonus'] * ml

            # Block bonus: does this block opponent's line?
            oml = game.max_line(q, r, 1 - player)
            if oml >= 4:
                score += self.weights['block_bonus'] * oml

            if score > best_score:
                best_score = score
                best_move = (q, r)

        return best_move

    def mutate(self, rate=0.3):
        """Create a mutated copy."""
        child = EvoBot(self.weights)
        for key in child.weights:
            if random.random() < rate:
                child.weights[key] *= random.uniform(0.7, 1.3)
        return child

    def __repr__(self):
        w = {k: round(v, 2) for k, v in self.weights.items()}
        return f'EvoBot({w})'


# Evolutionary training loop
POPULATION = 8
GENERATIONS = 5
GAMES_PER_MATCH = 6

print(f"Evolving {POPULATION} bots over {GENERATIONS} generations")
print(f"({GAMES_PER_MATCH} games per matchup)\n")

# Initialize random population
population = [EvoBot() for _ in range(POPULATION)]

for gen in range(GENERATIONS):
    # Round-robin tournament
    scores = [0] * POPULATION
    for i in range(POPULATION):
        for j in range(i + 1, POPULATION):
            result = Arena(population[i], population[j],
                          num_games=GAMES_PER_MATCH).play(verbose=False)
            scores[i] += result.wins[0]
            scores[j] += result.wins[1]

    # Sort by fitness
    ranked = sorted(range(POPULATION), key=lambda i: scores[i], reverse=True)
    best = population[ranked[0]]

    print(f"Gen {gen+1}: best={best} score={scores[ranked[0]]}")

    # Selection + mutation: top half survive, breed children
    survivors = [population[i] for i in ranked[:POPULATION // 2]]
    children = [s.mutate() for s in survivors]
    population = survivors + children

# Final champion vs heuristic
champion = population[0]
print(f"\nChampion: {champion}")
print("\nChampion vs Heuristic (20 games):")
from hexbot import Bot
Arena(champion, Bot.heuristic(), num_games=20).play()
