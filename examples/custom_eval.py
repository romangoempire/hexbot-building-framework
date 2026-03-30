"""Build a bot with a custom hand-tuned evaluation function.

This demonstrates how to use hexbot's raw analysis tools
to build a bot without any neural network or MCTS.
"""
import sys; sys.path.insert(0, '..')
from hexbot import HexGame, Arena, Bot, evaluate_moves, find_threats, find_winning_moves


def my_eval_bot(game):
    """Custom bot: prioritize wins > blocks > longest line extension."""
    player = game.current_player
    opponent = 1 - player

    # 1. Take the win if available
    wins = find_winning_moves(game, player)
    if wins:
        return wins[0]

    # 2. Block opponent's winning moves
    opp_wins = find_winning_moves(game, opponent)
    if opp_wins:
        return opp_wins[0]

    # 3. Play where we have the most threat potential
    my_threats = find_threats(game, player)
    if my_threats:
        return my_threats[0]

    # 4. Fall back to best heuristic move
    moves = evaluate_moves(game, top_n=1)
    if moves:
        return moves[0][0]

    # 5. Last resort: first legal move
    return game.legal_moves()[0]


# Test it against the built-in bots
print("Custom eval bot vs Random (20 games)")
result = Arena(my_eval_bot, Bot.random(), num_games=20).play()

print("\nCustom eval bot vs Heuristic (20 games)")
result = Arena(my_eval_bot, Bot.heuristic(), num_games=20).play()
