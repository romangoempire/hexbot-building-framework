"""Play a game: heuristic bot vs random bot."""
import sys; sys.path.insert(0, '..')
from hexbot import HexGame, Bot

game = HexGame()
bots = [Bot.heuristic(), Bot.random()]

while not game.is_over:
    bot = bots[game.current_player]
    move = bot.best_move(game)
    game.place(*move)

print(game)
print(f"Winner: Player {game.winner} ({['Heuristic', 'Random'][game.winner]})")
print(f"Total moves: {game.total_stones}")
