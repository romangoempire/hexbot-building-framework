"""Train a bot from scratch via self-play."""
import sys; sys.path.insert(0, '..')
from hexbot import train

# Quick training (5 iterations for demo)
bot = train(
    iterations=5,
    games_per_iter=10,
    sims=30,
    network_config='fast',
    checkpoint_prefix='demo_bot',
)
bot.save('demo_bot.pt')
print(f"\nBot saved: {bot}")
