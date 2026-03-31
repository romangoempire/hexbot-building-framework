# hexbot

A Python framework for building AI bots for [Hexagonal Tic-Tac-Toe](https://hexo.did.science) (Connect-6 on an infinite hex grid). Fast C game engine, neural network training pipeline, and a live training dashboard.

Home of the **Orca** bot.

## Installation

```bash
git clone https://github.com/Saiki77/hexbot-building-framework.git
cd hexbot-building-framework
pip install torch flask flask-socketio psutil
```

The C engine auto-compiles on first import (`cc`/`gcc`/`clang` required).

## Game Rules

| Rule | Detail |
|------|--------|
| Board | Infinite hexagonal grid (axial coordinates q, r) |
| Players | 2 players alternating turns |
| First turn | Player 0 places 1 stone |
| Other turns | 2 stones per turn |
| Win | 6 in a row on any of 3 hex axes |

## Quick Start

```python
from hexbot import HexGame, Bot, Arena

# Play a game
game = HexGame()
game.place(0, 0)       # P0: 1 stone first turn
game.place(1, 0)       # P1: first of 2 stones
game.place(1, -1)      # P1: second stone

# Build a bot (any function works)
def my_bot(game):
    moves = game.legal_moves()
    return moves[0]

# Test it
result = Arena(my_bot, Bot.heuristic(), num_games=20).play()
```

## Orca Bot

Pre-trained AlphaZero-style bot (3.9M params, 128 filters, 12 residual blocks). The included checkpoint is early-stage (65 iterations) - with more training on a GPU it gets significantly stronger.

```python
from hexbot import Bot

orca = Bot.orca()
move = orca.best_move(game)
```

Train from game collections or from scratch:
```bash
python -m orca.scrape --games 1000 --output games.jsonl  # download games
python -m orca.sft --data games.jsonl --then-selfplay 50  # SFT + self-play
python -m orca.train --iterations 100 --lr 0.002           # pure self-play
```

See [Orca documentation](docs/orca.md) for details.

## Training Dashboard

Live visualization with REST API + WebSocket. Works with any bot in any language.

```python
from dashboard import Dashboard
dash = Dashboard(port=5001)
dash.start()
dash.run_arena(Bot.heuristic(), Bot.random(), games=100)
```

See [Dashboard guide](docs/dashboard-guide.md) for the full API.

## Bot Approaches

6 ways to build a bot, from simple heuristics to neural networks:

| # | Approach | Complexity | Strength |
|---|----------|-----------|----------|
| 1 | [Hand-tuned evaluation](docs/bot-approaches.md#approach-1-hand-tuned-evaluation) | Low | Medium |
| 2 | [Evolutionary weights](docs/bot-approaches.md#approach-2-evolutionary-weights) | Low | Medium |
| 3 | [Minimax with alpha-beta](docs/bot-approaches.md#approach-3-minimax-with-alpha-beta) | Medium | High |
| 4 | [Monte Carlo playouts](docs/bot-approaches.md#approach-4-monte-carlo-random-playouts) | Medium | Medium |
| 5 | [Neural network (AlphaZero)](docs/bot-approaches.md#approach-5-neural-network-alphazero-style) | High | Highest |
| 6 | [Hybrid strategies](docs/bot-approaches.md#approach-6-combine-multiple-strategies) | High | Highest |

## Architecture

```
hexbot.py            Framework API (Bot, Arena, train)
hexgame.py           Game engine (wraps C engine via ctypes)
engine.c             2,300 lines of optimized C (bitboard win detection)
bot.py               Neural network, MCTS, training pipeline
dashboard.py         Live dashboard API (REST + WebSocket)
train_dashboard.py   Training dashboard with web UI
orca/
  __init__.py        Orca bot loader
  config.py          All tunable parameters
  train.py           Training pipeline (CLI + library)
  transformer_net.py Experimental transformer variant
  checkpoint.pt      Pre-trained weights (iter 60)
docs/                Detailed documentation
examples/            Example bot scripts
```

## Device Support

Auto-detects the best available device:

| Priority | Device | Platform |
|----------|--------|----------|
| 1 | CUDA | Linux/Windows with NVIDIA GPU |
| 2 | MPS | macOS with Apple Silicon |
| 3 | CPU | Any platform |

## Configuration

Every training parameter is adjustable via `orca/config.py`, CLI flags, or Python kwargs:

```bash
python -m orca.train --help   # see all options
```

```python
from orca.train import OrcaTrainer
OrcaTrainer(lr=0.002, mcts_sims=100, batch_size=512).run()
```

See [Configuration reference](docs/configuration.md) for all parameters.

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Game API, coordinates, turn structure |
| [Bot Approaches](docs/bot-approaches.md) | 6 approaches with full code examples |
| [API Reference](docs/api-reference.md) | Complete function signatures |
| [Training Guide](docs/training-guide.md) | Self-play, curriculum, ELO, checkpoints |
| [Configuration](docs/configuration.md) | All tunable parameters |
| [Dashboard](docs/dashboard-guide.md) | REST API, WebSocket, keyboard shortcuts |
| [Train Dashboard](docs/train-dashboard.md) | Web UI for Orca training pipeline |
| [Advanced](docs/advanced.md) | Batched inference, MCTS tricks, ONNX |
| [SFT Guide](docs/sft-guide.md) | Train from game collections |
| [Orca Bot](docs/orca.md) | Architecture, loading, training |

## Examples

| File | Description |
|------|-------------|
| `examples/play_random.py` | Heuristic vs random |
| `examples/custom_eval.py` | Hand-tuned evaluation |
| `examples/evolutionary.py` | Evolve scoring weights |
| `examples/train_bot.py` | Train AlphaZero bot |
| `examples/bot_vs_bot.py` | Compare bot types |
| `examples/play_orca.py` | Play against Orca |
| `examples/train_orca.py` | Quick training demo |
| `examples/dashboard_arena.py` | Dashboard with arena |
| `examples/dashboard_train.py` | Dashboard with training |
| `examples/dashboard_custom_bot.py` | Custom bot + dashboard |
| `test_dashboard.py` | Evolving bot demo |

## Performance

| Operation | Throughput |
|-----------|-----------|
| Random game playouts | ~30,000 games/sec |
| C engine move scoring | ~100,000 positions/sec |
| Alpha-beta search (depth 8) | ~50,000 nodes/sec |
| NN inference (batch=64) | ~2,300 positions/sec |
| Self-play (200 sims) | ~500 games/hour |

## Contributing

Built for the Hexagonal Tic-Tac-Toe community. Contributions welcome.

## License

MIT
