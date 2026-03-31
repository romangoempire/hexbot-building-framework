# API Reference

Complete reference for all public classes, methods, and functions in the hexbot framework.

See also: [Getting Started](getting-started.md) | [Bot Approaches](bot-approaches.md) | [Dashboard Guide](dashboard-guide.md)

---

## HexGame

The core game engine. Wraps a fast C library that auto-compiles on first import.

### Construction

```python
from hexbot import HexGame

game = HexGame()                          # empty board
game = HexGame(max_stones=300)            # custom max stones
game = HexGame.from_moves([(0,0), ...])   # replay a move sequence
game = HexGame.from_dict(data)            # reconstruct from serialized dict
game = HexGame.triangle()                 # pre-built triangle opening
```

### Actions

| Method | Description |
|--------|-------------|
| `game.place(q, r)` | Place a stone at `(q, r)`. Auto-advances the turn. |
| `game.undo()` | Undo the last placement. Restores all state including Zobrist hash. |
| `game.clone()` | Return an independent deep copy of the game. |

### State Properties

| Property | Type | Description |
|----------|------|-------------|
| `game.current_player` | `int` | `0` or `1` |
| `game.winner` | `int or None` | `None`, `0`, or `1` |
| `game.is_over` | `bool` | `True` if someone won or max stones reached |
| `game.total_stones` | `int` | Number of stones on the board |
| `game.stones_this_turn` | `int` | `0` or `1` -- how many placed so far this turn |
| `game.stones_per_turn` | `int` | `1` (first turn) or `2` |
| `game.moves` | `list[(q, r)]` | All moves in play order |
| `game.zhash` | `int` | 64-bit Zobrist hash of the current position |

### Move Generation

| Method | Returns | Description |
|--------|---------|-------------|
| `game.legal_moves()` | `[(q, r), ...]` | All legal positions |
| `game.scored_moves(limit=20)` | `[(q, r, score), ...]` | Top N moves ranked by C heuristic |
| `game.forcing_moves()` | `[(q, r), ...]` | Moves creating or blocking 4+ in a row |

### Threat Detection

| Method | Returns | Description |
|--------|---------|-------------|
| `game.has_winning_move(player)` | `bool` | Can `player` win in 1 stone? |
| `game.count_winning_moves(player)` | `int` | Number of instant-win cells |
| `game.max_line(q, r, player)` | `int` | Longest line through `(q, r)` for `player` |
| `game.would_win(q, r, player)` | `bool` | Would placing at `(q, r)` win? |

### Search

```python
result = game.search(depth=8)
```

Returns a dict:

| Key | Type | Description |
|-----|------|-------------|
| `'best_move'` | `(q, r)` | Best move found |
| `'value'` | `float` | Evaluation from -1 to +1 (P0's perspective) |
| `'nodes'` | `int` | Number of nodes searched |

Uses transposition tables, killer heuristics, and late move reduction.

### Serialization

| Method | Returns | Description |
|--------|---------|-------------|
| `game.to_dict()` | `dict` | JSON-compatible dict with moves and settings |
| `HexGame.from_dict(data)` | `HexGame` | Reconstruct from dict |

### Display

```python
print(game)   # ASCII hex board with P0/P1 markers
repr(game)    # "HexGame(5 stones, P1 to move)"
```

---

## Analysis Functions

Standalone functions for move evaluation and tactical analysis.

```python
from hexbot import (evaluate_moves, find_threats, find_winning_moves,
                     count_lines, rollout, alphabeta)
```

### evaluate_moves(game, top_n=10)

Score moves using the C heuristic (no neural network).

**Returns:** `[((q, r), score), ...]` sorted by score descending.

### find_threats(game, player=None)

Find cells where placing a stone gives 4 or more in a row.

**Parameters:**
- `player` -- which player to check. Defaults to current player.

**Returns:** `[(q, r), ...]`

### find_winning_moves(game, player=None)

Find cells that complete 6 in a row (instant win).

**Parameters:**
- `player` -- which player to check. Defaults to current player.

**Returns:** `[(q, r), ...]`

### count_lines(game, q, r, player=None)

Analyze lines through a specific cell.

**Returns:** `{'max_line': int}`

### rollout(game, num_games=1000)

Run random playouts from the current position to estimate win rates.

**Returns:**

| Key | Type | Description |
|-----|------|-------------|
| `'p0_wins'` | `float` | Player 0 win rate (0.0 to 1.0) |
| `'p1_wins'` | `float` | Player 1 win rate (0.0 to 1.0) |
| `'draw_rate'` | `float` | Draw rate (0.0 to 1.0) |

### alphabeta(game, depth=8)

Run alpha-beta search with the C engine.

**Returns:**

| Key | Type | Description |
|-----|------|-------------|
| `'best_move'` | `(q, r)` | Best move found |
| `'value'` | `float` | Evaluation from -1 to +1 |
| `'nodes'` | `int` | Nodes searched |

---

## Bot

Bot constructors and methods for creating players.

```python
from hexbot import Bot, BotProtocol
```

### Built-in Constructors

| Constructor | Description |
|-------------|-------------|
| `Bot.random()` | Plays random legal moves |
| `Bot.heuristic()` | Uses C engine scoring (no neural network) |
| `Bot.load('model.pt')` | Loads a trained neural network from file |
| `Bot(sims=400)` | Neural network bot with MCTS (random initial weights) |

### Bot Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `bot.best_move(game)` | `(q, r)` | Choose the best move for the current position |
| `bot.policy(game)` | `{(q,r): float}` | Move probability distribution |
| `bot.evaluate(game)` | `float` | Position evaluation from -1 to +1 |
| `bot.save('model.pt')` | `None` | Save model to file |

### Custom Bot via Subclass

```python
class MyBot(BotProtocol):
    def best_move(self, game):
        return game.legal_moves()[0]
```

### Custom Bot via Function

```python
def my_func(game):
    return game.legal_moves()[0]
```

Any function `(HexGame) -> (q, r)` works anywhere a bot is expected.

---

## Arena

Run matches between any two bots.

```python
from hexbot import Arena
```

### Construction and Play

```python
# Accepts Bot instances, BotProtocol subclasses, or plain functions
result = Arena(bot1, bot2, num_games=100).play()
result = Arena(my_func, Bot.heuristic(), num_games=50).play(verbose=False)
```

### Result Fields

| Field | Type | Description |
|-------|------|-------------|
| `result.wins` | `[int, int]` | Wins for bot1 and bot2 |
| `result.draws` | `int` | Number of draws |
| `result.total_games` | `int` | Total games played |
| `result.avg_length` | `float` | Average moves per game |
| `result.win_rate` | `[float, float]` | Win rates for bot1 and bot2 |
| `result.games` | `list[dict]` | Per-game detail dicts |

---

## train()

Train a neural network bot via self-play (requires PyTorch).

```python
from hexbot import train

bot = train(
    iterations=100,
    games_per_iter=50,
    sims=200,
    lr=0.001,
    network_config='standard',
    checkpoint_every=10,
    checkpoint_prefix='mybot',
    on_iteration=callback,
)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `iterations` | `100` | Number of training cycles |
| `games_per_iter` | `50` | Self-play games per cycle |
| `sims` | `200` | MCTS simulations per move |
| `lr` | `0.001` | Learning rate |
| `network_config` | `'standard'` | `'fast'` (700K), `'standard'` (3.9M), or `'large'` (14.5M) |
| `checkpoint_every` | `10` | Save checkpoint every N iterations |
| `checkpoint_prefix` | `'mybot'` | Filename prefix for checkpoints |
| `on_iteration` | `None` | Optional callback `(iter, losses)` called after each iteration |

**Returns:** A trained `Bot` instance.

GPU is auto-detected (CUDA or MPS on Apple Silicon). Game simulation always runs on CPU via the C engine.

---

## Advanced Threat Analysis

```python
find_forced_move(game) -> Optional[(q, r)]
```
Find undeniable forced move (instant win or must-block). Returns None if no forced move. No neural network needed.

```python
threat_search(game, depth=4) -> Optional[(q, r)]
```
Search for winning threat sequences (forks, double attacks) up to `depth` moves ahead. Returns the first move of the winning sequence.

```python
count_threats(game, player=None) -> int
```
Count cells where player can complete 6-in-a-row. If 3+, position is likely unstoppable.

```python
detect_fork(game, player=None) -> bool
```
Check if player has 3+ winning cells (unstoppable fork).

---

## Neural Network Access

```python
create_network(config='standard') -> nn.Module
```
Create a neural network. Configs: `'fast'` (500K params), `'standard'` (3.9M), `'large'` (15M), `'orca-transformer'` (4.3M, experimental).

```python
encode_state(game) -> (tensor, offset_q, offset_r)
```
Encode game as (7, 19, 19) tensor. Channels: stones, legal moves, player, turn, threat maps.

```python
decode_policy(logits, game, offset_q, offset_r) -> Dict[(q,r), float]
```
Convert 361-dim policy logits back to move probabilities.

```python
nn_evaluate(game, net=None) -> (policy_dict, value)
```
Evaluate position with neural network. Returns move probabilities and value in [-1, +1].

```python
nn_evaluate_batch(games, net=None) -> List[(policy_dict, value)]
```
Batch evaluate multiple positions (much faster than calling nn_evaluate in a loop).

---

## MCTS Search

```python
mcts_search(game, net=None, sims=200, batch_size=64) -> Dict[(q,r), float]
```
Full MCTS search returning visit distribution.

```python
mcts_policy(game, net=None, sims=200, temperature=1.0, add_noise=False) -> Dict[(q,r), float]
```
MCTS with temperature-based move selection. `temperature=1.0` for training, `0.01` for play.

---

## Training Building Blocks

```python
FastGame(max_stones=200) -> CGameState
```
Fast C-engine game state (~10x faster than HexGame for self-play).

```python
self_play(net=None, sims=200, batch_size=64) -> (List[TrainingSample], List[(q,r)])
```
Generate one game of self-play training data.

```python
train_step(net, optimizer, replay_buffer, device=None) -> Dict[str, float]
```
One gradient step. Returns `{'total', 'value', 'policy', 'threat'}` losses.

```python
augment_sample(sample) -> List[TrainingSample]
```
Hex-valid symmetry augmentation. Returns 3 augmented copies.

```python
TrainingSample  # dataclass: encoded_state, policy_target, result, threat_label, priority
ReplayBuffer    # prioritized experience replay: .push(sample), .sample(batch_size)
```

---

## Data & Curriculum

```python
load_games(path) -> List[TrainingSample]
```
Load training samples from JSONL game file.

```python
generate_puzzles(n=100) -> List[(position_dict, hint_moves)]
```
Generate random tactical puzzle positions.

```python
positions  # dict of pre-built starting positions for curriculum training
```

---

## SFT Pipeline (v4)

```python
from orca.sft import sft_train, import_games
```

### sft_train(net, games_path, epochs=5, lr=1e-3, device=None) -> nn.Module
Supervised fine-tuning on expert game data. Returns the fine-tuned network.

### import_games(path, format='sgf') -> List[TrainingSample]
Import games from SGF/JSONL files into training samples.

### Game Scraper

```python
from orca.sft import scrape_games
scrape_games(source='littlegolem', output='games.jsonl', limit=1000)
```

---

## Endgame Solver (v4)

```python
from orca.solver import solve, quick_solve, solver_or_mcts, TranspositionCache
```

### solve(game, max_depth=None) -> dict
Exact endgame solver. Returns `{'winner': int, 'pv': [(q,r), ...], 'nodes': int}`.

### quick_solve(game, time_limit=1.0) -> Optional[dict]
Time-limited solve attempt. Returns `None` if not solved in time.

### solver_or_mcts(game, net, sims=200) -> (q, r)
Use solver when position is solvable, fall back to MCTS otherwise.

### TranspositionCache(max_size=2**20)
Shared transposition table for solver. Methods: `.get(zhash)`, `.put(zhash, result)`, `.hit_rate()`.

---

## Opening Book (v4)

```python
from orca.openings import OpeningBook
```

### OpeningBook(path='openings.db')
Persistent opening book backed by SQLite.

| Method | Description |
|--------|-------------|
| `OpeningBook.build_from_games(games_path)` | Build book from game archive |
| `book.lookup(game) -> Dict[(q,r), float]` | Look up position, returns move weights |
| `book.blend(game, nn_policy, weight=0.3) -> Dict[(q,r), float]` | Blend book moves with NN policy |

```python
book = OpeningBook('openings.db')
book.build_from_games('games.jsonl')
policy = book.blend(game, nn_policy, weight=0.3)
```

---

## Distributed Training (v4)

```python
from orca.distributed import SelfPlayPool, MultiGPUTrainer, RayTrainer
```

### SelfPlayPool(num_workers=8, net=None)
Process pool for parallel self-play. Methods: `.generate(num_games)`, `.shutdown()`.

### MultiGPUTrainer(net, device_ids=None)
DataParallel training across multiple GPUs. Methods: `.train_step(batch)`, `.save()`.

### RayTrainer(net_config='standard', num_actors=16)
Ray-based distributed training with remote actors. Methods: `.run(iterations)`, `.status()`.

```python
pool = SelfPlayPool(num_workers=8)
samples = pool.generate(num_games=100)
```

---

## Skill Curriculum (v4)

```python
from orca.curriculum import SkillCurriculum
```

### SkillCurriculum(start_level=0)
6-level curriculum that progressively introduces harder training conditions.

| Level | Sims | Games/iter | Description |
|-------|------|------------|-------------|
| 0 | 30 | 80 | Random openings |
| 1 | 50 | 60 | Basic tactics |
| 2 | 100 | 50 | Threat patterns |
| 3 | 150 | 40 | Positional play |
| 4 | 200 | 30 | Complex endgames |
| 5 | 400 | 20 | Full strength |

Methods: `.current_level()`, `.advance()`, `.settings() -> (sims, games)`.

---

## Ensemble (v4)

```python
from orca.ensemble import Ensemble
```

### Ensemble.from_latest(n=3) -> Ensemble
Load ensemble from the N most recent checkpoints.

### ensemble.evaluate(game) -> (policy, value, uncertainty)
Evaluate with uncertainty estimation. `uncertainty` is the stddev across members.

### ensemble.best_move(game) -> (q, r)
Pick the best move by averaged policy.

---

## Model Zoo (v4)

```python
from orca.zoo import Zoo
```

| Method | Description |
|--------|-------------|
| `Zoo.list() -> List[dict]` | List available models with metadata |
| `Zoo.download(name) -> Path` | Download a model by name |
| `Zoo.package(net, name, metadata)` | Package and upload a model |
| `Zoo.load(name) -> nn.Module` | Download and load a model |

```python
Zoo.list()  # [{'name': 'orca-v4-std', 'elo': 1850, ...}]
net = Zoo.load('orca-v4-std')
```

---

## Leaderboard (v4)

```python
from orca.leaderboard import rate, compare, show
```

### rate(bot, opponents, num_games=50) -> float
Rate a bot's ELO against a set of opponents.

### compare(bot_a, bot_b, num_games=100) -> dict
Head-to-head comparison. Returns `{'wins_a', 'wins_b', 'draws', 'elo_diff'}`.

### show() -> DataFrame
Display the current leaderboard as a pandas DataFrame.

---

## Plugin System (v4)

```python
from hexbot import register_bot, register_network
```

### register_bot(name, cls_or_fn)
Register a custom bot that appears in Arena and the dashboard.

### register_network(name, factory_fn)
Register a custom network architecture for `create_network(name)`.

```python
register_bot('my-bot', MyBot)
register_network('my-net', lambda: MyNetwork())
```

---

## Installation (v4)

```bash
pip install hexbot
```

The package is defined in `pyproject.toml` and includes all dependencies.

---

## Next Steps

- [Getting Started](getting-started.md) - learn the HexGame API with examples
- [Bot Approaches](bot-approaches.md) - six strategies with full code
- [Training Guide](training-guide.md) - full training pipeline documentation
- [Configuration](configuration.md) - all tunable parameters
- [Dashboard Guide](dashboard-guide.md) - visualize training and arena matches
