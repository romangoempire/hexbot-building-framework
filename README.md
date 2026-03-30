# hexbot - Build AI Bots for Hexagonal Connect-6

A Python framework for building bots for [Hexagonal Tic-Tac-Toe](https://hexo.did.science) (Connect-6 on an infinite hex grid). Provides a fast C game engine, analysis tools, and optional neural network training - use any AI approach you want.

---

## Table of Contents

- [Installation](#installation)
- [Game Rules](#game-rules)
- [Quick Start](#quick-start)
- [Training Dashboard](#training-dashboard)
- [Building Your First Bot](#building-your-first-bot)
- [Six Bot Approaches (End-to-End)](#six-bot-approaches-end-to-end)
  - [1. Hand-Tuned Evaluation](#approach-1-hand-tuned-evaluation)
  - [2. Evolutionary Weights](#approach-2-evolutionary-weights)
  - [3. Minimax with Alpha-Beta](#approach-3-minimax-with-alpha-beta)
  - [4. Monte Carlo Playouts](#approach-4-monte-carlo-random-playouts)
  - [5. Neural Network (AlphaZero)](#approach-5-neural-network-alphazero-style)
  - [6. Hybrid Strategies](#approach-6-combine-multiple-strategies)
- [API Reference](#api-reference)
- [Performance Benchmarks](#performance-benchmarks)
- [Architecture](#architecture)
- [FAQ](#faq)

---

## Installation

```bash
git clone https://github.com/Saiki77/hexbot.git
cd hexbot
pip install torch  # only needed for neural network bots
```

The C engine auto-compiles on first import. Requires a C compiler (`cc`/`gcc`/`clang`). On macOS this comes with Xcode Command Line Tools (`xcode-select --install`). On Linux: `apt install build-essential`.

To verify everything works:

```bash
python hexbot.py    # runs self-test
python benchmark.py # runs performance benchmarks
```

---


## Game Rules

Hexagonal Connect-6 is played on an infinite hexagonal grid using axial coordinates `(q, r)`.

| Rule | Detail |
|------|--------|
| **Board** | Infinite hexagonal grid |
| **Players** | 2 players (Player 0 and Player 1), alternating turns |
| **First turn** | Player 0 places 1 stone at any position |
| **Subsequent turns** | Each player places 2 stones per turn |
| **Win condition** | 6 consecutive stones in a straight line |
| **Line directions** | Three axes: (1,0) horizontal, (0,1) vertical, (1,-1) diagonal |
| **Placement** | Any empty hex within range of existing stones |

The coordinate system uses axial coordinates where `q` runs along the horizontal axis and `r` runs diagonally. Every hex has six neighbors. The third implicit coordinate `s = -q - r` completes the cube coordinate system.

---


## Quick Start

### Play a Game

```python
from hexbot import HexGame

game = HexGame()
game.place(0, 0)       # Player 0 places at center (1 stone on first turn)
game.place(1, 0)       # Player 1 places first stone
game.place(1, -1)      # Player 1 places second stone (2 per turn after first)

print(game)
#   P0 to move | 3 stones
# . . . .
#  . . O .
#   . X O .
#    . . . .

print(game.current_player)   # 0
print(game.winner)           # None
print(game.legal_moves()[:5]) # [(2,-2), (2,-1), ...]
```

### Working with the Game

Understanding the game API is the foundation for any bot. Here is everything you need to know.

#### Placing Stones and Turn Structure

The game alternates between two players. Player 0 goes first and places only 1 stone on their first turn. After that, every turn is 2 stones.

```python
from hexbot import HexGame

game = HexGame()

# Turn 0: Player 0 places 1 stone
game.place(0, 0)
print(game.current_player)    # 1 (turn switches after 1 stone)

# Turn 1: Player 1 places 2 stones
game.place(2, 0)
print(game.current_player)    # 1 (still P1, needs one more)
game.place(2, -1)
print(game.current_player)    # 0 (P1 done, back to P0)

# Turn 2: Player 0 places 2 stones
game.place(1, 0)
game.place(0, 1)
print(game.current_player)    # 1
print(game.total_stones)      # 5
```

The `stones_this_turn` and `stones_per_turn` properties tell you where you are in the current turn. This matters for bots because the two stones in a turn should work together - the first stone sets up the second.

```python
print(game.stones_this_turn)  # 0 (haven't placed yet this turn)
print(game.stones_per_turn)   # 2 (need to place 2)
game.place(3, 0)
print(game.stones_this_turn)  # 1 (placed 1 of 2)
```

#### Exploring Moves Without Committing

The `place()` and `undo()` pattern lets you try moves and take them back. This is essential for search algorithms - you explore a move, evaluate the resulting position, then undo to try the next move. No board copying needed.

```python
game = HexGame()
game.place(0, 0)

# Try a move
game.place(1, 0)
print(game.total_stones)      # 2
score_a = game.scored_moves(1)[0][2]

# Undo and try a different move
game.undo()
print(game.total_stones)      # 1

game.place(0, 1)
score_b = game.scored_moves(1)[0][2]
game.undo()

print(f"Move A scored {score_a}, Move B scored {score_b}")
```

This is extremely fast - 1.4 million place/undo cycles per second on M4 Pro. The C engine stores undo information on a stack, so there is no allocation or garbage collection.

#### Reading the Board

The game provides several ways to understand the current position:

```python
game = HexGame()
for q, r in [(0,0), (3,0), (3,-1), (1,0), (2,0)]:
    game.place(q, r)

# Who is winning?
print(game.winner)             # None (game not over)
print(game.is_over)            # False

# What moves are available?
moves = game.legal_moves()     # all legal positions
print(f"{len(moves)} legal moves")

# Which moves are best? (C heuristic scoring)
for q, r, score in game.scored_moves(5):
    print(f"  ({q},{r}) score={score}")

# Where are the threats?
print(f"P0 can win: {game.has_winning_move(0)}")
print(f"P1 can win: {game.has_winning_move(1)}")
print(f"Winning cells for P0: {game.count_winning_moves(0)}")
```

#### Analyzing Specific Cells

You can analyze any cell without placing a stone there. This is useful for evaluating candidate moves:

```python
# How long is the line through (3,0) for Player 0?
line_len = game.max_line(3, 0, player=0)
print(f"Line through (3,0): {line_len}")  # e.g. 3

# Would placing at (4,0) win for Player 0?
would_win = game.would_win(4, 0, player=0)
print(f"(4,0) wins: {would_win}")
```

These functions check what would happen if a stone were placed, without actually placing it. Combined with `evaluate_moves()`, they give you all the raw data needed to build any evaluation function.

#### Deep Search

For tactical analysis, the built-in alpha-beta search looks several turns ahead:

```python
result = game.search(depth=8)  # depth 8 = 4 full turns ahead
print(f"Best move: {result['best_move']}")
print(f"Evaluation: {result['value']}")   # -1 to +1
print(f"Nodes searched: {result['nodes']}")
```

The search uses the C engine with transposition tables, killer heuristics, and late move reduction. At depth 8, it searches ~250K positions in about 1 second and can spot forced wins and forced losses that are invisible to simpler evaluation.

#### Cloning and Serialization

When you need independent copies of a game (for parallel evaluation or saving state):

```python
# Clone: independent deep copy
copy = game.clone()
copy.place(5, 0)             # doesn't affect original
print(game.total_stones)      # unchanged

# Serialize to dict (JSON-compatible)
data = game.to_dict()
print(data)  # {'moves': [(0,0), (3,0), ...], 'max_stones': 200}

# Reconstruct from dict
game2 = HexGame.from_dict(data)

# Reconstruct from move list
game3 = HexGame.from_moves([(0,0), (1,0), (1,-1)])
```

#### The Zobrist Hash

Every position has a unique 64-bit hash that changes incrementally as stones are placed. This is useful for transposition tables and position caching in your own search algorithms:

```python
game = HexGame()
game.place(0, 0)
hash_a = game.zhash

game.place(1, 0)
game.place(1, -1)
hash_b = game.zhash   # different position, different hash

game.undo()
game.undo()
assert game.zhash == hash_a  # undo restores the hash exactly
```

Two games with identical stone placements in different order may produce the same hash (transposition). This property is what makes transposition tables work - you can detect when two different move sequences reach the same position and reuse the evaluation.

#### Putting It All Together: A Manual Game

```python
from hexbot import HexGame, evaluate_moves, find_winning_moves

game = HexGame()

# Opening: Player 0 places center
game.place(0, 0)

# Player 1 responds
game.place(1, 0)
game.place(1, -1)

# Player 0 builds a line
game.place(-1, 1)
game.place(0, 1)

# Check the position
print(game)
print(f"Player {game.current_player} to move")
print(f"Top moves: {evaluate_moves(game, 3)}")
print(f"P0 winning moves: {find_winning_moves(game, 0)}")
print(f"P1 winning moves: {find_winning_moves(game, 1)}")

# Look ahead with alpha-beta
result = game.search(depth=6)
print(f"Best move: {result['best_move']}, eval: {result['value']:.2f}")
```

### Build a Bot in 10 Lines

```python
from hexbot import HexGame, evaluate_moves, find_winning_moves

def my_bot(game):
    # Win if we can
    wins = find_winning_moves(game)
    if wins:
        return wins[0]
    # Otherwise pick the highest-scored move
    moves = evaluate_moves(game, top_n=1)
    return moves[0][0] if moves else game.legal_moves()[0]
```

### Test Your Bot

```python
from hexbot import Arena, Bot

result = Arena(my_bot, Bot.heuristic(), num_games=100).play()
print(f"Your bot: {result.wins[0]}W, Heuristic: {result.wins[1]}W")
```

Any function that takes a game and returns a `(q, r)` move works as a bot. You can also create a class with a `best_move(game)` method.

---

## Training Dashboard

A live training dashboard that visualizes games, tracks ELO, loss curves, and more. Works with any bot -the framework's built-in bots or your own.

### Quick Start

```bash
python test_dashboard.py       # real bot games on port 5002
```

### Arena (2 lines)

Pit any two bots against each other. ELO is computed automatically from results.

```python
from hexbot import Bot
from dashboard import Dashboard

dash = Dashboard(port=5001)
dash.start()

# One line - dashboard handles everything: games, ELO, charts
dash.run_arena(Bot.heuristic(), Bot.random(), games=100)
```

### Training Loop (auto-ELO via snapshots)

The dashboard stores snapshots of your bot over time and automatically plays the current version against past versions to compute ELO. You don't have to do anything.

```python
from hexbot import Bot
from dashboard import Dashboard

dash = Dashboard(port=5001)
dash.start()

# Dashboard runs self-play, snapshots the bot, auto-computes ELO
dash.train(Bot.heuristic(), iterations=50, games_per_iter=20)
```

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bot` | required | function(game)->(q,r) or object with best_move(game) |
| `iterations` | 100 | Number of training iterations |
| `games_per_iter` | 20 | Self-play games per iteration |
| `opponent` | None | Opponent for self-play (default: bot plays itself) |
| `eval_every` | 5 | Run ELO evaluation every N iterations |
| `eval_games` | 10 | Games per ELO evaluation |
| `snapshot_every` | 3 | Snapshot bot every N iterations for ELO |

### Use with Your Own Bot

Any function that takes a game and returns a move works:

```python
from dashboard import Dashboard

dash = Dashboard(port=5001)
dash.start()

def my_bot(game):
    # your logic here
    return (0, 0)

dash.train(my_bot, iterations=50)  # auto-ELO, auto-charts, auto-everything
```

### Manual Control

For full control, push games and metrics yourself. ELO is still auto-computed from win rates.

```python
dash.add_game(moves=[[0,0],[1,0]], result=1.0)   # ELO auto-updates
dash.add_metric(iteration=1, loss=0.5)             # charts auto-update
dash.update_progress(step=50, total=100)           # progress bar
```

### REST API

Push data from any language or process via HTTP:

```bash
# Submit a game
curl -X POST http://localhost:5001/api/game \
  -H 'Content-Type: application/json' \
  -d '{"moves": [[0,0],[1,0],[0,1],[2,0]], "result": 1.0}'

# Submit metrics
curl -X POST http://localhost:5001/api/metric \
  -H 'Content-Type: application/json' \
  -d '{"iteration": 5, "loss": {"total": 0.82}, "elo": 1100}'

# Read stats
curl http://localhost:5001/api/stats
curl http://localhost:5001/api/elo
curl http://localhost:5001/api/games
```

### WebSocket API

Connect via Socket.IO for real-time streaming:

```javascript
const socket = io('http://localhost:5001');

// Send a game result
socket.emit('game_result', {
  moves: [[0,0],[1,0],[0,1]],
  result: 1.0
});

// Send metrics
socket.emit('metric', {
  iteration: 5,
  loss: { total: 0.82 },
  elo: 1100
});

// Listen for updates
socket.on('game_complete', d => console.log('Game:', d));
socket.on('stats_update', d => console.log('Stats:', d));
```

### Game Viewer

The left panel shows an animated replay of training games with numbered moves (black hexagons for P0, white hatched hexagons for P1). Gray dots show empty hex positions around the stones.

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `Space` | Pause / resume auto-advance |
| `Right arrow` | Step forward one move (pauses auto-advance) |
| `Left arrow` | Step backward one move |
| `R` | Restart current game from the beginning |

When you use arrow keys to step through moves, auto-advance pauses so you can analyze the position. Press Space to resume.

The game history bar below the board shows recent games. Click any game to re-watch it.

### Settings

Click the gear icon in the header to adjust:

| Setting | Default | Description |
|---------|---------|-------------|
| Replay speed | 120ms | Speed of game replay animation |
| Dot size | 2 | Size of empty hex grid dots |
| Grid radius | 2 | How many empty hexes shown around stones |
| Move numbers | On | Show move order numbers on stones |
| Auto-refresh | On | Periodically refresh charts |

All settings are saved to your browser's localStorage.

### Charts

The right panel has 6 collapsible chart panels. Click any header to collapse/expand:

- **ELO Progression** - rating over iterations
- **Loss Curves** - total, value, and policy loss
- **Win Rates** - P0 vs P1 win percentages
- **Game Length** - average moves per game
- **Training Speed** - games per second
- **Resources** - CPU and RAM usage

### Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/stats` | Current aggregate stats |
| `GET` | `/api/elo` | ELO history array |
| `GET` | `/api/losses` | Loss history array |
| `GET` | `/api/games` | Recent 50 game move histories |
| `GET` | `/api/resources` | CPU/RAM history |
| `GET` | `/api/winrates` | Win rate history |
| `GET` | `/api/gamelength` | Game length history |
| `GET` | `/api/speed` | Training speed history |
| `POST` | `/api/game` | Submit a completed game |
| `POST` | `/api/metric` | Submit training metrics |

---

## Building Your First Bot

The simplest way to build a bot is a function that takes a `HexGame` and returns a move `(q, r)`:

```python
from hexbot import HexGame

def my_first_bot(game):
    """Pick the first legal move."""
    return game.legal_moves()[0]
```

This bot is terrible - it always picks the same move without any strategy. Let's improve it step by step.

### Step 1: Use Heuristic Scoring

The C engine scores every legal move based on how much it extends lines and blocks the opponent:

```python
from hexbot import evaluate_moves

def better_bot(game):
    moves = evaluate_moves(game, top_n=1)
    return moves[0][0] if moves else game.legal_moves()[0]
```

This bot already beats random play consistently because it prefers moves that build longer lines.

### Step 2: Add Win/Block Detection

Always take a winning move if available, and always block the opponent's winning move:

```python
from hexbot import evaluate_moves, find_winning_moves

def good_bot(game):
    # Take the win
    wins = find_winning_moves(game, game.current_player)
    if wins:
        return wins[0]

    # Block opponent's win
    blocks = find_winning_moves(game, 1 - game.current_player)
    if blocks:
        return blocks[0]

    # Best heuristic move
    moves = evaluate_moves(game, top_n=1)
    return moves[0][0] if moves else game.legal_moves()[0]
```

### Step 3: Add Threat Awareness

Look for moves that create lines of 4 or more (threats that must be answered):

```python
from hexbot import evaluate_moves, find_winning_moves, find_threats

def strong_bot(game):
    player = game.current_player
    opponent = 1 - player

    wins = find_winning_moves(game, player)
    if wins: return wins[0]

    blocks = find_winning_moves(game, opponent)
    if blocks: return blocks[0]

    # Prefer moves that create threats
    threats = find_threats(game, player)
    if threats: return threats[0]

    moves = evaluate_moves(game, top_n=1)
    return moves[0][0] if moves else game.legal_moves()[0]
```

### Step 4: Test Against Built-in Bots

```python
from hexbot import Arena, Bot

# Test against random
result = Arena(strong_bot, Bot.random(), num_games=50).play()
print(f"vs Random: {result.wins[0]}W-{result.wins[1]}L")

# Test against heuristic
result = Arena(strong_bot, Bot.heuristic(), num_games=50).play()
print(f"vs Heuristic: {result.wins[0]}W-{result.wins[1]}L")
```

---

## Six Bot Approaches (End-to-End)

### Approach 1: Hand-Tuned Evaluation

The simplest approach - score each candidate move using domain knowledge and pick the best one. No machine learning needed. Fast to iterate and easy to understand.

```python
from hexbot import (HexGame, Arena, Bot, evaluate_moves,
                     find_threats, find_winning_moves)

def smart_bot(game):
    player = game.current_player
    opponent = 1 - player

    # Priority 1: Win immediately
    wins = find_winning_moves(game, player)
    if wins:
        return wins[0]

    # Priority 2: Block opponent's winning move
    blocks = find_winning_moves(game, opponent)
    if blocks:
        return blocks[0]

    # Priority 3: Extend our longest threats
    threats = find_threats(game, player)
    if threats:
        return threats[0]

    # Priority 4: Block opponent's threats
    opp_threats = find_threats(game, opponent)
    if opp_threats:
        return opp_threats[0]

    # Priority 5: Best positional move
    moves = evaluate_moves(game, top_n=1)
    return moves[0][0] if moves else game.legal_moves()[0]

# Test it
result = Arena(smart_bot, Bot.heuristic(), num_games=50).play()
```

To make this stronger: add preemptive awareness (2-in-a-row setups), colony detection (distant stone groups), and formation recognition (triangles, rhombuses).

### Approach 2: Evolutionary Weights

Instead of hand-tuning priorities, let evolution discover them. Create a population of bots with random scoring weights, play tournaments, and breed the winners.

```python
import random
from hexbot import Arena, evaluate_moves, find_winning_moves

class EvoBot:
    def __init__(self, weights=None):
        self.weights = weights or {
            'line_score': random.uniform(0.5, 2.0),
            'center_pull': random.uniform(0.0, 1.0),
            'threat_weight': random.uniform(1.0, 5.0),
            'block_weight': random.uniform(1.0, 5.0),
        }

    def best_move(self, game):
        # Always take wins/blocks
        for p in [game.current_player, 1 - game.current_player]:
            wins = find_winning_moves(game, p)
            if wins: return wins[0]

        # Score moves with evolved weights
        best, best_score = game.legal_moves()[0], -999
        for (q, r), base in evaluate_moves(game, 15):
            score = base * self.weights['line_score']
            score -= (abs(q) + abs(r)) * self.weights['center_pull']

            my_line = game.max_line(q, r, game.current_player)
            opp_line = game.max_line(q, r, 1 - game.current_player)
            if my_line >= 4: score += my_line * self.weights['threat_weight']
            if opp_line >= 4: score += opp_line * self.weights['block_weight']

            if score > best_score:
                best_score, best = score, (q, r)
        return best

    def mutate(self):
        child = EvoBot(dict(self.weights))
        for k in child.weights:
            if random.random() < 0.3:
                child.weights[k] *= random.uniform(0.7, 1.3)
        return child

# Evolution loop
POP = 8
population = [EvoBot() for _ in range(POP)]

for gen in range(10):
    scores = [0] * POP
    for i in range(POP):
        for j in range(i+1, POP):
            r = Arena(population[i], population[j], num_games=6).play(verbose=False)
            scores[i] += r.wins[0]
            scores[j] += r.wins[1]

    ranked = sorted(range(POP), key=lambda i: scores[i], reverse=True)
    survivors = [population[i] for i in ranked[:POP//2]]
    population = survivors + [s.mutate() for s in survivors]

    best = survivors[0]
    print(f"Gen {gen+1}: score={scores[ranked[0]]}, weights={best.weights}")
```

After 10 generations the winning weights typically converge to: high threat weight (4+), moderate line score (1.0-1.5), low center pull (0.1-0.3), and high block weight (3+). This matches human intuition - threats and blocks matter most.

### Approach 3: Minimax with Alpha-Beta

Use depth-first search with the built-in C engine for deep tactical analysis. The C engine includes transposition tables, killer move heuristics, and late move reduction.

```python
from hexbot import HexGame, alphabeta, Arena, Bot

def minimax_bot(game):
    result = alphabeta(game, depth=6)  # 3 full turns ahead
    return result['best_move']

# Alpha-beta returns:
# result['best_move']  - (q, r) best move
# result['value']      - evaluation (-1 to +1, from P0's perspective)
# result['nodes']      - nodes searched

result = Arena(minimax_bot, Bot.heuristic(), num_games=20).play()
```

For a custom minimax with your own evaluation function:

```python
def my_eval(game):
    """Custom position evaluation. Returns float (-1 to +1)."""
    p = game.current_player
    my_threats = len(find_threats(game, p))
    opp_threats = len(find_threats(game, 1 - p))
    return (my_threats - opp_threats) * 0.2

def my_minimax(game, depth):
    if depth == 0 or game.is_over:
        if game.winner == 0: return 1.0, None
        if game.winner == 1: return -1.0, None
        return my_eval(game), None

    best_val = -2.0
    best_move = None
    for q, r, _ in game.scored_moves(12):  # top 12 candidates
        game.place(q, r)
        val, _ = my_minimax(game, depth - 1)
        val = -val  # negamax
        game.undo()
        if val > best_val:
            best_val, best_move = val, (q, r)
    return best_val, best_move
```

The `game.place()` / `game.undo()` pattern lets you explore the game tree without copying the board. This is the same pattern the C engine uses internally, and it is very fast (973K place/undo cycles per second).

### Approach 4: Monte Carlo Random Playouts

Estimate move quality by playing random games from each candidate. The C engine's fast random playout (33K stones/sec) makes this practical.

```python
from hexbot import HexGame, rollout, evaluate_moves, Arena, Bot

def monte_carlo_bot(game):
    best_move = game.legal_moves()[0]
    best_rate = -1

    for (q, r), _ in evaluate_moves(game, 8):  # top 8 candidates
        game.place(q, r)
        result = rollout(game, num_games=100)
        game.undo()

        # Win rate for current player
        p = game.current_player
        rate = result['p0_wins'] if p == 0 else result['p1_wins']
        if rate > best_rate:
            best_rate, best_move = rate, (q, r)

    return best_move

result = Arena(monte_carlo_bot, Bot.heuristic(), num_games=20).play()
```

The strength of this bot scales with the number of rollouts per move. 100 rollouts gives a rough estimate; 1000 gives reliable evaluations. The bottleneck is that random playouts don't understand strategy - they just explore randomly. To improve, you could bias the rollouts toward better moves using `scored_moves()`.

### Approach 5: Neural Network (AlphaZero-Style)

Train a neural network to evaluate positions and guide Monte Carlo Tree Search. This is the most powerful approach but requires PyTorch and compute time.

```python
from hexbot import Bot, train, Arena

# Train from scratch
bot = train(
    iterations=50,         # training cycles
    games_per_iter=20,     # self-play games per cycle
    sims=50,               # MCTS simulations per move
    network_config='fast', # small network, quick training
)
bot.save('my_nn_bot.pt')

# Test
result = Arena(bot, Bot.heuristic(), num_games=30).play()
```

The network architecture is a ResNet with separate policy (where to play), value (who's winning), and threat (tactical awareness) heads. Training works by having the network play against itself, generating training data from MCTS visit counts, then training on that data.

For longer training with a stronger network:

```python
bot = train(
    iterations=200,
    games_per_iter=50,
    sims=200,
    network_config='standard',  # 3.9M params, 128 filters, 12 res blocks
    lr=0.001,
    checkpoint_every=20,
)
```

### Approach 6: Combine Multiple Strategies

The strongest bots often combine approaches. Use fast tactical checks for obvious moves, deep search for complex positions, and heuristics as a fallback:

```python
from hexbot import (evaluate_moves, find_winning_moves, find_threats,
                     alphabeta, Bot, Arena)

def hybrid_bot(game):
    player = game.current_player

    # Layer 1: Instant tactical responses (microseconds)
    wins = find_winning_moves(game, player)
    if wins: return wins[0]
    blocks = find_winning_moves(game, 1 - player)
    if blocks: return blocks[0]

    # Layer 2: Deep search for critical positions (milliseconds)
    result = alphabeta(game, depth=4)
    if abs(result['value']) > 0.8:
        return result['best_move']

    # Layer 3: Threat-based play (microseconds)
    threats = find_threats(game, player)
    if threats: return threats[0]

    # Layer 4: Positional heuristic (microseconds)
    moves = evaluate_moves(game, 1)
    return moves[0][0] if moves else game.legal_moves()[0]

result = Arena(hybrid_bot, Bot.heuristic(), num_games=50).play()
```

---

## API Reference

### Game Engine

```python
from hexbot import HexGame

# Creation
game = HexGame()                          # empty board
game = HexGame(max_stones=300)            # custom max stones
game = HexGame.from_moves([(0,0), ...])   # replay a move sequence
game = HexGame.triangle()                 # pre-built triangle opening

# Actions
game.place(q, r)          # place stone (auto-advances turn)
game.undo()                # undo last placement
copy = game.clone()        # independent deep copy

# State queries
game.current_player        # 0 or 1
game.winner                # None, 0, or 1
game.is_over               # bool (win or max stones)
game.total_stones          # int
game.stones_this_turn      # 0 or 1
game.stones_per_turn       # 1 (first turn) or 2
game.moves                 # list of (q, r) in play order
game.zhash                 # 64-bit Zobrist hash

# Move generation
game.legal_moves()            # all legal (q, r) positions
game.scored_moves(limit=20)   # top N by C heuristic: [(q, r, score), ...]
game.forcing_moves()          # moves creating/blocking 4+ in a row

# Threat detection
game.has_winning_move(player)     # can player win in 1 stone?
game.count_winning_moves(player)  # how many instant-win cells?
game.max_line(q, r, player)       # longest line through (q, r)
game.would_win(q, r, player)      # would placing here win?

# Search
game.search(depth=8)              # C alpha-beta: {'best_move', 'value', 'nodes'}

# Serialization
data = game.to_dict()             # JSON-compatible dict
game = HexGame.from_dict(data)    # reconstruct from dict

# Display
print(game)                       # ASCII hex board
repr(game)                        # "HexGame(5 stones, P1 to move)"
```

### Analysis Functions

```python
from hexbot import (evaluate_moves, find_threats, find_winning_moves,
                     count_lines, rollout, alphabeta)

# Move scoring (C heuristic, no NN)
evaluate_moves(game, top_n=10)
# Returns: [((q, r), score), ...] sorted by score descending

# Threat detection
find_threats(game, player=None)
# Returns: [(q, r), ...] cells where placing gives 4+ in a row

find_winning_moves(game, player=None)
# Returns: [(q, r), ...] cells that complete 6 in a row

# Line analysis
count_lines(game, q, r, player=None)
# Returns: {'max_line': N}

# Random playout evaluation
rollout(game, num_games=1000)
# Returns: {'p0_wins': float, 'p1_wins': float, 'draw_rate': float}

# Deep search
alphabeta(game, depth=8)
# Returns: {'best_move': (q, r), 'value': float, 'nodes': int}
```

### Bot Classes

```python
from hexbot import Bot, BotProtocol

# Built-in bots
Bot.random()                    # plays random legal moves
Bot.heuristic()                 # uses C engine scoring (no NN)
Bot.load('model.pt')            # loads trained neural network
Bot(sims=400)                   # NN bot with MCTS (random weights)

# Custom bot via subclass
class MyBot(BotProtocol):
    def best_move(self, game):
        return game.legal_moves()[0]

# Custom bot via function (no class needed)
def my_func(game):
    return game.legal_moves()[0]

# Bot methods
bot.best_move(game)             # returns (q, r)
bot.policy(game)                # returns {(q,r): probability}
bot.evaluate(game)              # returns float (-1 to +1)
bot.save('model.pt')            # save to file
```

### Arena

```python
from hexbot import Arena

# Accepts any combination: Bot, BotProtocol subclass, or function
result = Arena(bot1, bot2, num_games=100).play()
result = Arena(my_func, Bot.heuristic(), num_games=50).play(verbose=False)

# Result fields
result.wins           # [bot1_wins, bot2_wins]
result.draws          # int
result.total_games    # int
result.avg_length     # float (average moves per game)
result.win_rate       # [bot1_rate, bot2_rate]
result.games          # list of per-game dicts
```

### Training (requires PyTorch)

```python
from hexbot import train

bot = train(
    iterations=100,             # training cycles
    games_per_iter=50,          # self-play games per cycle
    sims=200,                   # MCTS simulations per move
    lr=0.001,                   # learning rate
    network_config='standard',  # 'fast' (700K), 'standard' (3.9M), 'large' (14.5M)
    checkpoint_every=10,        # save every N iterations
    checkpoint_prefix='mybot',  # filename prefix
    on_iteration=callback,      # optional callback(iter, losses)
)
```

---

## Performance Benchmarks

Measured on Apple M4 Pro (14-core), single thread, idle system:

### Game Simulation

| Operation | Speed | Notes |
|-----------|-------|-------|
| Random game playouts | **431 games/sec** (61K stones/sec) | Full games, avg 141 moves |
| Move scoring (C heuristic) | **15.4K positions/sec** | Top-N move ranking |
| Place + undo cycle | **1.4M ops/sec** | In-place board modification |
| Game cloning | **86.6K clones/sec** | Full deep copy |
| Threat detection | **2.8M checks/sec** | Win/threat cell scanning |

### Alpha-Beta Search

| Depth | Turns Ahead | Nodes | Time | Nodes/sec |
|-------|-------------|-------|------|-----------|
| 4 | 2 | 689 | 0.04s | 16.3K |
| 6 | 3 | 18,404 | 0.11s | 167.4K |
| 8 | 4 | 253,497 | 0.96s | 263K |
| 10 | 5 | 953,363 | 8.3s | 115K |

### Neural Network Training (with MCTS)

| Config | Params | Sims | Speed | Notes |
|--------|--------|------|-------|-------|
| fast | 700K | 50 | ~3 games/sec | Quick prototyping |
| standard | 3.9M | 50 | ~0.3 games/sec | Good quality |
| standard | 200 | 200 | ~0.1 games/sec | Full quality training |

### Comparison to Pure Python

The C engine provides 14-50x speedup over equivalent Python implementations:

| Operation | Python | C Engine | Speedup |
|-----------|--------|----------|---------|
| Random playout | ~2.3K stones/sec | 61K stones/sec | **27x** |
| Win detection | ~50K checks/sec | 2.8M checks/sec | **56x** |
| Place/undo | ~80K ops/sec | 1.4M ops/sec | **18x** |

Run `python benchmark.py` to measure performance on your hardware.

---

## Architecture

```
hexbot.py            High-level framework (Bot, Arena, train, analysis functions)
    |
    ├── hexgame.py         Game engine API (HexGame wrapping C engine)
    │       |
    │   engine.so          Compiled C library (auto-built from engine.c)
    │       |
    │   engine.c           2,331 lines of optimized C
    │
    ├── bot.py             Neural network + MCTS + training (optional, needs PyTorch)
    │       |
    │   main.py            Pure Python game engine (used by bot.py for training)
    │
    ├── dashboard.py       Live training dashboard (REST + WebSocket API)
    │
    ├── test_dashboard.py  Test the dashboard with fake data (no training needed)
    │
    └── examples/          Example bots demonstrating different approaches
```


### How the C Engine Works

The game board is stored as three sets of bitboards - one per hex axis. Each row/column/diagonal of the hex grid is a 64-bit integer where set bits represent stone positions. Win detection reduces to a single expression:

```c
(bits & (bits >> 1) & (bits >> 2) & (bits >> 3) & (bits >> 4) & (bits >> 5)) != 0
```

This checks for 6 consecutive bits in ~15 integer operations, compared to nested Python loops that would require dozens of dictionary lookups.

Candidate tracking uses pre-allocated arrays instead of Python sets, eliminating all garbage collection in the hot path. Every undo operation is a stack pop - no object creation or copying.

---

## FAQ

### What coordinate system does the game use?

Axial coordinates `(q, r)` where `q` is the horizontal axis and `r` is the diagonal axis. The implicit third coordinate is `s = -q - r`. The three win axes are directions `(1,0)`, `(0,1)`, and `(1,-1)`.

### Why is the board infinite?

The real game has no fixed boundaries - stones can be placed anywhere within 8 hexes of existing stones. The C engine uses a 31x31 internal array (centered at the origin) which covers most practical games. If play extends beyond this, stones near the edges may not be tracked correctly, but this is extremely rare in real games.

### Can I use GPU for training?

Yes. The `train()` function automatically detects CUDA or MPS (Apple Silicon) and uses it for neural network training. The game simulation always runs on CPU (via the C engine) since it's already very fast.

### How does the C engine auto-compile?

When you first `import hexgame` or `import hexbot`, the module checks if `engine.so` exists and is newer than `engine.c`. If not, it runs `cc -O3 -march=native -shared -fPIC -o engine.so engine.c` automatically. You need a C compiler installed but don't need to compile manually.

### Can I use my own neural network architecture?

Yes. The `Bot` class accepts any PyTorch `nn.Module` that takes a `(batch, 7, 19, 19)` tensor and returns `(policy_logits, value, threat_logits)`. See `bot.py` in the full project for the default `HexNet` architecture.

### What's the strongest bot approach?

For this game, a hybrid approach works best: use the C engine's alpha-beta search for tactical situations (forced wins/blocks within 4 turns), MCTS with a neural network for strategic evaluation, and heuristic scoring as a fast fallback. The neural network + MCTS approach (AlphaZero-style) is the most powerful long-term but requires significant training time.

### Is there a way to play against humans online?

The full project (not included in this framework repo) includes a Playwright-based browser bot that can play on [hexo.did.science](https://hexo.did.science). See the full project repository for details.

---

## Examples

| File | Approach | Description |
|------|----------|-------------|
| `examples/play_random.py` | Heuristic | Heuristic bot vs random bot |
| `examples/custom_eval.py` | Hand-tuned | Custom evaluation function with priority system |
| `examples/evolutionary.py` | Evolutionary | Evolve scoring weights through tournament selection |
| `examples/train_bot.py` | Neural network | Train an AlphaZero-style bot from scratch |
| `examples/bot_vs_bot.py` | Comparison | Compare multiple bot types head-to-head |

## Advanced: Speedup Techniques

### Batched Neural Network Evaluation

When using a neural network for position evaluation inside alpha-beta search, the naive approach calls Python from C for every leaf node - about 600 microseconds per call due to ctypes overhead. With thousands of leaves per move, this dominates search time.

The framework includes a batched evaluation system that eliminates this bottleneck. Instead of calling Python once per leaf, the C engine collects all leaf positions into a buffer during the first search pass, Python evaluates them all in a single GPU batch, and the C engine re-searches with the cached values.

```python
from bot import BatchedNNAlphaBeta, create_network

net = create_network('standard')
# ... load weights ...

searcher = BatchedNNAlphaBeta(net, depth=8, nn_depth=5)
policy = searcher.search(game)
best_move = max(policy, key=policy.get)
```

The two-phase approach works like this:

1. **Phase 1 (Collect):** C alpha-beta runs to completion using a fast heuristic at leaves. Positions that need NN evaluation are stored in a buffer (up to 2,048 per batch).

2. **Phase 2 (Evaluate):** Python reads all collected positions, encodes them as tensors, and runs one batched forward pass through the neural network. Results are injected into the C engine's value cache.

3. **Phase 3 (Re-search):** C alpha-beta runs again. This time leaf positions hit the NN cache instead of using the heuristic, producing much better move ordering and pruning.

This gives 5-10x speedup over the callback approach because there are zero Python-C transitions during search - just two bulk data transfers plus two search invocations.

### MCTS with Virtual Loss Batching

The `BatchedMCTS` class uses virtual loss to select multiple leaves simultaneously, then evaluates them all in one batched NN forward pass:

```python
from bot import BatchedMCTS, create_network

net = create_network('standard')
mcts = BatchedMCTS(net, num_simulations=200, batch_size=64)
policy = mcts.search(game, temperature=1.0, add_noise=True)
```

Virtual loss temporarily increments a node's visit count before evaluation, ensuring that parallel leaf selections explore different branches. After the batch evaluation completes, the virtual losses are corrected. This reduces the number of NN forward passes by the batch size factor (64x fewer calls with batch_size=64).

### AB Hybrid in MCTS

The `BatchedMCTS` also includes a shallow alpha-beta pre-check at the root. Before running MCTS simulations, it does a quick depth-4 C engine search (~10ms). If alpha-beta finds a proven win (value = ±1.0), MCTS is skipped entirely and the winning move is returned immediately. This catches forced tactical sequences that MCTS would need hundreds of simulations to discover.

### Transposition Cache

MCTS normally evaluates the same position multiple times when reached through different move orders. In Connect-6, this is common because placing stones A then B produces the same board as B then A. The `BatchedMCTS` caches NN evaluations by Zobrist hash, reusing results for transposed positions. This typically saves 20-40% of NN calls.

### C Engine Move Ordering

The C engine's `board_get_scored_moves()` function provides fast heuristic move ordering that benefits any search algorithm. Moves are scored by line extension potential, blocking value, and proximity. When used as the first pass before NN evaluation, it ensures the most promising moves are searched first, improving alpha-beta cutoff rates and MCTS convergence.

## Using the Pre-trained Model and Replay Buffer

The repo includes a pre-trained checkpoint and a compressed replay buffer so you can start playing or continue training immediately without starting from scratch.

### Loading the Pre-trained Bot

```python
from hexbot import Bot, Arena

bot = Bot.load('pretrained.pt')
print(bot)  # Bot(mcts, sims=200, 3,909,308 params)

# Play against it
result = Arena(bot, Bot.heuristic(), num_games=20).play()
```

The checkpoint contains a 3.9M parameter network (128 filters, 12 residual blocks) trained for 40 iterations of self-play. It includes the full optimizer state so you can resume training from where it left off.

### Continuing Training from Checkpoint

```python
import torch
from bot import create_network, BatchedMCTS, ReplayBuffer, self_play_game_v2, train_step

# Load model
net = create_network('standard')
ckpt = torch.load('pretrained.pt', map_location='cpu', weights_only=False)
net.load_state_dict(ckpt['model_state_dict'], strict=False)

# Restore optimizer for continued training
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
if 'optimizer_state_dict' in ckpt:
    try:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    except:
        pass  # fresh optimizer if architecture changed
```

### Using the Replay Buffer

The compressed replay buffer contains ~400K training samples from 40 iterations of self-play. Decompress and load it to skip the cold-start phase:

```bash
gunzip replay_buffer.pkl.gz  # decompresses to ~2GB
```

```python
import pickle

with open('replay_buffer.pkl', 'rb') as f:
    data = pickle.load(f)

buffer_samples = data['buffer']    # list of TrainingSample objects
priorities = data['priorities']     # sampling priorities

print(f"Loaded {len(buffer_samples)} training samples")

# Feed into ReplayBuffer
from bot import ReplayBuffer
replay_buffer = ReplayBuffer()
for sample in buffer_samples:
    replay_buffer.push(sample)
```

With the buffer pre-loaded, training starts producing meaningful gradient updates from iteration 1 instead of waiting 5-10 iterations to fill the buffer from scratch.

## Training Configuration

### Play Style

Set `PLAY_STYLE` in `bot.py` to choose your training strategy:

```python
PLAY_STYLE = 'distant'   # spread-out colony exploration (default)
PLAY_STYLE = 'close'     # classic adjacent-only play
```

When `'distant'` is active, four mechanisms encourage spread-out play:

| Setting | Default | Description |
|---------|---------|-------------|
| `DIRICHLET_ALPHA` | 0.3 | Exploration noise (0.03 = focused, 0.3 = diverse) |
| `DISTANT_EXPLORE_PROB` | 0.25 | Chance per move to force a gap placement |
| `DISTANT_RANGE` | (2, 5) | Min/max distance from nearest stone for forced moves |
| `C_BLEND_ADJACENT` | 0.15 | C heuristic weight for adjacent moves |
| `C_BLEND_DISTANT` | 0.05 | C heuristic weight for distant moves |
| `TEMP_THRESHOLD` | 35 | Moves before switching to greedy play |

All constants are at the top of `bot.py`. Set `PLAY_STYLE = 'close'` to disable all distant play mechanisms.

### Training Throughput

| Setting | Default | Description |
|---------|---------|-------------|
| `BATCH_SIZE` | 512 | Training batch size (larger = faster steps) |
| `BatchedMCTS batch_size` | 64 | Positions per NN forward pass in MCTS |
| Workers per iteration | 5 | Parallel self-play processes |
| Games per future | 2 | Games per worker batch (balances streaming vs overhead) |

### Threat Evaluation

The threat system only counts lines that can still extend to 6. A 4-in-a-row blocked on both ends by opponent stones is not counted as a threat -the evaluator checks that `consecutive + open_forward + open_backward >= 6` before scoring.

## Contributing

Built for the Hexagonal Tic-Tac-Toe community. Contributions welcome - especially new example bots, performance improvements, and documentation.

## License

MIT
