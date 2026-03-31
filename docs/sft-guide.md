# Supervised Fine-Tuning (SFT) Guide

Train a network by imitating moves from strong games. Much faster than pure self-play for bootstrapping.

See also: [Training Guide](training-guide.md) | [Configuration](configuration.md) | [Orca Bot](orca.md)

---

## Quick Start

### 1. Get games

Download games from the community:

```bash
python -m orca.scrape --games 1000 --output games.jsonl
python -m orca.scrape --games 5000 --min-elo 1200 --output strong_games.jsonl
```

Or use your own game file in JSONL, CSV, or text format.

### 2. Train

```bash
python -m orca.sft --data games.jsonl --epochs 10
```

### 3. Continue with self-play

```bash
python -m orca.sft --data games.jsonl --epochs 5 --then-selfplay 50
```

This chains SFT warmup (learn from games) with self-play refinement (improve beyond the games).

---

## How SFT Works

For each position in each game:
1. **Encode** the board state as a (7, 19, 19) tensor
2. **Policy target**: the move that was actually played (one-hot)
3. **Value target**: the game outcome (+1 for winner, -1 for loser)
4. **Threat target**: threat features computed from the position

The network learns to predict what strong players do. After SFT, self-play refines beyond imitation.

---

## Game File Formats

### JSONL (hexo.did.science format)
```json
{"moves": [{"x": 0, "y": 0, "playerId": "p1", "moveNumber": 1}, ...], "players": [{"elo": 1200}], "gameResult": {"winningPlayerId": "p1"}}
```

### Simple JSONL
```json
{"moves": [[0,0], [1,0], [0,1]], "result": 1.0}
```

### CSV
```
game_id,move_num,q,r,result
game1,1,0,0,1.0
game1,2,1,0,1.0
```

### Plain text
```
0,0 1,0 0,1 1,1 0,2 1,2 1.0
```

The parser auto-detects the format from the file extension and content.

---

## CLI Reference

```
python -m orca.sft [options]

Required:
  --data PATH          Game file (JSONL, CSV, or text)

Training:
  --epochs N           Training epochs (default: 10)
  --batch-size N       Batch size (default: 512)
  --lr FLOAT           Learning rate (default: 0.001)
  --config NAME        Network: standard, fast, large, orca-transformer
  --device NAME        Device: cuda, mps, cpu (auto if omitted)
  --checkpoint PATH    Resume from specific checkpoint
  --save PATH          Save path (default: sft_checkpoint.pt)

Filtering:
  --min-elo N          Only use games where max player ELO >= N
  --max-games N        Limit number of games loaded

Data:
  --no-augment         Disable hex symmetry augmentation

Chaining:
  --then-selfplay N    After SFT, continue with N iterations of self-play
```

---

## Scraper Reference

```
python -m orca.scrape [options]

  --output PATH        Output file (default: games.jsonl)
  --games N            Games to download (default: 1000)
  --min-elo N          Min player ELO filter (default: 0 = all)
  --min-moves N        Min moves per game (default: 6)
  --delay FLOAT        Delay between API calls (default: 0.5s)
```

---

## Python API

```python
from orca.sft import sft_train, sft_then_selfplay, import_games, games_to_samples

# Train from games
net = sft_train('games.jsonl', epochs=10, lr=0.001)

# SFT then self-play
sft_then_selfplay('games.jsonl', sft_epochs=5, selfplay_iters=50)

# Load and inspect games
games = import_games('games.jsonl', min_elo=1200)
print(f'{len(games)} games')

# Convert to training samples manually
samples = games_to_samples(games)
print(f'{len(samples)} training positions')

# From hexbot
from hexbot import import_games
games = import_games('games.jsonl')
```

---

## Tips

- **Start with SFT, refine with self-play.** SFT gives a strong baseline in minutes. Self-play finds moves humans miss.
- **Filter by ELO.** `--min-elo 1200` removes weak games that teach bad habits.
- **Use augmentation.** The 3x data multiplier from hex symmetries helps a lot with small datasets.
- **Chain with self-play.** `--then-selfplay 50` is the recommended workflow. The network starts from imitation and evolves beyond it.
- **More data > more epochs.** 10K games for 5 epochs beats 1K games for 50 epochs.
