# Orca Bot

Orca is the framework's built-in AlphaZero-style bot for Hexagonal Connect-6.
It combines a 3.9M parameter convolutional neural network with Monte Carlo
Tree Search to learn entirely from self-play.

---

## Architecture: HexNet

The default network (`HexNet`) is a residual CNN with three output heads.

### Input

7 input channels on a 19x19 grid (361 positions):

- 5 board planes (stone positions, current player, move history)
- 2 threat planes (friendly and opponent threat maps)

### Backbone

```
Input (7, 19, 19)
  -> Conv2d 7 -> 128, kernel 3x3, padding 1, no bias
  -> BatchNorm2d + ReLU
  -> ResBlock x12 (each: Conv 3x3 -> BN -> ReLU -> Conv 3x3 -> BN + skip -> ReLU)
```

Each ResBlock has 128 filters with 3x3 convolutions, batch normalization, and
skip connections. The tower is 12 blocks deep.

### Output Heads

| Head | Architecture | Output | Description |
|------|-------------|--------|-------------|
| **Policy** | Conv 1x1 -> BN -> ReLU -> FC | 361 logits | Move probabilities over all board positions |
| **Value** | Conv 1x1 -> BN -> ReLU -> FC(256) -> ReLU -> FC(1) -> tanh | scalar in [-1, 1] | Win probability estimate |
| **Threat** | Conv 1x1 -> BN -> ReLU -> FC | 4 floats | `[my_4, my_5, opp_4, opp_5]` in-a-row counts |

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| Initial conv + BN | ~8K |
| 12 ResBlocks (128 filters) | ~3.5M |
| Policy head | ~260K |
| Value head | ~93K |
| Threat head | ~365 |
| **Total** | **~3.9M** |

---

## Loading Orca

### Orca.load()

The simplest way to get a playing bot:

```python
from orca import Orca

bot = Orca.load()                        # auto-detect checkpoint
bot = Orca.load(sims=400)               # more MCTS simulations
bot = Orca.load('my_checkpoint.pt')     # specific checkpoint
bot = Orca.load(search='alphabeta')     # use alpha-beta instead of MCTS
```

Checkpoint search order (when no path is given):

1. `orca/checkpoint.pt`
2. `pretrained.pt`
3. Latest `hex_checkpoint_*.pt` (sorted by iteration number)

### Bot.load() / Bot.orca()

The lower-level `Bot` class also works:

```python
from hexbot import Bot

bot = Bot.load('pretrained.pt', sims=200)
print(bot)  # Bot(mcts, sims=200, 3,909,308 params)
```

---

## Playing

### best_move()

Get the best move for a position:

```python
from orca import Orca
from hexbot import HexGame

bot = Orca.load()
game = HexGame()
game.place(0, 0)

move = bot.best_move(game)  # returns (q, r) tuple
game.place(*move)
```

### Arena

Pit Orca against other bots:

```python
from hexbot import Arena, Bot
from orca import Orca

orca = Orca.load(sims=200)
heuristic = Bot.heuristic()

result = Arena(orca, heuristic, num_games=20).play()
print(f"Orca: {result.wins[0]}W, Heuristic: {result.wins[1]}W")
```

---

## Training

### OrcaTrainer

The full training pipeline:

```python
from orca.train import OrcaTrainer

trainer = OrcaTrainer(
    iterations=100,
    games_per_iter=30,
    train_steps=200,
    device='cuda',
)
trainer.run()
```

### CLI

```bash
python -m orca.train --iterations 100 --games-per-iter 30
```

### Convenience Function

```python
from orca import Orca
Orca.train(iterations=50, games_per_iter=20)
```

See [training-guide.md](training-guide.md) for the full training pipeline
documentation and [configuration.md](configuration.md) for all parameters.

---

## Pre-trained Checkpoint

The repository includes a pre-trained checkpoint (`pretrained.pt`) trained for
65 iterations of self-play. It contains:

- Model weights (3.9M parameters, 128 filters, 12 residual blocks)
- Full optimizer state (for continuing training)
- Metrics and ELO history

**Note:** This is an early-stage checkpoint with limited training. The bot has
learned basic tactics (blocking, line extension) but is far from what it can
achieve with more training time. On proper hardware (CUDA GPU), training for
500+ iterations with 200 sims should produce significantly stronger play.
This checkpoint is a starting point, not a finished product.

### Continue Training

```python
from orca.train import OrcaTrainer

# Automatically resumes from the latest checkpoint
trainer = OrcaTrainer(iterations=200)
trainer.run()
```

### Using the Replay Buffer

A compressed replay buffer (`replay_buffer.pkl.gz`) with ~400K training samples
is also available:

```bash
gunzip replay_buffer.pkl.gz  # decompresses to ~2GB
```

```python
import pickle
from bot import ReplayBuffer

with open('replay_buffer.pkl', 'rb') as f:
    data = pickle.load(f)

replay_buffer = ReplayBuffer()
for sample in data['buffer']:
    replay_buffer.push(sample)

print(f"Loaded {len(replay_buffer)} samples")
```

Pre-loading the buffer means training produces meaningful updates from iteration 1
instead of waiting 5-10 iterations to fill the buffer from scratch.

---

## Experimental: TransformerHexNet

An experimental variant that adds global self-attention after the CNN backbone.

### Architecture

```
Input (7, 19, 19)
  -> Conv2d 7 -> 128 + BN + ReLU            (local feature extraction)
  -> ResBlock x12 (128 filters)               (deep local patterns)
  -> Flatten to (361, 128) sequence           (spatial -> sequence)
  -> HexPositionalEncoding                    (learned spatial embeddings)
  -> TransformerEncoder x2 (8 heads, GELU)    (global attention)
  -> LayerNorm                                (post-attention normalization)
  -> Reshape to (128, 19, 19)                 (sequence -> spatial)
  -> Policy / Value / Threat heads            (same as HexNet)
```

### Key Differences from HexNet

| Property | HexNet | TransformerHexNet |
|----------|--------|-------------------|
| Parameters | ~3.9M | ~5.2M |
| Receptive field | Local (3x3 convolutions) | Global (full attention) |
| Speed | Baseline | ~30% slower per step |
| Distant reasoning | Limited by conv depth | Attends to all positions |
| Status | Production | Experimental, untested |

### Why Transformers for Hex

Standard convolutions have a limited receptive field. Even with 12 residual blocks,
the network can only "see" patterns within a local neighborhood. The transformer
layers let the network attend to ALL 361 positions simultaneously, which matters for:

- **Colony play** -- reasoning about distant stone clusters
- **Multi-threat detection** -- spotting threats across separate board regions
- **Long-range planning** -- coordinating stones that are far apart

### Usage

```python
from bot import create_network

net = create_network('orca-transformer')
print(sum(p.numel() for p in net.parameters()))  # ~5.2M
```

```bash
python -m orca.train --config orca-transformer
```

The transformer variant is not yet tested in competitive play. It may produce
stronger results but requires more compute per training step.

---

## New Architectures (v4)

### HexGNN (`hex-gnn`)

A Graph Neural Network that treats the hex board as a graph with hex-aware
adjacency. Each cell is a node; edges connect hex neighbors (6-connected).

```python
from orca.hex_gnn import HexGNN

net = create_network('hex-gnn')  # ~3.2M params
```

| Property | Value |
|----------|-------|
| Parameters | ~3.2M |
| Layers | 8 message-passing |
| Receptive field | Global (after 8 hops) |
| Advantage | Native hex topology, no grid artifacts |

### MultiscaleNet (`multiscale`)

Parallel multi-scale convolutions capture patterns at different spatial scales
simultaneously, then merge features before the residual tower.

```python
from orca.multiscale_net import MultiscaleNet

net = create_network('multiscale')  # ~6.1M params
```

| Property | Value |
|----------|-------|
| Parameters | ~6.1M |
| Branches | 3x3 + 5x5 + 7x7 parallel convolutions |
| Advantage | Captures both local tactics and broad structure |

### HexMaskedNet (`hex-masked`) -- Recommended

Standard CNN with hex-neighbor masking on 3x3 conv filters. Zeros out the
top-left and bottom-right kernel positions that correspond to non-hex-neighbors
in axial coordinates:

```
Standard 3x3:    Hex-masked 3x3:
[a] [b] [c]      [0] [b] [c]
[d] [e] [f]  ->  [d] [e] [f]
[g] [h] [i]      [g] [h] [0]
```

```python
net = create_network('hex-masked')  # ~3.9M params (same as standard)
```

| Property | Value |
|----------|-------|
| Parameters | ~3.9M (identical to standard HexNet) |
| Speed | Same as standard CNN |
| Advantage | Respects hex topology without GNN complexity |

The mask is applied during every forward pass so gradients never flow through
non-neighbor positions. Same speed and parameter count as standard HexNet but
the network can only learn hex-valid patterns.

All architectures share the same policy/value/threat head interface as HexNet
and are drop-in replacements via `--config hex-masked`, `--config hex-gnn`, etc.

---

## Ensemble (v4)

Combine multiple checkpoints for stronger play and uncertainty estimation.

```python
from orca.ensemble import Ensemble

ens = Ensemble.from_latest(n=3)         # last 3 checkpoints
policy, value, unc = ens.evaluate(game) # unc = stddev across members
move = ens.best_move(game)              # averaged policy
```

Use uncertainty to detect positions where the model is unsure -- high
uncertainty suggests the position would benefit from more MCTS simulations
or solver verification.

---

## Model Zoo (v4)

Share and download pre-trained models.

```python
from orca.zoo import Zoo

# Browse available models
for m in Zoo.list():
    print(f"{m['name']}  ELO={m['elo']}  params={m['params']}")

# Download and use
net = Zoo.load('orca-v4-std')

# Package and share your own
Zoo.package(net, name='my-orca', metadata={'elo': 1900, 'iters': 500})
```

| Method | Description |
|--------|-------------|
| `Zoo.list()` | List models with name, ELO, param count, description |
| `Zoo.download(name)` | Download checkpoint to local cache, return path |
| `Zoo.load(name)` | Download + load into `nn.Module` |
| `Zoo.package(net, name, metadata)` | Package a trained model for sharing |
