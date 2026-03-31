# Orca Training Guide

Complete guide to training the Orca bot using the AlphaZero-style self-play pipeline.

---

## Quick Start

```bash
# Start training with defaults (resumes from latest checkpoint)
python -m orca.train

# Common overrides
python -m orca.train --iterations 50 --games-per-iter 30
python -m orca.train --config orca-transformer --device cuda
python -m orca.train --fresh  # ignore existing checkpoints
```

Or from Python:

```python
from orca.train import OrcaTrainer

trainer = OrcaTrainer(iterations=100, games_per_iter=30)
trainer.run()
```

```python
from orca import Orca
Orca.train(iterations=50, games_per_iter=20)
```

### With Web UI

For a visual training dashboard with live game replay and charts:

```bash
python train_dashboard.py
```

Then open http://localhost:5001 and click play. See [Train Dashboard docs](train-dashboard.md).

### Disabling Features

Disable specific training features for experimentation:

```bash
python -m orca.train --no-curriculum --no-adaptive-lr  # fixed sims + fixed LR
python -m orca.train --no-auto-tuner --no-augmentation  # no tuner, no data aug
```

See [Configuration Reference](configuration.md) for all toggles.

---

## How Self-Play Works

Each training iteration follows this cycle:

1. **Export model** -- the current network is exported to ONNX for worker inference
2. **Self-play** -- parallel workers play games against themselves using MCTS
3. **Augmentation** -- collected samples are augmented with hex-valid symmetries
4. **Training** -- gradient descent on the replay buffer
5. **ELO evaluation** -- the current model plays against past generations
6. **Checkpoint** -- model, optimizer, buffer, and metrics are saved

### Workers

Self-play runs in parallel using `ProcessPoolExecutor`. Each worker loads the
network weights, creates a search engine, and plays a batch of games independently.

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_WORKERS` | 5 | Maximum parallel self-play processes |
| `GAMES_PER_FUTURE` | 2 | Games per subprocess future |

Workers are auto-sized: `min(MAX_WORKERS, cpu_count - 2)`.

There are two worker implementations:

- **V2 worker** (`_self_play_worker_v2`): Uses the C game engine (`CGameState`) with
  `BatchedMCTS` or `BatchedNNAlphaBeta`. Preferred when the C engine is available.
- **V1 worker** (`_self_play_worker_v1`): Uses `OnnxPredictor` for CPU inference with
  the pure-Python MCTS. Fallback when the C engine is unavailable.

### MCTS in Self-Play

Each move during self-play:

1. Run MCTS with the current simulation count (set by curriculum)
2. Sample a move from the visit count distribution (temperature-based)
3. Record the state, policy target, player, and threat label as a `TrainingSample`
4. After the game ends, fill in results and assign priority scores

Temperature control: for the first `TEMP_THRESHOLD` (35) moves, moves are sampled
proportionally to visit counts. After that, the best move is chosen greedily.

---

## Curriculum Learning

The curriculum dynamically adjusts MCTS simulations and games per iteration based
on both wall-clock time and iteration count.

### Adaptive Simulations

| Time / Iteration | Simulations | Games/iter |
|------------------|-------------|------------|
| < 0.5h / iter < 10 | 50 | 60 |
| 0.5-1.5h / iter 10-30 | 100 | 50 |
| 1.5-3.0h / iter 30-60 | 150 | 40 |
| > 3.0h / iter 60+ | 200 | 30 |

The actual sim count is `max(time_based, iteration_based)`. More games are played
when sims are low (fast exploration), fewer when sims are high (quality over quantity).

### Plateau Detection

When ELO stalls, the curriculum boosts search depth:

- **Threshold**: ELO delta < 15 between evaluations
- **Trigger**: 10 consecutive stalled iterations (`PLATEAU_ITERS`)
- **Boost**: +50 simulations (capped at 400)

```python
# From orca/config.py
PLATEAU_THRESHOLD = 15   # ELO delta to detect plateau
PLATEAU_ITERS = 10       # iterations of stall before boosting sims
PLATEAU_SIM_BOOST = 50   # extra sims on plateau (capped at 400)
```

### AutoTuner

The `AutoTuner` class makes rule-based adjustments each iteration:

- Caps MCTS sims at 50 during training (curriculum provides the real count)
- Decays hint blend over time: `max(0.0, 0.3 - iteration * 0.015)`
- Increases train steps (up to 600) when the buffer is >90% full and loss is decreasing
- Locks game mix to 100% normal self-play

---

## ELO Evaluation

### Model Vault

The `ModelVault` stores compressed (fp16) weights for every evaluated generation.
When the vault exceeds `max_models` (200), it prunes to keep:

- The first and last models
- The 20 most recent models
- Evenly spaced models across the full history

### Generational Arena

Every 2 iterations (`ELO_EVAL_EVERY`), the `GenerationalArena` runs a mini
round-robin tournament:

1. Select up to 6 opponents from the vault (first, last, and evenly spaced)
2. Play `ELO_EVAL_GAMES` (4) games per opponent, alternating colors
3. Compute new ELO: `current_elo + 16 * (score - 0.5) * num_opponents`

Games use 30 simulations and temperature 0.1 for near-deterministic play.

---

## Checkpoints

### Save / Load

Checkpoints are saved every 5 iterations (`CHECKPOINT_EVERY`) as
`hex_checkpoint_{iteration}.pt`. They contain:

- `model_state_dict` -- network weights
- `optimizer_state_dict` -- Adam optimizer state
- `scheduler_state_dict` -- cosine annealing LR scheduler state
- `iteration` -- current iteration number
- `metrics` -- ELO history, total games, iteration metrics
- `auto_tuner` -- hyperparameter tuner state

### Resume

By default, training resumes from the latest checkpoint. The trainer searches for:

1. `hex_checkpoint_*.pt` files, sorted by iteration number
2. Loads model weights, optimizer, scheduler, metrics, and AutoTuner state
3. Restores the replay buffer from `replay_buffer.pkl` if present

Use `--fresh` to start from scratch.

### Migration

Checkpoints are automatically migrated when the architecture changes:

- **5-to-7 channel migration**: Old 5-channel models are expanded to 7 channels
  (adding threat planes) with zero-initialized weights
- **Filter migration**: Models with different filter counts are resized with
  padding or truncation

---

## Replay Buffer

### Priority Sampling

The `ReplayBuffer` uses priority-weighted sampling. Samples are drawn proportionally
to their priority scores:

| Source | Priority | Rationale |
|--------|----------|-----------|
| Normal self-play | 1.0 | Baseline |
| Human games (loaded from `human_games.jsonl`) | 0.8-1.5 | Expert demonstrations |
| Online games (real opponents) | 2.0 | Real human play |
| Late-game positions (last 15 moves) | 2.0 | Stronger learning signal |
| Final 5 positions | 3.0 | Decisive game-ending positions |
| Fork/multi-threat moves (2+ threats) | 3.5 | Tactical patterns |
| Unstoppable forks (3+ threats) | 5.0 | Critical tactical patterns |
| Augmented samples | original * 0.8 | Slightly lower than originals |
| Short games (< 30 moves) | original * 0.5 | Penalized: less mid/late-game signal |
| Spread-out games (spread >= 8) | original * 1.5 | Rewarded: distant play diversity |

### TD-Error Updates

After each training step (with 50% probability to save time), the buffer updates
priorities based on temporal difference error:

```python
value_err = abs(predicted_value - target_value)
new_priority = value_err + 0.1  # ensures non-zero priority
```

Positions where the network's value prediction is most wrong get sampled more
frequently, focusing learning on the hardest positions.

### Capacity

Default capacity is 400,000 samples (`REPLAY_BUFFER_SIZE`). When full, oldest
samples are evicted (FIFO via `collections.deque`).

---

## Data Augmentation

### Hex-Valid Symmetries

The `augment_sample()` function generates 3 additional samples per original using
symmetries that are valid on the 19x19 hex grid:

1. **180-degree rotation**: `rot180(state)` -- flip both axes
2. **Transpose**: `state.transpose(1, 2)` -- swap q and r axes
3. **Transpose + 180-degree rotation**: combine both transforms

These are the only valid augmentations for a hex grid mapped to a square tensor.
Single-axis flips and 90-degree rotations break the hex topology and produce
invalid board states.

Policy targets are remapped to match the transformed board coordinates.
Augmented samples receive 80% of the original's priority.

---

## Training Step

### Loss Functions

Each training step optimizes three losses jointly:

```
total_loss = value_loss + policy_loss + 0.5 * threat_loss
```

| Loss | Function | Description |
|------|----------|-------------|
| **Value loss** | `MSE(predicted_value, game_result)` | How well the network predicts who wins |
| **Policy loss** | `-sum(target_policy * log_softmax(logits))` | Cross-entropy between MCTS policy and network output |
| **Threat loss** | `BCE_with_logits(threat_pred, threat_label)` | Auxiliary head predicting `[my_4, my_5, opp_4, opp_5]` in-a-row |

The threat loss weight (0.5) is lower than value and policy because it is an
auxiliary signal that improves tactical awareness without dominating the main
objectives.

### Optimizer

- **Adam** with learning rate 0.001 and weight decay 1e-4
- **CosineAnnealingWarmRestarts** scheduler: T_0=50, T_mult=2, eta_min=1e-4
- If LR drops below 1e-4 on resume, it is reset to 0.001

### Batch Size

Default training batch size is 1024 (`BATCH_SIZE`). Training is skipped if the
replay buffer contains fewer samples than the batch size.

---

## SFT Pre-Training (v4)

Supervised fine-tuning bootstraps the network from expert games before self-play.

```python
from orca.sft import sft_train, import_games, scrape_games

# Scrape games from an online source
scrape_games(source='littlegolem', output='expert_games.jsonl', limit=5000)

# Import and train
samples = import_games('expert_games.jsonl')
net = create_network('standard')
net = sft_train(net, 'expert_games.jsonl', epochs=5, lr=1e-3)

# Then continue with self-play
trainer = OrcaTrainer(iterations=100)
trainer.net = net
trainer.run()
```

SFT trains on cross-entropy policy loss only (no value head) since expert games
provide move labels but not reliable value targets. After SFT, switch to the
full self-play pipeline which trains all three heads.

### CLI

```bash
python -m orca.sft --games expert_games.jsonl --epochs 5 --lr 1e-3
python -m orca.train --resume  # continues from SFT checkpoint
```

---

## Mixed-Precision Training (v4)

Enable automatic mixed precision (AMP) for ~2x training speedup on CUDA GPUs.

```python
from bot import train_step
import torch

scaler = torch.amp.GradScaler()
losses = train_step(net, optimizer, replay_buffer, device='cuda', grad_scaler=scaler)
```

The `grad_scaler` parameter in `train_step()` enables fp16 forward passes with
fp32 gradient accumulation. Loss scaling prevents underflow in fp16 gradients.

### OrcaTrainer with Mixed Precision

```python
trainer = OrcaTrainer(
    iterations=200,
    device='cuda',
    mixed_precision=True,  # auto-creates GradScaler
)
trainer.run()
```

### CLI

```bash
python -m orca.train --mixed-precision --device cuda
```

Mixed precision is only effective on CUDA GPUs with Tensor Cores (RTX 20+, A100, etc.).
MPS and CPU fall back to fp32 automatically.

---

## Distributed Training (v4)

Scale training across multiple GPUs or machines.

### Multi-GPU (Single Machine)

```python
from orca.distributed import MultiGPUTrainer

trainer = MultiGPUTrainer(net, device_ids=[0, 1, 2, 3])
losses = trainer.train_step(batch)
```

Uses `torch.nn.DataParallel` to split batches across GPUs. Linear speedup for
large batch sizes.

### Distributed Self-Play

```python
from orca.distributed import SelfPlayPool

pool = SelfPlayPool(num_workers=16, net=net)
samples = pool.generate(num_games=200)
pool.shutdown()
```

`SelfPlayPool` manages worker processes with automatic load balancing. Workers
run on CPU; the training loop runs on GPU.

### Ray-Based Distributed

```python
from orca.distributed import RayTrainer

trainer = RayTrainer(net_config='standard', num_actors=32)
trainer.run(iterations=100)
```

`RayTrainer` distributes self-play across a Ray cluster. Each actor is a
remote process that can run on a different machine. The trainer aggregates
samples and runs gradient updates centrally.

```bash
python -m orca.train --distributed ray --num-actors 32
```

---

## Skill Curriculum (v4)

The `SkillCurriculum` replaces the time-based curriculum with a 6-level
progression tied to training milestones.

```python
from orca.curriculum import SkillCurriculum

curriculum = SkillCurriculum(start_level=0)
sims, games = curriculum.settings()
```

### Levels

| Level | Sims | Games/iter | Focus |
|-------|------|------------|-------|
| 0 | 30 | 80 | Random openings, basic legality |
| 1 | 50 | 60 | Line extension, simple blocking |
| 2 | 100 | 50 | Threat detection, 4-in-a-row patterns |
| 3 | 150 | 40 | Positional evaluation, colony play |
| 4 | 200 | 30 | Complex tactical sequences |
| 5 | 400 | 20 | Full-strength search |

Advancement is triggered by ELO thresholds: when the bot's ELO exceeds the
level's target, the curriculum advances automatically.

### CLI

```bash
python -m orca.train --skill-curriculum --start-level 0
```

---

## CLI Reference

```
python -m orca.train [OPTIONS]

Pipeline:
  --iterations N          Training iterations (default: infinite)
  --games-per-iter N      Games per iteration (default: from curriculum)
  --train-steps N         Gradient steps per iteration (default: 200)
  --resume                Resume from latest checkpoint (default)
  --fresh                 Start fresh, ignore existing checkpoints
  --workers N             Parallel self-play workers (default: auto)

Network:
  --config NAME           Architecture: fast, standard, large, hybrid,
                          orca-transformer (default: standard)
  --device DEVICE         Device: cuda, mps, cpu (default: auto)

Optimizer:
  --lr FLOAT              Learning rate (default: 0.001)
  --weight-decay FLOAT    L2 regularization (default: 1e-4)
  --scheduler-t0 N        CosineAnnealing T_0 (default: 50)
  --scheduler-tmult N     CosineAnnealing T_mult (default: 2)
  --scheduler-eta-min F   CosineAnnealing eta_min (default: 1e-4)

Search:
  --mcts-sims N           MCTS simulations per move (default: 400)
  --mcts-batch N          MCTS batch size for NN eval (default: 64)

Replay Buffer:
  --buffer-size N         Replay buffer capacity (default: 400000)
  --batch-size N          Training batch size (default: 1024)

Evaluation:
  --elo-every N           ELO eval frequency in iterations (default: 2)
  --elo-games N           Games per ELO opponent (default: 4)
```
