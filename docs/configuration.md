# Configuration Reference

Complete reference for all configurable parameters in the Orca training pipeline.
All defaults are defined in `orca/config.py`.

---

## How to Override Parameters

Parameters can be overridden in three ways (highest priority first):

1. **CLI arguments** -- `python -m orca.train --lr 0.002 --mcts-sims 200`
2. **OrcaTrainer kwargs** -- `OrcaTrainer(lr=0.002, mcts_sims=200)`
3. **Edit `orca/config.py`** -- change the default values directly

When a parameter is `None` in OrcaTrainer, the value from `orca/config.py` is used.

---

## Network Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BOARD_SIZE` | 19 | Board dimension (19x19 grid = 361 positions) |
| `NUM_CHANNELS` | 7 | Input planes: 5 board planes + 2 threat planes |
| `NUM_FILTERS` | 128 | Convolutional filter width. 128 = fast, 256 = stronger but slower |
| `NUM_RES_BLOCKS` | 12 | Depth of the residual tower |

The standard HexNet with these defaults has ~3.9M parameters.

### Network Configs

The `--config` / `net_config` parameter selects a preset architecture:

| Config | Description |
|--------|-------------|
| `fast` | Smaller network for quick experiments |
| `standard` | Default 128-filter, 12-block HexNet (3.9M params) |
| `large` | Larger network with more filters |
| `hybrid` | Hybrid architecture |
| `orca-transformer` | Experimental transformer variant (~5.2M params) |

---

## Search (MCTS)

| Parameter | Default | CLI Flag | Description |
|-----------|---------|----------|-------------|
| `C_PUCT` | 1.5 | -- | UCB exploration constant. Higher = more exploration |
| `NUM_SIMULATIONS` | 400 | `--mcts-sims` | MCTS simulations per move. Curriculum may override |
| `MCTS_BATCH_SIZE` | 64 | `--mcts-batch` | Positions per batched NN forward pass |
| `DIRICHLET_ALPHA` | 0.3 | -- | Root noise for exploration. 0.03 = focused, 0.3 = diverse |
| `DIRICHLET_EPSILON` | 0.25 | -- | Fraction of root prior replaced by Dirichlet noise |
| `TEMP_THRESHOLD` | 35 | -- | Moves before switching from sampling to greedy play |

---

## Distant Play (Colony Strategy)

Controls the "distant" play style where stones are placed away from existing
clusters, encouraging colony-based strategies.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PLAY_STYLE` | `'distant'` | `'distant'` for spread-out play, `'close'` for adjacent-only |
| `C_BLEND_ADJACENT` | 0.15 | C heuristic weight for moves adjacent to existing stones |
| `C_BLEND_DISTANT` | 0.05 | C heuristic weight for moves far from existing stones |
| `DISTANT_EXPLORE_PROB` | 0.25 | Probability per move of injecting distant candidates |
| `DISTANT_RANGE` | (2, 5) | Min/max distance from nearest stone for distant moves |

Set `PLAY_STYLE = 'close'` to disable all distant play mechanisms.

---

## Training

| Parameter | Default | CLI Flag | Description |
|-----------|---------|----------|-------------|
| `BATCH_SIZE` | 1024 | `--batch-size` | Training batch size for gradient updates |
| `LEARNING_RATE` | 0.001 | `--lr` | Adam optimizer learning rate |
| `L2_REG` | 1e-4 | `--weight-decay` | Weight decay (L2 regularization) |
| `REPLAY_BUFFER_SIZE` | 400,000 | `--buffer-size` | Maximum samples in replay buffer |

---

## Pipeline

| Parameter | Default | CLI Flag | Description |
|-----------|---------|----------|-------------|
| `DEFAULT_TRAIN_STEPS` | 200 | `--train-steps` | Gradient steps per iteration |
| `DEFAULT_GAMES_PER_ITER` | 100 | `--games-per-iter` | Base games per iteration (curriculum adjusts) |
| `CHECKPOINT_EVERY` | 5 | -- | Save checkpoint every N iterations |
| `MAX_WORKERS` | 5 | `--workers` | Maximum parallel self-play workers |
| `GAMES_PER_FUTURE` | 2 | -- | Games per subprocess future |
| `ELO_EVAL_EVERY` | 2 | `--elo-every` | Run ELO evaluation every N iterations |
| `ELO_EVAL_GAMES` | 4 | `--elo-games` | Games played per ELO opponent |

---

## Curriculum (Adaptive Scaling)

The curriculum dictionary maps wall-clock hours to `(simulations, games_per_iter)`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CURRICULUM` | see below | Time-based sim/game schedule |
| `PLATEAU_THRESHOLD` | 15 | ELO delta below which training is considered stalled |
| `PLATEAU_ITERS` | 10 | Consecutive stall iterations before boosting sims |
| `PLATEAU_SIM_BOOST` | 50 | Extra sims added on plateau (capped at 400 total) |

Default curriculum schedule:

```python
CURRICULUM = {
    0.0: (50, 60),    # fast exploration
    0.5: (100, 50),   # better quality
    1.5: (150, 40),   # deeper search
    3.0: (200, 30),   # full depth
}
```

---

## LR Scheduler (CosineAnnealingWarmRestarts)

| Parameter | Default | CLI Flag | Description |
|-----------|---------|----------|-------------|
| `COSINE_T0` | 50 | `--scheduler-t0` | Number of iterations for the first restart period |
| `COSINE_T_MULT` | 2 | `--scheduler-tmult` | Factor by which T_i increases after each restart |
| `COSINE_ETA_MIN` | 1e-4 | `--scheduler-eta-min` | Minimum learning rate |

The scheduler cycles the learning rate between `LEARNING_RATE` and `COSINE_ETA_MIN`
using a cosine curve. Each restart period is `T_MULT` times longer than the previous.

---

## Mixed Precision (v4)

| Parameter | Default | CLI Flag | Description |
|-----------|---------|----------|-------------|
| `MIXED_PRECISION` | `False` | `--mixed-precision` | Enable AMP (fp16 forward, fp32 grads) |
| `GRAD_CLIP_NORM` | `1.0` | `--grad-clip` | Max gradient norm for clipping. `0` to disable |
| `GRAD_SCALER_INIT` | `2**16` | -- | Initial loss scale for GradScaler |
| `GRAD_SCALER_GROWTH` | `2.0` | -- | Scale growth factor on successful steps |

Mixed precision is only effective on CUDA with Tensor Cores. On MPS/CPU it is
silently ignored.

Gradient clipping applies regardless of mixed precision and prevents exploding
gradients during early training or with large learning rates.

---

## New Architecture Configs (v4)

| Config | Params | Description |
|--------|--------|-------------|
| `hex-masked` | 3.9M | CNN with hex-neighbor masking on 3x3 filters (recommended) |
| `hex-gnn` | 432K | Graph Neural Network on hex topology (experimental) |
| `multiscale` | 1.1M | Local CNN + global attention two-tower (experimental) |

### hex-gnn Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GNN_LAYERS` | 8 | Message-passing layers |
| `GNN_HIDDEN` | 128 | Hidden dimension per node |
| `GNN_AGGR` | `'mean'` | Aggregation: `'mean'`, `'sum'`, `'max'` |

### multiscale Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MS_BRANCHES` | `[3, 5, 7]` | Kernel sizes for parallel branches |
| `MS_FILTERS` | 64 | Filters per branch (total = 3 * 64 = 192) |
| `MS_RES_BLOCKS` | 8 | Residual blocks after branch merge |

```bash
python -m orca.train --config hex-gnn
python -m orca.train --config multiscale
```

---

## Experimental: Transformer Variant

These parameters only apply when using `--config orca-transformer`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRANSFORMER_LAYERS` | 2 | Number of transformer encoder layers |
| `TRANSFORMER_HEADS` | 8 | Attention heads per layer |
| `TRANSFORMER_DROPOUT` | 0.1 | Dropout rate in transformer layers |

The transformer adds global self-attention after the CNN backbone, producing a
~5.2M parameter model that trains ~30% slower per step.

---

## Example Configurations

### Fast Iteration (Quick Experiments)

Rapid feedback loop for testing ideas. Trades quality for speed.

```bash
python -m orca.train \
    --config fast \
    --iterations 20 \
    --games-per-iter 15 \
    --train-steps 100 \
    --mcts-sims 50 \
    --batch-size 256 \
    --buffer-size 50000 \
    --workers 2
```

### Deep Training (Overnight GPU)

Maximum quality for unattended training runs.

```bash
python -m orca.train \
    --config standard \
    --iterations 200 \
    --train-steps 400 \
    --mcts-sims 400 \
    --batch-size 1024 \
    --buffer-size 400000 \
    --device cuda \
    --workers 5
```

### Weak Hardware (CPU Only)

For machines without a GPU. Reduces all compute-heavy settings.

```bash
python -m orca.train \
    --config fast \
    --iterations 50 \
    --games-per-iter 10 \
    --train-steps 50 \
    --mcts-sims 30 \
    --mcts-batch 16 \
    --batch-size 128 \
    --buffer-size 50000 \
    --device cpu \
    --workers 2
```

### Strong Hardware (Multi-GPU)

For machines with powerful GPUs. Maximizes batch sizes and search depth.

```bash
python -m orca.train \
    --config large \
    --iterations 500 \
    --train-steps 600 \
    --mcts-sims 400 \
    --mcts-batch 128 \
    --batch-size 2048 \
    --buffer-size 800000 \
    --device cuda \
    --workers 8 \
    --lr 0.002
```

---

## OrcaTrainer Constructor Parameters

All parameters accepted by `OrcaTrainer.__init__()`:

```python
OrcaTrainer(
    # Pipeline
    iterations=999999,        # int: training iterations
    games_per_iter=None,      # int: games per iteration (None = config default)
    train_steps=None,         # int: gradient steps per iteration (None = 200)
    resume=True,              # bool: resume from latest checkpoint
    device=None,              # str: 'cuda', 'mps', 'cpu' (None = auto-detect)
    num_workers=None,         # int: parallel workers (None = auto)

    # Network
    net_config='standard',    # str: architecture preset

    # Optimizer
    lr=None,                  # float: learning rate (None = 0.001)
    weight_decay=None,        # float: L2 reg (None = 1e-4)

    # LR Scheduler
    scheduler_T0=None,        # int: cosine T_0 (None = 50)
    scheduler_Tmult=None,     # int: cosine T_mult (None = 2)
    scheduler_eta_min=None,   # float: cosine eta_min (None = 1e-4)

    # Search
    mcts_sims=None,           # int: MCTS sims per move (None = 400)
    mcts_batch_size=None,     # int: positions per NN batch (None = 64)

    # Replay Buffer
    buffer_size=None,         # int: buffer capacity (None = 400000)
    batch_size=None,          # int: training batch size (None = 1024)

    # ELO Evaluation
    elo_every=None,           # int: eval frequency (None = 2)
    elo_games=None,           # int: games per opponent (None = 4)

    # Observer
    observer=None,            # TrainingObserver: event handler (None = PrintObserver)

    # Vault
    vault_size=200,           # int: max stored model generations

    # Feature toggles
    use_curriculum=True,      # bool: adaptive sim/game scaling by time
    use_auto_tuner=True,      # bool: rule-based hyperparameter adjustment
    use_adaptive_lr=True,     # bool: cosine annealing LR schedule
    use_augmentation=True,    # bool: hex symmetry data augmentation
)
```

## Feature Toggles

Disable specific training features for experimentation:

| Toggle | CLI Flag | Default | Description |
|--------|----------|---------|-------------|
| `use_curriculum` | `--no-curriculum` | True | Adaptive sim/game scaling based on time and ELO plateau |
| `use_auto_tuner` | `--no-auto-tuner` | True | AutoTuner adjusts hyperparams based on training metrics |
| `use_adaptive_lr` | `--no-adaptive-lr` | True | CosineAnnealingWarmRestarts LR schedule |
| `use_augmentation` | `--no-augmentation` | True | Hex-valid symmetry augmentation (3x data) |

Example: fixed LR training without curriculum:
```bash
python -m orca.train --lr 0.001 --no-adaptive-lr --no-curriculum --mcts-sims 200
```

Example: minimal training (no bells and whistles):
```bash
python -m orca.train --no-curriculum --no-auto-tuner --no-adaptive-lr --no-augmentation
```

## Plateau Detection

When ELO stalls, the curriculum can auto-boost MCTS simulations.

| Parameter | Default | CLI | Description |
|-----------|---------|-----|-------------|
| `PLATEAU_THRESHOLD` | 15 | `--plateau-threshold` | ELO must change by this much or it's a stall |
| `PLATEAU_ITERS` | 10 | `--plateau-iters` | Consecutive stalled iterations before boosting |
| `PLATEAU_SIM_BOOST` | 50 | `--plateau-boost` | Extra sims added on plateau (capped at 400) |

Disable with `--no-curriculum` (plateau detection is part of the curriculum system).

## ELO Arena

| Parameter | Default | CLI | Description |
|-----------|---------|-----|-------------|
| `ELO_EVAL_EVERY` | 2 | `--elo-every` | Evaluate ELO every N iterations |
| `ELO_EVAL_GAMES` | 4 | `--elo-games` | Games per opponent in arena |
| `ELO_EVAL_SIMS` | 30 | `--elo-sims` | MCTS sims during ELO games (lower = faster eval) |
| `ELO_MAX_OPPONENTS` | 6 | `--elo-max-opponents` | Max past versions to evaluate against |
| `VAULT_MAX_MODELS` | 200 | `vault_size` kwarg | Max stored model snapshots |
