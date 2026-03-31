"""
bot.py - backwards compatible re-export layer.

All code has moved to orca/ submodules:
  - orca/network.py   : ResBlock, HexNet, HybridHexNet, create_network, ONNX, migrations
  - orca/search.py    : MCTSNode, MCTS, BatchedMCTS, NNAlphaBeta, BatchedNNAlphaBeta
  - orca/encoding.py  : encode_state, decode_policy, CGameState, c_encode_state, etc.
  - orca/threats.py   : find_forced_move, detect_finisher, _threat_search, etc.
  - orca/data.py      : TrainingSample, ReplayBuffer, self_play_game, train_step, etc.

This file re-exports everything so that `from bot import X` continues to work.

Usage:
    python bot.py train          # start self-play training
    python bot.py test           # quick smoke test
    python bot.py play           # play interactively
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch

from main import HexGame

# ---------------------------------------------------------------------------
# Training observer protocol (for dashboard integration)
# ---------------------------------------------------------------------------

@runtime_checkable
class TrainingObserver(Protocol):
    def on_iteration_start(self, iteration: int, total: int) -> None: ...
    def on_game_complete(self, game_idx: int, total_games: int,
                         move_history: List[Tuple[int, int]], result: float,
                         num_samples: int) -> None: ...
    def on_iteration_complete(self, metrics: dict) -> None: ...
    def on_training_complete(self) -> None: ...
    def should_stop(self) -> bool: ...

# ---------------------------------------------------------------------------
# Config - importable from bot.py AND orca.config
# ---------------------------------------------------------------------------

try:
    from orca.config import (
        BOARD_SIZE, NUM_CHANNELS, NUM_FILTERS, NUM_RES_BLOCKS,
        C_PUCT, PLAY_STYLE, C_BLEND_ADJACENT, C_BLEND_DISTANT,
        DISTANT_EXPLORE_PROB, DISTANT_RANGE,
        DIRICHLET_ALPHA, DIRICHLET_EPSILON, NUM_SIMULATIONS, TEMP_THRESHOLD,
        REPLAY_BUFFER_SIZE, BATCH_SIZE, LEARNING_RATE, L2_REG,
        MCTS_BATCH_SIZE,
    )
except ImportError:
    BOARD_SIZE = 19
    NUM_CHANNELS = 7
    NUM_FILTERS = 128
    NUM_RES_BLOCKS = 12
    C_PUCT = 1.5
    PLAY_STYLE = 'distant'
    C_BLEND_ADJACENT = 0.15
    C_BLEND_DISTANT = 0.05
    DISTANT_EXPLORE_PROB = 0.25
    DISTANT_RANGE = (2, 5)
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPSILON = 0.25
    NUM_SIMULATIONS = 400
    TEMP_THRESHOLD = 35
    REPLAY_BUFFER_SIZE = 400_000
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.001
    L2_REG = 1e-4
    MCTS_BATCH_SIZE = 64

HALF = BOARD_SIZE // 2


def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Re-exports from orca submodules
# ---------------------------------------------------------------------------

# orca.encoding
from orca.encoding import (
    _threat_line_at, encode_state, decode_policy,
    compute_threat_label,
    _get_lib, _setup_c_signatures,
    CGameState, c_encode_state, c_decode_policy, c_compute_threat_label,
    _AXES,
)

# orca.network
from orca.network import (
    ResBlock, SEResBlock,
    HexNet, HybridHexNet,
    OnnxPredictor, export_onnx,
    create_network,
    migrate_checkpoint_5to7, migrate_checkpoint_filters,
)

# orca.threats
from orca.threats import (
    AXES_3,
    _count_line, _max_line_at, _line_through_candidate,
    _get_threat_moves, _count_winning_cells, _count_multi_axis_threats,
    _threat_search,
    find_forced_move, detect_finisher, compute_threat_bonus,
)

# orca.search
from orca.search import (
    MCTSNode, MCTS,
    NNAlphaBeta, BatchedNNAlphaBeta,
    BatchedMCTS,
    _get_existing_stones,
)

# orca.data
from orca.data import (
    TrainingSample, ReplayBuffer,
    POSITION_CATALOG, GUIDED_POSITIONS,
    get_guided_positions_by_level, setup_position,
    generate_puzzles,
    load_human_games, load_online_games,
    augment_sample,
    self_play_game, self_play_game_v2,
    train_step, train,
)


# ---------------------------------------------------------------------------
# Entry points (kept here since they're CLI commands)
# ---------------------------------------------------------------------------

def smoke_test():
    """Quick test of all components."""
    device = get_device()
    print(f"Device: {device}")

    # Test encoding
    game = HexGame(candidate_radius=2)
    enc, oq, orr = encode_state(game)
    assert enc.shape == (5, BOARD_SIZE, BOARD_SIZE)
    print(f"  encode_state: shape={enc.shape}, offsets=({oq},{orr})")

    # Test network
    net = HexNet().to(device)
    param_count = sum(p.numel() for p in net.parameters())
    print(f"  HexNet: {param_count:,} parameters")

    policy_logits, value = net.predict(enc)
    assert policy_logits.shape == (BOARD_SIZE * BOARD_SIZE,)
    print(f"  predict: policy shape={policy_logits.shape}, value={value:.4f}")

    # Test decode_policy
    policy = decode_policy(policy_logits, game, oq, orr)
    assert abs(sum(policy.values()) - 1.0) < 1e-5 or len(policy) == 1
    print(f"  decode_policy: {len(policy)} moves, sum={sum(policy.values()):.4f}")

    # Test MCTS with tiny sim count
    mcts = MCTS(net, num_simulations=10)
    game.place_stone(0, 0)  # first move
    policy = mcts.search(game, temperature=1.0, add_noise=False)
    print(f"  MCTS (10 sims): {len(policy)} moves, top={max(policy.values()):.3f}")

    # Play a quick game with 5 sims
    mcts_fast = MCTS(net, num_simulations=5)
    game2 = HexGame(candidate_radius=2, max_total_stones=50)
    move_count = 0
    while not game2.is_terminal:
        p = mcts_fast.search(game2, temperature=1.0, add_noise=False)
        if not p:
            break
        moves = list(p.keys())
        probs = [p[m] for m in moves]
        idx = np.random.choice(len(moves), p=probs)
        game2.place_stone(*moves[idx])
        move_count += 1
    print(
        f"  Quick game: {move_count} moves, "
        f"winner={game2.winner}, stones={game2.total_stones}"
    )

    print("All smoke tests passed!")


def play_interactive():
    """Play against the bot interactively."""
    device = get_device()
    net = HexNet().to(device)

    # Try to load latest checkpoint
    import glob
    checkpoints = sorted(glob.glob("hex_checkpoint_*.pt"))
    if checkpoints:
        ckpt = torch.load(checkpoints[-1], map_location=device, weights_only=True)
        net.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded {checkpoints[-1]}")
    else:
        print("No checkpoint found, using random network")

    net.eval()
    mcts = MCTS(net, num_simulations=NUM_SIMULATIONS)

    game = HexGame(candidate_radius=3, max_total_stones=200)
    print("You are Player 0 (first move: 1 stone, then 2 per turn)")
    print("Enter moves as: q r  (e.g. '0 0')")
    print()

    while not game.is_terminal:
        print(f"Turn {game.turn}, Player {game.current_player}, "
              f"stones this turn: {game.stones_this_turn}/{game.stones_per_turn}")
        print(f"Total stones: {game.total_stones}")

        if game.current_player == 0:
            # Human
            while True:
                try:
                    line = input("Your move (q r): ").strip()
                    q, r = map(int, line.split())
                    if (q, r) not in game.candidates and game.occupied:
                        print("Illegal move (not in candidates)")
                        continue
                    if (q, r) in game.occupied:
                        print("Cell already occupied")
                        continue
                    break
                except (ValueError, EOFError):
                    print("Enter two integers: q r")
            game.place_stone(q, r)
        else:
            # Bot
            t0 = time.perf_counter()
            policy = mcts.search(game, temperature=0.1, add_noise=False)
            elapsed = time.perf_counter() - t0
            best_move = max(policy, key=policy.get)
            print(f"Bot plays: {best_move[0]} {best_move[1]} ({elapsed:.1f}s)")
            game.place_stone(*best_move)

    if game.winner is not None:
        print(f"\nPlayer {game.winner} wins!")
    else:
        print("\nDraw!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "train":
            train()
        elif cmd == "test":
            smoke_test()
        elif cmd == "play":
            play_interactive()
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python bot.py [train|test|play]")
    else:
        smoke_test()
