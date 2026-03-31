"""
orca/data.py - Training data: self-play, replay buffer, augmentation, positions.

Contains:
- TrainingSample dataclass
- ReplayBuffer
- Position catalogs (POSITION_CATALOG, GUIDED_POSITIONS)
- setup_position, generate_puzzles
- load_human_games, load_online_games
- augment_sample
- self_play_game, self_play_game_v2
- train_step, train
"""

from __future__ import annotations

import collections
import contextlib
import json as _json
import os as _os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from main import HexGame

# Import config constants
try:
    from orca.config import (
        BOARD_SIZE, NUM_CHANNELS, NUM_FILTERS, NUM_RES_BLOCKS,
        BATCH_SIZE, LEARNING_RATE, L2_REG, NUM_SIMULATIONS,
        TEMP_THRESHOLD, REPLAY_BUFFER_SIZE,
        PLAY_STYLE, C_BLEND_ADJACENT, C_BLEND_DISTANT,
        DISTANT_EXPLORE_PROB, DISTANT_RANGE,
        DIRICHLET_ALPHA, DIRICHLET_EPSILON,
    )
except ImportError:
    BOARD_SIZE = 19
    NUM_CHANNELS = 7
    NUM_FILTERS = 128
    NUM_RES_BLOCKS = 12
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.001
    L2_REG = 1e-4
    NUM_SIMULATIONS = 400
    TEMP_THRESHOLD = 35
    REPLAY_BUFFER_SIZE = 400_000
    PLAY_STYLE = 'distant'
    C_BLEND_ADJACENT = 0.15
    C_BLEND_DISTANT = 0.05
    DISTANT_EXPLORE_PROB = 0.25
    DISTANT_RANGE = (2, 5)
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPSILON = 0.25

# Imports from sibling modules
from orca.encoding import (
    encode_state, decode_policy, compute_threat_label,
    c_encode_state, c_decode_policy, c_compute_threat_label,
    CGameState,
)
from orca.network import HexNet
from orca.threats import find_forced_move, _line_through_candidate
from orca.search import MCTS, BatchedMCTS, MCTSNode, _get_existing_stones


# ---------------------------------------------------------------------------
# TrainingSample
# ---------------------------------------------------------------------------

@dataclass
class TrainingSample:
    encoded_state: torch.Tensor  # (5, BS, BS)
    policy_target: np.ndarray    # (BS*BS,) float32
    player: int
    result: Optional[float] = None
    threat_label: Optional[np.ndarray] = None  # (4,) float32: [my4, my5, opp4, opp5]
    priority: float = 1.0


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Prioritized replay buffer - samples proportional to priority."""

    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE):
        self.buffer: collections.deque = collections.deque(maxlen=capacity)
        self.priorities: collections.deque = collections.deque(maxlen=capacity)

    def push(self, sample: TrainingSample) -> None:
        self.buffer.append(sample)
        self.priorities.append(sample.priority)

    def sample(self, batch_size: int) -> Tuple[List[TrainingSample], List[int]]:
        """Returns (samples, indices) for priority updating."""
        n = min(batch_size, len(self.buffer))
        priors = np.array(self.priorities, dtype=np.float64)
        priors /= priors.sum()
        indices = np.random.choice(len(self.buffer), size=n, replace=False, p=priors)
        buf = list(self.buffer)
        return [buf[i] for i in indices], indices.tolist()

    def update_priorities(self, indices: List[int], errors: List[float]) -> None:
        """Update priorities based on |prediction_error| + epsilon."""
        for idx, err in zip(indices, errors):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = abs(err) + 0.01

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Position catalog - predefined starting formations
# ---------------------------------------------------------------------------

POSITION_CATALOG = {
    'empty': {},
    'center': {(0, 0): 0},

    # Triangle formations
    'triangle_p0': {
        (0, 0): 0, (1, 0): 0, (0, 1): 0, (1, -1): 0,
        (3, 0): 1, (3, 1): 1, (2, 2): 1, (4, -1): 1,
    },
    'triangle_p1': {
        (0, 0): 1, (1, 0): 1, (0, 1): 1, (1, -1): 1,
        (3, 0): 0, (3, 1): 0, (2, 2): 0, (4, -1): 0,
    },

    # Line starters (3-in-a-row each)
    'line3_h': {
        (0, 0): 0, (1, 0): 0, (2, 0): 0,
        (-2, 0): 1, (-3, 0): 1, (-2, 1): 1,
    },
    'line3_v': {
        (0, 0): 0, (0, 1): 0, (0, 2): 0,
        (2, 0): 1, (2, 1): 1, (3, 0): 1,
    },
    'line3_d': {
        (0, 0): 0, (1, -1): 0, (2, -2): 0,
        (0, 2): 1, (1, 2): 1, (0, 3): 1,
    },

    # Near-win: 5-in-a-row (must complete or block)
    'threat5_h': {
        (0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (4, 0): 0,
        (0, 1): 1, (1, 1): 1, (2, 1): 1, (3, 1): 1, (0, 2): 1,
    },
    'threat5_d': {
        (0, 0): 0, (1, -1): 0, (2, -2): 0, (3, -3): 0, (4, -4): 0,
        (0, 1): 1, (1, 1): 1, (2, 1): 1, (3, 1): 1, (0, 2): 1,
    },

    # Diamond cluster
    'diamond': {
        (0, 0): 0, (1, 0): 0, (0, 1): 0, (-1, 1): 0, (-1, 0): 0, (1, -1): 0,
        (3, 0): 1, (4, 0): 1, (3, 1): 1, (4, -1): 1, (2, 1): 1, (2, 0): 1,
    },

    # Fork (two directions of 3)
    'fork_p0': {
        (0, 0): 0, (1, 0): 0, (2, 0): 0, (0, 1): 0, (0, 2): 0,
        (-2, 0): 1, (-2, 1): 1, (-2, 2): 1, (4, 0): 1, (4, 1): 1,
    },
}


# ---------------------------------------------------------------------------
# Guided positions - theory-based curriculum for faster learning
# ---------------------------------------------------------------------------

GUIDED_POSITIONS = {
    # === LEVEL 1: Endgame Puzzles (5-in-a-row complete/block) ===
    'endgame_complete_h': {
        'level': 1,
        'position': {
            (0,0):0, (1,0):0, (2,0):0, (3,0):0, (4,0):0,
            (0,1):1, (1,1):1, (2,1):1, (3,1):1, (0,2):1,
        },
        'hint_moves': [(-1,0)],
    },
    'endgame_complete_v': {
        'level': 1,
        'position': {
            (0,0):0, (0,1):0, (0,2):0, (0,3):0, (0,4):0,
            (1,0):1, (1,1):1, (2,0):1, (2,1):1, (3,0):1,
        },
        'hint_moves': [(0,5)],
    },
    'endgame_complete_d': {
        'level': 1,
        'position': {
            (0,0):0, (1,-1):0, (2,-2):0, (3,-3):0, (4,-4):0,
            (0,1):1, (1,1):1, (2,1):1, (3,1):1, (0,2):1,
        },
        'hint_moves': [(5,-5)],
    },
    'endgame_block_h': {
        'level': 1,
        'position': {
            (0,0):1, (1,0):1, (2,0):1, (3,0):1, (4,0):1,
            (0,2):0, (1,2):0, (2,2):0, (3,2):0, (0,3):0,
        },
        'hint_moves': [(-1,0)],
    },
    'endgame_race': {
        'level': 1,
        'position': {
            (0,0):0, (1,0):0, (2,0):0, (3,0):0, (4,0):0,
            (0,2):1, (1,2):1, (2,2):1, (3,2):1, (4,2):1,
            (-2,0):1, (-2,1):1,
        },
        'hint_moves': [(-1,0)],
    },
    'endgame_gap_fill': {
        'level': 1,
        'position': {
            (0,0):0, (1,0):0, (3,0):0, (4,0):0, (5,0):0,
            (0,1):1, (1,1):1, (2,1):1, (3,1):1, (0,2):1,
        },
        'hint_moves': [(2,0)],
    },

    # === LEVEL 2: Formation Puzzles ===
    'formation_rhombus': {
        'level': 2,
        'position': {
            (0,0):0, (1,0):0, (0,1):0, (-1,1):0,
            (3,0):1, (3,1):1, (2,2):1, (4,-1):1,
        },
        'hint_moves': [(-1,0), (-2,1)],
    },
    'formation_rhombus_blocked': {
        'level': 2,
        'position': {
            (0,0):0, (1,0):0, (0,1):0, (-1,1):0,
            (2,0):1, (-2,1):1, (3,0):1, (3,-1):1,
        },
        'hint_moves': [(-1,0), (0,2)],
    },
    'formation_line4_open': {
        'level': 2,
        'position': {
            (0,0):0, (1,0):0, (2,0):0, (3,0):0,
            (0,2):1, (1,2):1, (2,2):1, (3,2):1,
        },
        'hint_moves': [],
    },
    'formation_trapezoid': {
        'level': 2,
        'position': {
            (0,0):0, (1,0):0, (0,1):0, (-1,1):0, (-1,0):0,
            (3,0):1, (3,1):1, (2,2):1, (4,-1):1, (-3,0):1,
        },
        'hint_moves': [(0,-1), (1,-1)],
    },
    'formation_fork': {
        'level': 2,
        'position': {
            (0,0):0, (1,0):0, (2,0):0, (3,0):0,
            (0,1):0, (0,2):0, (0,3):0,
            (-2,0):1, (-2,1):1, (-2,2):1, (5,0):1,
            (0,5):1, (1,5):1, (2,5):1,
        },
        'hint_moves': [],
    },
    'formation_arch': {
        'level': 2,
        'position': {
            (0,0):0, (1,0):0, (-1,1):0, (1,1):0,
            (3,0):1, (3,1):1, (-3,0):1, (-3,1):1,
        },
        'hint_moves': [(0,1), (2,0)],
    },
    'formation_open3': {
        'level': 2,
        'position': {
            (0,0):0, (1,0):0, (2,0):0,
            (-2,1):1, (-2,2):1, (5,0):1,
        },
        'hint_moves': [],
    },

    # === LEVEL 3: Known Winning Sequences ===
    'sequence_std_defense': {
        'level': 3,
        'position': {
            (0,0):0, (1,0):0, (0,1):0,
            (3,0):1, (3,1):1,
            (0,2):1, (2,-1):1,
        },
        'hint_moves': [(-1,1), (-1,0)],
    },
    'sequence_pair_defense': {
        'level': 3,
        'position': {
            (0,0):0, (1,0):0, (0,1):0,
            (3,1):1, (4,0):1,
            (2,0):1, (2,-1):1,
        },
        'hint_moves': [(-1,1), (0,2)],
    },
    'sequence_gap_defense': {
        'level': 3,
        'position': {
            (0,0):0, (1,0):0, (0,1):0,
            (3,0):1, (3,1):1,
            (0,-1):1, (2,-1):1,
        },
        'hint_moves': [(-1,1), (-1,0)],
    },
    'sequence_b8c16_defense': {
        'level': 3,
        'position': {
            (0,0):0, (1,0):0, (0,1):0,
            (3,0):1, (3,1):1,
            (0,-2):1, (3,-2):1,
        },
        'hint_moves': [(-1,1), (-1,0)],
    },
    'sequence_post_trapezoid': {
        'level': 3,
        'position': {
            (0,0):0, (1,0):0, (0,1):0, (-1,1):0, (-1,0):0,
            (0,2):1, (2,-1):1, (1,-1):1, (0,-1):1,
        },
        'hint_moves': [(2,0), (1,1)],
    },
    'sequence_split_defense': {
        'level': 3,
        'position': {
            (0,0):0, (1,0):0, (0,1):0,
            (3,0):1, (3,1):1,
            (0,2):1, (2,0):1,
        },
        'hint_moves': [(-1,1), (-1,0)],
    },
    'sequence_rhombus_continue': {
        'level': 3,
        'position': {
            (0,0):0, (1,0):0, (0,1):0, (-1,1):0,
            (2,0):1, (0,2):1, (2,-1):1, (-2,1):1,
        },
        'hint_moves': [(-1,0), (-2,0)],
    },
}


# Helper to get positions by level
def get_guided_positions_by_level(level: int) -> List[dict]:
    """Return list of (position_dict, hint_moves) for given level."""
    return [(v['position'], v.get('hint_moves', []))
            for v in GUIDED_POSITIONS.values() if v['level'] == level]


def setup_position(position: Dict[Tuple[int, int], int]) -> HexGame:
    """Create a game from a position dict {(q,r): player}.
    Directly sets stones and state without going through turn logic."""
    game = HexGame(candidate_radius=3, max_total_stones=200)
    if not position:
        return game

    p0_stones = [pos for pos, p in position.items() if p == 0]
    p1_stones = [pos for pos, p in position.items() if p == 1]

    # Place stones respecting turn order: turn 0 = p0 x 1, then alternating x 2
    moves = []
    i0, i1 = 0, 0
    # Turn 0: p0 places 1
    if i0 < len(p0_stones):
        moves.append(p0_stones[i0]); i0 += 1
    # Then alternate: p1 x 2, p0 x 2, p1 x 2, ...
    while i0 < len(p0_stones) or i1 < len(p1_stones):
        for _ in range(2):
            if i1 < len(p1_stones):
                moves.append(p1_stones[i1]); i1 += 1
        for _ in range(2):
            if i0 < len(p0_stones):
                moves.append(p0_stones[i0]); i0 += 1

    for m in moves:
        if not game.is_terminal:
            game.place_stone(*m)
    return game


# ---------------------------------------------------------------------------
# Puzzle generator
# ---------------------------------------------------------------------------

def generate_puzzles(num_puzzles: int = 50, rng: Optional[random.Random] = None) -> List[dict]:
    """Generate tactical puzzle positions. Returns list of position dicts."""
    if rng is None:
        rng = random.Random()

    puzzles: List[dict] = []
    axes = [(1, 0), (0, 1), (1, -1)]

    for _ in range(num_puzzles * 3):  # overshoot, not all attempts succeed
        if len(puzzles) >= num_puzzles:
            break

        game = HexGame(candidate_radius=3, max_total_stones=150)
        # Random mid-game: 10-30 moves
        n_random = rng.randint(10, 30)
        for _ in range(n_random):
            if game.is_terminal:
                break
            moves = game.legal_moves()
            if not moves:
                break
            game.place_stone(*rng.choice(moves))

        if game.is_terminal:
            continue

        # Try to inject a 4-in-a-row for current player on a random axis
        player = game.current_player
        rng.shuffle(axes)
        for dq, dr in axes:
            p_stones = game.get_stones(player)
            candidates = list(p_stones)[:10]
            rng.shuffle(candidates)

            placed = False
            for sq, sr in candidates:
                # Try line starting from this stone
                line = [(sq + dq * i, sr + dr * i) for i in range(5)]
                # Check all 5 cells are either already ours or empty
                valid = True
                to_place = []
                for pos in line:
                    if pos in game.occupied and pos not in p_stones:
                        valid = False
                        break
                    if pos not in game.occupied:
                        to_place.append(pos)

                if not valid or len(to_place) < 1 or len(to_place) > 2:
                    continue

                # Place all but one (leave the winning move)
                g2 = game.clone()
                winning_move = to_place[-1]
                success = True
                for pos in to_place[:-1]:
                    if g2.is_terminal:
                        success = False
                        break
                    g2.place_stone(*pos)

                if success and not g2.is_terminal:
                    # Export as position dict
                    pos_dict = {}
                    for s in g2.stones_0:
                        pos_dict[s] = 0
                    for s in g2.stones_1:
                        pos_dict[s] = 1
                    puzzles.append(pos_dict)
                    placed = True
                    break

            if placed:
                break

    return puzzles


# ---------------------------------------------------------------------------
# Human game importer
# ---------------------------------------------------------------------------

def load_human_games(path: str = "human_games.jsonl",
                     max_games: int = 5000,
                     min_elo: int = 800) -> List[TrainingSample]:
    """Load human games from JSONL file and convert to training samples."""
    if not _os.path.exists(path):
        return []

    samples: List[TrainingSample] = []
    games_loaded = 0

    with open(path, 'r') as f:
        for line in f:
            if games_loaded >= max_games:
                break
            try:
                record = _json.loads(line)
            except (_json.JSONDecodeError, ValueError):
                continue

            # Filter by ELO
            players = record.get("players", [])
            if min_elo > 0:
                elos = [p.get("elo", 0) or 0 for p in players]
                if elos and max(elos) < min_elo:
                    continue

            # Get moves sorted by moveNumber
            moves = record.get("moves", [])
            if not moves or len(moves) < 10:
                continue
            moves = sorted(moves, key=lambda m: m["moveNumber"])

            # Map playerIds to 0/1
            player_map = {}
            for m in moves:
                pid = m["playerId"]
                if pid not in player_map:
                    player_map[pid] = len(player_map)

            # Determine winner
            result = record.get("gameResult", {})
            winner_pid = result.get("winningPlayerId")
            winner = player_map.get(winner_pid) if winner_pid else None

            # Replay game and create samples
            game = HexGame(candidate_radius=3, max_total_stones=300)
            game_samples = []
            valid = True

            for m in moves:
                q, r = m["x"], m["y"]
                player = player_map[m["playerId"]]

                # Verify turn matches
                if game.current_player != player:
                    valid = False
                    break

                # Encode state
                try:
                    encoded, oq, orr = encode_state(game)
                except Exception:
                    valid = False
                    break

                # Policy: human's move is the target
                policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
                i, j = q - oq, r - orr
                if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
                    policy[i * BOARD_SIZE + j] = 1.0
                else:
                    try:
                        game.place_stone(q, r)
                    except Exception:
                        valid = False
                        break
                    continue

                threat = compute_threat_label(game)
                game_samples.append(TrainingSample(
                    encoded_state=encoded,
                    policy_target=policy,
                    player=player,
                    threat_label=threat,
                    priority=1.5,
                ))

                try:
                    game.place_stone(q, r)
                except Exception:
                    valid = False
                    break

            if not valid or not game_samples:
                continue

            # Fill results
            for s in game_samples:
                if winner is not None:
                    s.result = 1.0 if s.player == winner else -1.0
                else:
                    s.result = 0.0

            samples.extend(game_samples)
            games_loaded += 1

    return samples


def load_online_games(path: str = "online_games.jsonl",
                      start_line: int = 0) -> Tuple[List[TrainingSample], int]:
    """Load online bot games and convert to training samples."""
    if not _os.path.exists(path):
        return [], 0

    samples: List[TrainingSample] = []
    lines_read = 0

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i < start_line:
                lines_read = i + 1
                continue
            lines_read = i + 1

            try:
                record = _json.loads(line)
            except (_json.JSONDecodeError, ValueError):
                continue

            moves = record.get('moves', [])
            if not moves or len(moves) < 4:
                continue

            result_str = record.get('result', '')
            my_player = record.get('my_player', 0)

            # Determine winner from bot's perspective
            if result_str == 'WIN':
                winner = my_player
            elif result_str == 'LOSS':
                winner = 1 - my_player
            else:
                winner = None

            # Replay game and create samples
            game = HexGame(candidate_radius=3, max_total_stones=300)
            game_samples = []
            valid = True

            for move in moves:
                if not isinstance(move, (list, tuple)) or len(move) < 2:
                    continue
                q, r = int(move[0]), int(move[1])
                player = game.current_player

                try:
                    encoded, oq, orr = encode_state(game)
                except Exception:
                    valid = False
                    break

                # Policy: the played move is the target
                policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
                ci, cj = q - oq, r - orr
                if 0 <= ci < BOARD_SIZE and 0 <= cj < BOARD_SIZE:
                    policy[ci * BOARD_SIZE + cj] = 1.0
                else:
                    try:
                        game.place_stone(q, r)
                    except Exception:
                        valid = False
                        break
                    continue

                threat = compute_threat_label(game)
                game_samples.append(TrainingSample(
                    encoded_state=encoded,
                    policy_target=policy,
                    player=player,
                    threat_label=threat,
                    priority=2.0,
                ))

                try:
                    game.place_stone(q, r)
                except Exception:
                    valid = False
                    break

            if not valid or not game_samples:
                continue

            for s in game_samples:
                if winner is not None:
                    s.result = 1.0 if s.player == winner else -1.0
                else:
                    s.result = 0.0

            samples.extend(game_samples)

    return samples, lines_read


# ---------------------------------------------------------------------------
# Symmetry augmentation
# ---------------------------------------------------------------------------

def augment_sample(sample: TrainingSample) -> List[TrainingSample]:
    """Apply valid hex-on-grid symmetry augmentations.
    Returns 3 new samples: 180 rotation, transpose, transpose+180."""
    state = sample.encoded_state.numpy()  # (C, N, N)
    policy = sample.policy_target.reshape(BOARD_SIZE, BOARD_SIZE)
    aug = []

    transforms = [
        # 180 rotation: (i,j) -> (N-1-i, N-1-j)
        (lambda s: s[:, ::-1, ::-1].copy(),
         lambda p: p[::-1, ::-1].copy()),
        # Transpose: (i,j) -> (j,i)
        (lambda s: s.transpose(0, 2, 1).copy(),
         lambda p: p.T.copy()),
        # Transpose + 180: (i,j) -> (N-1-j, N-1-i)
        (lambda s: s[:, ::-1, ::-1].transpose(0, 2, 1).copy(),
         lambda p: p[::-1, ::-1].T.copy()),
    ]

    for s_fn, p_fn in transforms:
        s_new = s_fn(state)
        p_new = p_fn(policy).flatten()
        ps = p_new.sum()
        if ps > 0:
            p_new = p_new / ps
        aug.append(TrainingSample(
            encoded_state=torch.from_numpy(np.ascontiguousarray(s_new)),
            policy_target=np.ascontiguousarray(p_new),
            player=sample.player,
            result=sample.result,
            threat_label=sample.threat_label,
            priority=sample.priority * 0.8,
        ))

    return aug


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------

def self_play_game(
    net,
    mcts: MCTS,
    temp_threshold: int = TEMP_THRESHOLD,
    start_position: Optional[Dict[Tuple[int, int], int]] = None,
    hint_moves: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[List[TrainingSample], List[Tuple[int, int]]]:
    """Play one self-play game. Returns (samples, move_history)."""
    if start_position:
        game = setup_position(start_position)
    else:
        game = HexGame(candidate_radius=3, max_total_stones=200)
    samples: List[TrainingSample] = []
    move_history: List[Tuple[int, int]] = []
    move_count = 0

    while not game.is_terminal:
        temperature = 1.0 if move_count < temp_threshold else 0.01
        add_noise = move_count < temp_threshold

        # Fast forced-move detection - skip NN for obvious wins/blocks
        forced = find_forced_move(game)
        if forced:
            policy = {forced: 1.0}
        else:
            # MCTS search
            policy = mcts.search(game, temperature=temperature, add_noise=add_noise)
            if not policy:
                break

            # Hint move blending: soft guide toward known-correct moves
            if hint_moves and move_count < len(hint_moves):
                hint = hint_moves[move_count]
                if hint in policy:
                    for m in policy:
                        policy[m] *= 0.7
                    policy[hint] += 0.3
                    total_p = sum(policy.values())
                    if total_p > 0:
                        policy = {m: p / total_p for m, p in policy.items()}

        # Encode state for training
        encoded, oq, orr = encode_state(game)
        policy_target = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        for (q, r), prob in policy.items():
            i, j = q - oq, r - orr
            if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
                policy_target[i * BOARD_SIZE + j] = prob
        # Renormalize
        s = policy_target.sum()
        if s > 0:
            policy_target /= s

        threat = compute_threat_label(game)
        samples.append(TrainingSample(
            encoded_state=encoded,
            policy_target=policy_target,
            player=game.current_player,
            threat_label=threat,
        ))

        # Sample and play move
        moves = list(policy.keys())
        probs = np.array([policy[m] for m in moves], dtype=np.float64)
        probs /= probs.sum()
        idx = np.random.choice(len(moves), p=probs)
        chosen = moves[idx]
        move_history.append(chosen)
        game.place_stone(*chosen)
        move_count += 1

    # Fill in results + hindsight: boost priority for late-game positions
    n = len(samples)
    for i, sample in enumerate(samples):
        sample.result = game.result_for(sample.player)
        if i >= n - 5:
            sample.priority = 3.0
        elif i >= n - 15:
            sample.priority = 2.0

    return samples, move_history


# ---------------------------------------------------------------------------
# C-engine self-play (combines CGameState + BatchedMCTS)
# ---------------------------------------------------------------------------

def self_play_game_v2(
    net,
    mcts: BatchedMCTS,
    temp_threshold: int = TEMP_THRESHOLD,
    start_position: Optional[Dict[Tuple[int, int], int]] = None,
    hint_moves: Optional[List[Tuple[int, int]]] = None,
    use_c_engine: bool = True,
    move_callback=None,
) -> Tuple[List[TrainingSample], List[Tuple[int, int]]]:
    """Fast self-play using C engine + batched MCTS."""
    if use_c_engine:
        game = CGameState(max_total_stones=200)
        enc_fn = c_encode_state
        threat_fn = c_compute_threat_label
    else:
        game = HexGame(candidate_radius=3, max_total_stones=200)
        enc_fn = encode_state
        threat_fn = compute_threat_label

    if start_position:
        # Apply position
        p0 = [(pos, p) for pos, p in start_position.items() if p == 0]
        p1 = [(pos, p) for pos, p in start_position.items() if p == 1]
        moves = []
        i0 = i1 = 0
        if p0:
            moves.append(p0[0][0]); i0 = 1
        while i0 < len(p0) or i1 < len(p1):
            for _ in range(2):
                if i1 < len(p1):
                    moves.append(p1[i1][0]); i1 += 1
            for _ in range(2):
                if i0 < len(p0):
                    moves.append(p0[i0][0]); i0 += 1
        for m in moves:
            game.place_stone(*m)

    samples: List[TrainingSample] = []
    move_history: List[Tuple[int, int]] = []
    move_count = 0
    consecutive_bad = 0  # resign threshold counter

    while not game.is_terminal:
        temperature = 1.0 if move_count < temp_threshold else 0.01
        add_noise = move_count < temp_threshold

        # Let MCTS decide everything - no forced moves during training.
        policy = mcts.search(game, temperature=temperature, add_noise=add_noise)
        if not policy:
            break

        # --- DISTANT EXPLORATION: inject gap candidates into policy ---
        if PLAY_STYLE == 'distant' and move_count < 15:
            existing = _get_existing_stones(game)
            if existing and len(policy) > 0:
                lo, hi = DISTANT_RANGE
                gap_candidates = []
                for sq, sr in existing:
                    for dq in range(-hi, hi + 1):
                        for dr in range(-hi, hi + 1):
                            d = abs(dq) + abs(dr)
                            if d < lo or d > hi:
                                continue
                            cq, cr = sq + dq, sr + dr
                            if (cq, cr) not in existing and (cq, cr) not in policy:
                                gap_candidates.append((cq, cr))
                if gap_candidates:
                    n_inject = min(3, len(gap_candidates))
                    injected = random.sample(gap_candidates, n_inject)
                    inject_each = 0.05 / n_inject
                    for m in injected:
                        policy[m] = inject_each
                    # Renormalize so probabilities sum to exactly 1.0
                    total = sum(policy.values())
                    if total > 0:
                        policy = {m: p / total for m, p in policy.items()}

            # Hint move blending: soft guide toward known-correct moves
            if hint_moves and move_count < len(hint_moves):
                hint = hint_moves[move_count]
                if hint in policy:
                    for m in policy:
                        policy[m] *= 0.7
                    policy[hint] += 0.3
                    total_p = sum(policy.values())
                    if total_p > 0:
                        policy = {m: p / total_p for m, p in policy.items()}

        encoded, oq, orr = enc_fn(game)
        policy_target = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        for (q, r), prob in policy.items():
            i, j = q - oq, r - orr
            if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
                policy_target[i * BOARD_SIZE + j] = prob
        s = policy_target.sum()
        if s > 0:
            policy_target /= s

        threat = threat_fn(game)
        samples.append(TrainingSample(
            encoded_state=encoded,
            policy_target=policy_target,
            player=game.current_player,
            threat_label=threat,
        ))

        moves_list = list(policy.keys())
        probs = np.array([policy[m] for m in moves_list], dtype=np.float64)
        probs /= probs.sum()
        idx = np.random.choice(len(moves_list), p=probs)
        chosen = moves_list[idx]
        move_history.append(chosen)
        prev_player = game.current_player
        game.place_stone(*chosen)
        move_count += 1

        # Stream move to live display if callback provided
        if move_callback is not None:
            try:
                move_callback(chosen, prev_player)
            except Exception:
                pass

        # Fork detection: boost priority if this move created 2+ winning threats
        if samples:
            try:
                if hasattr(game, '_ptr'):
                    threats = game._lib.board_count_winning_moves(game._ptr, prev_player)
                else:
                    threats = sum(1 for m in game.legal_moves()
                                  if _line_through_candidate(game, m[0], m[1], prev_player) >= 6)
                if threats >= 3:
                    samples[-1].priority = max(samples[-1].priority, 5.0)
                elif threats >= 2:
                    samples[-1].priority = max(samples[-1].priority, 3.5)
            except Exception:
                pass  # don't crash training on threat counting

    n = len(samples)
    for i, sample in enumerate(samples):
        sample.result = game.result_for(sample.player)
        if i >= n - 5:
            sample.priority = 3.0
        elif i >= n - 15:
            sample.priority = 2.0

    # Penalize short games
    if n < 30:
        for sample in samples:
            sample.priority *= 0.5

    # --- DIVERSITY BONUS ---
    if PLAY_STYLE == 'distant' and game.total_stones > 10:
        all_stones = list(_get_existing_stones(game))
        if all_stones:
            qs = [s[0] for s in all_stones]
            rs = [s[1] for s in all_stones]
            spread = max(max(qs) - min(qs), max(rs) - min(rs))
            if spread >= 8:
                for sample in samples:
                    sample.priority *= 1.5

    return samples, move_history


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

_ZERO_THREAT = np.zeros(4, dtype=np.float32)

def train_step(
    net: HexNet,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    grad_scaler=None,
) -> Dict[str, float]:
    """One training step with threat aux loss + priority updates."""
    net.train()
    batch, indices = replay_buffer.sample(batch_size)
    n = len(batch)

    # Pre-allocate tensors
    states = torch.zeros(n, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    policies = np.empty((n, BOARD_SIZE * BOARD_SIZE), dtype=np.float32)
    values = np.empty(n, dtype=np.float32)
    threats = np.empty((n, 4), dtype=np.float32)

    for i, s in enumerate(batch):
        t = s.encoded_state
        c = t.shape[0]
        states[i, :c] = t
        policies[i] = s.policy_target
        values[i] = s.result
        threats[i] = s.threat_label if s.threat_label is not None else _ZERO_THREAT

    # Single transfer to device
    states = states.to(device, non_blocking=True)
    target_policies = torch.from_numpy(policies).to(device, non_blocking=True)
    target_values = torch.from_numpy(values).to(device, non_blocking=True)
    threat_targets = torch.from_numpy(threats).to(device, non_blocking=True)

    # Mixed-precision
    use_amp = grad_scaler is not None and device.type == 'cuda'
    amp_ctx = torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()

    with amp_ctx:
        policy_logits, value_preds, threat_preds = net(states)
        value_preds = value_preds.squeeze(-1)

        # Losses
        value_loss = F.mse_loss(value_preds, target_values)
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -torch.sum(target_policies * log_probs, dim=1).mean()
        threat_loss = F.binary_cross_entropy_with_logits(threat_preds, threat_targets)
        loss = value_loss + policy_loss + 0.5 * threat_loss

    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        loss.backward()
        optimizer.step()

    # TD-error priority updates
    if indices is not None and random.random() < 0.5:
        with torch.no_grad():
            value_err = (value_preds - target_values).abs()
            td_errors = (value_err + 0.1).cpu().numpy().tolist()
        replay_buffer.update_priorities(indices, td_errors)

    return {
        'total': loss.item(),
        'value': value_loss.item(),
        'policy': policy_loss.item(),
        'threat': threat_loss.item(),
    }


def train(
    num_iterations: int = 100,
    games_per_iter: int = 25,
    train_steps_per_iter: int = 200,
    num_simulations: int = NUM_SIMULATIONS,
    observer=None,
):
    from orca.network import HexNet as _HexNet

    def _get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = _get_device()
    print(f"Device: {device}")

    net = _HexNet().to(device)
    param_count = sum(p.numel() for p in net.parameters())
    print(f"HexNet: {param_count:,} parameters")

    optimizer = torch.optim.Adam(
        net.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG
    )
    replay_buffer = ReplayBuffer()
    mcts = MCTS(net, num_simulations=num_simulations)

    for iteration in range(num_iterations):
        if observer and observer.should_stop():
            break
        if observer:
            observer.on_iteration_start(iteration, num_iterations)

        t0 = time.perf_counter()

        # --- Self-play ---
        net.eval()
        total_samples = 0
        total_moves = 0
        wins = [0, 0, 0]  # p0, p1, draw
        for g in range(games_per_iter):
            if observer and observer.should_stop():
                break
            samples, move_history = self_play_game(net, mcts)
            for s in samples:
                replay_buffer.push(s)
            total_samples += len(samples)
            total_moves += len(move_history)

            result = 0.0
            if samples:
                result = samples[0].result
                if result > 0:
                    wins[0] += 1
                elif result < 0:
                    wins[1] += 1
                else:
                    wins[2] += 1

            if observer:
                observer.on_game_complete(
                    g, games_per_iter, move_history, result, len(samples)
                )

        sp_time = time.perf_counter() - t0

        # --- Training ---
        if len(replay_buffer) < BATCH_SIZE:
            print(
                f"Iter {iteration+1}: {total_samples} samples from "
                f"{games_per_iter} games ({sp_time:.1f}s), "
                f"buffer too small to train ({len(replay_buffer)}/{BATCH_SIZE})"
            )
            continue

        t1 = time.perf_counter()
        losses = None
        for _ in range(train_steps_per_iter):
            losses = train_step(net, optimizer, replay_buffer, device)
        train_time = time.perf_counter() - t1

        avg_game_len = total_moves / games_per_iter if games_per_iter else 0
        metrics = {
            'iteration': iteration + 1,
            'games': games_per_iter,
            'samples': total_samples,
            'wins': wins,
            'self_play_time': round(sp_time, 1),
            'train_time': round(train_time, 1),
            'loss': losses,
            'buffer_size': len(replay_buffer),
            'avg_game_length': round(avg_game_len, 1),
        }

        print(
            f"Iter {iteration+1}: "
            f"games={games_per_iter} samples={total_samples} "
            f"wins=P0:{wins[0]}/P1:{wins[1]}/D:{wins[2]} "
            f"sp={sp_time:.1f}s train={train_time:.1f}s | "
            f"loss={losses['total']:.4f} "
            f"(v={losses['value']:.4f} p={losses['policy']:.4f}) "
            f"buf={len(replay_buffer)}"
        )

        if observer:
            observer.on_iteration_complete(metrics)

        # --- Checkpoint ---
        if (iteration + 1) % 10 == 0:
            path = f"hex_checkpoint_{iteration+1}.pt"
            torch.save({
                'iteration': iteration,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f"  Saved {path}")

    if observer:
        observer.on_training_complete()

    return net, replay_buffer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    'TrainingSample', 'ReplayBuffer',
    'POSITION_CATALOG', 'GUIDED_POSITIONS',
    'get_guided_positions_by_level', 'setup_position',
    'generate_puzzles',
    'load_human_games', 'load_online_games',
    'augment_sample',
    'self_play_game', 'self_play_game_v2',
    '_get_existing_stones',
    'train_step', 'train',
]
