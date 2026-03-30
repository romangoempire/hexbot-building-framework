"""
Neural network + MCTS training system for Hexagonal Connect-6.

Contains the full AlphaZero-style training pipeline:
- HexNet: ResNet with policy, value, and threat heads
- MCTS / BatchedMCTS: Monte Carlo Tree Search with progressive widening
- BatchedNNAlphaBeta: NN-guided alpha-beta with batched leaf evaluation
- ReplayBuffer: prioritized experience replay
- Self-play and training loop

This module powers the neural network features in hexbot.py.
For the game engine, see main.py and hexgame.py.

Usage:
    python bot.py train          # start self-play training
    python bot.py test           # quick smoke test
    python bot.py play           # play interactively
"""

from __future__ import annotations

import collections
import ctypes
import math
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# Config
# ---------------------------------------------------------------------------

BOARD_SIZE = 19
NUM_CHANNELS = 7  # was 5: added 2 threat channels for deeper tactical awareness
NUM_FILTERS = 128     # 128 filters — fast training, search compensates for width
NUM_RES_BLOCKS = 12   # 12 blocks — deeper tower for complex pattern recognition

C_PUCT = 1.5

# Play style: 'distant' explores spread-out colony placements,
#              'close' keeps classic adjacent-only play.
# Switch this before training to choose your strategy.
PLAY_STYLE = 'distant'   # 'distant' or 'close'

# --- Distant play tuning (only used when PLAY_STYLE == 'distant') ---
C_BLEND_ADJACENT = 0.15      # C heuristic weight for adjacent moves (was 0.30)
C_BLEND_DISTANT  = 0.05      # C heuristic weight for far moves (was 0.15)
DISTANT_EXPLORE_PROB = 0.25   # chance to force a gap placement per move
DISTANT_RANGE = (2, 5)        # min/max distance from nearest existing stone

DIRICHLET_ALPHA = 0.3 if PLAY_STYLE == 'distant' else 0.15
DIRICHLET_EPSILON = 0.25
NUM_SIMULATIONS = 400
TEMP_THRESHOLD = 35  # was 20 — explore longer into mid-game where we're weakest

REPLAY_BUFFER_SIZE = 400_000  # 4x larger for diverse experience
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
L2_REG = 1e-4

HALF = BOARD_SIZE // 2  # 9


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Threat helpers for encoding
# ---------------------------------------------------------------------------

_AXES = ((1, 0), (0, 1), (1, -1))


def _threat_line_at(game, q: int, r: int, player: int) -> int:
    """Max consecutive stones for player if they placed at (q,r).
    Counts existing stones on each axis through (q,r)."""
    if hasattr(game, 'stones_0'):
        stones = game.stones_0 if player == 0 else game.stones_1
    elif hasattr(game, '_get_stones_set'):
        stones = game._get_stones_set(player)
    else:
        return 0
    best = 0
    for dq, dr in _AXES:
        c = 1
        nq, nr = q + dq, r + dr
        while (nq, nr) in stones:
            c += 1
            nq += dq
            nr += dr
        nq, nr = q - dq, r - dr
        while (nq, nr) in stones:
            c += 1
            nq -= dq
            nr -= dr
        if c > best:
            best = c
    return best


# ---------------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------------

def encode_state(game: HexGame) -> Tuple[torch.Tensor, int, int]:
    """Encode game state as (7, BOARD_SIZE, BOARD_SIZE) tensor.
    Returns (tensor, offset_q, offset_r) where offsets map window to axial coords.
    """
    s0 = game.stones_0
    s1 = game.stones_1
    occ = game.occupied
    player = game.current_player

    # Centroid
    if occ:
        sum_q = sum_r = 0
        for q, r in occ:
            sum_q += q
            sum_r += r
        n = len(occ)
        cq = round(sum_q / n)
        cr = round(sum_r / n)
    else:
        cq, cr = 0, 0

    offset_q = cq - HALF
    offset_r = cr - HALF

    planes = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Current player's stones = plane 0, opponent = plane 1
    cur_stones = s0 if player == 0 else s1
    opp_stones = s1 if player == 0 else s0

    for q, r in cur_stones:
        i, j = q - offset_q, r - offset_r
        if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
            planes[0, i, j] = 1.0

    for q, r in opp_stones:
        i, j = q - offset_q, r - offset_r
        if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
            planes[1, i, j] = 1.0

    # Legal moves = plane 2
    if occ:
        for q, r in game.candidates:
            i, j = q - offset_q, r - offset_r
            if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
                planes[2, i, j] = 1.0
    else:
        # First move: only (0,0)
        i, j = -offset_q, -offset_r
        if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
            planes[2, i, j] = 1.0

    # Plane 3: current player indicator
    planes[3, :, :] = float(player)

    # Plane 4: stones remaining this turn (normalized)
    remaining = game.stones_per_turn - game.stones_this_turn
    planes[4, :, :] = remaining / 2.0

    # Plane 5: current player's threat map (where placing creates 4+ in a row)
    # Plane 6: opponent's threat map (where opponent has 4+ in a row potential)
    # These give the network INFORMATION about threats — it decides what to do
    if occ:
        cands = game.candidates if hasattr(game, 'candidates') else []
        for q, r in cands:
            i, j = q - offset_q, r - offset_r
            if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
                my_line = _threat_line_at(game, q, r, player)
                opp_line = _threat_line_at(game, q, r, 1 - player)
                if my_line >= 4:
                    planes[5, i, j] = min(my_line / 6.0, 1.0)
                if opp_line >= 4:
                    planes[6, i, j] = min(opp_line / 6.0, 1.0)

    return torch.from_numpy(planes), offset_q, offset_r


def decode_policy(
    policy_logits: torch.Tensor,
    game: HexGame,
    offset_q: int,
    offset_r: int,
) -> Dict[Tuple[int, int], float]:
    """Convert raw policy logits to {(q,r): probability} over legal moves."""
    logits = policy_logits.cpu().numpy().astype(np.float64)

    if not game.occupied:
        return {(0, 0): 1.0}

    legal = game.candidates
    mask = np.full(BOARD_SIZE * BOARD_SIZE, -1e9, dtype=np.float64)

    for q, r in legal:
        i, j = q - offset_q, r - offset_r
        if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
            idx = i * BOARD_SIZE + j
            mask[idx] = logits[idx]

    # Softmax
    mask -= mask.max()
    exp = np.exp(mask)
    total = exp.sum()
    if total < 1e-30:
        # Fallback: uniform over legal moves
        p = 1.0 / len(legal)
        return {m: p for m in legal}

    probs = exp / total
    result = {}
    for q, r in legal:
        i, j = q - offset_q, r - offset_r
        if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
            idx = i * BOARD_SIZE + j
            result[(q, r)] = probs[idx]
        # Moves outside the window get no probability — that's fine

    # Renormalize
    s = sum(result.values())
    if s > 1e-30:
        for k in result:
            result[k] /= s
    else:
        p = 1.0 / len(legal)
        return {m: p for m in legal}

    return result


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


def migrate_checkpoint_5to7(state_dict: dict) -> dict:
    """Migrate a 5-channel checkpoint to 7-channel architecture.
    Preserves all learned weights; new channels initialized to zero."""
    key = 'conv_init.weight'
    if key in state_dict:
        w = state_dict[key]
        if w.shape[1] == 5:
            nf = w.shape[0]
            new_w = torch.zeros(nf, 7, 3, 3, dtype=w.dtype, device=w.device)
            new_w[:, :5, :, :] = w
            state_dict[key] = new_w
            print(f'  ✓ Migrated conv_init.weight: {w.shape} → {new_w.shape} (2 threat channels added)')
    return state_dict


def migrate_checkpoint_filters(state_dict: dict, target_filters: int = NUM_FILTERS,
                                target_blocks: int = NUM_RES_BLOCKS) -> dict:
    """Migrate a smaller network checkpoint to a larger one.
    Pads filter dimensions with small random values. Copies existing blocks,
    initializes new blocks randomly."""
    migrated = False
    new_sd = {}

    for key, tensor in state_dict.items():
        new_sd[key] = tensor  # default: copy as-is

    # Check if migration is needed by looking at conv_init
    init_key = 'conv_init.weight'
    if init_key in new_sd:
        old_filters = new_sd[init_key].shape[0]
        if old_filters < target_filters:
            print(f'  ✓ Migrating {old_filters}→{target_filters} filters, expanding network...')
            migrated = True

            # Helper: pad a tensor's filter dimensions
            def pad_filters(t, dim0=None, dim1=None):
                if t.dim() == 4:  # Conv weight: (out, in, H, W)
                    out_f = dim0 or t.shape[0]
                    in_f = dim1 or t.shape[1]
                    new_t = torch.randn(out_f, in_f, t.shape[2], t.shape[3],
                                        dtype=t.dtype) * 0.01
                    new_t[:t.shape[0], :t.shape[1], :, :] = t
                    return new_t
                elif t.dim() == 1:  # BN weight/bias/running_mean/running_var
                    new_t = torch.zeros(dim0 or t.shape[0], dtype=t.dtype)
                    new_t[:t.shape[0]] = t
                    if 'weight' in key:  # BN weight should be 1, not 0
                        new_t[t.shape[0]:] = 1.0
                    return new_t
                elif t.dim() == 2:  # Linear weight: (out, in)
                    out_f = dim0 or t.shape[0]
                    in_f = dim1 or t.shape[1]
                    new_t = torch.randn(out_f, in_f, dtype=t.dtype) * 0.01
                    new_t[:t.shape[0], :t.shape[1]] = t
                    return new_t
                return t

            nf = target_filters
            nc = new_sd[init_key].shape[1]  # input channels (7)
            bs2 = BOARD_SIZE * BOARD_SIZE

            # conv_init: (old_f, 7, 3, 3) → (nf, 7, 3, 3)
            new_sd[init_key] = pad_filters(new_sd[init_key], nf, nc)
            new_sd['bn_init.weight'] = pad_filters(new_sd['bn_init.weight'], nf)
            new_sd['bn_init.bias'] = pad_filters(new_sd['bn_init.bias'], nf)
            new_sd['bn_init.running_mean'] = pad_filters(new_sd['bn_init.running_mean'], nf)
            new_sd['bn_init.running_var'] = pad_filters(new_sd.get('bn_init.running_var',
                                                        torch.ones(old_filters)), nf)
            if 'bn_init.num_batches_tracked' in new_sd:
                pass  # scalar, no migration needed

            # Res blocks: migrate existing, leave new ones for random init
            for i in range(target_blocks):
                prefix = f'res_blocks.{i}'
                for suffix in ['.conv1.weight', '.bn1.weight', '.bn1.bias',
                               '.bn1.running_mean', '.bn1.running_var',
                               '.conv2.weight', '.bn2.weight', '.bn2.bias',
                               '.bn2.running_mean', '.bn2.running_var']:
                    k = prefix + suffix
                    if k in new_sd:
                        t = new_sd[k]
                        if 'conv' in suffix and t.dim() == 4:
                            new_sd[k] = pad_filters(t, nf, nf)
                        elif t.dim() == 1:
                            new_sd[k] = pad_filters(t, nf)
                    # If key doesn't exist (new block), it'll be randomly initialized by the model

            # Policy head: conv (nf→2), fc (2*bs2 → bs2)
            if 'policy_conv.weight' in new_sd:
                new_sd['policy_conv.weight'] = pad_filters(
                    new_sd['policy_conv.weight'], 2, nf)
            # policy_fc stays same size (2*bs2 → bs2)

            # Value head: conv (nf→1), fc1 (bs2→256), fc2 (256→1)
            if 'value_conv.weight' in new_sd:
                new_sd['value_conv.weight'] = pad_filters(
                    new_sd['value_conv.weight'], 1, nf)

            # Threat head: conv (nf→1)
            if 'threat_conv.weight' in new_sd:
                new_sd['threat_conv.weight'] = pad_filters(
                    new_sd['threat_conv.weight'], 1, nf)

            print(f'  ✓ Network expanded: {old_filters}→{nf} filters')

    return new_sd


class HexNet(nn.Module):
    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        num_channels: int = NUM_CHANNELS,
        num_filters: int = NUM_FILTERS,
        num_res_blocks: int = NUM_RES_BLOCKS,
    ):
        super().__init__()
        self.board_size = board_size
        bs2 = board_size * board_size

        # Initial block
        self.conv_init = nn.Conv2d(num_channels, num_filters, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(num_filters)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * bs2, bs2)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(bs2, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Threat detection head: predicts [my_4, my_5, opp_4, opp_5] in a row
        self.threat_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.threat_bn = nn.BatchNorm2d(1)
        self.threat_fc = nn.Linear(bs2, 4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Initial block
        x = F.relu(self.bn_init(self.conv_init(x)))
        # Residual tower
        x = self.res_blocks(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)  # raw logits (batch, BS*BS)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # (batch, 1)

        # Threat head
        t = F.relu(self.threat_bn(self.threat_conv(x)))
        t = t.view(t.size(0), -1)
        t = self.threat_fc(t)  # (batch, 4)

        return p, v, t

    def forward_pv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Policy + value only (for ONNX export and inference)."""
        x = F.relu(self.bn_init(self.conv_init(x)))
        x = self.res_blocks(x)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return p, v

    @torch.no_grad()
    def predict(
        self, encoded_state: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Single-state inference. Returns (policy_logits, value_scalar)."""
        self.eval()
        x = encoded_state.unsqueeze(0)
        if next(self.parameters()).device.type != x.device.type:
            x = x.to(next(self.parameters()).device)
        p, v = self.forward_pv(x)
        return p.squeeze(0).cpu(), v.item()


# ---------------------------------------------------------------------------
# ONNX Runtime predictor (6x faster than PyTorch for CPU inference)
# ---------------------------------------------------------------------------

class OnnxPredictor:
    """Drop-in replacement for HexNet.predict() using ONNX Runtime."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(
            onnx_path, sess_options=opts,
            providers=['CPUExecutionProvider']
        )

    def predict(self, encoded_state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        inp = {'state': encoded_state.unsqueeze(0).numpy()}
        policy, value = self.sess.run(None, inp)
        return torch.from_numpy(policy[0]), float(value[0][0])


def export_onnx(net: HexNet, path: str = '/tmp/hex_model.onnx'):
    """Export the policy+value heads to ONNX for fast CPU inference."""
    import warnings
    net_cpu = net.cpu()
    net_cpu.eval()
    dummy = torch.randn(1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)

    # Use a wrapper that calls forward_pv
    class _PVWrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        def forward(self, x):
            return self.net.forward_pv(x)

    wrapper = _PVWrapper(net_cpu)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            wrapper, dummy, path,
            input_names=['state'],
            output_names=['policy', 'value'],
        )
    return path


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = (
        'parent', 'move', 'player',
        'prior', 'visit_count', 'value_sum',
        'children', 'is_expanded',
        '_pending_moves',  # progressive widening: moves not yet expanded
    )

    def __init__(
        self,
        parent: Optional[MCTSNode],
        move: Optional[Tuple[int, int]],
        prior: float,
        player: int,
    ):
        self.parent = parent
        self.move = move
        self.prior = prior
        self.player = player  # current_player when this node is reached
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[Tuple[int, int], MCTSNode] = {}
        self.is_expanded = False
        self._pending_moves: list = []  # sorted (move, prob) pairs waiting to be added

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(
        self,
        net: HexNet,
        c_puct: float = C_PUCT,
        num_simulations: int = NUM_SIMULATIONS,
    ):
        self.net = net
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def search(
        self,
        game: HexGame,
        temperature: float = 1.0,
        add_noise: bool = True,
    ) -> Dict[Tuple[int, int], float]:
        """Run MCTS from current state. Returns {move: probability}."""
        root = MCTSNode(parent=None, move=None, prior=0.0, player=game.current_player)

        # Expand root
        self._expand(root, game)

        # Add Dirichlet noise to root
        if add_noise and root.children:
            noise = np.random.dirichlet(
                [DIRICHLET_ALPHA] * len(root.children)
            )
            for (_, child), n in zip(root.children.items(), noise):
                child.prior = (
                    (1 - DIRICHLET_EPSILON) * child.prior + DIRICHLET_EPSILON * n
                )

        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root, game)

        return self._get_policy(root, temperature)

    def _simulate(self, root: MCTSNode, game: HexGame) -> None:
        node = root
        depth = 0

        # SELECT: descend tree via PUCT, widening as needed
        while node.is_expanded and not game.is_terminal:
            if not node.children:
                break
            # Progressive widening: add more children if this node has enough visits
            self._maybe_widen(node, game)
            move, node = self._select_child(node)
            game.place_stone(move[0], move[1])
            depth += 1

        # EXPAND & EVALUATE
        if game.is_terminal:
            # Value from root player's perspective
            value = game.result_for(root.player)
        else:
            value = self._expand(node, game)

        # BACKUP
        self._backup(node, value, root.player)

        # UNDO
        for _ in range(depth):
            game.undo()

    def _select_child(
        self, node: MCTSNode
    ) -> Tuple[Tuple[int, int], MCTSNode]:
        """Select child with highest PUCT score."""
        c_puct = self.c_puct
        sqrt_parent = math.sqrt(node.visit_count + 1)

        best_score = -float('inf')
        best_move = None
        best_child = None

        for move, child in node.children.items():
            q = child.q_value
            # Negate q if child's player differs from parent's player
            if child.player != node.player:
                q = -q
            score = q + c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    # Progressive widening: start narrow → forces deeper search
    INITIAL_WIDTH = 6    # only 6 children at first → deep tree
    WIDEN_AT_20 = 10     # after 20 visits, expand to 10
    WIDEN_AT_50 = 16     # after 50 visits, expand to 16
    WIDEN_AT_100 = 25    # after 100 visits, expand to 25

    def _expand(self, node: MCTSNode, game: HexGame) -> float:
        """Expand node using NN with progressive widening.
        Only creates top-K children initially. More are added as visits increase.
        This forces the tree DEEPER instead of wider — key for multi-move lookahead."""
        encoded, oq, orr = encode_state(game)
        policy_logits, value = self.net.predict(encoded)
        policy = decode_policy(policy_logits, game, oq, orr)

        # Quiescence boost: if position has double/triple threats, adjust value
        try:
            if hasattr(game, '_lib'):  # CGameState
                p = game.current_player
                wm = game._lib.board_count_winning_moves(game._ptr, p)
                opp_wm = game._lib.board_count_winning_moves(game._ptr, 1 - p)
                if wm >= 3:
                    value = 1.0 if p == 0 else -1.0  # forced win (3+ winning cells, opponent can block max 2)
                elif opp_wm >= 3:
                    value = -1.0 if p == 0 else 1.0  # forced loss
                elif wm >= 2:
                    boost = 0.3 if p == 0 else -0.3
                    value = value * 0.7 + boost  # strong advantage
                elif opp_wm >= 2:
                    boost = -0.3 if p == 0 else 0.3
                    value = value * 0.7 + boost  # strong disadvantage
        except Exception:
            pass  # don't crash on threat detection failure

        # Sort moves by policy probability (highest first)
        sorted_moves = sorted(policy.items(), key=lambda x: x[1], reverse=True)

        # Create children for top-K moves only
        k = self.INITIAL_WIDTH
        for move, prob in sorted_moves[:k]:
            child_player = self._next_player(game, move)
            child = MCTSNode(parent=node, move=move, prior=prob, player=child_player)
            node.children[move] = child

        # Store remaining moves for later widening
        node._pending_moves = sorted_moves[k:]
        node.is_expanded = True
        return value

    def _maybe_widen(self, node: MCTSNode, game: HexGame) -> None:
        """Progressive widening: add more children as node gets more visits."""
        if not node._pending_moves:
            return
        n = node.visit_count
        current = len(node.children)
        if n >= 100 and current < self.WIDEN_AT_100:
            target = self.WIDEN_AT_100
        elif n >= 50 and current < self.WIDEN_AT_50:
            target = self.WIDEN_AT_50
        elif n >= 20 and current < self.WIDEN_AT_20:
            target = self.WIDEN_AT_20
        else:
            return
        to_add = target - current
        for move, prob in node._pending_moves[:to_add]:
            if move not in node.children:
                child_player = self._next_player(game, move)
                child = MCTSNode(parent=node, move=move, prior=prob, player=child_player)
                node.children[move] = child
        node._pending_moves = node._pending_moves[to_add:]

    def _next_player(self, game: HexGame, move: Tuple[int, int]) -> int:
        """Determine current_player after placing this move."""
        stt = game.stones_this_turn + 1
        if stt >= game.stones_per_turn:
            return (game.turn + 1) & 1
        return game.current_player

    def _backup(self, node: MCTSNode, value: float, root_player: int) -> None:
        """Propagate value up the tree. Value is from root_player's perspective."""
        while node is not None:
            node.visit_count += 1
            # value_sum stored from this node's player's perspective
            if node.player == root_player:
                node.value_sum += value
            else:
                node.value_sum -= value
            node = node.parent

    def _get_policy(
        self, root: MCTSNode, temperature: float
    ) -> Dict[Tuple[int, int], float]:
        if not root.children:
            return {}

        if temperature < 1e-8:
            # Greedy: all weight on max visit count
            best = max(root.children.values(), key=lambda c: c.visit_count)
            return {
                m: (1.0 if c is best else 0.0)
                for m, c in root.children.items()
            }

        visits = {}
        for m, c in root.children.items():
            visits[m] = c.visit_count ** (1.0 / temperature)
        total = sum(visits.values())
        if total < 1e-30:
            p = 1.0 / len(visits)
            return {m: p for m in visits}
        return {m: v / total for m, v in visits.items()}


# ---------------------------------------------------------------------------
# Self-play & training data
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
# Forced-move detection — skip NN for obvious moves (huge speedup)
# ---------------------------------------------------------------------------

AXES_3 = ((1, 0), (0, 1), (1, -1))


def _count_line(stones: set, q: int, r: int, dq: int, dr: int) -> int:
    c = 0
    nq, nr = q + dq, r + dr
    while (nq, nr) in stones:
        c += 1
        nq += dq
        nr += dr
    return c


def _max_line_at(stones: set, q: int, r: int) -> int:
    best = 0
    for dq, dr in AXES_3:
        c = 1 + _count_line(stones, q, r, dq, dr) + _count_line(stones, q, r, -dq, -dr)
        if c > best:
            best = c
    return best


def _get_threat_moves(game, player: int, min_line: int = 4) -> List[Tuple[int, int]]:
    """Get moves that create min_line+ in a row for player. Fast scan."""
    threats = []
    if hasattr(game, 'candidates'):
        cands = game.candidates
    else:
        cands = game.legal_moves()
    for q, r in cands:
        line = _line_through_candidate(game, q, r, player)
        if line >= min_line:
            threats.append((q, r))
    return threats


def _count_winning_cells(game, player: int) -> int:
    """Count how many cells would give player 6-in-a-row."""
    count = 0
    if hasattr(game, 'candidates'):
        cands = game.candidates
    else:
        cands = game.legal_moves()
    for q, r in cands:
        if _line_through_candidate(game, q, r, player) >= 6:
            count += 1
    return count


def _count_multi_axis_threats(game, q: int, r: int, player: int) -> int:
    """Count how many axes have 3+ consecutive stones through (q,r) for player."""
    if hasattr(game, 'candidates'):
        stones = game.stones_0 if player == 0 else game.stones_1
    else:
        stones = game.stones_0 if player == 0 else game.stones_1
    test = stones | {(q, r)}
    count = 0
    for dq, dr in AXES_3:
        c = 1 + _count_line(test, q, r, dq, dr) + _count_line(test, q, r, -dq, -dr)
        if c >= 3:
            count += 1
    return count


def _threat_search(game, depth: int = 4) -> Optional[Tuple[int, int]]:
    """Fast threat-space search to find moves creating unstoppable forks.

    Searches forcing lines (moves creating 3+ on multiple axes or 4+ on one axis).
    Much cheaper than MCTS — branching factor ~3-8 instead of 30+.

    Returns the first move of a winning forcing sequence, or None.
    """
    p = game.current_player
    opp = 1 - p

    if hasattr(game, 'candidates'):
        cands = list(game.candidates)
    else:
        cands = game.legal_moves()

    # Score each candidate by threat potential
    scored_moves = []
    for q, r in cands:
        my_line = _line_through_candidate(game, q, r, p)
        opp_line = _line_through_candidate(game, q, r, opp)
        n_axes = _count_multi_axis_threats(game, q, r, p)

        # Only consider threatening moves
        if my_line >= 4 or (my_line >= 3 and n_axes >= 2) or opp_line >= 4:
            score = my_line * 100 + n_axes * 50 + (opp_line * 30 if opp_line >= 4 else 0)
            scored_moves.append(((q, r), score, my_line, n_axes))

    if not scored_moves:
        return None

    scored_moves.sort(key=lambda x: x[1], reverse=True)

    # Check if we have 2 stones this turn — if so, try PAIRS of moves
    stt = game.stones_this_turn if hasattr(game, 'stones_this_turn') else 0
    stones_left_this_turn = (2 - stt) if hasattr(game, 'stones_per_turn') and game.stones_per_turn == 2 else 1

    if stones_left_this_turn >= 2 and len(scored_moves) >= 2:
        # Try pairs: place move1, then check if move2 creates unstoppable fork
        for i, (move1, s1, ml1, ax1) in enumerate(scored_moves[:6]):
            game.place_stone(*move1)
            # After our first stone, check what second stone creates
            for j, (move2, s2, ml2, ax2) in enumerate(scored_moves[:6]):
                if i == j or move2 == move1:
                    continue
                if move2 not in game.candidates:
                    continue
                # Simulate second stone
                game.place_stone(*move2)
                # Check: unstoppable 6-in-row threats?
                win_cells = _count_winning_cells(game, p)
                # Also check: multiple 5-in-row threats (opponent blocks 2, but we have 3+)
                five_threats = len(_get_threat_moves(game, p, min_line=5))
                game.undo()  # undo move2
                if win_cells >= 3 or five_threats >= 3:
                    # This PAIR creates an unstoppable fork!
                    game.undo()  # undo move1
                    return move1  # play the first stone of the pair
            game.undo()  # undo move1

    for move, score, my_line, n_axes in scored_moves[:8]:  # limit breadth
        game.place_stone(*move)

        # Check: did this create an unstoppable position?
        win_cells = _count_winning_cells(game, p)

        if win_cells >= 3:
            # Unstoppable! Opponent gets 2 stones per turn, can't block 3+ threats
            game.undo()
            return move

        # Check: did this create 2 winning cells? If both are far apart, it's a fork.
        if win_cells >= 2:
            game.undo()
            return move

        # Deeper search: does this move lead to a fork after opponent responds?
        if depth > 1 and (my_line >= 4 or n_axes >= 2):
            # Opponent's best responses: block our threats or make their own
            opp_responses = _get_threat_moves(game, opp, min_line=3)
            if not opp_responses:
                # Opponent has no threats — any reasonable move
                opp_responses = _get_threat_moves(game, p, min_line=4)  # block ours

            forces_win = True
            tested = 0
            for opp_move in opp_responses[:4]:  # limit opponent breadth
                game.place_stone(*opp_move)
                tested += 1

                # After opponent blocks, do we still have a forcing sequence?
                sub = _threat_search(game, depth - 2)
                game.undo()

                if sub is None:
                    forces_win = False
                    break

            if forces_win and tested > 0:
                game.undo()
                return move

        game.undo()

    return None


def _line_through_candidate(game, q: int, r: int, player: int) -> int:
    """Max consecutive line length if player places at (q,r). Works on CGameState & HexGame."""
    if hasattr(game, '_ptr'):
        qi, ri = q + 15, r + 15  # OFF=15 in C engine
        return game._lib.board_max_line_through(game._ptr, qi, ri, player)
    # HexGame path
    stones = game.stones_0 if player == 0 else game.stones_1
    test = stones | {(q, r)}
    return _max_line_at(test, q, r)


def find_forced_move(game) -> Optional[Tuple[int, int]]:
    """Only force the ONE truly undeniable move: completing 6-in-a-row to win.

    Everything else (blocking, extending, forking) is left to MCTS so it can
    weigh strategic nuance — e.g. blocking one cell further out may be better
    than the adjacent block if it also builds your own position.

    Returns winning move or None.
    """
    p = game.current_player

    # Get candidates — works for both CGameState and HexGame
    if hasattr(game, 'candidates'):
        cands = game.candidates
    else:
        cands = game.legal_moves()

    for q, r in cands:
        if _line_through_candidate(game, q, r, p) >= 6:
            return (q, r)

    return None


def detect_finisher(game, player: int) -> bool:
    """Detect known winning formations (finishers) for a player.

    A finisher is a stone arrangement that guarantees a win within a few moves
    regardless of opponent's response. Known finishers:
    - Triangle: 3 adjacent stones (the starting formation)
    - Trapezoid: triangle + extension on a second axis
    - Double threat: 3+ cells that would complete 6-in-a-row

    Returns True if player has a finisher (position is won).
    """
    try:
        if hasattr(game, '_lib'):  # CGameState
            # 3+ winning moves = unstoppable (opponent can block max 2 per turn)
            wm = game._lib.board_count_winning_moves(game._ptr, player)
            if wm >= 3:
                return True
        else:
            # HexGame: count winning cells
            stones = game.stones_0 if player == 0 else game.stones_1
            cands = game.candidates if hasattr(game, 'candidates') else game.legal_moves()
            wm = 0
            for q, r in cands:
                test = stones | {(q, r)}
                for dq, dr in [(1, 0), (0, 1), (1, -1)]:
                    c = 1
                    nq, nr = q + dq, r + dr
                    while (nq, nr) in test:
                        c += 1; nq += dq; nr += dr
                    nq, nr = q - dq, r - dr
                    while (nq, nr) in test:
                        c += 1; nq -= dq; nr -= dr
                    if c >= 6:
                        wm += 1
                        break
            if wm >= 3:
                return True
    except Exception:
        pass
    return False


def compute_threat_bonus(game, move: Tuple[int, int], player: int) -> float:
    """Compute a training priority bonus for a move based on threat potential.

    Used in self-play to boost priority of samples where forks/threats are created.
    NOT used to force moves — just to make the training signal richer.

    Returns bonus value (0.0 = normal, up to 5.0 for unstoppable forks).
    """
    game.place_stone(*move)
    bonus = 0.0

    # Count winning cells (6-in-a-row completions)
    win_cells = _count_winning_cells(game, player)
    if win_cells >= 3:
        bonus = 5.0  # unstoppable fork
    elif win_cells >= 2:
        bonus = 3.0  # double threat

    # Count 5-in-a-row threats (one stone from winning)
    if bonus < 3.0:
        five_threats = len(_get_threat_moves(game, player, min_line=5))
        if five_threats >= 3:
            bonus = max(bonus, 4.0)  # multiple 5-threats = near-unstoppable
        elif five_threats >= 2:
            bonus = max(bonus, 2.0)

    # Multi-axis buildup (3+ on 2+ axes)
    if bonus < 2.0:
        n_axes = _count_multi_axis_threats(game, move[0], move[1], player)
        if n_axes >= 2:
            bonus = max(bonus, 1.5)

    game.undo()
    return bonus


def compute_threat_label(game: HexGame) -> np.ndarray:
    """Compute threat features including preemptives (2-in-a-row).

    Returns 4 floats encoding the full threat landscape:
    [0] my threat level — continuous 0-1:
        2-in-a-row = 0.15 (preemptive)
        3-in-a-row = 0.35 (strong preemptive)
        4-in-a-row = 0.60 (active threat)
        5-in-a-row = 0.85 (imminent win)
        6-in-a-row = 1.00 (won)
    [1] my multi-axis score — how many axes have 2+ in a row:
        1 axis = 0.25, 2 axes = 0.60 (proto-fork), 3 axes = 1.0 (dominant)
    [2] opp threat level (same scale)
    [3] opp multi-axis score (same scale)

    This teaches the network to value preemptives (2-in-a-row setups)
    which humans use to build threats 2-3 moves before they become critical.
    """
    player = game.current_player
    my_stones = game.stones_0 if player == 0 else game.stones_1
    opp_stones = game.stones_1 if player == 0 else game.stones_0

    _AXES = ((1, 0), (0, 1), (1, -1))

    all_stones = my_stones | opp_stones

    def analyze_stones(stones: set, blockers: set):
        """Returns (max_live_consecutive, axes_with_2plus, axes_with_3plus).
        Only counts lines that can still extend to 6 (have enough open space)."""
        if not stones:
            return 0, 0, 0
        best = 0
        axes_2plus = 0
        axes_3plus = 0
        for dq, dr in _AXES:
            axis_best = 0
            for q, r in stones:
                # Count consecutive run
                count = 1
                nq, nr = q + dq, r + dr
                while (nq, nr) in stones:
                    count += 1
                    nq += dq
                    nr += dr
                # Check if this line can extend to 6: count open spaces
                # in both directions until blocked
                open_fwd = 0
                fq, fr = q + dq * count, r + dr * count
                while (fq, fr) not in blockers and open_fwd < 6:
                    open_fwd += 1
                    fq += dq
                    fr += dr
                open_bwd = 0
                bq, br = q - dq, r - dr
                while (bq, br) not in blockers and open_bwd < 6:
                    open_bwd += 1
                    bq -= dq
                    br -= dr
                potential = count + open_fwd + open_bwd
                if potential < 6:
                    continue  # dead line, can never reach 6
                if count > axis_best:
                    axis_best = count
                if count > best:
                    best = count
            if axis_best >= 2:
                axes_2plus += 1
            if axis_best >= 3:
                axes_3plus += 1
        return best, axes_2plus, axes_3plus

    my_max, my_axes2, my_axes3 = analyze_stones(my_stones, opp_stones)
    opp_max, opp_axes2, opp_axes3 = analyze_stones(opp_stones, my_stones)

    # Continuous threat level: preemptives start at 2-in-a-row
    # 4 and 5 are nearly identical — both can finish with 1 stone
    def threat_level(max_line):
        if max_line <= 1: return 0.0
        if max_line == 2: return 0.15   # preemptive
        if max_line == 3: return 0.40   # strong preemptive (1 move from threat)
        if max_line == 4: return 0.80   # active threat (can win with 2 more)
        if max_line == 5: return 0.85   # nearly same as 4 (both finish next move)
        return 1.0                       # won

    # Multi-axis score: proto-forks from 2-in-a-rows on multiple axes
    def axis_score(axes_2plus, axes_3plus):
        if axes_3plus >= 2: return 1.0   # fork with 3+ on 2 axes
        if axes_2plus >= 3: return 0.80  # preemptives on all 3 axes
        if axes_3plus >= 1 and axes_2plus >= 2: return 0.70  # mixed
        if axes_2plus >= 2: return 0.50  # proto-fork (2-in-a-row on 2 axes)
        if axes_2plus >= 1: return 0.25  # single axis preemptive
        return 0.0

    return np.array([
        threat_level(my_max),
        axis_score(my_axes2, my_axes3),
        threat_level(opp_max),
        axis_score(opp_axes2, opp_axes3),
    ], dtype=np.float32)


class ReplayBuffer:
    """Prioritized replay buffer — samples proportional to priority."""

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
# Position catalog — predefined starting formations
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
# Guided positions — theory-based curriculum for faster learning
# Level 1: Endgame (trivial wins/blocks)
# Level 2: Formation (mid-game patterns)
# Level 3: Known sequences (with hint moves)
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

    # Place stones respecting turn order: turn 0 = p0 × 1, then alternating × 2
    moves = []
    i0, i1 = 0, 0
    # Turn 0: p0 places 1
    if i0 < len(p0_stones):
        moves.append(p0_stones[i0]); i0 += 1
    # Then alternate: p1 × 2, p0 × 2, p1 × 2, ...
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
# Puzzle generator — tactical positions with forced wins/blocks
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
# Human game importer — learn from real games on hexo.did.science
# ---------------------------------------------------------------------------

def load_human_games(path: str = "human_games.jsonl",
                     max_games: int = 5000,
                     min_elo: int = 800) -> List[TrainingSample]:
    """Load human games from JSONL file and convert to training samples.

    Each game's moves become policy targets (the human's chosen move = 1.0).
    Game outcome determines the value target.
    """
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
                    # Coordinate mismatch or turn issue — skip game
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
                    # Move outside encoding window — skip this sample but continue
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
                    priority=1.5,  # slight boost over self-play
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


import json as _json  # for load_human_games


def load_online_games(path: str = "online_games.jsonl",
                      start_line: int = 0) -> Tuple[List[TrainingSample], int]:
    """Load online bot games and convert to training samples.

    Returns (samples, lines_read) so the caller can track position for incremental loading.
    Online games have higher priority (2.0) since they're against real humans.
    """
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
                    priority=2.0,  # high priority: real human opponent
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
# Symmetry augmentation — 4x data multiplier (2D rotations + flip)
# ---------------------------------------------------------------------------

def augment_sample(sample: TrainingSample) -> List[TrainingSample]:
    """Apply 2D augmentations (rot90, rot180, rot270, hflip) to a sample.
    Returns 4 new augmented samples."""
    state = sample.encoded_state.numpy()  # (5, 19, 19)
    policy = sample.policy_target.reshape(BOARD_SIZE, BOARD_SIZE)
    aug = []

    # 90°, 180°, 270° rotations
    for k in (1, 2, 3):
        s_rot = np.rot90(state, k, axes=(1, 2)).copy()
        p_rot = np.rot90(policy, k).copy().flatten()
        # Renormalize policy
        ps = p_rot.sum()
        if ps > 0:
            p_rot /= ps
        aug.append(TrainingSample(
            encoded_state=torch.from_numpy(s_rot),
            policy_target=p_rot,
            player=sample.player,
            result=sample.result,
            threat_label=sample.threat_label,
            priority=sample.priority * 0.8,  # slightly lower priority for augmented
        ))

    # Horizontal flip
    s_flip = np.ascontiguousarray(state[:, :, ::-1])
    p_flip = np.ascontiguousarray(policy[:, ::-1]).flatten()
    ps = p_flip.sum()
    if ps > 0:
        p_flip /= ps
    aug.append(TrainingSample(
        encoded_state=torch.from_numpy(s_flip),
        policy_target=p_flip,
        player=sample.player,
        result=sample.result,
        threat_label=sample.threat_label,
        priority=sample.priority * 0.8,
    ))

    return aug


def self_play_game(
    net,
    mcts: MCTS,
    temp_threshold: int = TEMP_THRESHOLD,
    start_position: Optional[Dict[Tuple[int, int], int]] = None,
    hint_moves: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[List[TrainingSample], List[Tuple[int, int]]]:
    """Play one self-play game. Returns (samples, move_history).
    Optionally starts from a predefined position with hint moves."""
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

        # Fast forced-move detection — skip NN for obvious wins/blocks
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
        probs = [policy[m] for m in moves]
        idx = np.random.choice(len(moves), p=probs)
        chosen = moves[idx]
        move_history.append(chosen)
        game.place_stone(*chosen)
        move_count += 1

    # Fill in results + hindsight: boost priority for late-game positions
    n = len(samples)
    for i, sample in enumerate(samples):
        sample.result = game.result_for(sample.player)
        # Last 5 positions get higher priority (stronger signal near game end)
        if i >= n - 5:
            sample.priority = 3.0
        elif i >= n - 15:
            sample.priority = 2.0

    return samples, move_history


# ---------------------------------------------------------------------------
# C Engine Integration — CGameState (50x faster game simulation)
# ---------------------------------------------------------------------------

import ctypes
import os as _os

_engine_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'engine.so')
_c_lib = None

def _get_lib():
    global _c_lib
    if _c_lib is None:
        if not _os.path.exists(_engine_path):
            raise RuntimeError(f"C engine not found at {_engine_path}")
        _c_lib = ctypes.CDLL(_engine_path)
        _setup_c_signatures(_c_lib)
    return _c_lib

def _setup_c_signatures(lib):
    lib.board_sizeof.restype = ctypes.c_int
    lib.board_reset.argtypes = [ctypes.c_void_p]
    lib.board_setup_triangle.argtypes = [ctypes.c_void_p]
    lib.board_place.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.board_undo.argtypes = [ctypes.c_void_p]
    lib.board_get_winner.argtypes = [ctypes.c_void_p]
    lib.board_get_winner.restype = ctypes.c_int
    lib.board_get_current_player.argtypes = [ctypes.c_void_p]
    lib.board_get_current_player.restype = ctypes.c_int
    lib.board_get_total_stones.argtypes = [ctypes.c_void_p]
    lib.board_get_total_stones.restype = ctypes.c_int
    lib.board_get_cand_count.argtypes = [ctypes.c_void_p]
    lib.board_get_cand_count.restype = ctypes.c_int
    lib.board_get_stones_this_turn.argtypes = [ctypes.c_void_p]
    lib.board_get_stones_this_turn.restype = ctypes.c_int
    lib.board_get_stones_per_turn.argtypes = [ctypes.c_void_p]
    lib.board_get_stones_per_turn.restype = ctypes.c_int
    _IA = ctypes.POINTER(ctypes.c_int)
    lib.board_get_candidates.argtypes = [ctypes.c_void_p, _IA, _IA]
    lib.board_get_candidates.restype = ctypes.c_int
    _FA = ctypes.POINTER(ctypes.c_float)
    lib.board_encode_state.argtypes = [ctypes.c_void_p, _FA, _IA, _IA]
    lib.board_get_legal_mask.argtypes = [ctypes.c_void_p, _FA, ctypes.c_int, ctypes.c_int]
    lib.board_get_legal_mask.restype = ctypes.c_int
    lib.board_compute_threat_label.argtypes = [ctypes.c_void_p, _FA]
    lib.board_copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

# Pre-allocated ctypes arrays (thread-local would be ideal, but these work for single-threaded)
_IntArr600 = ctypes.c_int * 1200  # radius 3 generates more candidates
_FloatArr = ctypes.c_float * (5 * BOARD_SIZE * BOARD_SIZE)
_FloatArr361 = ctypes.c_float * (BOARD_SIZE * BOARD_SIZE)
_FloatArr4 = ctypes.c_float * 4


class CGameState:
    """Drop-in replacement for HexGame using C engine (50x faster)."""

    __slots__ = ('_buf', '_ptr', '_lib', 'max_total_stones', '_move_log')

    def __init__(self, max_total_stones: int = 200):
        self._lib = _get_lib()
        sz = self._lib.board_sizeof()
        self._buf = ctypes.create_string_buffer(sz)
        self._ptr = ctypes.cast(self._buf, ctypes.c_void_p)
        self._lib.board_reset(self._ptr)
        self.max_total_stones = max_total_stones
        self._move_log: list = []  # track moves for NNAlphaBeta sync

    def place_stone(self, q: int, r: int) -> None:
        self._lib.board_place(self._ptr, q, r)
        self._move_log.append((q, r))

    def undo(self) -> None:
        self._lib.board_undo(self._ptr)
        if self._move_log:
            self._move_log.pop()

    @property
    def current_player(self) -> int:
        return self._lib.board_get_current_player(self._ptr)

    @property
    def winner(self) -> Optional[int]:
        w = self._lib.board_get_winner(self._ptr)
        return w if w >= 0 else None

    @property
    def is_terminal(self) -> bool:
        return (self._lib.board_get_winner(self._ptr) >= 0 or
                self._lib.board_get_total_stones(self._ptr) >= self.max_total_stones)

    @property
    def stones_this_turn(self) -> int:
        return self._lib.board_get_stones_this_turn(self._ptr)

    @property
    def stones_per_turn(self) -> int:
        return self._lib.board_get_stones_per_turn(self._ptr)

    @property
    def turn(self) -> int:
        # Derive from total stones + turn structure
        return self._lib.board_get_total_stones(self._ptr)  # approximate

    @property
    def total_stones(self) -> int:
        return self._lib.board_get_total_stones(self._ptr)

    @property
    def occupied(self) -> bool:
        """Returns truthy if board has stones."""
        return self._lib.board_get_total_stones(self._ptr) > 0

    @property
    def candidates(self) -> List[Tuple[int, int]]:
        q_arr = _IntArr600()
        r_arr = _IntArr600()
        n = self._lib.board_get_candidates(self._ptr, q_arr, r_arr)
        return [(q_arr[i], r_arr[i]) for i in range(n)]

    def legal_moves(self) -> List[Tuple[int, int]]:
        if not self.occupied:
            return [(0, 0)]
        return self.candidates

    def result_for(self, player: int) -> float:
        w = self._lib.board_get_winner(self._ptr)
        if w < 0:
            return 0.0
        return 1.0 if w == player else -1.0

    def clone(self) -> 'CGameState':
        new = CGameState.__new__(CGameState)
        new._lib = self._lib
        sz = self._lib.board_sizeof()
        new._buf = ctypes.create_string_buffer(sz)
        new._ptr = ctypes.cast(new._buf, ctypes.c_void_p)
        self._lib.board_copy(new._ptr, self._ptr)
        new.max_total_stones = self.max_total_stones
        return new


def c_encode_state(game: CGameState) -> Tuple[torch.Tensor, int, int]:
    """Encode CGameState using C engine + Python threat channels."""
    buf = _FloatArr()
    oq = ctypes.c_int(0)
    orr = ctypes.c_int(0)
    game._lib.board_encode_state(game._ptr, buf, ctypes.byref(oq), ctypes.byref(orr))
    arr5 = np.ctypeslib.as_array(buf).reshape(5, BOARD_SIZE, BOARD_SIZE).copy()

    # Add 2 threat channels (planes 5-6) in Python
    arr = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    arr[:5] = arr5

    # Threat planes: use C engine's line counting
    offset_q, offset_r = oq.value, orr.value
    player = game.current_player
    cands = game.candidates
    if cands:
        for q, r in cands:
            i, j = q - offset_q, r - offset_r
            if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
                qi, ri = q + 15, r + 15  # C engine uses OFF=15
                my_l = game._lib.board_max_line_through(game._ptr, qi, ri, player)
                opp_l = game._lib.board_max_line_through(game._ptr, qi, ri, 1 - player)
                if my_l >= 4:
                    arr[5, i, j] = min(my_l / 6.0, 1.0)
                if opp_l >= 4:
                    arr[6, i, j] = min(opp_l / 6.0, 1.0)

    return torch.from_numpy(arr), offset_q, offset_r


def c_decode_policy(
    policy_logits: torch.Tensor,
    game: CGameState,
    offset_q: int,
    offset_r: int,
) -> Dict[Tuple[int, int], float]:
    """Decode policy for CGameState."""
    logits = policy_logits.cpu().numpy().astype(np.float64)

    if not game.occupied:
        return {(0, 0): 1.0}

    legal = game.candidates
    mask = np.full(BOARD_SIZE * BOARD_SIZE, -1e9, dtype=np.float64)
    for q, r in legal:
        i, j = q - offset_q, r - offset_r
        if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
            idx = i * BOARD_SIZE + j
            mask[idx] = logits[idx]

    mask -= mask.max()
    exp = np.exp(mask)
    total = exp.sum()
    if total < 1e-30:
        p = 1.0 / len(legal)
        return {m: p for m in legal}

    probs = exp / total
    result = {}
    for q, r in legal:
        i, j = q - offset_q, r - offset_r
        if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
            result[(q, r)] = probs[i * BOARD_SIZE + j]

    s = sum(result.values())
    if s > 1e-30:
        for k in result:
            result[k] /= s
    else:
        p = 1.0 / len(legal)
        return {m: p for m in legal}
    return result


def c_compute_threat_label(game: CGameState) -> np.ndarray:
    """Compute threat label with preemptive awareness using C engine data.
    Maps C's basic 4+/5+ detection to our continuous scale + adds axis info."""
    try:
        buf = _FloatArr4()
        game._lib.board_compute_threat_label(game._ptr, buf)
        # buf = [my_has_4+, my_has_5+, opp_has_4+, opp_has_5+]

        # Map to continuous threat level
        if buf[1] > 0.5:      my_level = 0.85  # 5+ in a row
        elif buf[0] > 0.5:    my_level = 0.80  # 4+ in a row
        else:                  my_level = 0.15  # assume at least preemptive

        if buf[3] > 0.5:      opp_level = 0.85
        elif buf[2] > 0.5:    opp_level = 0.80
        else:                  opp_level = 0.15

        # Axis score: use winning move count as proxy for multi-axis threats
        p = game.current_player
        my_wm = game._lib.board_count_winning_moves(game._ptr, p)
        opp_wm = game._lib.board_count_winning_moves(game._ptr, 1 - p)
        my_axis = min(1.0, my_wm * 0.35) if my_wm > 0 else 0.25
        opp_axis = min(1.0, opp_wm * 0.35) if opp_wm > 0 else 0.25

        return np.array([my_level, my_axis, opp_level, opp_axis], dtype=np.float32)
    except Exception:
        return np.zeros(4, dtype=np.float32)


# ---------------------------------------------------------------------------
# SE-ResBlock + HybridHexNet (KataGo-inspired architecture)
# ---------------------------------------------------------------------------

class SEResBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation (channel attention)."""

    def __init__(self, num_filters: int, se_ratio: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        # SE: global avg pool → FC → ReLU → FC → sigmoid
        mid = max(num_filters // se_ratio, 16)
        self.se_fc1 = nn.Linear(num_filters, mid)
        self.se_fc2 = nn.Linear(mid, num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # SE attention
        se = out.mean(dim=(2, 3))  # (B, C)
        se = F.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        out = out * se.unsqueeze(-1).unsqueeze(-1)
        return F.relu(out + residual)


class HybridHexNet(nn.Module):
    """CNN + Global Attention network for hex connect-6.

    Architecture:
      1. Stem: Conv2d → BN → ReLU
      2. Body: N × SE-ResBlock (local patterns + channel attention)
      3. Global Attention: 2 × MultiheadAttention (long-range dependencies)
      4. Heads: policy (361), value (1), threat (4)
    """

    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        num_channels: int = NUM_CHANNELS,
        num_filters: int = 256,
        num_res_blocks: int = 12,
        num_attention_heads: int = 8,
        num_attention_layers: int = 2,
    ):
        super().__init__()
        self.board_size = board_size
        bs2 = board_size * board_size

        # Stem
        self.conv_init = nn.Conv2d(num_channels, num_filters, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(num_filters)

        # SE-ResBlock tower
        self.res_blocks = nn.Sequential(
            *[SEResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Global attention layers
        self.attn_layers = nn.ModuleList()
        self.attn_norms = nn.ModuleList()
        for _ in range(num_attention_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(num_filters, num_attention_heads, batch_first=True)
            )
            self.attn_norms.append(nn.LayerNorm(num_filters))

        # Hex positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, bs2, num_filters) * 0.02)

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * bs2, bs2)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(bs2, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Threat head
        self.threat_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.threat_bn = nn.BatchNorm2d(1)
        self.threat_fc = nn.Linear(bs2, 4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x.size(0)
        bs = self.board_size
        bs2 = bs * bs

        # Stem + ResBlocks
        x = F.relu(self.bn_init(self.conv_init(x)))
        x = self.res_blocks(x)

        # Global attention: (B, C, H, W) → (B, H*W, C)
        x_flat = x.view(B, -1, bs2).permute(0, 2, 1)  # (B, bs2, C)
        x_flat = x_flat + self.pos_embedding

        for attn, norm in zip(self.attn_layers, self.attn_norms):
            residual = x_flat
            x_flat = norm(x_flat)
            attn_out, _ = attn(x_flat, x_flat, x_flat)
            x_flat = residual + attn_out

        # Back to spatial: (B, bs2, C) → (B, C, H, W)
        x = x_flat.permute(0, 2, 1).view(B, -1, bs, bs)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(B, -1)
        p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(B, -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        # Threat head
        t = F.relu(self.threat_bn(self.threat_conv(x)))
        t = t.view(B, -1)
        t = self.threat_fc(t)

        return p, v, t

    def forward_pv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p, v, _ = self.forward(x)
        return p, v

    @torch.no_grad()
    def predict(self, encoded_state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        self.eval()
        x = encoded_state.unsqueeze(0)
        dev = next(self.parameters()).device
        if x.device != dev:
            x = x.to(dev)
        p, v = self.forward_pv(x)
        return p.squeeze(0).cpu(), v.item()


def create_network(config: str = 'standard', board_size: int = BOARD_SIZE) -> nn.Module:
    """Factory for network architectures."""
    if config == 'fast':
        return HexNet(board_size=board_size, num_filters=64, num_res_blocks=4)
    elif config == 'standard':
        return HexNet(board_size=board_size, num_filters=128, num_res_blocks=12)
    elif config == 'large':
        return HexNet(board_size=board_size, num_filters=256, num_res_blocks=12)
    elif config == 'hybrid':
        return HybridHexNet(board_size=board_size, num_filters=256, num_res_blocks=12)
    elif config == 'hybrid-small':
        return HybridHexNet(board_size=board_size, num_filters=128, num_res_blocks=6,
                            num_attention_heads=4, num_attention_layers=1)
    else:
        raise ValueError(f"Unknown network config: {config}")


# ---------------------------------------------------------------------------
# NN-Guided Alpha-Beta Search (like Stockfish NNUE)
# ---------------------------------------------------------------------------

class NNAlphaBeta:
    """Alpha-beta search with NN evaluation at leaves.
    Uses C engine for fast move ordering + pruning, NN for position evaluation.
    Reaches depth 8-12 (vs MCTS depth 3-5) — sees 4-6 full turns ahead.
    """

    def __init__(self, net, depth: int = 12, nn_depth: int = 5):
        self.net = net
        self.depth = depth
        self.nn_depth = nn_depth  # call NN when remaining_depth == nn_depth (5 is sweet spot)
        self._lib = None
        self._board_buf = None
        self._board_ptr = None
        self._active_cb = None
        # Pre-allocated encoding buffers (avoid GC during C callback)
        self._enc_buf = (ctypes.c_float * (5 * BOARD_SIZE * BOARD_SIZE))()
        self._enc_full = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self._setup_c_engine()

    def _setup_c_engine(self):
        """Load C engine library."""
        import ctypes
        import os
        lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'engine.so')
        if not os.path.exists(lib_path):
            raise RuntimeError(f"C engine not found at {lib_path}")
        self._lib = ctypes.CDLL(lib_path)

        # Board allocation
        board_size = self._lib.board_sizeof()
        self._board_buf = ctypes.create_string_buffer(board_size)
        self._board_ptr = ctypes.cast(self._board_buf, ctypes.c_void_p)

        # Setup function signatures
        self._lib.board_reset.argtypes = [ctypes.c_void_p]
        self._lib.board_place.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self._lib.board_undo.argtypes = [ctypes.c_void_p]
        self._lib.board_setup_triangle.argtypes = [ctypes.c_void_p]
        self._lib.board_get_current_player.argtypes = [ctypes.c_void_p]
        self._lib.board_get_current_player.restype = ctypes.c_int
        self._lib.board_get_winner.argtypes = [ctypes.c_void_p]
        self._lib.board_get_winner.restype = ctypes.c_int

        # Board encoding for NN evaluation in callback
        self._lib.board_encode_state.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
        ]
        self._lib.board_encode_state.restype = None

        # NN callback type
        self._EVAL_FN = ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)

        # NN-guided search
        self._lib.c_nn_ab_search.argtypes = [
            ctypes.c_void_p,    # Board*
            ctypes.c_int,       # depth
            ctypes.c_int,       # nn_depth (call NN at this remaining depth)
            self._EVAL_FN,      # eval callback
            ctypes.c_void_p,    # context (unused)
            ctypes.POINTER(ctypes.c_float),     # out_value
            ctypes.POINTER(ctypes.c_int),       # out_best_q
            ctypes.POINTER(ctypes.c_int),       # out_best_r
            ctypes.POINTER(ctypes.c_longlong),  # out_nn_evals
        ]
        self._lib.c_nn_ab_search.restype = ctypes.c_longlong

    def _sync_board(self, game):
        """Sync C board from Python/C game state by replaying move history."""
        self._lib.board_reset(self._board_ptr)

        if hasattr(game, '_history') and game._history:
            # HexGame — extract (q, r) from undo records
            for rec in game._history:
                q, r = rec[0], rec[1]
                self._lib.board_place(self._board_ptr, q, r)
        elif hasattr(game, '_move_log') and game._move_log:
            # CGameState with move log
            for q, r in game._move_log:
                self._lib.board_place(self._board_ptr, q, r)
        elif hasattr(game, 'move_history') and game.move_history:
            for q, r in game.move_history:
                self._lib.board_place(self._board_ptr, q, r)

    def _nn_eval_callback(self, board_ptr, current_player, ctx):
        """Called by C engine at leaf nodes. Evaluates position with NN."""
        try:
            # Use C engine's 5-channel encoding, pad to NUM_CHANNELS
            buf = self._enc_buf
            oq = ctypes.c_int(0)
            orr = ctypes.c_int(0)
            self._lib.board_encode_state(board_ptr, buf, ctypes.byref(oq), ctypes.byref(orr))

            # Copy 5 channels into full tensor
            arr = np.ctypeslib.as_array(buf).reshape(5, BOARD_SIZE, BOARD_SIZE)
            self._enc_full[:5] = arr
            # Channels 5-6 (threat) left as zero for speed

            tensor = torch.from_numpy(self._enc_full).unsqueeze(0)
            with torch.no_grad():
                _, v = self.net.forward_pv(tensor)
            value = v.item()

            # C engine wants: positive = good for P0, negative = good for P1
            if current_player == 1:
                value = -value
            # Clamp to prevent extreme values from confusing alpha-beta
            return max(-0.99, min(0.99, value))
        except Exception:
            # Fallback to 0 on any error (avoids segfault)
            return 0.0

    def search(self, game, **kwargs) -> Dict[Tuple[int, int], float]:
        """Search from current position. Returns {move: probability} (1-hot for best move)."""
        move, _, _, _ = self.search_with_info(game)
        return {move: 1.0}

    def search_with_info(self, game) -> Tuple[Tuple[int, int], float, int, int]:
        """Search and return (best_move, value, nodes, nn_evals)."""
        import ctypes

        self._sync_board(game)

        # Create callback and KEEP REFERENCE (prevents GC during C call)
        self._active_cb = self._EVAL_FN(self._nn_eval_callback)

        out_val = ctypes.c_float(0)
        out_q = ctypes.c_int(0)
        out_r = ctypes.c_int(0)
        out_nn = ctypes.c_longlong(0)

        nodes = self._lib.c_nn_ab_search(
            self._board_ptr, self.depth, self.nn_depth,
            self._active_cb, None,
            ctypes.byref(out_val), ctypes.byref(out_q), ctypes.byref(out_r),
            ctypes.byref(out_nn)
        )

        self._active_cb = None  # release
        return (out_q.value, out_r.value), out_val.value, nodes, out_nn.value


# ---------------------------------------------------------------------------
# Batched NN Alpha-Beta (collect-inject, zero callbacks, 15x faster)
# ---------------------------------------------------------------------------

class BatchedNNAlphaBeta:
    """NN-guided alpha-beta with batched evaluation (no per-leaf callbacks).

    Two-phase search:
      Phase 1: C alpha-beta with C heuristic, collecting leaf positions
      Phase 2: Python batch-evaluates all leaves in one NN forward pass
      Phase 3: C alpha-beta with cached NN values

    ~15x faster than callback-based NNAlphaBeta at same depth.
    """

    MAX_LEAVES = 4096

    def __init__(self, net, depth: int = 8, nn_depth: int = 5):
        self.net = net
        self.depth = depth
        self.nn_depth = nn_depth
        self._lib = _get_lib()
        self._setup_c_engine()

        # Pre-allocate board buffer
        sz = self._lib.board_sizeof()
        self._board_buf = ctypes.create_string_buffer(sz)
        self._board_ptr = ctypes.cast(self._board_buf, ctypes.c_void_p)

    def _setup_c_engine(self):
        lib = self._lib

        # Batched search (no callback params)
        lib.c_batched_ab_search.argtypes = [
            ctypes.c_void_p,                    # Board*
            ctypes.c_int,                       # depth
            ctypes.c_int,                       # nn_depth
            ctypes.POINTER(ctypes.c_float),     # out_value
            ctypes.POINTER(ctypes.c_int),       # out_best_q
            ctypes.POINTER(ctypes.c_int),       # out_best_r
            ctypes.POINTER(ctypes.c_longlong),  # out_nn_hits
        ]
        lib.c_batched_ab_search.restype = ctypes.c_longlong

        # NN cache
        lib.c_nn_cache_clear.argtypes = []
        lib.c_nn_cache_clear.restype = None

        lib.c_nn_cache_inject_batch.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        lib.c_nn_cache_inject_batch.restype = None

        # Leaf buffer accessors
        lib.c_get_leaf_count.argtypes = []
        lib.c_get_leaf_count.restype = ctypes.c_int

        lib.c_get_leaf_encodings.argtypes = []
        lib.c_get_leaf_encodings.restype = ctypes.POINTER(ctypes.c_float)

        lib.c_get_leaf_hashes.argtypes = []
        lib.c_get_leaf_hashes.restype = ctypes.POINTER(ctypes.c_uint64)

        lib.c_get_leaf_players.argtypes = []
        lib.c_get_leaf_players.restype = ctypes.POINTER(ctypes.c_int)

        lib.c_set_collect_mode.argtypes = [ctypes.c_int]
        lib.c_set_collect_mode.restype = None

        lib.c_clear_leaves.argtypes = []
        lib.c_clear_leaves.restype = None

        lib.c_tt_clear.argtypes = []
        lib.c_tt_clear.restype = None

    def _sync_board(self, game):
        """Sync C board from Python/C game state."""
        self._lib.board_reset(self._board_ptr)
        if hasattr(game, '_move_log') and game._move_log:
            for q, r in game._move_log:
                self._lib.board_place(self._board_ptr, q, r)
        elif hasattr(game, '_history') and game._history:
            for rec in game._history:
                q, r = rec[0], rec[1]
                self._lib.board_place(self._board_ptr, q, r)

    def search_with_info(self, game):
        """Three-phase batched search. Returns ((q,r), value, nodes, nn_hits)."""
        self._sync_board(game)
        lib = self._lib

        out_val = ctypes.c_float(0)
        out_q = ctypes.c_int(0)
        out_r = ctypes.c_int(0)
        out_nn = ctypes.c_longlong(0)

        # Phase 1: Collect leaves
        lib.c_nn_cache_clear()
        lib.c_clear_leaves()
        lib.c_set_collect_mode(1)
        lib.c_batched_ab_search(
            self._board_ptr, self.depth, self.nn_depth,
            ctypes.byref(out_val), ctypes.byref(out_q), ctypes.byref(out_r),
            ctypes.byref(out_nn)
        )

        n_leaves = lib.c_get_leaf_count()

        if n_leaves > 0:
            # Phase 2: Batch NN evaluation
            enc_ptr = lib.c_get_leaf_encodings()
            hash_ptr = lib.c_get_leaf_hashes()
            player_ptr = lib.c_get_leaf_players()

            # Zero-copy numpy views of C buffers
            enc_arr = np.ctypeslib.as_array(enc_ptr, shape=(n_leaves * 5 * BOARD_SIZE * BOARD_SIZE,))
            enc_arr = enc_arr.reshape(n_leaves, 5, BOARD_SIZE, BOARD_SIZE)

            # Pad 5→NUM_CHANNELS (7) channels
            batch = np.zeros((n_leaves, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            batch[:, :5] = enc_arr

            # Batched NN forward pass
            tensor = torch.from_numpy(batch)
            with torch.no_grad():
                _, values = self.net.forward_pv(tensor)

            # Prepare values: flip for P1, clamp
            vals = values.cpu().numpy().flatten().astype(np.float32)
            players = np.ctypeslib.as_array(player_ptr, shape=(n_leaves,))
            # Flip sign for player 1 (C convention: positive = good for P0)
            p1_mask = players == 1
            vals[p1_mask] = -vals[p1_mask]
            vals = np.clip(vals, -0.99, 0.99).astype(np.float32)

            # Read hashes (must copy — C buffer may be reused)
            hashes = np.ctypeslib.as_array(hash_ptr, shape=(n_leaves,)).copy()

            # Bulk inject into C cache
            h_c = hashes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
            v_c = vals.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            lib.c_nn_cache_inject_batch(h_c, v_c, n_leaves)

        # Phase 3: Re-search with cached NN values
        lib.c_tt_clear()  # clear stale TT from phase 1
        lib.c_set_collect_mode(0)
        nodes = lib.c_batched_ab_search(
            self._board_ptr, self.depth, self.nn_depth,
            ctypes.byref(out_val), ctypes.byref(out_q), ctypes.byref(out_r),
            ctypes.byref(out_nn)
        )

        return (out_q.value, out_r.value), out_val.value, nodes, out_nn.value

    def search(self, game, temperature=0.1, add_noise=False):
        """Compatible interface: returns {(q,r): probability} dict.

        Evaluates top root moves and returns soft distribution for exploration.
        """
        self._sync_board(game)
        lib = self._lib

        # Get top root moves from C engine
        q_arr = (ctypes.c_int * 25)()
        r_arr = (ctypes.c_int * 25)()
        s_arr = (ctypes.c_int * 25)()
        n = lib.board_get_scored_moves(self._board_ptr, q_arr, r_arr, s_arr, 20)

        if n == 0:
            # Empty board: first move is always (0,0)
            if lib.board_get_total_stones(self._board_ptr) == 0:
                return {(0, 0): 1.0}
            return {}
        if n == 1:
            return {(q_arr[0], r_arr[0]): 1.0}

        # Phase 1: Collect leaves for all root moves
        lib.c_nn_cache_clear()
        lib.c_clear_leaves()
        lib.c_set_collect_mode(1)

        out_val = ctypes.c_float(0)
        out_q = ctypes.c_int(0)
        out_r = ctypes.c_int(0)
        out_nn = ctypes.c_longlong(0)

        lib.c_batched_ab_search(
            self._board_ptr, self.depth, self.nn_depth,
            ctypes.byref(out_val), ctypes.byref(out_q), ctypes.byref(out_r),
            ctypes.byref(out_nn)
        )

        # Phase 2: Batch NN eval
        n_leaves = lib.c_get_leaf_count()
        if n_leaves > 0:
            enc_ptr = lib.c_get_leaf_encodings()
            hash_ptr = lib.c_get_leaf_hashes()
            player_ptr = lib.c_get_leaf_players()

            enc_arr = np.ctypeslib.as_array(enc_ptr, shape=(n_leaves * 5 * BOARD_SIZE * BOARD_SIZE,))
            enc_arr = enc_arr.reshape(n_leaves, 5, BOARD_SIZE, BOARD_SIZE)

            batch = np.zeros((n_leaves, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            batch[:, :5] = enc_arr

            tensor = torch.from_numpy(batch)
            with torch.no_grad():
                _, values = self.net.forward_pv(tensor)

            vals = values.cpu().numpy().flatten().astype(np.float32)
            players = np.ctypeslib.as_array(player_ptr, shape=(n_leaves,))
            p1_mask = players == 1
            vals[p1_mask] = -vals[p1_mask]
            vals = np.clip(vals, -0.99, 0.99).astype(np.float32)

            hashes = np.ctypeslib.as_array(hash_ptr, shape=(n_leaves,)).copy()
            h_c = hashes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
            v_c = vals.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            lib.c_nn_cache_inject_batch(h_c, v_c, n_leaves)

        # Phase 3: Evaluate each root move individually
        lib.c_tt_clear()
        lib.c_set_collect_mode(0)

        move_values = {}
        for i in range(min(n, 15)):  # evaluate top 15 root moves
            mq, mr = q_arr[i], r_arr[i]
            lib.board_place(self._board_ptr, mq, mr)

            child_out = ctypes.c_float(0)
            child_q = ctypes.c_int(0)
            child_r = ctypes.c_int(0)
            child_nn = ctypes.c_longlong(0)
            lib.c_batched_ab_search(
                self._board_ptr, self.depth - 1, self.nn_depth,
                ctypes.byref(child_out), ctypes.byref(child_q), ctypes.byref(child_r),
                ctypes.byref(child_nn)
            )
            lib.board_undo(self._board_ptr)

            # From current player's perspective
            v = child_out.value
            p = lib.board_get_current_player(self._board_ptr)
            if p == 0:  # maximizing
                move_values[(mq, mr)] = v
            else:  # minimizing
                move_values[(mq, mr)] = -v

        if not move_values:
            return {(q_arr[0], r_arr[0]): 1.0}

        # Convert to probability distribution
        vals_arr = np.array(list(move_values.values()), dtype=np.float32)

        if temperature > 0 and temperature < 100:
            # Scale by temperature, softmax
            scaled = vals_arr / max(temperature, 0.01)
            scaled -= scaled.max()  # numerical stability
            exp_vals = np.exp(scaled)
            probs = exp_vals / exp_vals.sum()
        else:
            # Greedy
            probs = np.zeros_like(vals_arr)
            probs[vals_arr.argmax()] = 1.0

        if add_noise:
            noise = np.random.dirichlet([0.3] * len(probs))
            probs = 0.75 * probs + 0.25 * noise

        policy = {}
        for (move, _), prob in zip(move_values.items(), probs):
            if prob > 0.001:
                policy[move] = float(prob)

        # Normalize
        total = sum(policy.values())
        if total > 0:
            policy = {m: p / total for m, p in policy.items()}

        return policy


# ---------------------------------------------------------------------------
# Batched MCTS with Virtual Loss (8x fewer NN calls)
# ---------------------------------------------------------------------------

class BatchedMCTS:
    """MCTS with batched NN inference using virtual loss.

    Instead of evaluating one leaf at a time, selects batch_size leaves
    simultaneously using virtual loss to ensure diversity, then evaluates
    them all in a single batched NN forward pass.
    """

    def __init__(
        self,
        net,
        c_puct: float = C_PUCT,
        num_simulations: int = NUM_SIMULATIONS,
        batch_size: int = 64,
    ):
        self.net = net
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.batch_size = batch_size

    # Transposition cache: reuse NN evaluations across positions
    _eval_cache: dict = {}
    _cache_hits: int = 0

    def search(
        self,
        game,  # HexGame or CGameState
        temperature: float = 1.0,
        add_noise: bool = True,
    ) -> Dict[Tuple[int, int], float]:
        """Run batched MCTS with search improvements. Returns {move: probability}.

        Improvements over standard MCTS:
        - AB hybrid: shallow alpha-beta guides root move ordering
        - Rollout blend: C heuristic scores blended with NN priors
        - Transposition cache: reuse NN evals for same positions
        - Quiescence: double/triple threats boost value estimate
        """
        root = MCTSNode(parent=None, move=None, prior=0.0, player=game.current_player)

        # --- AB HYBRID: quick depth-4 search to detect forced wins ---
        try:
            if hasattr(game, '_lib'):
                ab_val = ctypes.c_float(0)
                ab_ste = ctypes.c_int(0)
                game._lib.c_ab_solve(game._ptr, 4,
                                      ctypes.byref(ab_val), ctypes.byref(ab_ste))
                if abs(ab_val.value) >= 1.0:
                    # Proven win/loss — skip MCTS entirely
                    moves = game.legal_moves()
                    if moves:
                        # Return best move from scored moves
                        q_arr = (ctypes.c_int * 10)()
                        r_arr = (ctypes.c_int * 10)()
                        s_arr = (ctypes.c_int * 10)()
                        n = game._lib.board_get_scored_moves(game._ptr, q_arr, r_arr, s_arr, 1)
                        if n > 0:
                            return {(q_arr[0], r_arr[0]): 1.0}
        except Exception:
            pass

        # Use C-engine encoding if CGameState, else Python
        if isinstance(game, CGameState):
            enc_fn = c_encode_state
            dec_fn = c_decode_policy
        else:
            enc_fn = encode_state
            dec_fn = decode_policy

        # --- TRANSPOSITION CACHE: check if we've seen this position ---
        zhash = game._lib.board_get_zhash(game._ptr) if hasattr(game, '_lib') else None
        cached = BatchedMCTS._eval_cache.get(zhash) if zhash else None

        if cached is not None:
            policy, value = cached
            BatchedMCTS._cache_hits += 1
        else:
            # Expand root with NN
            encoded, oq, orr = enc_fn(game)
            policy_logits, value = self.net.predict(encoded)
            policy = dec_fn(policy_logits, game, oq, orr)

            # --- ROLLOUT BLEND: mix NN priors with C heuristic scores ---
            try:
                if hasattr(game, '_lib'):
                    q_arr = (ctypes.c_int * 25)()
                    r_arr = (ctypes.c_int * 25)()
                    s_arr = (ctypes.c_int * 25)()
                    n_scored = game._lib.board_get_scored_moves(
                        game._ptr, q_arr, r_arr, s_arr, 20)
                    if n_scored > 0:
                        # Softmax of C scores
                        scores = [s_arr[i] for i in range(n_scored)]
                        max_s = max(scores)
                        exp_scores = [math.exp(min(s - max_s, 20)) for s in scores]
                        total = sum(exp_scores)
                        c_priors = {}
                        for i in range(n_scored):
                            c_priors[(q_arr[i], r_arr[i])] = exp_scores[i] / total
                        # Blend NN + C heuristic
                        if PLAY_STYLE == 'distant':
                            existing = _get_existing_stones(game)
                            for move in policy:
                                c_prob = c_priors.get(move, 0.01)
                                adj = any(abs(move[0]-s[0]) + abs(move[1]-s[1]) <= 1
                                          for s in existing) if existing else True
                                blend = C_BLEND_ADJACENT if adj else C_BLEND_DISTANT
                                policy[move] = (1 - blend) * policy[move] + blend * c_prob
                        else:
                            for move in policy:
                                c_prob = c_priors.get(move, 0.01)
                                policy[move] = 0.7 * policy[move] + 0.3 * c_prob
                        # Renormalize
                        total_p = sum(policy.values())
                        if total_p > 0:
                            policy = {m: p / total_p for m, p in policy.items()}
            except Exception:
                pass

            # --- QUIESCENCE: boost value for double/triple threats ---
            try:
                if hasattr(game, '_lib'):
                    p = game.current_player
                    wm = game._lib.board_count_winning_moves(game._ptr, p)
                    opp_wm = game._lib.board_count_winning_moves(game._ptr, 1 - p)
                    if wm >= 3:
                        value = 1.0 if p == 0 else -1.0
                    elif opp_wm >= 3:
                        value = -1.0 if p == 0 else 1.0
                    elif wm >= 2:
                        boost = 0.15 if p == 0 else -0.15
                        value = value * 0.85 + boost
                    elif opp_wm >= 2:
                        boost = -0.15 if p == 0 else 0.15
                        value = value * 0.85 + boost
            except Exception:
                pass

            # Store in cache
            if zhash is not None and len(BatchedMCTS._eval_cache) < 100000:
                BatchedMCTS._eval_cache[zhash] = (dict(policy), value)

        sorted_moves = sorted(policy.items(), key=lambda x: x[1], reverse=True)
        k = MCTS.INITIAL_WIDTH
        for move, prob in sorted_moves[:k]:
            child_player = self._next_player(game, move)
            child = MCTSNode(parent=root, move=move, prior=prob, player=child_player)
            root.children[move] = child
        root._pending_moves = sorted_moves[k:]
        root.is_expanded = True

        # Add Dirichlet noise
        if add_noise and root.children:
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(root.children))
            for (_, child), n in zip(root.children.items(), noise):
                child.prior = (1 - DIRICHLET_EPSILON) * child.prior + DIRICHLET_EPSILON * n

        # Run simulations in batches
        sims_done = 0
        while sims_done < self.num_simulations:
            batch = min(self.batch_size, self.num_simulations - sims_done)
            self._run_batch(root, game, batch, enc_fn, dec_fn)
            sims_done += batch

        return self._get_policy(root, temperature)

    def _run_batch(self, root, game, batch_size, enc_fn, dec_fn):
        """Select, evaluate, and backup a batch of leaves."""
        leaves = []      # (node, depth, game_state_for_encoding)
        depths = []

        for _ in range(batch_size):
            node = root
            depth = 0

            # SELECT with virtual loss + progressive widening
            while node.is_expanded and not game.is_terminal:
                if not node.children:
                    break
                # Widen if enough visits
                if node._pending_moves and node.visit_count >= 20:
                    self._widen_node(node, game)
                move, node = self._select_child(node)
                game.place_stone(move[0], move[1])
                depth += 1

            if game.is_terminal:
                # Terminal: backup immediately
                value = game.result_for(root.player)
                self._backup(node, value, root.player)
                for _ in range(depth):
                    game.undo()
                continue

            if node.is_expanded:
                # Already expanded (rare collision) — just backup
                for _ in range(depth):
                    game.undo()
                continue

            # Apply virtual loss
            node.visit_count += 1

            # Encode state for batch evaluation
            encoded, oq, orr = enc_fn(game)
            leaves.append((node, depth, encoded, oq, orr, game.current_player))
            depths.append(depth)

            # Undo moves to restore root state
            for _ in range(depth):
                game.undo()

        if not leaves:
            return

        # Batch NN inference
        states = torch.stack([l[2] for l in leaves])
        dev = next(self.net.parameters()).device
        if states.device != dev:
            states = states.to(dev)

        with torch.no_grad():
            self.net.eval()
            policy_batch, value_batch = self.net.forward_pv(states)
            policy_batch = policy_batch.cpu()
            value_batch = value_batch.cpu()

        # Expand each leaf and backup
        for i, (node, depth, encoded, oq, orr, player) in enumerate(leaves):
            # Undo virtual loss
            node.visit_count -= 1

            # Re-traverse to this node to expand
            # (We need the game state to decode the policy)
            path = []
            n = node
            while n.parent is not None:
                path.append(n.move)
                n = n.parent
            path.reverse()

            for move in path:
                game.place_stone(move[0], move[1])

            # Decode policy with progressive widening
            policy = dec_fn(policy_batch[i], game, oq, orr)
            sorted_moves = sorted(policy.items(), key=lambda x: x[1], reverse=True)
            k = MCTS.INITIAL_WIDTH
            for move, prob in sorted_moves[:k]:
                child_player = self._next_player(game, move)
                child = MCTSNode(parent=node, move=move, prior=prob, player=child_player)
                node.children[move] = child
            node._pending_moves = sorted_moves[k:]
            node.is_expanded = True

            # Backup
            value = value_batch[i].item()
            # Convert from current player perspective to root player perspective
            if player != root.player:
                value = -value
            self._backup(node, value, root.player)

            # Undo
            for _ in range(len(path)):
                game.undo()

    def _widen_node(self, node, game):
        """Progressive widening: add more children as visits increase."""
        if not node._pending_moves:
            return
        n = node.visit_count
        current = len(node.children)
        if n >= 100 and current < MCTS.WIDEN_AT_100:
            target = MCTS.WIDEN_AT_100
        elif n >= 50 and current < MCTS.WIDEN_AT_50:
            target = MCTS.WIDEN_AT_50
        elif n >= 20 and current < MCTS.WIDEN_AT_20:
            target = MCTS.WIDEN_AT_20
        else:
            return
        to_add = target - current
        for move, prob in node._pending_moves[:to_add]:
            if move not in node.children:
                child_player = self._next_player(game, move)
                child = MCTSNode(parent=node, move=move, prior=prob, player=child_player)
                node.children[move] = child
        node._pending_moves = node._pending_moves[to_add:]

    def _select_child(self, node):
        c_puct = self.c_puct
        sqrt_parent = math.sqrt(node.visit_count + 1)
        best_score = -float('inf')
        best_move = best_child = None
        for move, child in node.children.items():
            q = child.q_value
            if child.player != node.player:
                q = -q
            score = q + c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

    def _next_player(self, game, move):
        stt = game.stones_this_turn + 1
        if stt >= game.stones_per_turn:
            return (game.current_player + 1) & 1
        return game.current_player

    def _backup(self, node, value, root_player):
        while node is not None:
            node.visit_count += 1
            if node.player == root_player:
                node.value_sum += value
            else:
                node.value_sum -= value
            node = node.parent

    def _get_policy(self, root, temperature):
        if not root.children:
            return {}
        if temperature < 1e-8:
            best = max(root.children.values(), key=lambda c: c.visit_count)
            return {m: (1.0 if c is best else 0.0) for m, c in root.children.items()}
        visits = {m: c.visit_count ** (1.0 / temperature) for m, c in root.children.items()}
        total = sum(visits.values())
        if total < 1e-30:
            p = 1.0 / len(visits)
            return {m: p for m in visits}
        return {m: v / total for m, v in visits.items()}


# ---------------------------------------------------------------------------
# Helper: get all stone positions from any game type
# ---------------------------------------------------------------------------

def _get_existing_stones(game) -> set:
    """Return set of (q,r) for all placed stones, works with CGameState and HexGame."""
    if hasattr(game, '_move_log') and game._move_log:
        return set(game._move_log)
    if hasattr(game, 'stones_0'):
        return game.stones_0 | game.stones_1
    if hasattr(game, 'board'):
        return set(game.board.keys())
    return set()


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

        # Let MCTS decide everything — no forced moves during training.
        policy = mcts.search(game, temperature=temperature, add_noise=add_noise)
        if not policy:
            break

        # --- DISTANT EXPLORATION: inject gap candidates into policy ---
        # Instead of forcing moves, add distant positions as options with
        # small probability so MCTS can discover their value naturally.
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
                    # Add a few random gap positions with small weight
                    n_inject = min(3, len(gap_candidates))
                    injected = random.sample(gap_candidates, n_inject)
                    # Give each injected move ~5% of total probability
                    inject_weight = 0.05 / n_inject
                    total = sum(policy.values())
                    scale = (1.0 - inject_weight * n_inject)
                    policy = {m: p * scale / total for m, p in policy.items()}
                    for m in injected:
                        policy[m] = inject_weight

        # --- RESIGN THRESHOLD: stop hopeless games early ---
        # Disabled: causes false "draws" on an infinite board where draws don't exist
        # The game must always play to completion (6-in-a-row)

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

        moves = list(policy.keys())
        probs = [policy[m] for m in moves]
        idx = np.random.choice(len(moves), p=probs)
        chosen = moves[idx]
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

    # Penalize short games — they don't teach mid/late-game skills
    if n < 30:
        for sample in samples:
            sample.priority *= 0.5

    # --- DIVERSITY BONUS: reward games with spread-out stone placement ---
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
) -> Dict[str, float]:
    """One training step with threat aux loss + priority updates.

    Optimized: vectorized tensor construction, minimal Python loops.
    """
    net.train()
    batch, indices = replay_buffer.sample(batch_size)
    n = len(batch)

    # Pre-allocate tensors (avoid per-sample Python loop)
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

    policy_logits, value_preds, threat_preds = net(states)
    value_preds = value_preds.squeeze(-1)

    # Losses
    value_loss = F.mse_loss(value_preds, target_values)
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -torch.sum(target_policies * log_probs, dim=1).mean()
    threat_loss = F.binary_cross_entropy_with_logits(threat_preds, threat_targets)
    loss = value_loss + policy_loss + 0.5 * threat_loss

    optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()
    loss.backward()
    optimizer.step()

    # TD-error priority updates (skip every other step to save time)
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


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def train(
    num_iterations: int = 100,
    games_per_iter: int = 25,
    train_steps_per_iter: int = 200,
    num_simulations: int = NUM_SIMULATIONS,
    observer: Optional[TrainingObserver] = None,
):
    device = get_device()
    print(f"Device: {device}")

    net = HexNet().to(device)
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
