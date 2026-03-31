"""
orca/encoding.py - Board encoding and decoding for neural network input/output.

Contains:
- encode_state / decode_policy (Python HexGame)
- c_encode_state / c_decode_policy (C engine CGameState)
- CGameState class (C engine wrapper)
- compute_threat_label / c_compute_threat_label
- C engine loading (_get_lib, _setup_c_signatures)
"""

from __future__ import annotations

import ctypes
import os as _os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from main import HexGame

# Import config constants
try:
    from orca.config import BOARD_SIZE, NUM_CHANNELS
except ImportError:
    BOARD_SIZE = 19
    NUM_CHANNELS = 7

HALF = BOARD_SIZE // 2

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
    # These give the network INFORMATION about threats - it decides what to do
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
        # Moves outside the window get no probability - that's fine

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
# Threat label computation (Python path)
# ---------------------------------------------------------------------------

def compute_threat_label(game: HexGame) -> np.ndarray:
    """Compute threat features including preemptives (2-in-a-row).

    Returns 4 floats encoding the full threat landscape:
    [0] my threat level - continuous 0-1:
        2-in-a-row = 0.15 (preemptive)
        3-in-a-row = 0.35 (strong preemptive)
        4-in-a-row = 0.60 (active threat)
        5-in-a-row = 0.85 (imminent win)
        6-in-a-row = 1.00 (won)
    [1] my multi-axis score - how many axes have 2+ in a row:
        1 axis = 0.25, 2 axes = 0.60 (proto-fork), 3 axes = 1.0 (dominant)
    [2] opp threat level (same scale)
    [3] opp multi-axis score (same scale)

    This teaches the network to value preemptives (2-in-a-row setups)
    which humans use to build threats 2-3 moves before they become critical.
    """
    player = game.current_player
    my_stones = game.stones_0 if player == 0 else game.stones_1
    opp_stones = game.stones_1 if player == 0 else game.stones_0

    _AXES_LOCAL = ((1, 0), (0, 1), (1, -1))

    all_stones = my_stones | opp_stones

    def analyze_stones(stones: set, blockers: set):
        """Returns (max_live_consecutive, axes_with_2plus, axes_with_3plus).
        Only counts lines that can still extend to 6 (have enough open space)."""
        if not stones:
            return 0, 0, 0
        best = 0
        axes_2plus = 0
        axes_3plus = 0
        for dq, dr in _AXES_LOCAL:
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
    # 4 and 5 are nearly identical - both can finish with 1 stone
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


# ---------------------------------------------------------------------------
# C Engine Integration - CGameState (50x faster game simulation)
# ---------------------------------------------------------------------------

_engine_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'engine.so')
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
# Public API for `from orca.encoding import *`
# ---------------------------------------------------------------------------

__all__ = [
    '_threat_line_at', 'encode_state', 'decode_policy',
    'compute_threat_label',
    '_get_lib', '_setup_c_signatures',
    'CGameState', 'c_encode_state', 'c_decode_policy', 'c_compute_threat_label',
    '_AXES', 'HALF',
    '_IntArr600', '_FloatArr', '_FloatArr361', '_FloatArr4',
]
