"""
orca/threats.py - Forced-move detection and threat analysis.

Contains:
- Line counting helpers (_count_line, _max_line_at, etc.)
- Threat search (_threat_search, find_forced_move)
- Finisher detection (detect_finisher)
- Threat bonus for training (compute_threat_bonus)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Axis constants
# ---------------------------------------------------------------------------

AXES_3 = ((1, 0), (0, 1), (1, -1))


# ---------------------------------------------------------------------------
# Line counting helpers
# ---------------------------------------------------------------------------

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


def _line_through_candidate(game, q: int, r: int, player: int) -> int:
    """Max consecutive line length if player places at (q,r). Works on CGameState & HexGame."""
    if hasattr(game, '_ptr'):
        qi, ri = q + 15, r + 15  # OFF=15 in C engine
        return game._lib.board_max_line_through(game._ptr, qi, ri, player)
    # HexGame path
    stones = game.stones_0 if player == 0 else game.stones_1
    test = stones | {(q, r)}
    return _max_line_at(test, q, r)


# ---------------------------------------------------------------------------
# Threat move helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Threat search
# ---------------------------------------------------------------------------

def _threat_search(game, depth: int = 4) -> Optional[Tuple[int, int]]:
    """Fast threat-space search to find moves creating unstoppable forks.

    Searches forcing lines (moves creating 3+ on multiple axes or 4+ on one axis).
    Much cheaper than MCTS - branching factor ~3-8 instead of 30+.

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

    # Check if we have 2 stones this turn - if so, try PAIRS of moves
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
                # Opponent has no threats - any reasonable move
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


# ---------------------------------------------------------------------------
# Forced move detection
# ---------------------------------------------------------------------------

def find_forced_move(game) -> Optional[Tuple[int, int]]:
    """Only force the ONE truly undeniable move: completing 6-in-a-row to win.

    Everything else (blocking, extending, forking) is left to MCTS so it can
    weigh strategic nuance - e.g. blocking one cell further out may be better
    than the adjacent block if it also builds your own position.

    Returns winning move or None.
    """
    p = game.current_player

    # Get candidates - works for both CGameState and HexGame
    if hasattr(game, 'candidates'):
        cands = game.candidates
    else:
        cands = game.legal_moves()

    for q, r in cands:
        if _line_through_candidate(game, q, r, p) >= 6:
            return (q, r)

    return None


# ---------------------------------------------------------------------------
# Finisher detection
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Threat bonus for training priority
# ---------------------------------------------------------------------------

def compute_threat_bonus(game, move: Tuple[int, int], player: int) -> float:
    """Compute a training priority bonus for a move based on threat potential.

    Used in self-play to boost priority of samples where forks/threats are created.
    NOT used to force moves - just to make the training signal richer.

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    'AXES_3',
    '_count_line', '_max_line_at', '_line_through_candidate',
    '_get_threat_moves', '_count_winning_cells', '_count_multi_axis_threats',
    '_threat_search',
    'find_forced_move', 'detect_finisher', 'compute_threat_bonus',
]
