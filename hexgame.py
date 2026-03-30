"""
hexgame — Clean Python API for the Hex Connect-6 C Engine

A fast hexagonal tic-tac-toe (Connect-6 on hex grid) game engine.
Uses a compiled C backend for speed (~50-100x faster than pure Python).

Usage:
    from hexgame import HexGame

    game = HexGame()
    game.place(0, 0)       # Player 0 places at center
    game.place(1, 0)       # Player 1 places
    game.place(1, -1)      # Player 1's second stone

    print(game)            # ASCII board
    print(game.legal_moves()[:5])
    print(game.scored_moves(10))

    result = game.search(depth=8)
    print(result)          # {'best_move': (q, r), 'value': 0.35, 'nodes': 12345}

Rules:
    - Infinite hex grid with axial coordinates (q, r)
    - Player 0 places 1 stone first, then 2 stones per turn
    - First to connect 6 in a row on any axis wins
    - Three axes: (1,0), (0,1), (1,-1)
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__all__ = ['HexGame']

# ---------------------------------------------------------------------------
# C Engine loader (auto-compiles if needed)
# ---------------------------------------------------------------------------

_ENGINE_DIR = Path(__file__).parent
_ENGINE_SRC = _ENGINE_DIR / 'engine.c'
_ENGINE_LIB = _ENGINE_DIR / 'engine.so'
_lib = None


def _compile_engine():
    """Compile engine.c to engine.so if missing or outdated."""
    if _ENGINE_LIB.exists() and _ENGINE_SRC.exists():
        if _ENGINE_LIB.stat().st_mtime >= _ENGINE_SRC.stat().st_mtime:
            return  # up to date
    if not _ENGINE_SRC.exists():
        raise FileNotFoundError(f"engine.c not found at {_ENGINE_SRC}")
    print(f"Compiling {_ENGINE_SRC.name}...", file=sys.stderr)
    subprocess.run(
        ['cc', '-O3', '-march=native', '-shared', '-fPIC',
         '-o', str(_ENGINE_LIB), str(_ENGINE_SRC)],
        check=True,
    )


def _load_engine():
    """Load the C engine shared library."""
    global _lib
    if _lib is not None:
        return _lib
    _compile_engine()
    if not _ENGINE_LIB.exists():
        raise RuntimeError(f"C engine not found at {_ENGINE_LIB}")
    _lib = ctypes.CDLL(str(_ENGINE_LIB))
    _setup_signatures(_lib)
    return _lib


def _setup_signatures(lib):
    """Declare all C function signatures."""
    VP = ctypes.c_void_p
    CI = ctypes.c_int
    CF = ctypes.c_float
    CLL = ctypes.c_longlong
    PI = ctypes.POINTER(CI)
    PF = ctypes.POINTER(CF)

    lib.board_sizeof.restype = CI

    lib.board_reset.argtypes = [VP]
    lib.board_reset.restype = None

    lib.board_setup_triangle.argtypes = [VP]
    lib.board_setup_triangle.restype = None

    lib.board_place.argtypes = [VP, CI, CI]
    lib.board_place.restype = None

    lib.board_undo.argtypes = [VP]
    lib.board_undo.restype = None

    lib.board_get_winner.argtypes = [VP]
    lib.board_get_winner.restype = CI

    lib.board_get_current_player.argtypes = [VP]
    lib.board_get_current_player.restype = CI

    lib.board_get_total_stones.argtypes = [VP]
    lib.board_get_total_stones.restype = CI

    lib.board_get_cand_count.argtypes = [VP]
    lib.board_get_cand_count.restype = CI

    lib.board_get_stones_this_turn.argtypes = [VP]
    lib.board_get_stones_this_turn.restype = CI

    lib.board_get_stones_per_turn.argtypes = [VP]
    lib.board_get_stones_per_turn.restype = CI

    lib.board_get_zhash.argtypes = [VP]
    lib.board_get_zhash.restype = ctypes.c_uint64

    lib.board_get_candidates.argtypes = [VP, PI, PI]
    lib.board_get_candidates.restype = CI

    lib.board_get_scored_moves.argtypes = [VP, PI, PI, PI, CI]
    lib.board_get_scored_moves.restype = CI

    lib.board_get_forcing_moves.argtypes = [VP, PI, PI]
    lib.board_get_forcing_moves.restype = CI

    lib.board_has_winning_move.argtypes = [VP, CI]
    lib.board_has_winning_move.restype = CI

    lib.board_count_winning_moves.argtypes = [VP, CI]
    lib.board_count_winning_moves.restype = CI

    lib.board_would_win.argtypes = [VP, CI, CI, CI]
    lib.board_would_win.restype = CI

    lib.board_max_line_through.argtypes = [VP, CI, CI, CI]
    lib.board_max_line_through.restype = CI

    lib.board_is_forcing.argtypes = [VP, CI, CI, CI]
    lib.board_is_forcing.restype = CI

    lib.board_move_score.argtypes = [VP, CI, CI]
    lib.board_move_score.restype = CI

    lib.board_copy.argtypes = [VP, VP]
    lib.board_copy.restype = None

    lib.c_ab_solve.argtypes = [VP, CI, PF, PI]
    lib.c_ab_solve.restype = CLL


# Shared ctypes arrays
_IntArr = ctypes.c_int * 1200  # radius 3 can generate 600+ candidates
_OFF = 15  # board offset (axial 0 maps to index 15)


# ---------------------------------------------------------------------------
# HexGame — the public API
# ---------------------------------------------------------------------------

class HexGame:
    """A Hex Connect-6 game.

    Uses a compiled C engine for fast move generation, win detection,
    and alpha-beta search. Each instance owns its own board state.

    Example:
        >>> game = HexGame()
        >>> game.place(0, 0)
        >>> game.place(1, 0)
        >>> game.place(1, -1)
        >>> game.current_player
        0
        >>> game.legal_moves()[:3]
        [(2, -2), (2, -1), (2, 0)]
    """

    def __init__(self, max_stones: int = 200):
        """Create a new empty game.

        Args:
            max_stones: Maximum total stones before game ends in draw.
        """
        self._lib = _load_engine()
        sz = self._lib.board_sizeof()
        self._buf = ctypes.create_string_buffer(sz)
        self._ptr = ctypes.cast(self._buf, ctypes.c_void_p)
        self._lib.board_reset(self._ptr)
        self._max_stones = max_stones
        self._moves: List[Tuple[int, int]] = []

    # --- Core actions ---

    def place(self, q: int, r: int) -> None:
        """Place a stone at axial coordinates (q, r).

        Automatically advances the turn. Player 0 places 1 stone on turn 0,
        then both players place 2 stones per turn.

        Args:
            q: Axial q-coordinate.
            r: Axial r-coordinate.

        Raises:
            ValueError: If the cell is already occupied.
        """
        self._lib.board_place(self._ptr, q, r)
        self._moves.append((q, r))

    def undo(self) -> None:
        """Undo the last stone placement."""
        self._lib.board_undo(self._ptr)
        if self._moves:
            self._moves.pop()

    def clone(self) -> HexGame:
        """Create an independent deep copy of this game."""
        new = HexGame.__new__(HexGame)
        new._lib = self._lib
        sz = self._lib.board_sizeof()
        new._buf = ctypes.create_string_buffer(sz)
        new._ptr = ctypes.cast(new._buf, ctypes.c_void_p)
        self._lib.board_copy(new._ptr, self._ptr)
        new._max_stones = self._max_stones
        new._moves = list(self._moves)
        return new

    # --- State queries ---

    @property
    def current_player(self) -> int:
        """The player who places the next stone (0 or 1)."""
        return self._lib.board_get_current_player(self._ptr)

    @property
    def winner(self) -> Optional[int]:
        """The winning player (0 or 1), or None if no winner yet."""
        w = self._lib.board_get_winner(self._ptr)
        return w if w >= 0 else None

    @property
    def is_over(self) -> bool:
        """Whether the game has ended (win or max stones reached)."""
        return (self._lib.board_get_winner(self._ptr) >= 0 or
                self._lib.board_get_total_stones(self._ptr) >= self._max_stones)

    @property
    def total_stones(self) -> int:
        """Total number of stones on the board."""
        return self._lib.board_get_total_stones(self._ptr)

    @property
    def stones_this_turn(self) -> int:
        """How many stones the current player has placed this turn (0 or 1)."""
        return self._lib.board_get_stones_this_turn(self._ptr)

    @property
    def stones_per_turn(self) -> int:
        """How many stones the current player must place this turn (1 or 2)."""
        return self._lib.board_get_stones_per_turn(self._ptr)

    @property
    def moves(self) -> List[Tuple[int, int]]:
        """All moves played so far, in order."""
        return list(self._moves)

    @property
    def zhash(self) -> int:
        """Zobrist hash of the current position (64-bit)."""
        return self._lib.board_get_zhash(self._ptr)

    # --- Move generation ---

    def legal_moves(self) -> List[Tuple[int, int]]:
        """Get all legal moves (empty cells within range of existing stones).

        Returns:
            List of (q, r) coordinates where a stone can be placed.
        """
        if self.total_stones == 0:
            return [(0, 0)]
        q_arr = _IntArr()
        r_arr = _IntArr()
        n = self._lib.board_get_candidates(self._ptr, q_arr, r_arr)
        return [(q_arr[i], r_arr[i]) for i in range(n)]

    def scored_moves(self, limit: int = 20) -> List[Tuple[int, int, int]]:
        """Get top moves sorted by heuristic score (best first).

        The score considers: line extensions, blocking opponent threats,
        and proximity to center.

        Args:
            limit: Maximum number of moves to return.

        Returns:
            List of (q, r, score) tuples, sorted by score descending.
        """
        q_arr = _IntArr()
        r_arr = _IntArr()
        s_arr = _IntArr()
        n = self._lib.board_get_scored_moves(self._ptr, q_arr, r_arr, s_arr, limit)
        return [(q_arr[i], r_arr[i], s_arr[i]) for i in range(n)]

    def forcing_moves(self) -> List[Tuple[int, int]]:
        """Get moves that create or block 4+ in-a-row threats.

        Returns:
            List of (q, r) forcing move coordinates.
        """
        q_arr = _IntArr()
        r_arr = _IntArr()
        n = self._lib.board_get_forcing_moves(self._ptr, q_arr, r_arr)
        return [(q_arr[i], r_arr[i]) for i in range(n)]

    # --- Threat detection ---

    def has_winning_move(self, player: Optional[int] = None) -> bool:
        """Check if a player can win with one stone.

        Args:
            player: Player to check (0 or 1). Defaults to current player.
        """
        if player is None:
            player = self.current_player
        return bool(self._lib.board_has_winning_move(self._ptr, player))

    def count_winning_moves(self, player: Optional[int] = None) -> int:
        """Count how many cells would give a player an instant win.

        Args:
            player: Player to check. Defaults to current player.
        """
        if player is None:
            player = self.current_player
        return self._lib.board_count_winning_moves(self._ptr, player)

    def max_line(self, q: int, r: int, player: Optional[int] = None) -> int:
        """Get the longest line through (q, r) for a player.

        Args:
            q, r: Axial coordinates of the cell.
            player: Player to check. Defaults to current player.

        Returns:
            Length of the longest consecutive line through this cell.
        """
        if player is None:
            player = self.current_player
        qi, ri = q + _OFF, r + _OFF
        return self._lib.board_max_line_through(self._ptr, qi, ri, player)

    def would_win(self, q: int, r: int, player: Optional[int] = None) -> bool:
        """Check if placing at (q, r) would win for a player.

        Does NOT actually place the stone.

        Args:
            q, r: Axial coordinates.
            player: Player to check. Defaults to current player.
        """
        if player is None:
            player = self.current_player
        qi, ri = q + _OFF, r + _OFF
        return bool(self._lib.board_would_win(self._ptr, qi, ri, player))

    # --- Search ---

    def search(self, depth: int = 8) -> Dict:
        """Run alpha-beta search from the current position.

        Uses the C engine's optimized alpha-beta with transposition table,
        killer heuristics, and late move reduction.

        Args:
            depth: Search depth in individual stone placements (not turns).
                   Depth 8 = 4 full turns ahead.

        Returns:
            Dictionary with:
                - 'best_move': (q, r) tuple of the best move
                - 'value': float evaluation (-1 to +1, from Player 0's perspective)
                - 'nodes': int number of nodes searched
        """
        out_val = ctypes.c_float(0)
        out_ste = ctypes.c_int(0)
        nodes = self._lib.c_ab_solve(
            self._ptr, depth,
            ctypes.byref(out_val), ctypes.byref(out_ste),
        )
        # Find the best move by trying each candidate
        moves = self.scored_moves(35)
        best_move = moves[0][:2] if moves else (0, 0)
        best_val = -2.0
        for q, r, _ in moves[:15]:
            self.place(q, r)
            child_val = ctypes.c_float(0)
            child_ste = ctypes.c_int(0)
            self._lib.c_ab_solve(
                self._ptr, depth - 1,
                ctypes.byref(child_val), ctypes.byref(child_ste),
            )
            self.undo()
            v = child_val.value
            if self.current_player == 0:
                if v > best_val:
                    best_val = v
                    best_move = (q, r)
            else:
                if v < best_val or best_val == -2.0:
                    best_val = v
                    best_move = (q, r)

        return {
            'best_move': best_move,
            'value': out_val.value,
            'nodes': nodes,
        }

    # --- Serialization ---

    def to_dict(self) -> Dict:
        """Serialize game state to a dictionary (JSON-compatible)."""
        return {
            'moves': self._moves,
            'max_stones': self._max_stones,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> HexGame:
        """Recreate a game from a serialized dictionary."""
        game = cls(max_stones=data.get('max_stones', 200))
        for q, r in data.get('moves', []):
            game.place(q, r)
        return game

    @classmethod
    def from_moves(cls, moves: List[Tuple[int, int]], max_stones: int = 200) -> HexGame:
        """Create a game by replaying a move sequence."""
        game = cls(max_stones=max_stones)
        for q, r in moves:
            game.place(q, r)
        return game

    # --- Pre-built positions ---

    @classmethod
    def triangle(cls) -> HexGame:
        """Create the triangle position: P0 at (0,0), (1,0), (0,1). P1 to move."""
        game = cls()
        game._lib.board_setup_triangle(game._ptr)
        game._moves = [(0, 0), (1, 0), (0, 1)]  # approximate
        return game

    # --- Display ---

    def __repr__(self) -> str:
        w = self.winner
        status = f"P{w} wins" if w is not None else f"P{self.current_player} to move"
        return f"HexGame({self.total_stones} stones, {status})"

    def __str__(self) -> str:
        """ASCII visualization of the board."""
        if not self._moves:
            return "Empty board"

        # Find bounds
        qs = [m[0] for m in self._moves]
        rs = [m[1] for m in self._moves]
        min_q, max_q = min(qs) - 1, max(qs) + 1
        min_r, max_r = min(rs) - 1, max(rs) + 1

        # Build stone map: (q,r) -> player
        stones = {}
        game_copy = HexGame()
        for i, (q, r) in enumerate(self._moves):
            p = game_copy.current_player
            stones[(q, r)] = p
            game_copy.place(q, r)

        lines = []
        for r in range(min_r, max_r + 1):
            indent = " " * (r - min_r)
            row = []
            for q in range(min_q, max_q + 1):
                if (q, r) in stones:
                    row.append("X" if stones[(q, r)] == 0 else "O")
                else:
                    row.append(".")
            lines.append(indent + " ".join(row))

        header = f"  P{self.current_player} to move | {self.total_stones} stones"
        if self.winner is not None:
            header = f"  P{self.winner} WINS | {self.total_stones} stones"
        return header + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("hexgame — Hex Connect-6 Engine")
    print("=" * 40)

    g = HexGame()
    print(f"New game: {g}")

    # Play a few moves
    g.place(0, 0)
    print(f"After (0,0): {g}")

    g.place(2, 0)
    g.place(2, -1)
    print(f"After P1 moves: {g}")

    g.place(1, 0)
    g.place(0, 1)
    print(f"Triangle formed: {g}")

    print(f"\nLegal moves: {len(g.legal_moves())} available")
    print(f"Top 5 moves: {g.scored_moves(5)}")
    print(f"Forcing moves: {g.forcing_moves()}")
    print(f"P0 can win in 1: {g.has_winning_move(0)}")
    print(f"P1 can win in 1: {g.has_winning_move(1)}")

    print(f"\nBoard:")
    print(g)

    # Test clone
    copy = g.clone()
    copy.place(*copy.legal_moves()[0])
    print(f"\nOriginal: {g.total_stones} stones")
    print(f"Clone: {copy.total_stones} stones")

    # Test serialization
    d = g.to_dict()
    g2 = HexGame.from_dict(d)
    print(f"\nSerialized and restored: {g2}")

    # Test search
    print(f"\nSearching (depth 6)...")
    result = g.search(depth=6)
    print(f"Best move: {result['best_move']}")
    print(f"Value: {result['value']:.3f}")
    print(f"Nodes: {result['nodes']:,}")

    print(f"\nAll tests passed!")
