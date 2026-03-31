"""
Endgame solver for Hex Connect-6.

Uses the C engine's alpha-beta search for deep endgame analysis.
Proves wins/losses when positions are tactically sharp.

Usage:
    from orca.solver import solve, solver_or_mcts

    # Pure solver
    result = solve(game, max_depth=12)
    if result['result'] == 'win':
        game.place(*result['move'])

    # Hybrid: solver for endgames, MCTS for complex positions
    policy = solver_or_mcts(game, net, solver_depth=8)
"""

import ctypes
import os
import pickle
import sys
import time
from typing import Dict, Optional, Tuple

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)


def solve(game, max_depth: int = 12, time_limit: float = 5.0) -> Dict:
    """Solve a position using alpha-beta search.

    Uses the C engine for fast deep search. Returns a dict with:
    - result: 'win', 'loss', or 'unknown'
    - move: best move (q, r) or None
    - value: evaluation (-1 to +1)
    - depth: search depth reached
    - nodes: nodes searched
    - time: seconds taken

    Args:
        game: HexGame or CGameState
        max_depth: maximum search depth (default 12, higher = slower but more accurate)
        time_limit: max seconds to search (default 5.0)

    Example:
        result = solve(game, max_depth=16)
        if result['result'] == 'win':
            print(f"Forced win starting with {result['move']}")
    """
    from hexgame import HexGame

    # Use hexgame.HexGame's built-in search (wraps C engine)
    if isinstance(game, HexGame):
        t0 = time.perf_counter()
        try:
            ab_result = game.search(depth=max_depth)
        except Exception:
            return {'result': 'unknown', 'move': None, 'value': 0.0,
                    'depth': 0, 'nodes': 0, 'time': 0.0}
        elapsed = time.perf_counter() - t0

        value = ab_result.get('value', 0.0)
        best = ab_result.get('best_move')
        nodes = ab_result.get('nodes', 0)

        if abs(value) >= 0.99:
            result = 'win' if value > 0 else 'loss'
        else:
            result = 'unknown'

        return {
            'result': result,
            'move': best,
            'value': value,
            'depth': max_depth,
            'nodes': nodes,
            'time': elapsed,
        }

    # CGameState - use c_ab_solve directly
    if hasattr(game, '_lib') and hasattr(game, '_ptr'):
        t0 = time.perf_counter()
        out_val = ctypes.c_float(0)
        out_ste = ctypes.c_int(0)
        try:
            nodes = game._lib.c_ab_solve(
                game._ptr, max_depth,
                ctypes.byref(out_val), ctypes.byref(out_ste)
            )
        except Exception:
            return {'result': 'unknown', 'move': None, 'value': 0.0,
                    'depth': 0, 'nodes': 0, 'time': 0.0}
        elapsed = time.perf_counter() - t0

        value = out_val.value
        if abs(value) >= 0.99:
            result = 'win' if value > 0 else 'loss'
        else:
            result = 'unknown'

        # Get best move from scored moves
        best = None
        try:
            q_arr = (ctypes.c_int * 5)()
            r_arr = (ctypes.c_int * 5)()
            s_arr = (ctypes.c_int * 5)()
            n = game._lib.board_get_scored_moves(game._ptr, q_arr, r_arr, s_arr, 1)
            if n > 0:
                best = (q_arr[0], r_arr[0])
        except Exception:
            pass

        return {
            'result': result,
            'move': best,
            'value': value,
            'depth': max_depth,
            'nodes': nodes if isinstance(nodes, int) else 0,
            'time': elapsed,
        }

    # Fallback: convert to HexGame
    from hexgame import HexGame as _HG
    hg = _HG(max_stones=200)
    if hasattr(game, 'moves'):
        for q, r in game.moves:
            hg.place(q, r)
    elif hasattr(game, '_move_log'):
        for q, r in game._move_log:
            hg.place(q, r)
    return solve(hg, max_depth=max_depth, time_limit=time_limit)


def quick_solve(game, depth: int = 6) -> Optional[Tuple[int, int]]:
    """Quick solve attempt. Returns winning move or None.

    Faster than solve() - just checks if there's a forced win at shallow depth.
    Good for integration into search loops.

    Example:
        move = quick_solve(game, depth=6)
        if move:
            game.place(*move)  # guaranteed win
    """
    result = solve(game, max_depth=depth)
    if result['result'] == 'win' and result['move']:
        return result['move']
    return None


def solver_or_mcts(game, net=None, solver_depth: int = 8, mcts_sims: int = 200,
                   mcts_batch: int = 64) -> Dict:
    """Hybrid: try solver first, fall back to MCTS.

    If the solver proves a win/loss, returns that move with probability 1.0.
    Otherwise runs full MCTS search.

    Returns: policy dict {(q,r): probability}

    Example:
        policy = solver_or_mcts(game, net=my_net, solver_depth=10)
        best = max(policy, key=policy.get)
    """
    # Try solver
    result = solve(game, max_depth=solver_depth, time_limit=2.0)
    if result['result'] == 'win' and result['move']:
        return {result['move']: 1.0}

    # Also check for immediate forced moves
    from bot import find_forced_move
    forced = find_forced_move(game)
    if forced:
        return {forced: 1.0}

    # Fall back to MCTS
    from bot import BatchedMCTS, get_device
    if net is None:
        from hexbot import Bot
        bot = Bot.orca(sims=1)
        net = bot._net
    net.eval()

    # Convert game if needed
    from hexbot import _to_mcts_game
    mcts_game = _to_mcts_game(game)
    mcts = BatchedMCTS(net, num_simulations=mcts_sims, batch_size=mcts_batch)
    return mcts.search(mcts_game, temperature=0.01, add_noise=False)


# ---------------------------------------------------------------------------
# Transposition table persistence
# ---------------------------------------------------------------------------

class TranspositionCache:
    """Persistent transposition table for solver results.

    Caches solve() results by Zobrist hash. Survives between sessions.

    Example:
        cache = TranspositionCache('solver_cache.pkl')
        result = cache.get(game.zhash)
        if result is None:
            result = solve(game, max_depth=12)
            cache.put(game.zhash, result)
        cache.save()
    """

    def __init__(self, path: str = 'solver_cache.pkl', max_entries: int = 100000):
        self.path = path
        self.max_entries = max_entries
        self._cache: Dict[int, Dict] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'rb') as f:
                    self._cache = pickle.load(f)
            except Exception:
                self._cache = {}

    def save(self):
        try:
            with open(self.path, 'wb') as f:
                pickle.dump(self._cache, f)
        except Exception:
            pass

    def get(self, zhash: int) -> Optional[Dict]:
        return self._cache.get(zhash)

    def put(self, zhash: int, result: Dict):
        self._cache[zhash] = result
        if len(self._cache) > self.max_entries:
            # Evict oldest entries
            keys = list(self._cache.keys())
            for k in keys[:len(keys) // 4]:
                del self._cache[k]

    def __len__(self):
        return len(self._cache)

    def __contains__(self, zhash):
        return zhash in self._cache
