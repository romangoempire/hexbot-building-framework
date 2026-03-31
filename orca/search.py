"""
orca/search.py - Search algorithms (MCTS, Alpha-Beta).

Contains:
- MCTSNode
- MCTS (single-threaded)
- BatchedMCTS (batched NN inference)
- NNAlphaBeta (callback-based)
- BatchedNNAlphaBeta (collect-inject)
"""

from __future__ import annotations

import ctypes
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from main import HexGame

# Import config constants
try:
    from orca.config import (
        BOARD_SIZE, NUM_CHANNELS, NUM_SIMULATIONS, C_PUCT,
        DIRICHLET_ALPHA, DIRICHLET_EPSILON,
        PLAY_STYLE, C_BLEND_ADJACENT, C_BLEND_DISTANT,
        MCTS_BATCH_SIZE,
    )
except ImportError:
    BOARD_SIZE = 19
    NUM_CHANNELS = 7
    NUM_SIMULATIONS = 400
    C_PUCT = 1.5
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPSILON = 0.25
    PLAY_STYLE = 'distant'
    C_BLEND_ADJACENT = 0.15
    C_BLEND_DISTANT = 0.05
    MCTS_BATCH_SIZE = 64

# Lazy imports from sibling modules to avoid circular imports
from orca.encoding import (
    encode_state, decode_policy,
    c_encode_state, c_decode_policy,
    CGameState, _get_lib,
)
from orca.threats import _line_through_candidate


# ---------------------------------------------------------------------------
# MCTSNode
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


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

class MCTS:
    # Progressive widening constants (class-level for access by BatchedMCTS)
    INITIAL_WIDTH = 6    # only 6 children at first -> deep tree
    WIDEN_AT_20 = 10     # after 20 visits, expand to 10
    WIDEN_AT_50 = 16     # after 50 visits, expand to 16
    WIDEN_AT_100 = 25    # after 100 visits, expand to 25

    def __init__(
        self,
        net,
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

    def _expand(self, node: MCTSNode, game: HexGame) -> float:
        """Expand node using NN with progressive widening.
        Only creates top-K children initially. More are added as visits increase.
        This forces the tree DEEPER instead of wider - key for multi-move lookahead."""
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
# NN-Guided Alpha-Beta Search
# ---------------------------------------------------------------------------

class NNAlphaBeta:
    """Alpha-beta search with NN evaluation at leaves.
    Uses C engine for fast move ordering + pruning, NN for position evaluation.
    Reaches depth 8-12 (vs MCTS depth 3-5) - sees 4-6 full turns ahead.
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
        import os
        lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'engine.so')
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
            # HexGame - extract (q, r) from undo records
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
# Batched NN Alpha-Beta
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

            # Pad 5->NUM_CHANNELS (7) channels
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

            # Read hashes (must copy - C buffer may be reused)
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
# Batched MCTS with Virtual Loss
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
                    # Proven win/loss - skip MCTS entirely
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
                # Already expanded (rare collision) - just backup
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
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    'MCTSNode', 'MCTS',
    'NNAlphaBeta', 'BatchedNNAlphaBeta',
    'BatchedMCTS',
    '_get_existing_stones',
]
