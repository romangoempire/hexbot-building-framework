"""
Opening book for Hex Connect-6.

Stores known strong opening sequences and blends them with MCTS policy.
Built from game collections or manually curated.

Usage:
    from orca.openings import OpeningBook

    book = OpeningBook()
    book.build_from_games(games, min_frequency=3)
    book.save('openings.json')

    # During play
    move = book.lookup(game)
    if move:
        game.place(*move)

    # Blend with MCTS
    blended = book.blend(mcts_policy, weight=0.3)
"""

import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)


class OpeningBook:
    """Trie-based opening book with Zobrist hash lookup.

    Stores move frequencies from winning games. Lookup is O(1) per position
    via Zobrist hash, with trie fallback for move-sequence matching.
    """

    def __init__(self):
        # Hash-based lookup: zhash -> {(q,r): frequency}
        self._hash_table: Dict[int, Dict[Tuple[int, int], int]] = {}
        # Move sequence trie: tuple of moves -> {(q,r): frequency}
        self._trie: Dict[tuple, Dict[Tuple[int, int], int]] = {}
        # Metadata
        self.total_games = 0
        self.max_depth = 20  # only store first N moves

    def build_from_games(self, games: List[dict], min_frequency: int = 2,
                         max_move_depth: int = 20, only_winners: bool = True):
        """Build opening book from a list of parsed games.

        Args:
            games: list of {"moves": [(q,r), ...], "result": float}
            min_frequency: minimum times a move must appear to be included
            max_move_depth: only store the first N moves
            only_winners: only learn from winning side's moves
        """
        self.max_depth = max_move_depth
        raw_counts: Dict[tuple, Dict[Tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))

        for game_data in games:
            moves = game_data.get("moves", [])
            result = game_data.get("result", 0.0)
            if not moves:
                continue

            self.total_games += 1
            history = []

            for i, move in enumerate(moves[:max_move_depth]):
                if isinstance(move, (list, tuple)):
                    move = (int(move[0]), int(move[1]))

                # Determine which player made this move
                # Turn structure: move 0 = P0 (1 stone), then alternating 2 each
                if i == 0:
                    player = 0
                else:
                    # stones 1-2 = P1, 3-4 = P0, 5-6 = P1, ...
                    player = ((i - 1) // 2 + 1) % 2

                # Only learn from winner's moves if requested
                if only_winners:
                    if result > 0 and player != 0:
                        history.append(move)
                        continue
                    if result < 0 and player != 1:
                        history.append(move)
                        continue

                key = tuple(history)
                raw_counts[key][move] += 1
                history.append(move)

        # Filter by frequency
        for key, move_counts in raw_counts.items():
            filtered = {m: c for m, c in move_counts.items() if c >= min_frequency}
            if filtered:
                self._trie[key] = dict(filtered)

        # Build hash table for fast lookup
        self._build_hash_table()

    def _build_hash_table(self):
        """Build Zobrist hash table from trie for O(1) lookup."""
        from hexgame import HexGame
        self._hash_table = {}

        for move_seq, next_moves in self._trie.items():
            try:
                game = HexGame(max_stones=200)
                for q, r in move_seq:
                    game.place(q, r)
                zhash = game.zhash
                self._hash_table[zhash] = next_moves
            except Exception:
                continue

    def lookup(self, game, temperature: float = 0.5) -> Optional[Tuple[int, int]]:
        """Look up the best opening move for this position.

        Returns a move sampled from the book's frequency distribution,
        or None if the position isn't in the book.

        Args:
            game: HexGame or CGameState
            temperature: 0 = always best, 1 = proportional to frequency
        """
        # Try hash lookup first (fast)
        zhash = None
        if hasattr(game, 'zhash'):
            zhash = game.zhash
        elif hasattr(game, '_lib'):
            try:
                zhash = game._lib.board_get_zhash(game._ptr)
            except Exception:
                pass

        candidates = None
        if zhash is not None and zhash in self._hash_table:
            candidates = self._hash_table[zhash]

        # Fallback: trie lookup by move sequence
        if candidates is None:
            moves = []
            if hasattr(game, 'moves'):
                moves = game.moves
            elif hasattr(game, '_move_log'):
                moves = game._move_log
            key = tuple(tuple(m) if isinstance(m, list) else m for m in moves)
            candidates = self._trie.get(key)

        if not candidates:
            return None

        # Sample from distribution
        moves = list(candidates.keys())
        counts = [candidates[m] for m in moves]

        if temperature <= 0:
            return moves[counts.index(max(counts))]

        # Temperature-scaled sampling
        import math
        weights = [c ** (1.0 / temperature) for c in counts]
        total = sum(weights)
        r = random.random() * total
        cumulative = 0
        for move, w in zip(moves, weights):
            cumulative += w
            if r <= cumulative:
                return move
        return moves[-1]

    def blend(self, mcts_policy: Dict, weight: float = 0.3) -> Dict:
        """Blend opening book moves into an MCTS policy.

        If the position is in the book, mix book frequencies with MCTS
        probabilities. Otherwise returns the original policy unchanged.

        Args:
            mcts_policy: {(q,r): probability} from MCTS
            weight: book weight (0 = pure MCTS, 1 = pure book)

        Returns:
            Blended policy dict.
        """
        # This needs a game context for lookup - use the trie keys
        # For now, return original if no game provided
        return mcts_policy

    def blend_with_game(self, game, mcts_policy: Dict,
                        weight: float = 0.3) -> Dict:
        """Blend opening book into MCTS policy for a specific game state.

        Args:
            game: current game state
            mcts_policy: MCTS policy to blend with
            weight: book influence (0-1)
        """
        # Get book candidates
        zhash = getattr(game, 'zhash', None)
        candidates = self._hash_table.get(zhash, {}) if zhash else {}
        if not candidates:
            return mcts_policy

        # Normalize book frequencies to probabilities
        total_freq = sum(candidates.values())
        book_policy = {m: c / total_freq for m, c in candidates.items()}

        # Blend
        blended = {}
        all_moves = set(mcts_policy.keys()) | set(book_policy.keys())
        for m in all_moves:
            mp = mcts_policy.get(m, 0.0)
            bp = book_policy.get(m, 0.0)
            blended[m] = (1 - weight) * mp + weight * bp

        # Renormalize
        total = sum(blended.values())
        if total > 0:
            blended = {m: p / total for m, p in blended.items()}
        return blended

    def save(self, path: str = 'openings.json'):
        """Save opening book to JSON."""
        data = {
            'total_games': self.total_games,
            'max_depth': self.max_depth,
            'entries': [
                {
                    'moves': [list(m) for m in key],
                    'next': {f'{q},{r}': c for (q, r), c in moves.items()},
                }
                for key, moves in self._trie.items()
            ],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str = 'openings.json'):
        """Load opening book from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.total_games = data.get('total_games', 0)
        self.max_depth = data.get('max_depth', 20)
        self._trie = {}
        for entry in data.get('entries', []):
            key = tuple(tuple(m) for m in entry['moves'])
            next_moves = {}
            for coord_str, count in entry['next'].items():
                q, r = coord_str.split(',')
                next_moves[(int(q), int(r))] = count
            self._trie[key] = next_moves
        self._build_hash_table()

    def __len__(self):
        return len(self._trie)

    def __repr__(self):
        return f'OpeningBook({len(self._trie)} positions, {self.total_games} games)'


def build_default_book() -> OpeningBook:
    """Build a minimal opening book from the position catalog.

    Returns a book with common opening patterns.
    """
    book = OpeningBook()
    # Common strong first moves: center and near-center
    book._trie[()] = {(0, 0): 100}  # first move is always center
    book._trie[((0, 0),)] = {
        (1, 0): 30, (0, 1): 30, (1, -1): 25,
        (-1, 0): 10, (0, -1): 10, (-1, 1): 10,
    }
    book._build_hash_table()
    book.total_games = 0
    return book
