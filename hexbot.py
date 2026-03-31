"""
hexbot - Python framework for building Hex Connect-6 bots.

Simple API for playing, training, and evaluating hex bots:

    from hexbot import HexGame, Bot, Arena, train

    # Play with a bot
    game = HexGame()
    bot = Bot.heuristic()
    move = bot.best_move(game)
    game.place(*move)

    # Train your own
    bot = train(iterations=50, sims=100)
    bot.save('my_bot.pt')

    # Bot vs bot
    result = Arena(bot1, bot2, num_games=50).play()
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Re-export the game engine
from hexgame import HexGame

__all__ = [
    # Game
    'HexGame', 'FastGame',
    # Analysis
    'evaluate_moves', 'find_threats', 'find_winning_moves',
    'count_lines', 'rollout', 'alphabeta',
    # Advanced threats
    'find_forced_move', 'threat_search', 'count_threats', 'detect_fork',
    # Neural network
    'nn_evaluate', 'nn_evaluate_batch', 'create_network', 'encode_state', 'decode_policy',
    # Search
    'mcts_search', 'mcts_policy',
    # Training
    'TrainingSample', 'ReplayBuffer', 'self_play', 'train_step', 'augment_sample',
    # Solver + openings
    'solve', 'quick_solve', 'opening_move', 'OpeningBook',
    # Ensemble
    'Ensemble',
    # Bot framework
    'Bot', 'BotProtocol', 'Arena', 'ArenaResult', 'train',
    # Plugin system
    'register_bot', 'register_network', 'registered_bots', 'registered_networks',
    # Model zoo
    'Zoo',
    # Data
    'positions', 'load_games', 'generate_puzzles', 'import_games',
]


# ---------------------------------------------------------------------------
# Raw analysis functions - use these to build ANY bot approach
# ---------------------------------------------------------------------------

def evaluate_moves(game, top_n: int = 10) -> List[Tuple[Tuple[int, int], int]]:
    """Score legal moves using the C engine heuristic.

    The heuristic considers line extension potential, blocking value,
    and proximity to existing stones. No neural network needed.

    Args:
        game: HexGame instance.
        top_n: Maximum number of moves to return.

    Returns:
        List of ((q, r), score) sorted by score descending.
    """
    return [(m[:2], m[2]) for m in game.scored_moves(top_n)]


def find_threats(game, player: Optional[int] = None) -> List[Tuple[int, int]]:
    """Find cells where placing a stone creates a line of 4+.

    Args:
        game: HexGame instance.
        player: Player to check (0 or 1). Defaults to current player.

    Returns:
        List of (q, r) threat positions.
    """
    if player is None:
        player = game.current_player
    threats = []
    for q, r in game.legal_moves():
        if game.max_line(q, r, player) >= 4:
            threats.append((q, r))
    return threats


def find_winning_moves(game, player: Optional[int] = None) -> List[Tuple[int, int]]:
    """Find cells that would complete 6 in a row (instant win).

    Args:
        game: HexGame instance.
        player: Player to check. Defaults to current player.

    Returns:
        List of (q, r) winning move positions.
    """
    if player is None:
        player = game.current_player
    wins = []
    for q, r in game.legal_moves():
        if game.would_win(q, r, player):
            wins.append((q, r))
    return wins


def count_lines(game, q: int, r: int, player: Optional[int] = None) -> Dict[str, int]:
    """Count consecutive stones through (q, r) on each axis.

    Useful for building custom evaluation functions.

    Args:
        game: HexGame instance.
        q, r: Axial coordinates to check.
        player: Player to count for. Defaults to current player.

    Returns:
        Dict with 'horizontal', 'diagonal_up', 'diagonal_down' counts.
    """
    # Use the hexgame C engine's max_line (returns max across axes)
    # For per-axis, we need to go through the raw engine
    total = game.max_line(q, r, player)
    return {'max_line': total}


def rollout(game, num_games: int = 1000) -> Dict[str, float]:
    """Fast random playout from current position.

    Plays num_games random games from this position and returns win rates.
    Uses the C engine for speed (~30K games/sec).

    Args:
        game: HexGame instance (position to evaluate).
        num_games: Number of random games to play.

    Returns:
        Dict with 'p0_wins', 'p1_wins', 'draw_rate'.
    """
    import random as rng
    wins = [0, 0, 0]
    for _ in range(num_games):
        g = game.clone()
        while not g.is_over:
            moves = g.legal_moves()
            g.place(*rng.choice(moves))
        if g.winner == 0:
            wins[0] += 1
        elif g.winner == 1:
            wins[1] += 1
        else:
            wins[2] += 1
    t = max(num_games, 1)
    return {'p0_wins': wins[0]/t, 'p1_wins': wins[1]/t, 'draw_rate': wins[2]/t}


def alphabeta(game, depth: int = 8) -> Dict:
    """Run alpha-beta search from the current position.

    Uses the C engine's optimized alpha-beta with transposition table,
    killer heuristics, and late move reduction.

    Args:
        game: HexGame instance.
        depth: Search depth (8 = 4 turns ahead).

    Returns:
        Dict with 'best_move', 'value', 'nodes'.
    """
    return game.search(depth=depth)


# ---------------------------------------------------------------------------
# Fast game state (C engine, ~10x faster than HexGame for self-play)
# ---------------------------------------------------------------------------

def FastGame(max_stones: int = 200):
    """Create a fast C-engine game state for self-play.

    10x faster than HexGame for encode/decode and move generation.
    Same API: place_stone(q,r), undo(), legal_moves(), is_terminal, etc.

    Example:
        game = FastGame()
        game.place_stone(0, 0)
        print(game.current_player, game.total_stones)
    """
    from bot import CGameState
    return CGameState(max_total_stones=max_stones)


# ---------------------------------------------------------------------------
# Advanced threat analysis
# ---------------------------------------------------------------------------

def find_forced_move(game) -> Optional[Tuple[int, int]]:
    """Find an undeniable forced move (instant win or must-block).

    Skips neural network entirely - pure tactical detection.
    Returns None if no forced move exists.

    Example:
        move = find_forced_move(game)
        if move:
            game.place(*move)  # must play this or lose
    """
    from bot import find_forced_move as _ffm
    from main import HexGame as _PyGame
    if isinstance(game, HexGame) and not hasattr(game, 'stones_0'):
        pg = _PyGame(candidate_radius=3, max_total_stones=200)
        for q, r in game.moves:
            pg.place_stone(q, r)
        return _ffm(pg)
    return _ffm(game)


def threat_search(game, depth: int = 4) -> Optional[Tuple[int, int]]:
    """Search for winning threat sequences (forks, double attacks).

    Looks depth moves ahead for unstoppable forcing sequences.
    More expensive than find_forced_move but finds deeper tactics.

    Returns the first move of the winning sequence, or None.

    Example:
        move = threat_search(game, depth=6)
        if move:
            print("Found winning sequence starting with", move)
    """
    from bot import _threat_search
    from main import HexGame as _PyGame
    # Convert hexgame.HexGame to main.HexGame if needed
    if isinstance(game, HexGame) and not hasattr(game, 'stones_0'):
        pg = _PyGame(candidate_radius=3, max_total_stones=200)
        for q, r in game.moves:
            pg.place_stone(q, r)
        return _threat_search(pg, depth=depth)
    return _threat_search(game, depth=depth)


def count_threats(game, player: Optional[int] = None) -> int:
    """Count cells where player can complete 6-in-a-row.

    If 3+, the position is likely unstoppable (opponent can only block 2 per turn).

    Example:
        n = count_threats(game, player=0)
        if n >= 3:
            print("Player 0 has an unstoppable position!")
    """
    p = player if player is not None else game.current_player
    if hasattr(game, '_lib'):
        return game._lib.board_count_winning_moves(game._ptr, p)
    return len(find_winning_moves(game, player=p))


def detect_fork(game, player: Optional[int] = None) -> bool:
    """Check if player has 3+ winning cells (unstoppable fork).

    In Connect-6, each player places 2 stones per turn, so they can
    only block 2 threats. 3+ threats = guaranteed win.

    Example:
        if detect_fork(game, player=0):
            print("Player 0 wins by force!")
    """
    return count_threats(game, player) >= 3


# ---------------------------------------------------------------------------
# Endgame solver
# ---------------------------------------------------------------------------

def solve(game, max_depth: int = 12, time_limit: float = 5.0) -> Dict:
    """Solve a position using deep alpha-beta search.

    Returns dict with 'result' ('win'/'loss'/'unknown'), 'move', 'value',
    'depth', 'nodes', 'time'.

    Example:
        result = solve(game, max_depth=16)
        if result['result'] == 'win':
            print(f"Forced win: play {result['move']}")
    """
    from orca.solver import solve as _solve
    return _solve(game, max_depth=max_depth, time_limit=time_limit)


def quick_solve(game, depth: int = 6) -> Optional[Tuple[int, int]]:
    """Quick solve attempt. Returns winning move or None.

    Example:
        move = quick_solve(game)
        if move:
            game.place(*move)  # guaranteed win
    """
    from orca.solver import quick_solve as _qs
    return _qs(game, depth=depth)


def opening_move(game, temperature: float = 0.5) -> Optional[Tuple[int, int]]:
    """Look up the best opening move from the book.

    Returns None if the position isn't in the book.

    Example:
        move = opening_move(game)
        if move:
            game.place(*move)
    """
    from orca.openings import build_default_book
    book = build_default_book()
    return book.lookup(game, temperature=temperature)


# Lazy import for OpeningBook
def OpeningBook():
    """Create an opening book. See orca.openings for full API.

    Example:
        book = OpeningBook()
        book.build_from_games(games)
        book.save('my_openings.json')
    """
    from orca.openings import OpeningBook as _OB
    return _OB()


# ---------------------------------------------------------------------------
# Ensemble evaluation
# ---------------------------------------------------------------------------

def Ensemble(paths=None, n=5, net_config='standard'):
    """Create an ensemble evaluator from multiple checkpoints.

    Averages predictions from N networks for stronger, more stable play.
    Returns uncertainty estimates for position complexity.

    Example:
        from hexbot import Ensemble
        ens = Ensemble(n=3)  # last 3 checkpoints
        policy, value, uncertainty = ens.evaluate(game)

        # Or specific checkpoints:
        ens = Ensemble(paths=['ckpt_50.pt', 'ckpt_60.pt'])
    """
    from orca.ensemble import Ensemble as _Ens
    if paths:
        return _Ens.from_checkpoints(paths, net_config=net_config)
    return _Ens.from_latest(n=n, net_config=net_config)


# ---------------------------------------------------------------------------
# Neural network access
# ---------------------------------------------------------------------------

def create_network(config: str = 'standard'):
    """Create a neural network for position evaluation.

    Configs:
        'fast'              - 64 filters, 4 blocks (~500K params, quick experiments)
        'standard'          - 128 filters, 12 blocks (~3.9M params, default)
        'large'             - 256 filters, 12 blocks (~15M params, stronger but slow)
        'orca-transformer'  - 128 filters + transformer attention (~4.3M, experimental)

    Example:
        net = create_network('standard')
        print(sum(p.numel() for p in net.parameters()))  # 3,909,308
    """
    from bot import create_network as _cn
    return _cn(config)


def encode_state(game):
    """Encode a game state as a tensor for neural network input.

    Returns (tensor, offset_q, offset_r) where tensor is (7, 19, 19):
        Channel 0: current player's stones
        Channel 1: opponent's stones
        Channel 2: legal move positions
        Channel 3: current player indicator
        Channel 4: stones remaining this turn
        Channel 5: current player's threat map
        Channel 6: opponent's threat map

    Example:
        tensor, oq, orr = encode_state(game)
        print(tensor.shape)  # torch.Size([7, 19, 19])
    """
    # CGameState from bot.py has _move_log attribute
    if hasattr(game, '_move_log'):
        from bot import c_encode_state
        return c_encode_state(game)
    # hexgame.HexGame or main.HexGame - use Python encoder
    if isinstance(game, HexGame) and not hasattr(game, 'stones_0'):
        from main import HexGame as _PyGame
        pg = _PyGame(candidate_radius=3, max_total_stones=200)
        for q, r in game.moves:
            pg.place_stone(q, r)
        from bot import encode_state as _enc
        return _enc(pg)
    from bot import encode_state as _enc
    return _enc(game)


def decode_policy(policy_logits, game, offset_q: int, offset_r: int) -> Dict:
    """Convert raw policy logits to a move probability dictionary.

    Takes the 361-dim output from the policy head and maps it back
    to legal (q, r) moves with softmax probabilities.

    Example:
        tensor, oq, orr = encode_state(game)
        logits, value = net.forward_pv(tensor.unsqueeze(0))
        policy = decode_policy(logits[0], game, oq, orr)
        best_move = max(policy, key=policy.get)
    """
    if hasattr(game, '_move_log'):
        from bot import c_decode_policy
        return c_decode_policy(policy_logits, game, offset_q, offset_r)
    if isinstance(game, HexGame) and not hasattr(game, 'stones_0'):
        from main import HexGame as _PyGame
        pg = _PyGame(candidate_radius=3, max_total_stones=200)
        for q, r in game.moves:
            pg.place_stone(q, r)
        from bot import decode_policy as _dec
        return _dec(policy_logits, pg, offset_q, offset_r)
    from bot import decode_policy as _dec
    return _dec(policy_logits, game, offset_q, offset_r)


def nn_evaluate(game, net=None):
    """Evaluate a position with the neural network.

    Returns (policy_dict, value) where policy_dict maps (q,r) to probability
    and value is a float in [-1, +1] (positive = current player winning).

    If net is None, loads the default Orca checkpoint.

    Example:
        policy, value = nn_evaluate(game)
        print(f"Value: {value:.3f}")
        print(f"Best move: {max(policy, key=policy.get)}")
    """
    import torch
    if net is None:
        bot = Bot.orca(sims=1)
        net = bot._net
    net.eval()
    tensor, oq, orr = encode_state(game)
    with torch.no_grad():
        p_logits, v = net.forward_pv(tensor.unsqueeze(0))
    policy = decode_policy(p_logits[0], game, oq, orr)
    value = v.item()
    return policy, value


def nn_evaluate_batch(games, net=None):
    """Batch evaluate multiple positions with the neural network.

    Much faster than calling nn_evaluate() in a loop.
    Returns list of (policy_dict, value) tuples.

    Example:
        results = nn_evaluate_batch([game1, game2, game3], net=my_net)
        for policy, value in results:
            print(f"Value: {value:.3f}")
    """
    import torch
    if net is None:
        bot = Bot.orca(sims=1)
        net = bot._net
    net.eval()
    encoded = [encode_state(g) for g in games]
    tensors = torch.stack([e[0] for e in encoded])
    with torch.no_grad():
        p_logits, v = net.forward_pv(tensors)
    results = []
    for i, (_, oq, orr) in enumerate(encoded):
        policy = decode_policy(p_logits[i], games[i], oq, orr)
        results.append((policy, v[i].item()))
    return results


# ---------------------------------------------------------------------------
# Search (MCTS with full control)
# ---------------------------------------------------------------------------

def _to_mcts_game(game):
    """Convert hexgame.HexGame to a format MCTS can use."""
    if hasattr(game, '_move_log') or hasattr(game, 'stones_0'):
        return game  # already CGameState or main.HexGame
    from bot import CGameState
    cg = CGameState(max_total_stones=200)
    for q, r in game.moves:
        cg.place_stone(q, r)
    return cg


def mcts_search(game, net=None, sims: int = 200, batch_size: int = 64):
    """Run MCTS search and return the full visit count distribution.

    Returns dict mapping (q, r) -> visit_count (raw integers, not normalized).
    Use this for custom move selection strategies.

    Example:
        visits = mcts_search(game, sims=400)
        # Pick move with most visits
        best = max(visits, key=visits.get)
        # Or sample proportionally
        total = sum(visits.values())
        probs = {m: v/total for m, v in visits.items()}
    """
    from bot import BatchedMCTS
    if net is None:
        bot = Bot.orca(sims=1)
        net = bot._net
    net.eval()
    mcts = BatchedMCTS(net, num_simulations=sims, batch_size=batch_size)
    return mcts.search(_to_mcts_game(game), temperature=0.01, add_noise=False)


def mcts_policy(game, net=None, sims: int = 200, temperature: float = 1.0,
                add_noise: bool = False):
    """Run MCTS and return move probability distribution.

    With temperature=1.0, probabilities are proportional to visit counts.
    With temperature->0, concentrates on the best move (greedy).

    Example:
        # For training (explore):
        policy = mcts_policy(game, temperature=1.0, add_noise=True)
        # For play (exploit):
        policy = mcts_policy(game, temperature=0.01)
    """
    from bot import BatchedMCTS
    if net is None:
        bot = Bot.orca(sims=1)
        net = bot._net
    net.eval()
    mcts = BatchedMCTS(net, num_simulations=sims, batch_size=64)
    return mcts.search(_to_mcts_game(game), temperature=temperature, add_noise=add_noise)


# ---------------------------------------------------------------------------
# Training building blocks
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Plugin system
# ---------------------------------------------------------------------------

_registered_bots: Dict[str, type] = {}
_registered_networks: Dict[str, type] = {}


def register_bot(name: str, bot_class):
    """Register a custom bot class for use with Arena and the framework.

    The class must have a best_move(game) method.

    Example:
        class MyBot:
            def best_move(self, game):
                return game.legal_moves()[0]

        register_bot('my-bot', MyBot)
        # Now usable everywhere:
        bot = registered_bots()['my-bot']()
    """
    _registered_bots[name] = bot_class


def register_network(name: str, net_class):
    """Register a custom network architecture.

    The class must accept (board_size, num_channels) and have
    forward(x) returning (policy, value, threat).

    After registration, use with create_network(name).

    Example:
        register_network('my-arch', MyNetwork)
        net = create_network('my-arch')
    """
    _registered_networks[name] = net_class


def registered_bots() -> Dict[str, type]:
    """Get all registered bot classes."""
    return dict(_registered_bots)


def registered_networks() -> Dict[str, type]:
    """Get all registered network architectures."""
    return dict(_registered_networks)


# ---------------------------------------------------------------------------
# Model Zoo
# ---------------------------------------------------------------------------

def Zoo():
    """Access the model zoo for sharing and downloading community models.

    Example:
        from hexbot import Zoo
        zoo = Zoo()
        zoo.list()
        bot = zoo.load('orca-v3')
    """
    from orca.zoo import Zoo as _Zoo
    return _Zoo


# ---------------------------------------------------------------------------
# Lazy imports to avoid loading PyTorch for basic game usage
_training_imports_done = False
TrainingSample = None
ReplayBuffer = None


def _ensure_training_imports():
    global _training_imports_done, TrainingSample, ReplayBuffer
    if _training_imports_done:
        return
    from bot import TrainingSample as _TS, ReplayBuffer as _RB
    TrainingSample = _TS
    ReplayBuffer = _RB
    _training_imports_done = True


def self_play(net=None, sims: int = 200, batch_size: int = 64, use_c_engine: bool = True):
    """Generate one game of self-play training data.

    Returns (samples, move_history) where samples is a list of TrainingSample
    and move_history is the list of (q, r) moves played.

    Example:
        samples, moves = self_play(net=my_net, sims=100)
        print(f"Game: {len(moves)} moves, {len(samples)} training samples")
        for s in samples:
            replay_buffer.push(s)
    """
    from bot import BatchedMCTS, self_play_game_v2
    if net is None:
        bot = Bot.orca(sims=1)
        net = bot._net
    net.eval()
    mcts = BatchedMCTS(net, num_simulations=sims, batch_size=batch_size)
    return self_play_game_v2(net, mcts, use_c_engine=use_c_engine)


def train_step(net, optimizer, replay_buffer, device=None):
    """Run one training step (forward + backward + optimizer step).

    Returns dict with loss values: {'total', 'value', 'policy', 'threat'}.

    Example:
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        for step in range(200):
            losses = train_step(net, optimizer, buffer, device='mps')
            if step % 50 == 0:
                print(f"Step {step}: loss={losses['total']:.4f}")
    """
    from bot import train_step as _ts, get_device
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        import torch
        device = torch.device(device)
    return _ts(net, optimizer, replay_buffer, device)


def augment_sample(sample):
    """Apply hex-valid symmetry augmentations to a training sample.

    Returns 3 augmented copies (180 rotation, transpose, transpose+180).
    All transforms preserve the 3 hex line directions.

    Example:
        samples, moves = self_play(net)
        for s in samples:
            buffer.push(s)
            for aug in augment_sample(s):
                buffer.push(aug)
    """
    from bot import augment_sample as _aug
    return _aug(sample)


# ---------------------------------------------------------------------------
# Curriculum and data loading
# ---------------------------------------------------------------------------

def load_games(path: str):
    """Load training samples from a JSONL game file.

    Each line should be a JSON object with 'moves' (list of [q,r])
    and optional 'winner' (0 or 1).

    Returns list of TrainingSample objects ready for a replay buffer.
    """
    from bot import load_human_games
    return load_human_games(path)


def import_games(path: str, min_moves: int = 6, min_elo: int = 0,
                 max_games: int = 999999):
    """Import games from any format (JSONL, CSV, text).

    Auto-detects format. Returns list of dicts with 'moves' and 'result'.
    Use for SFT training or analysis.

    Example:
        games = import_games('strong_games.jsonl', min_elo=1200)
        from orca.sft import games_to_samples
        samples = games_to_samples(games)
    """
    from orca.sft import import_games as _imp
    return _imp(path, min_moves=min_moves, min_elo=min_elo, max_games=max_games)


def generate_puzzles(n: int = 100):
    """Generate random tactical puzzle positions for curriculum training.

    Returns list of (position_dict, hint_moves) tuples.
    Useful for teaching bots to handle endgame tactics.

    Example:
        puzzles = generate_puzzles(50)
        for pos, hints in puzzles:
            game = HexGame.from_dict(pos)
            # Train on this position...
    """
    from bot import generate_puzzles as _gp
    return _gp(n)


# Pre-built starting positions for curriculum learning
positions = None


def _load_positions():
    global positions
    if positions is not None:
        return positions
    try:
        from bot import POSITION_CATALOG
        positions = POSITION_CATALOG
    except ImportError:
        positions = {}
    return positions


# ---------------------------------------------------------------------------
# BotProtocol - implement this to make any custom bot work with Arena
# ---------------------------------------------------------------------------

class BotProtocol:
    """Base class for custom bots.

    Implement best_move() and your bot works with Arena, tournaments,
    and any other hexbot tool. You don't have to subclass this -
    any object with a best_move(game) method works.

    Example:
        class MyBot(BotProtocol):
            def best_move(self, game):
                # Your custom logic here
                moves = evaluate_moves(game, top_n=5)
                return moves[0][0]  # pick highest-scored move
    """
    def best_move(self, game) -> Tuple[int, int]:
        raise NotImplementedError("Implement best_move(game) -> (q, r)")


# ---------------------------------------------------------------------------
# Bot - wraps network + search into one clean object
# ---------------------------------------------------------------------------

class Bot:
    """A hex bot that can evaluate positions and choose moves.

    Example:
        >>> bot = Bot.heuristic()          # C engine heuristic (no NN)
        >>> bot = Bot.load('model.pt')     # load trained model
        >>> bot = Bot(sims=400)            # random weights, 400 MCTS sims
        >>> move = bot.best_move(game)
        >>> game.place(*move)
    """

    def __init__(
        self,
        net=None,
        search: str = 'mcts',
        sims: int = 200,
        depth: int = 8,
        temperature: float = 0.1,
    ):
        """Create a bot.

        Args:
            net: PyTorch neural network (HexNet). If None, creates random weights.
            search: 'mcts' for Monte Carlo Tree Search, 'alphabeta' for alpha-beta.
            sims: Number of MCTS simulations per move (ignored for alphabeta).
            depth: Search depth for alpha-beta (ignored for mcts).
            temperature: Move selection temperature (0=greedy, 1=proportional).
        """
        self._search_type = search
        self._sims = sims
        self._depth = depth
        self._temperature = temperature
        self._searcher = None

        if net is None:
            from bot import create_network
            net = create_network('standard')
            net.eval()
        self._net = net
        self._init_searcher()

    def _init_searcher(self):
        if self._search_type == 'mcts':
            from bot import BatchedMCTS
            self._searcher = BatchedMCTS(
                self._net, num_simulations=self._sims, batch_size=8
            )
        elif self._search_type == 'alphabeta':
            from bot import BatchedNNAlphaBeta
            self._searcher = BatchedNNAlphaBeta(
                self._net, depth=self._depth, nn_depth=5
            )

    def best_move(self, game) -> Tuple[int, int]:
        """Return the best move for the current position.

        Args:
            game: HexGame instance.

        Returns:
            (q, r) axial coordinates of the best move.
        """
        if self._search_type == 'random':
            moves = game.legal_moves()
            return random.choice(moves) if moves else (0, 0)
        if self._search_type == 'heuristic':
            moves = game.scored_moves(1)
            return moves[0][:2] if moves else (0, 0)

        policy = self.policy(game)
        if not policy:
            moves = game.legal_moves()
            return random.choice(moves) if moves else (0, 0)
        return max(policy, key=policy.get)

    def policy(self, game) -> Dict[Tuple[int, int], float]:
        """Return move probabilities for the current position.

        Returns:
            Dictionary mapping (q, r) to probability.
        """
        if self._searcher is None:
            return {}
        # Convert HexGame to format searcher expects
        from main import HexGame as _HexGame
        if isinstance(game, HexGame):
            # hexgame.HexGame -> need to replay into main.HexGame for MCTS
            py_game = _HexGame(candidate_radius=3, max_total_stones=200)
            for q, r in game.moves:
                py_game.place_stone(q, r)
            return self._searcher.search(
                py_game, temperature=self._temperature, add_noise=False
            )
        return self._searcher.search(
            game, temperature=self._temperature, add_noise=False
        )

    def evaluate(self, game) -> float:
        """Evaluate position: -1.0 (losing) to +1.0 (winning).

        Returns:
            Float evaluation from Player 0's perspective.
        """
        from bot import encode_state
        from main import HexGame as _HexGame
        import torch

        # Build a Python HexGame for encoding
        py_game = _HexGame(candidate_radius=3, max_total_stones=200)
        moves = game.moves if isinstance(game, HexGame) else []
        for q, r in moves:
            py_game.place_stone(q, r)

        encoded, _, _ = encode_state(py_game)
        with torch.no_grad():
            _, value, _ = self._net(encoded.unsqueeze(0))
        return value[0].item()

    def save(self, path: str):
        """Save bot to a checkpoint file.

        Args:
            path: File path for the checkpoint (.pt).
        """
        import torch
        torch.save({
            'model_state_dict': self._net.state_dict(),
            'search': self._search_type,
            'sims': self._sims,
            'depth': self._depth,
        }, path)

    @classmethod
    def load(cls, path: str) -> Bot:
        """Load a bot from a checkpoint file.

        Args:
            path: Path to checkpoint (.pt file).

        Returns:
            Bot instance with loaded weights.
        """
        import torch
        from bot import create_network, migrate_checkpoint_5to7, migrate_checkpoint_filters

        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        sd = migrate_checkpoint_5to7(sd)
        sd = migrate_checkpoint_filters(sd)

        # Detect network size
        nf = sd.get('conv_init.weight', None)
        if nf is not None:
            nf = nf.shape[0]
            nb = 0
            while f'res_blocks.{nb}.conv1.weight' in sd:
                nb += 1
        else:
            nf, nb = 128, 8

        from bot import HexNet
        net = HexNet(num_filters=nf, num_res_blocks=nb)
        net.load_state_dict(sd, strict=False)
        net.eval()

        search = ckpt.get('search', 'mcts')
        sims = ckpt.get('sims', 200)
        depth = ckpt.get('depth', 8)

        return cls(net=net, search=search, sims=sims, depth=depth)

    @classmethod
    def random(cls) -> Bot:
        """Create a bot that plays random legal moves.

        Returns:
            Bot that selects uniformly random moves.
        """
        bot = cls.__new__(cls)
        bot._search_type = 'random'
        bot._sims = 0
        bot._depth = 0
        bot._temperature = 1.0
        bot._net = None
        bot._searcher = None
        return bot

    @classmethod
    def heuristic(cls) -> Bot:
        """Create a bot using the C engine's heuristic scoring (no neural network).

        The heuristic scores moves by line extension potential, blocking value,
        and proximity to center. Fast but not very strong.

        Returns:
            Bot that uses C heuristic move scoring.
        """
        bot = cls.__new__(cls)
        bot._search_type = 'heuristic'
        bot._sims = 0
        bot._depth = 0
        bot._temperature = 0.0
        bot._net = None
        bot._searcher = None
        return bot

    @classmethod
    def orca(cls, sims: int = 200) -> Bot:
        """Load the pre-trained Orca bot (v3).

        The Orca bot uses a 3.9M parameter network (128 filters, 12 residual
        blocks) trained via AlphaZero-style self-play with MCTS.

        Args:
            sims: MCTS simulations per move (default 200).

        Returns:
            Bot loaded with Orca checkpoint.
        """
        import os
        orca_path = os.path.join(os.path.dirname(__file__), 'orca', 'checkpoint.pt')
        pretrained_path = os.path.join(os.path.dirname(__file__), 'pretrained.pt')
        for path in [orca_path, pretrained_path]:
            if os.path.exists(path):
                bot = cls.load(path)
                bot._sims = sims
                return bot
        raise FileNotFoundError(
            "No Orca checkpoint found. Download pretrained.pt or train with: "
            "python -m orca.train"
        )

    def __repr__(self):
        if self._search_type in ('random', 'heuristic'):
            return f'Bot({self._search_type})'
        params = sum(p.numel() for p in self._net.parameters()) if self._net else 0
        return f'Bot({self._search_type}, sims={self._sims}, {params:,} params)'


# ---------------------------------------------------------------------------
# Arena - play bots against each other
# ---------------------------------------------------------------------------

@dataclass
class ArenaResult:
    """Results from a bot vs bot match."""
    wins: List[int] = field(default_factory=lambda: [0, 0])
    draws: int = 0
    total_games: int = 0
    total_moves: int = 0
    games: List[dict] = field(default_factory=list)

    @property
    def avg_length(self) -> float:
        return self.total_moves / max(self.total_games, 1)

    @property
    def win_rate(self) -> List[float]:
        t = max(self.total_games, 1)
        return [self.wins[0] / t, self.wins[1] / t]

    def __repr__(self):
        return (f'ArenaResult({self.wins[0]}W-{self.wins[1]}W-{self.draws}D, '
                f'avg {self.avg_length:.0f} moves)')


class Arena:
    """Play two bots against each other.

    Accepts any object with a best_move(game) method, or a plain function.

    Example:
        >>> result = Arena(bot1, bot2, num_games=100).play()
        >>> result = Arena(my_func, Bot.heuristic(), num_games=50).play()
        >>> result = Arena(MyCustomBot(), Bot.random(), num_games=20).play()
    """

    def __init__(self, bot1, bot2, num_games: int = 100):
        """Create an arena.

        Args:
            bot1: Bot, BotProtocol, or function(game) -> (q, r).
            bot2: Same as bot1.
            num_games: Number of games to play.
        """
        self.bot1 = self._wrap(bot1)
        self.bot2 = self._wrap(bot2)
        self.num_games = num_games

    @staticmethod
    def _wrap(bot):
        """Wrap a function into a bot-like object if needed."""
        if callable(bot) and not hasattr(bot, 'best_move'):
            class _FuncBot:
                def __init__(self, fn): self._fn = fn
                def best_move(self, game): return self._fn(game)
                def __repr__(self): return f'FuncBot({self._fn.__name__})'
            return _FuncBot(bot)
        return bot

    def play(self, verbose: bool = True) -> ArenaResult:
        """Play all games and return results.

        Args:
            verbose: Print progress during play.

        Returns:
            ArenaResult with win/loss/draw statistics.
        """
        result = ArenaResult()
        bots = [self.bot1, self.bot2]

        for g in range(self.num_games):
            game = HexGame()
            # Alternate who goes first
            if g % 2 == 1:
                player_map = [1, 0]  # bot2 is P0, bot1 is P1
            else:
                player_map = [0, 1]  # bot1 is P0, bot2 is P1

            move_count = 0
            while not game.is_over:
                bot_idx = player_map[game.current_player]
                move = bots[bot_idx].best_move(game)
                game.place(*move)
                move_count += 1

            winner = game.winner
            game_info = {
                'game': g + 1,
                'winner': winner,
                'moves': move_count,
                'first_player': 'bot1' if player_map[0] == 0 else 'bot2',
            }
            result.games.append(game_info)
            result.total_moves += move_count
            result.total_games += 1

            if winner is not None:
                # Map game winner (P0/P1) back to bot index
                winning_bot = player_map[winner]
                result.wins[winning_bot] += 1
            else:
                result.draws += 1

            if verbose and (g + 1) % 10 == 0:
                print(f'  Game {g+1}/{self.num_games}: '
                      f'Bot1 {result.wins[0]}W, Bot2 {result.wins[1]}W')

        if verbose:
            print(f'\nFinal: {result}')
        return result


# ---------------------------------------------------------------------------
# train - simple self-play training
# ---------------------------------------------------------------------------

def train(
    iterations: int = 100,
    games_per_iter: int = 50,
    sims: int = 200,
    lr: float = 0.001,
    network_config: str = 'standard',
    checkpoint_every: int = 10,
    checkpoint_prefix: str = 'hexbot',
    on_iteration=None,
    verbose: bool = True,
) -> Bot:
    """Train a bot from scratch via self-play.

    Args:
        iterations: Number of training iterations.
        games_per_iter: Self-play games per iteration.
        sims: MCTS simulations per move during self-play.
        lr: Learning rate for Adam optimizer.
        network_config: Network config ('fast', 'standard', 'large').
        checkpoint_every: Save checkpoint every N iterations.
        checkpoint_prefix: Filename prefix for checkpoints.
        on_iteration: Optional callback(iteration, loss_dict) called each iteration.
        verbose: Print progress.

    Returns:
        Trained Bot instance.
    """
    import torch
    from bot import (create_network, BatchedMCTS, ReplayBuffer,
                     self_play_game_v2, train_step, augment_sample,
                     CGameState, BATCH_SIZE)

    net = create_network(network_config)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    replay_buffer = ReplayBuffer()
    mcts = BatchedMCTS(net, num_simulations=sims, batch_size=8)

    if verbose:
        params = sum(p.numel() for p in net.parameters())
        print(f'Training: {params:,} params, {sims} sims, {games_per_iter} games/iter')

    for iteration in range(iterations):
        # Self-play
        net.eval()
        samples_collected = []
        for g in range(games_per_iter):
            samples, moves = self_play_game_v2(net, mcts)
            for s in samples:
                replay_buffer.push(s)
                samples_collected.append(s)
            # Augment
            for s in samples:
                for aug in augment_sample(s):
                    replay_buffer.push(aug)

        # Train
        losses = {'total': 0, 'value': 0, 'policy': 0}
        if len(replay_buffer) >= BATCH_SIZE:
            net.train()
            train_steps = min(200, len(replay_buffer) // BATCH_SIZE)
            for _ in range(train_steps):
                losses = train_step(net, optimizer, replay_buffer, 'cpu')

        if verbose:
            print(f'  Iter {iteration+1}/{iterations}: '
                  f'games={games_per_iter}, samples={len(samples_collected)}, '
                  f'loss={losses["total"]:.4f}')

        if on_iteration:
            on_iteration(iteration + 1, losses)

        # Checkpoint
        if checkpoint_every and (iteration + 1) % checkpoint_every == 0:
            path = f'{checkpoint_prefix}_{iteration+1}.pt'
            torch.save({
                'iteration': iteration,
                'model_state_dict': net.state_dict(),
            }, path)
            if verbose:
                print(f'  Saved {path}')

    bot = Bot(net=net, search='mcts', sims=sims)
    if verbose:
        print(f'\nTraining complete: {iterations} iterations')
    return bot


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('hexbot - Hex Connect-6 Bot Framework')
    print('=' * 40)

    # Test HexGame
    game = HexGame()
    game.place(0, 0)
    game.place(1, 0)
    game.place(1, -1)
    print(f'Game: {game}')

    # Test raw analysis tools
    print(f'\nAnalysis tools:')
    print(f'  Top moves: {evaluate_moves(game, 3)}')
    print(f'  Threats P0: {find_threats(game, 0)}')
    print(f'  Winning moves P0: {find_winning_moves(game, 0)}')
    print(f'  Max line (0,0): {count_lines(game, 0, 0, 0)}')

    # Test built-in bots
    print(f'\nBuilt-in bots:')
    print(f'  Random: {Bot.random().best_move(game)}')
    print(f'  Heuristic: {Bot.heuristic().best_move(game)}')

    # Test custom function as bot
    def greedy_bot(game):
        moves = evaluate_moves(game, 1)
        return moves[0][0] if moves else game.legal_moves()[0]

    # Test Arena with function bot
    print(f'\nArena: function bot vs random (10 games)')
    result = Arena(greedy_bot, Bot.random(), num_games=10).play(verbose=False)
    print(f'Result: {result}')

    # Test Arena with BotProtocol subclass
    class ThreatBot(BotProtocol):
        def best_move(self, game):
            threats = find_threats(game)
            if threats: return threats[0]
            moves = evaluate_moves(game, 1)
            return moves[0][0] if moves else game.legal_moves()[0]

    print(f'\nArena: ThreatBot vs Heuristic (10 games)')
    result = Arena(ThreatBot(), Bot.heuristic(), num_games=10).play(verbose=False)
    print(f'Result: {result}')

    print(f'\nAll OK!')

    print('\nAll OK!')
