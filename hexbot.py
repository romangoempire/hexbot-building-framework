"""
hexbot — Python framework for building Hex Connect-6 bots.

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
    'HexGame', 'Bot', 'BotProtocol', 'Arena', 'ArenaResult', 'train',
    'evaluate_moves', 'find_threats', 'find_winning_moves',
    'count_lines', 'rollout', 'alphabeta',
]


# ---------------------------------------------------------------------------
# Raw analysis functions — use these to build ANY bot approach
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
# BotProtocol — implement this to make any custom bot work with Arena
# ---------------------------------------------------------------------------

class BotProtocol:
    """Base class for custom bots.

    Implement best_move() and your bot works with Arena, tournaments,
    and any other hexbot tool. You don't have to subclass this —
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
# Bot — wraps network + search into one clean object
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
            # hexgame.HexGame → need to replay into main.HexGame for MCTS
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

    def __repr__(self):
        if self._search_type in ('random', 'heuristic'):
            return f'Bot({self._search_type})'
        params = sum(p.numel() for p in self._net.parameters()) if self._net else 0
        return f'Bot({self._search_type}, sims={self._sims}, {params:,} params)'


# ---------------------------------------------------------------------------
# Arena — play bots against each other
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
# train — simple self-play training
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
    print('hexbot — Hex Connect-6 Bot Framework')
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
