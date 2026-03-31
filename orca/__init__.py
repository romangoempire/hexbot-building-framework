"""
Orca - AlphaZero-style Hex Connect-6 bot.

Quick start:
    from orca import Orca
    bot = Orca.load()
    move = bot.best_move(game)

Training:
    python -m orca.train --iterations 100
"""

__version__ = '4.0.0'
BOT_NAME = 'Orca'

import os
import sys

# Ensure parent directory is importable
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)


class Orca:
    """The Orca bot - wraps hexbot.Bot with auto-checkpoint loading.

    Usage:
        bot = Orca.load()                    # load latest checkpoint
        bot = Orca.load(sims=400)            # more MCTS simulations
        bot = Orca.load('my_checkpoint.pt')  # specific checkpoint
        move = bot.best_move(game)
    """

    @staticmethod
    def load(checkpoint_path=None, sims=200, search='mcts'):
        """Load the Orca bot from a checkpoint.

        Searches for checkpoints in this order:
        1. Explicit path (if provided)
        2. orca/checkpoint.pt
        3. pretrained.pt
        4. Latest hex_checkpoint_*.pt

        Args:
            checkpoint_path: Path to .pt file. Auto-detected if None.
            sims: MCTS simulations per move (default 200).
            search: 'mcts' or 'alphabeta'.

        Returns:
            hexbot.Bot instance loaded with weights.
        """
        from hexbot import Bot
        import glob

        if checkpoint_path is None:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidates = [
                os.path.join(root, 'orca', 'checkpoint.pt'),
                os.path.join(root, 'pretrained.pt'),
            ]
            # Also check for latest hex_checkpoint_*.pt
            ckpts = sorted(glob.glob(os.path.join(root, 'hex_checkpoint_*.pt')))
            if ckpts:
                candidates.append(ckpts[-1])

            for path in candidates:
                if os.path.exists(path):
                    checkpoint_path = path
                    break

        if checkpoint_path is None:
            raise FileNotFoundError(
                "No checkpoint found. Download pretrained.pt or train with: "
                "python -m orca.train"
            )

        return Bot.load(checkpoint_path, sims=sims, search=search)

    @staticmethod
    def train(**kwargs):
        """Start training. See orca.train.OrcaTrainer for all options.

        Quick start:
            Orca.train(iterations=50, games_per_iter=20)
        """
        from orca.train import OrcaTrainer
        trainer = OrcaTrainer(**kwargs)
        trainer.run()
        return trainer


def train_orca(**kwargs):
    """Convenience function to train the Orca bot.

    Args:
        iterations: Number of training iterations (default 100).
        games_per_iter: Self-play games per iteration (default from curriculum).
        train_steps: Gradient steps per iteration (default 200).
        device: 'cuda', 'mps', or 'cpu' (auto-detected if None).
        resume: Resume from latest checkpoint (default True).
        config: Network config: 'standard', 'large', 'orca-transformer' (default 'standard').

    Example:
        from orca import train_orca
        train_orca(iterations=50, games_per_iter=20)
    """
    return Orca.train(**kwargs)
