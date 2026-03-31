"""
Ensemble evaluation from multiple checkpoints or architectures.

Averages predictions from N networks for stronger, more stable play.
Uncertainty estimation from disagreement can guide exploration.

Usage:
    from orca.ensemble import Ensemble

    ensemble = Ensemble.from_checkpoints(['ckpt_50.pt', 'ckpt_55.pt', 'ckpt_60.pt'])
    policy, value, uncertainty = ensemble.evaluate(game)

    # Or auto-load last N checkpoints
    ensemble = Ensemble.from_latest(n=5)
"""

import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)


class Ensemble:
    """Ensemble of multiple neural networks for evaluation.

    Averages policy and value predictions from multiple networks.
    Disagreement between networks indicates position uncertainty.
    """

    def __init__(self, nets: List = None, device=None):
        from bot import get_device
        self.device = device or get_device()
        self.nets = nets or []
        for net in self.nets:
            net.to(self.device)
            net.eval()

    @classmethod
    def from_checkpoints(cls, paths: List[str], net_config: str = 'standard',
                         device=None):
        """Load ensemble from checkpoint files.

        Example:
            ensemble = Ensemble.from_checkpoints([
                'hex_checkpoint_50.pt',
                'hex_checkpoint_55.pt',
                'hex_checkpoint_60.pt',
            ])
        """
        from bot import (create_network, migrate_checkpoint_5to7,
                         migrate_checkpoint_filters, get_device)
        dev = device or get_device()
        nets = []
        for path in paths:
            if not os.path.exists(path):
                continue
            net = create_network(net_config)
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            sd = ckpt.get('model_state_dict', ckpt)
            sd = migrate_checkpoint_5to7(sd)
            sd = migrate_checkpoint_filters(sd)
            net.load_state_dict(sd, strict=False)
            nets.append(net)
        if not nets:
            raise FileNotFoundError(f"No valid checkpoints in {paths}")
        return cls(nets=nets, device=dev)

    @classmethod
    def from_latest(cls, n: int = 5, pattern: str = 'hex_checkpoint_*.pt',
                    net_config: str = 'standard', device=None):
        """Auto-load the last N checkpoints.

        Example:
            ensemble = Ensemble.from_latest(n=3)
        """
        ckpts = sorted(glob.glob(pattern))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints matching {pattern}")
        paths = ckpts[-n:]
        return cls.from_checkpoints(paths, net_config=net_config, device=device)

    def evaluate(self, game) -> Tuple[Dict, float, float]:
        """Evaluate a position with the ensemble.

        Returns (policy_dict, mean_value, uncertainty).
        Uncertainty is the standard deviation of value predictions across
        networks. High uncertainty = position is complex/ambiguous.

        Example:
            policy, value, uncertainty = ensemble.evaluate(game)
            if uncertainty > 0.3:
                print("Position is ambiguous - explore more")
        """
        from hexbot import encode_state, decode_policy

        tensor, oq, orr = encode_state(game)
        batch = tensor.unsqueeze(0).to(self.device)

        all_policies = []
        all_values = []

        with torch.no_grad():
            for net in self.nets:
                p_logits, v = net.forward_pv(batch)
                policy = decode_policy(p_logits[0].cpu(), game, oq, orr)
                all_policies.append(policy)
                all_values.append(v.item())

        # Average policy
        all_moves = set()
        for p in all_policies:
            all_moves.update(p.keys())

        avg_policy = {}
        for m in all_moves:
            avg_policy[m] = sum(p.get(m, 0.0) for p in all_policies) / len(self.nets)

        # Normalize
        total = sum(avg_policy.values())
        if total > 0:
            avg_policy = {m: p / total for m, p in avg_policy.items()}

        mean_value = sum(all_values) / len(all_values)
        uncertainty = float(np.std(all_values)) if len(all_values) > 1 else 0.0

        return avg_policy, mean_value, uncertainty

    def best_move(self, game) -> Tuple[int, int]:
        """Get the ensemble's best move.

        Example:
            move = ensemble.best_move(game)
            game.place(*move)
        """
        policy, _, _ = self.evaluate(game)
        return max(policy, key=policy.get)

    def __len__(self):
        return len(self.nets)

    def __repr__(self):
        params = sum(p.numel() for p in self.nets[0].parameters()) if self.nets else 0
        return f'Ensemble({len(self.nets)} nets, {params:,} params each)'
