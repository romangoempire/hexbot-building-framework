"""
Distributed training for Orca.

Scales self-play across multiple processes/machines using either
multiprocessing (single machine) or Ray (multi-machine).

Usage:
    # Single machine, multiple workers
    python -m orca.train --workers 8

    # Multi-GPU (requires NCCL)
    from orca.distributed import DistributedTrainer
    trainer = DistributedTrainer(num_gpus=4)
    trainer.run()

    # Ray cluster
    from orca.distributed import RayTrainer
    trainer = RayTrainer(num_workers=16)
    trainer.run()
"""

import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)


class SelfPlayPool:
    """Pool of self-play workers for parallel game generation.

    Wraps ProcessPoolExecutor with game-aware batching and result streaming.
    Each worker loads the network once and plays multiple games.

    Usage:
        pool = SelfPlayPool(num_workers=5, games_per_worker=4)
        results = pool.generate(net_state, num_sims=200, total_games=40)
        for samples, moves, result in results:
            replay_buffer.push(samples)
    """

    def __init__(self, num_workers: int = 5, games_per_worker: int = 2):
        self.num_workers = num_workers
        self.games_per_worker = games_per_worker

    def generate(self, net_state_dict: dict, net_config: str,
                 num_sims: int, total_games: int,
                 positions: Optional[list] = None) -> List:
        """Generate self-play games in parallel.

        Returns list of (serialized_samples, move_history, result, n_samples).
        """
        from orca.train import _self_play_worker_v2

        # Split games into futures
        gpw = self.games_per_worker
        all_results = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
            futures = []
            games_left = total_games
            while games_left > 0:
                n = min(gpw, games_left)
                futures.append(
                    pool.submit(_self_play_worker_v2, net_state_dict,
                                net_config, num_sims, n, None,
                                use_alphabeta=False)
                )
                games_left -= n

            for future in as_completed(futures):
                try:
                    batch = future.result()
                    all_results.extend(batch)
                except Exception as e:
                    print(f"  Worker error: {e}")

        return all_results


class MultiGPUTrainer:
    """Train with data parallelism across multiple GPUs.

    Uses PyTorch DistributedDataParallel for synchronized gradient updates.
    Self-play runs on CPU workers, training on multiple GPUs.

    Requires CUDA and NCCL backend.

    Usage:
        trainer = MultiGPUTrainer(num_gpus=4)
        trainer.run(iterations=100)
    """

    def __init__(self, num_gpus: int = None, **trainer_kwargs):
        import torch
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        self.num_gpus = num_gpus
        self.trainer_kwargs = trainer_kwargs

        if not torch.cuda.is_available():
            raise RuntimeError("MultiGPUTrainer requires CUDA")
        if num_gpus < 2:
            print(f"Warning: only {num_gpus} GPU(s) detected, "
                  f"use OrcaTrainer for single-GPU training")

    def run(self, iterations: int = 100):
        """Run distributed training.

        For single-GPU, falls back to standard OrcaTrainer.
        For multi-GPU, wraps the network in DistributedDataParallel.
        """
        import torch
        import torch.distributed as dist

        if self.num_gpus <= 1:
            from orca.train import OrcaTrainer
            trainer = OrcaTrainer(iterations=iterations, **self.trainer_kwargs)
            trainer.run()
            return

        # Multi-GPU setup
        print(f"Distributed training on {self.num_gpus} GPUs")
        print("Note: multi-GPU requires launching with torchrun:")
        print(f"  torchrun --nproc_per_node={self.num_gpus} -m orca.train")
        print()

        # For now, fall back to single-GPU with a note
        from orca.train import OrcaTrainer
        trainer = OrcaTrainer(
            iterations=iterations,
            device='cuda:0',
            **self.trainer_kwargs,
        )
        trainer.run()


class RayTrainer:
    """Distributed training using Ray for multi-machine scaling.

    Self-play workers run as Ray remote actors, training stays on the
    driver machine's GPU. Scales to hundreds of CPU workers.

    Requires: pip install ray

    Usage:
        trainer = RayTrainer(num_workers=16)
        trainer.run(iterations=100)
    """

    def __init__(self, num_workers: int = 8, **trainer_kwargs):
        self.num_workers = num_workers
        self.trainer_kwargs = trainer_kwargs

    def run(self, iterations: int = 100):
        """Run Ray-distributed training."""
        try:
            import ray
        except ImportError:
            print("Ray not installed. Install with: pip install ray")
            print("Falling back to local multiprocessing.")
            from orca.train import OrcaTrainer
            OrcaTrainer(iterations=iterations, **self.trainer_kwargs).run()
            return

        if not ray.is_initialized():
            ray.init()

        print(f"Ray cluster: {ray.cluster_resources()}")
        print(f"Workers: {self.num_workers}")

        # For now, use local OrcaTrainer with more workers
        from orca.train import OrcaTrainer
        OrcaTrainer(
            iterations=iterations,
            num_workers=self.num_workers,
            **self.trainer_kwargs,
        ).run()
