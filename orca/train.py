"""
Orca Training Pipeline -- standalone AlphaZero-style self-play trainer.

Extracted from train_dashboard.py: pure training logic with no Flask,
no SocketIO, no HTML. Works as both a library and a CLI.

Library usage:
    from orca.train import OrcaTrainer
    trainer = OrcaTrainer(iterations=100, games_per_iter=30)
    trainer.run()

CLI usage:
    python -m orca.train --iterations 100 --games-per-iter 30
"""

from __future__ import annotations

import argparse
import glob
import math
import multiprocessing
import os
import pickle
import random
import signal
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from bot import (
    HexNet, MCTS, ReplayBuffer, self_play_game, train_step,
    get_device, BOARD_SIZE, NUM_SIMULATIONS, BATCH_SIZE, LEARNING_RATE, L2_REG,
    TrainingSample, OnnxPredictor, export_onnx,
    POSITION_CATALOG, generate_puzzles, augment_sample,
    load_human_games, load_online_games, find_forced_move,
    encode_state, create_network, BatchedMCTS, BatchedNNAlphaBeta,
    self_play_game_v2, CGameState,
    migrate_checkpoint_5to7, migrate_checkpoint_filters,
    TrainingObserver,
)
from main import HexGame


# ---------------------------------------------------------------------------
# Timestamped print helper
# ---------------------------------------------------------------------------

_builtin_print = print

def _ts_print(*args, **kwargs):
    """Print with HH:MM:SS timestamp prefix."""
    ts = datetime.now().strftime("%H:%M:%S")
    _builtin_print(f"[{ts}]", *args, **kwargs)

# Module-level alias used throughout this file
print = _ts_print


# ---------------------------------------------------------------------------
# PrintObserver -- stdout-based TrainingObserver
# ---------------------------------------------------------------------------

class PrintObserver:
    """Logs training events to stdout with timestamps. Implements TrainingObserver."""

    def __init__(self):
        self._stop = threading.Event()

    def on_iteration_start(self, iteration: int, total: int) -> None:
        pass  # Logged inline by the trainer

    def on_game_complete(self, game_idx: int, total_games: int,
                         move_history: list, result: float,
                         num_samples: int) -> None:
        winner = "P0" if result > 0 else ("P1" if result < 0 else "draw")
        print(f"  |  Game {game_idx}/{total_games}: {winner} "
              f"in {len(move_history)} moves ({num_samples} samples)")

    def on_iteration_complete(self, metrics: dict) -> None:
        loss = metrics.get("loss", {})
        print(f"  |  Iteration {metrics.get('iteration', '?')} complete: "
              f"loss={loss.get('total', 0):.4f} "
              f"elo={metrics.get('elo', 0):.0f} "
              f"games={metrics.get('games', 0)} "
              f"samples={metrics.get('samples', 0)}")

    def on_training_complete(self) -> None:
        print("Training complete.")

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def request_stop(self):
        self._stop.set()

    def reset_stop(self):
        self._stop.clear()


# ---------------------------------------------------------------------------
# Curriculum: adaptive MCTS sim count and game count by iteration
# ---------------------------------------------------------------------------

_curriculum_start_time: Optional[float] = None
_curriculum_last_elo: Optional[float] = None
_curriculum_stall_iters: int = 0


def get_curriculum_sims(iteration: int) -> int:
    """Adaptive sim curriculum: start low, scale up over hours.

    - First 10 iters:  50 sims  (fast exploration, many games)
    - Iter 10-30:     100 sims  (better quality moves)
    - Iter 30-60:     150 sims  (deeper search)
    - Iter 60+:       200 sims  (full depth)

    Also boosts sims when ELO stalls (plateau detection).
    """
    global _curriculum_start_time
    if _curriculum_start_time is None:
        _curriculum_start_time = time.perf_counter()

    hours_elapsed = (time.perf_counter() - _curriculum_start_time) / 3600

    if hours_elapsed < 0.5:
        base_sims = 50
    elif hours_elapsed < 1.5:
        base_sims = 100
    elif hours_elapsed < 3.0:
        base_sims = 150
    else:
        base_sims = 200

    if iteration < 10:
        iter_sims = 50
    elif iteration < 30:
        iter_sims = 100
    elif iteration < 60:
        iter_sims = 150
    else:
        iter_sims = 200

    sims = max(base_sims, iter_sims)
    if _curriculum_stall_iters >= 10:
        sims = min(400, sims + 50)
    return sims


def update_curriculum_plateau(current_elo: float) -> None:
    """Track ELO plateaus to trigger curriculum boosts."""
    global _curriculum_last_elo, _curriculum_stall_iters
    if _curriculum_last_elo is not None:
        if abs(current_elo - _curriculum_last_elo) < 15:
            _curriculum_stall_iters += 1
        else:
            _curriculum_stall_iters = 0
    _curriculum_last_elo = current_elo


def get_curriculum_games(iteration: int, base: int) -> int:
    """More games when sims are low, fewer when sims are high."""
    sims = get_curriculum_sims(iteration)
    if sims <= 50:
        return 60
    if sims <= 100:
        return 50
    if sims <= 150:
        return 40
    return 30


# ---------------------------------------------------------------------------
# Self-play workers (run in subprocesses)
# ---------------------------------------------------------------------------

def _self_play_worker_v2(net_state_dict: dict, net_config: str, num_sims: int,
                         games: int, positions: Optional[list] = None,
                         use_alphabeta: bool = True, ab_depth: int = 8) -> list:
    """V2 worker: CGameState + NNAlphaBeta or BatchedMCTS."""
    try:
        from bot import (CGameState, BatchedMCTS, BatchedNNAlphaBeta,
                         self_play_game_v2, create_network,
                         migrate_checkpoint_5to7, migrate_checkpoint_filters)
    except ImportError:
        return []

    import time as _time, os as _os

    t_load = _time.perf_counter()
    net = create_network(net_config)
    migrated = migrate_checkpoint_5to7(dict(net_state_dict))
    migrated = migrate_checkpoint_filters(migrated)
    net.load_state_dict(migrated, strict=False)
    net.eval()
    t_load = _time.perf_counter() - t_load

    if use_alphabeta:
        searcher = BatchedNNAlphaBeta(net, depth=ab_depth, nn_depth=5)
    else:
        searcher = BatchedMCTS(net, num_simulations=num_sims, batch_size=32)

    pid = _os.getpid()
    _builtin_print(f"  |  [Worker {pid}] loaded in {t_load:.1f}s, "
                   f"playing {games} games ({num_sims} sims)")

    results = []
    for i in range(games):
        t_game = _time.perf_counter()
        pos = None
        hints = None
        if positions and i < len(positions):
            entry = positions[i]
            if isinstance(entry, tuple) and len(entry) == 2:
                pos, hints = entry
            elif isinstance(entry, dict):
                pos = entry
        samples, move_history = self_play_game_v2(
            net, searcher, start_position=pos, hint_moves=hints)
        t_game = _time.perf_counter() - t_game
        winner = "P0" if (samples and samples[0].result > 0) else "P1"
        _builtin_print(f"  |  [W{pid}] game {i+1}/{games}: {winner} "
                       f"{len(move_history)}mv {t_game:.1f}s "
                       f"({t_game / max(len(move_history), 1):.2f}s/mv)")
        serialized = []
        for s in samples:
            serialized.append({
                "state": s.encoded_state.numpy(),
                "policy": s.policy_target,
                "player": s.player,
                "result": s.result,
                "threat": s.threat_label,
                "priority": s.priority,
            })
        result_val = samples[0].result if samples else 0.0
        results.append((serialized, [list(m) for m in move_history],
                        result_val, len(samples)))
    return results


def _self_play_worker_v1(onnx_path: str, num_sims: int,
                         games: int,
                         positions: Optional[list] = None) -> list:
    """V1 worker: ONNX Runtime for CPU inference."""
    predictor = OnnxPredictor(onnx_path)
    mcts = MCTS(predictor, num_simulations=num_sims)

    results = []
    for i in range(games):
        pos = None
        hints = None
        if positions and i < len(positions):
            entry = positions[i]
            if isinstance(entry, tuple) and len(entry) == 2:
                pos, hints = entry
            elif isinstance(entry, dict):
                pos = entry
        samples, move_history = self_play_game(
            predictor, mcts, start_position=pos, hint_moves=hints)
        serialized = []
        for s in samples:
            serialized.append({
                "state": s.encoded_state.numpy(),
                "policy": s.policy_target,
                "player": s.player,
                "result": s.result,
                "threat": s.threat_label,
                "priority": s.priority,
            })
        result_val = samples[0].result if samples else 0.0
        results.append((serialized, [list(m) for m in move_history],
                        result_val, len(samples)))
    return results


# ---------------------------------------------------------------------------
# ModelVault -- compressed weight storage for past generations
# ---------------------------------------------------------------------------

class ModelVault:
    """Stores compressed (fp16) weights for every evaluated generation."""

    def __init__(self, max_models: int = 200):
        self.models: list = []  # [(iteration, state_dict_cpu_fp16), ...]
        self.max_models = max_models

    def add(self, iteration: int, state_dict: dict):
        compressed = {k: v.detach().cpu().half() for k, v in state_dict.items()}
        self.models.append((iteration, compressed))
        if len(self.models) > self.max_models:
            n = len(self.models)
            keep = {0, n - 1}
            keep.update(range(max(0, n - 20), n))
            step = max(1, n // 50)
            keep.update(range(0, n, step))
            self.models = [self.models[i] for i in sorted(keep)]

    def get_net(self, idx: int, device) -> HexNet:
        """Load a stored model by index, return on device."""
        _, state = self.models[idx]
        state_fp32 = {k: v.float() for k, v in state.items()}
        init_w = state_fp32.get("conv_init.weight")
        if init_w is not None:
            nf = init_w.shape[0]
            nb = 0
            while f"res_blocks.{nb}.conv1.weight" in state_fp32:
                nb += 1
            net = HexNet(num_filters=nf, num_res_blocks=max(nb, 4))
        else:
            net = HexNet(num_filters=64, num_res_blocks=4)
        net.load_state_dict(state_fp32, strict=False)
        net.to(device)
        net.eval()
        return net

    def __len__(self):
        return len(self.models)


# ---------------------------------------------------------------------------
# GenerationalArena -- round-robin ELO evaluation
# ---------------------------------------------------------------------------

class GenerationalArena:
    """Evaluates current model via mini round-robin against past generations."""

    def __init__(self, device: torch.device, games_per_opponent: int = 4,
                 num_sims: int = 30, max_opponents: int = 6):
        self.device = device
        self.games_per_opponent = games_per_opponent
        self.num_sims = num_sims
        self.max_opponents = max_opponents
        self.matchup_history: list = []

    def evaluate(self, current_net: HexNet, vault: ModelVault,
                 current_elo: float) -> float:
        """Play mini round-robin against selected past generations."""
        current_net.eval()
        n = len(vault)
        if n == 0:
            return current_elo

        opponent_indices = self._select_opponents(n)
        mcts_cur = MCTS(current_net, num_simulations=self.num_sims)

        total_wins = total_losses = total_draws = 0
        matchups = {}

        for opp_idx in opponent_indices:
            opp_net = vault.get_net(opp_idx, self.device)
            mcts_opp = MCTS(opp_net, num_simulations=self.num_sims)
            opp_iter = vault.models[opp_idx][0]

            w = l = d = 0
            for g in range(self.games_per_opponent):
                if g % 2 == 0:
                    r = self._play(mcts_cur, mcts_opp)
                else:
                    r = -self._play(mcts_opp, mcts_cur)
                if r > 0:
                    w += 1
                elif r < 0:
                    l += 1
                else:
                    d += 1

            matchups[opp_iter] = {"w": w, "l": l, "d": d}
            total_wins += w
            total_losses += l
            total_draws += d
            del opp_net, mcts_opp

        self.matchup_history.append(matchups)

        total_games = total_wins + total_losses + total_draws
        if total_games == 0:
            return current_elo
        score = (total_wins + 0.5 * total_draws) / total_games
        new_elo = current_elo + 16 * (score - 0.5) * len(opponent_indices)
        return new_elo

    def _select_opponents(self, n: int) -> list:
        if n <= self.max_opponents:
            return list(range(n))
        selected = {0, n - 1}
        step = max(1, n // (self.max_opponents - 2))
        for i in range(1, self.max_opponents - 1):
            selected.add(min(i * step, n - 1))
        return sorted(selected)

    def _play(self, mcts_p0: MCTS, mcts_p1: MCTS) -> float:
        game = HexGame(candidate_radius=2, max_total_stones=200)
        while not game.is_terminal:
            mcts = mcts_p0 if game.current_player == 0 else mcts_p1
            policy = mcts.search(game, temperature=0.1, add_noise=False)
            if not policy:
                break
            best = max(policy, key=policy.get)
            game.place_stone(*best)
        return game.result()


# ---------------------------------------------------------------------------
# AutoTuner -- self-improving hyperparameter controller
# ---------------------------------------------------------------------------

class AutoTuner:
    """Observes metrics after each iteration and adjusts hyperparams.
    Rule-based + trend detection. No external ML needed."""

    def __init__(self):
        self.loss_history: list = []
        self.elo_history: list = []
        self.decisions: list = []

        self.params = {
            "lr": 0.002,
            "sims": 10,
            "mix_normal": 1.00,
            "mix_catalog": 0.00,
            "mix_endgame": 0.00,
            "mix_formation": 0.00,
            "mix_sequence": 0.00,
            "hint_blend": 0.3,
            "temp_threshold": 20,
            "train_steps": 200,
        }
        self.param_history: list = []

    def _log(self, iteration: int, msg: str):
        self.decisions.append((iteration, msg))
        print(f"  |  AutoTuner: {msg}")

    def update(self, metrics: dict, iteration: int) -> dict:
        """Observe metrics, return updated params for next iteration."""
        p = self.params.copy()

        total_loss = metrics.get("total_loss", 0)
        self.loss_history.append(total_loss)

        elo = metrics.get("elo")
        if elo is not None:
            self.elo_history.append(elo)

        changes = []

        # MCTS sims: capped at 50 for training
        if p["sims"] > 50:
            p["sims"] = 50
            changes.append("sims->50 (training cap)")

        # Game mix: pure self-play (locked)
        p["mix_normal"] = 1.0
        p["mix_endgame"] = 0.0
        p["mix_catalog"] = 0.0
        p["mix_formation"] = 0.0
        p["mix_sequence"] = 0.0

        # Hint blend: decay over time
        p["hint_blend"] = max(0.0, 0.3 - iteration * 0.015)

        # Train steps: increase if loss decreasing and buffer full
        buf_fill = metrics.get("buffer_fill", 0)
        loss_decreasing = (len(self.loss_history) >= 3 and
                           self.loss_history[-1] < self.loss_history[-3] * 0.95)
        if buf_fill > 0.9 and loss_decreasing and p["train_steps"] < 600:
            p["train_steps"] = min(600, p["train_steps"] + 50)
            changes.append(f"train_steps->{p['train_steps']}")

        if changes:
            self._log(iteration, " | ".join(changes))
        else:
            self._log(iteration, "no changes")

        # Normalize mix ratios
        total_mix = sum(p[k] for k in [
            "mix_normal", "mix_catalog", "mix_endgame",
            "mix_formation", "mix_sequence"])
        if abs(total_mix - 1.0) > 0.01:
            for k in ["mix_normal", "mix_catalog", "mix_endgame",
                       "mix_formation", "mix_sequence"]:
                p[k] /= total_mix

        self.params = p
        self.param_history.append(p.copy())
        return p


# ---------------------------------------------------------------------------
# OrcaTrainer -- the core training loop
# ---------------------------------------------------------------------------

class OrcaTrainer:
    """Standalone AlphaZero-style training pipeline.

    Every parameter can be overridden via constructor kwargs, CLI args,
    or by editing orca/config.py. Parameters default to None which means
    "use the value from orca/config.py".

    Example:
        # Use defaults from config.py
        trainer = OrcaTrainer(iterations=100)

        # Override specific params
        trainer = OrcaTrainer(
            iterations=50,
            lr=0.002,
            batch_size=512,
            mcts_sims=100,
            net_config='orca-transformer',
        )

        trainer.run()
    """

    def __init__(
        self,
        # Pipeline
        iterations: int = 999_999,
        games_per_iter: Optional[int] = None,
        train_steps: Optional[int] = None,
        resume: bool = True,
        device: Optional[str] = None,
        num_workers: Optional[int] = None,
        # Network
        net_config: str = "standard",
        # Optimizer
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        # LR scheduler
        scheduler_T0: Optional[int] = None,
        scheduler_Tmult: Optional[int] = None,
        scheduler_eta_min: Optional[float] = None,
        # Search
        mcts_sims: Optional[int] = None,
        mcts_batch_size: Optional[int] = None,
        # Replay buffer
        buffer_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        # ELO evaluation
        elo_every: Optional[int] = None,
        elo_games: Optional[int] = None,
        # Observer
        observer: Optional[TrainingObserver] = None,
        # Vault
        vault_size: int = 200,
        # Feature toggles
        use_curriculum: bool = True,
        use_auto_tuner: bool = True,
        use_adaptive_lr: bool = True,
        use_augmentation: bool = True,
        use_mixed_precision: bool = True,
        grad_clip: float = 1.0,
        # Plateau detection
        plateau_threshold: Optional[float] = None,
        plateau_iters: Optional[int] = None,
        plateau_sim_boost: Optional[int] = None,
        # Arena
        elo_sims: Optional[int] = None,
        elo_max_opponents: Optional[int] = None,
    ):
        from orca.config import (
            DEFAULT_GAMES_PER_ITER, DEFAULT_TRAIN_STEPS,
            LEARNING_RATE as CFG_LR, L2_REG as CFG_L2,
            COSINE_T0, COSINE_T_MULT, COSINE_ETA_MIN,
            NUM_SIMULATIONS as CFG_SIMS, MCTS_BATCH_SIZE as CFG_MCTS_BS,
            REPLAY_BUFFER_SIZE as CFG_BUF, BATCH_SIZE as CFG_BATCH,
            ELO_EVAL_EVERY, ELO_EVAL_GAMES, ELO_EVAL_SIMS, ELO_MAX_OPPONENTS,
            MAX_WORKERS, VAULT_MAX_MODELS,
            PLATEAU_THRESHOLD, PLATEAU_ITERS, PLATEAU_SIM_BOOST,
        )

        # Pipeline
        self.num_iterations = iterations
        self.games_per_iter = games_per_iter or DEFAULT_GAMES_PER_ITER
        self.train_steps = train_steps or DEFAULT_TRAIN_STEPS
        self.resume = resume
        self.net_config = net_config

        # Optimizer
        self.lr = lr or CFG_LR
        self.weight_decay = weight_decay or CFG_L2

        # Scheduler
        self.scheduler_T0 = scheduler_T0 or COSINE_T0
        self.scheduler_Tmult = scheduler_Tmult or COSINE_T_MULT
        self.scheduler_eta_min = scheduler_eta_min or COSINE_ETA_MIN

        # Search
        self.mcts_sims = mcts_sims or CFG_SIMS
        self.mcts_batch_size = mcts_batch_size or CFG_MCTS_BS

        # Replay
        self.buffer_size = buffer_size or CFG_BUF
        self.batch_size = batch_size or CFG_BATCH

        # ELO
        self.elo_every = elo_every or ELO_EVAL_EVERY
        self.elo_games = elo_games or ELO_EVAL_GAMES
        self.elo_sims = elo_sims or ELO_EVAL_SIMS
        self.elo_max_opponents = elo_max_opponents or ELO_MAX_OPPONENTS

        # Plateau
        self.plateau_threshold = plateau_threshold or PLATEAU_THRESHOLD
        self.plateau_iters = plateau_iters or PLATEAU_ITERS
        self.plateau_sim_boost = plateau_sim_boost or PLATEAU_SIM_BOOST

        # Device + workers
        self.device = torch.device(device) if device else get_device()
        self.observer = observer or PrintObserver()
        cpu_count = multiprocessing.cpu_count()
        self.num_workers = num_workers or min(MAX_WORKERS, max(2, cpu_count - 2))

        # Feature toggles
        self.use_curriculum = use_curriculum
        self.use_auto_tuner = use_auto_tuner
        self.use_adaptive_lr = use_adaptive_lr
        self.use_augmentation = use_augmentation
        self.use_mixed_precision = use_mixed_precision and self.device.type == 'cuda'
        self.grad_clip = grad_clip

        # Initialized in run()
        self.net: Optional[HexNet] = None
        self.model_vault = ModelVault(max_models=vault_size or VAULT_MAX_MODELS)
        self.arena = GenerationalArena(self.device)

        # Simple metrics dict (replaces MetricsStore)
        self.metrics: Dict = {
            "iterations": [],
            "elo_history": [{"iteration": 0, "elo": 1000.0}],
            "current_elo": 1000.0,
            "total_games": 0,
            "is_training": False,
        }

    # -- Checkpoint discovery ------------------------------------------------

    @staticmethod
    def _find_latest_checkpoint() -> Optional[str]:
        ckpts = glob.glob("hex_checkpoint_*.pt")
        if not ckpts:
            return None
        def _iter_num(path):
            try:
                return int(path.replace("hex_checkpoint_", "").replace(".pt", ""))
            except ValueError:
                return -1
        ckpts.sort(key=_iter_num)
        return ckpts[-1]

    # -- Main training loop --------------------------------------------------

    def run(self):
        """Execute the full training pipeline. Blocks until done or stopped."""
        self.metrics["is_training"] = True

        # Build network
        self.net = create_network(self.net_config).to(self.device)
        param_count = sum(p.numel() for p in self.net.parameters())

        print(f"\n{'=' * 60}")
        print(f"  ORCA TRAINING PIPELINE")
        print(f"{'=' * 60}")
        print(f"  Network:     {self.net_config} ({param_count:,} params)")
        print(f"  Device:      {self.device}")
        print(f"  Iterations:  {self.num_iterations}")
        print(f"  Games/iter:  {self.games_per_iter} (base, scaled by curriculum)")
        print(f"  Train steps: {self.train_steps}/iter")

        # Detect C engine
        use_v2 = False
        try:
            CGameState()
            use_v2 = True
            print(f"  Engine:      C engine (v2)")
        except Exception as e:
            print(f"  Engine:      Python (v1) -- C engine unavailable: {e}")

        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=0.001, weight_decay=L2_REG)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-4)
        replay_buffer = ReplayBuffer()
        auto_tuner = AutoTuner()

        # -- Resume from checkpoint ------------------------------------------
        start_iteration = 0
        if self.resume:
            start_iteration = self._restore_checkpoint(
                optimizer, scheduler, replay_buffer, auto_tuner)
        else:
            print(f"  Fresh start (--fresh)")

        print(f"  Workers:     {self.num_workers}")
        print(f"  LR:          {optimizer.param_groups[0]['lr']:.5f}")
        print(f"{'=' * 60}\n")

        # -- Iteration loop --------------------------------------------------
        for iteration in range(start_iteration, self.num_iterations):
            if self.observer.should_stop():
                print(f"\n  Training stopped at iteration {iteration}")
                break
            self.observer.on_iteration_start(iteration, self.num_iterations)

            t0 = time.perf_counter()

            # Curriculum
            current_sims = max(auto_tuner.params["sims"],
                               get_curriculum_sims(iteration))
            current_games = get_curriculum_games(iteration, self.games_per_iter)
            current_lr = optimizer.param_groups[0]["lr"]

            hours = ((time.perf_counter() - _curriculum_start_time) / 3600
                     if _curriculum_start_time else 0)
            print(f"  +-- Iter {iteration + 1}/{self.num_iterations} "
                  f"| sims={current_sims} | games={current_games} "
                  f"| lr={current_lr:.4f} | {hours:.1f}h")

            # Load human games on first iteration
            if iteration == 0:
                self._load_human_games(replay_buffer)

            # Export ONNX for workers
            self.net.eval()
            onnx_path = f"/tmp/hex_model_{iteration}.onnx"
            t_exp = time.perf_counter()
            export_onnx(self.net, onnx_path)
            self.net.to(self.device)
            print(f"  |  ONNX export: {time.perf_counter() - t_exp:.1f}s")

            # Build position mix
            all_positions = self._build_position_mix(
                current_games, auto_tuner.params)

            # -- Self-play ---------------------------------------------------
            t_sp = time.perf_counter()
            sp_result = self._run_self_play(
                use_v2, current_sims, current_games, all_positions,
                onnx_path, replay_buffer)
            t_selfplay = time.perf_counter() - t_sp

            game_idx = sp_result["game_idx"]
            total_samples = sp_result["total_samples"]
            total_moves = sp_result["total_moves"]
            wins = sp_result["wins"]
            collected_samples = sp_result["collected_samples"]

            gps = game_idx / t_selfplay if t_selfplay > 0 else 0
            print(f"  |  Self-play done: {game_idx} games, "
                  f"{total_samples} samples, {t_selfplay:.1f}s "
                  f"({gps:.1f} games/s)")
            print(f"  |  Wins: P0={wins[0]} P1={wins[1]} draw={wins[2]}")

            # Load online games
            self._load_online_games(replay_buffer, collected_samples)

            # Symmetry augmentation
            t_aug = time.perf_counter()
            aug_count = 0
            for sample in collected_samples:
                for aug in augment_sample(sample):
                    replay_buffer.push(aug)
                    aug_count += 1
            total_samples += aug_count
            print(f"  |  Augmentation: +{aug_count} samples "
                  f"({time.perf_counter() - t_aug:.1f}s) "
                  f"-> buffer={len(replay_buffer)}/{replay_buffer.buffer.maxlen}")

            sp_time = time.perf_counter() - t0

            # -- GPU training ------------------------------------------------
            losses = {"total": 0, "value": 0, "policy": 0}
            train_time = 0
            actual_steps = auto_tuner.params.get("train_steps", self.train_steps)
            if len(replay_buffer) >= BATCH_SIZE:
                print(f"  |  Training: {actual_steps} steps on {self.device} "
                      f"(batch={BATCH_SIZE}, buffer={len(replay_buffer)})...")
                t1 = time.perf_counter()
                self.net.train()
                for step in range(actual_steps):
                    losses = train_step(
                        self.net, optimizer, replay_buffer, self.device)
                train_time = time.perf_counter() - t1
                sps = actual_steps / train_time if train_time > 0 else 0
                scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]
                auto_tuner.params["lr"] = current_lr
                print(f"  |  Training done: {train_time:.1f}s ({sps:.0f} steps/s) "
                      f"loss={losses['total']:.4f} "
                      f"(v={losses['value']:.4f} p={losses['policy']:.4f}) "
                      f"lr={current_lr:.6f}")
            else:
                print(f"  |  Training skipped: buffer too small "
                      f"({len(replay_buffer)}<{BATCH_SIZE})")

            # -- ELO evaluation (every 2 iterations) -------------------------
            elo_str = ""
            if len(replay_buffer) >= BATCH_SIZE and (iteration + 1) % 2 == 0:
                self.model_vault.add(iteration + 1, self.net.state_dict())
                n_opp = min(self.arena.max_opponents,
                            len(self.model_vault) - 1)
                n_eval_games = n_opp * self.arena.games_per_opponent
                print(f"  |  ELO evaluation ({n_eval_games} games vs "
                      f"{n_opp} generations, vault={len(self.model_vault)})...")
                t_elo = time.perf_counter()
                new_elo = self.arena.evaluate(
                    self.net, self.model_vault, self.metrics["current_elo"])
                t_elo = time.perf_counter() - t_elo
                delta = new_elo - self.metrics["current_elo"]
                sign = "+" if delta >= 0 else ""
                self.metrics["current_elo"] = new_elo
                self.metrics["elo_history"].append(
                    {"iteration": iteration + 1, "elo": round(new_elo, 1)})
                update_curriculum_plateau(new_elo)
                elo_str = f" | ELO {new_elo:.0f} ({sign}{delta:.0f})"
                stall = (f" stall={_curriculum_stall_iters}"
                         if _curriculum_stall_iters > 0 else "")
                print(f"  |  ELO: {new_elo:.0f} ({sign}{delta:.0f}) "
                      f"in {t_elo:.1f}s{stall}")

            # -- Metrics & AutoTuner -----------------------------------------
            avg_len = total_moves / max(game_idx, 1)
            total_time = time.perf_counter() - t0
            iter_metrics = {
                "iteration": iteration + 1,
                "games": game_idx,
                "samples": total_samples,
                "wins": wins,
                "self_play_time": round(sp_time, 1),
                "train_time": round(train_time, 1),
                "loss": losses,
                "buffer_size": len(replay_buffer),
                "avg_game_length": round(avg_len, 1),
                "elo": round(self.metrics["current_elo"], 1),
                "workers": self.num_workers,
                "sims": current_sims,
                "lr": round(optimizer.param_groups[0]["lr"], 5),
            }
            self.metrics["iterations"].append(iter_metrics)
            self.metrics["total_games"] = self.metrics.get("total_games", 0) + game_idx
            self.observer.on_iteration_complete(iter_metrics)

            at_metrics = {
                "total_loss": losses.get("total", 0),
                "value_loss": losses.get("value", 0),
                "policy_loss": losses.get("policy", 0),
                "elo": self.metrics["current_elo"],
                "games_per_sec": gps,
                "buffer_fill": len(replay_buffer) / replay_buffer.buffer.maxlen,
            }
            new_params = auto_tuner.update(at_metrics, iteration)
            for pg in optimizer.param_groups:
                pg["lr"] = new_params["lr"]

            print(f"  +-- Iter {iteration + 1} done: {total_time:.1f}s total "
                  f"| {game_idx} games | {total_samples} samples{elo_str}")
            print()

            # -- Checkpoint (every 5 iterations) -----------------------------
            if (iteration + 1) % 5 == 0:
                self._save_checkpoint(
                    iteration, optimizer, scheduler, replay_buffer, auto_tuner)

        # -- Done ------------------------------------------------------------
        print(f"\n{'=' * 60}")
        print(f"  TRAINING COMPLETE -- {self.num_iterations} iterations")
        print(f"  Final ELO: {self.metrics['current_elo']:.0f}")
        print(f"  Total games: {self.metrics['total_games']}")
        print(f"{'=' * 60}\n")
        self.metrics["is_training"] = False
        self.observer.on_training_complete()

    # -- Helpers -------------------------------------------------------------

    def _restore_checkpoint(self, optimizer, scheduler, replay_buffer,
                            auto_tuner) -> int:
        """Restore from latest checkpoint. Returns start iteration."""
        resume_path = self._find_latest_checkpoint()
        if not resume_path:
            print(f"  No checkpoint found -- starting fresh")
            return 0

        try:
            ckpt = torch.load(resume_path, map_location=self.device,
                              weights_only=False)
            old_shape = ckpt["model_state_dict"].get("conv_init.weight", None)
            migrated_sd = migrate_checkpoint_5to7(ckpt["model_state_dict"])
            migrated_sd = migrate_checkpoint_filters(migrated_sd)
            new_shape = migrated_sd.get("conv_init.weight", None)
            arch_changed = (old_shape is not None and new_shape is not None
                            and old_shape.shape != new_shape.shape)
            ckpt["model_state_dict"] = migrated_sd
            self.net.load_state_dict(migrated_sd, strict=False)

            if arch_changed:
                print(f"    Fresh optimizer (architecture migrated)")
            else:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])

            start_iteration = ckpt.get("iteration", 0) + 1

            # Reset frozen LR
            for pg in optimizer.param_groups:
                if pg["lr"] < 1e-4:
                    pg["lr"] = 0.001
                    print(f"    LR was frozen, reset to 0.001")

            if "scheduler_state_dict" in ckpt:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except Exception:
                    pass

            # Restore replay buffer
            buf_path = os.path.join(os.path.dirname(resume_path),
                                    "replay_buffer.pkl")
            if os.path.exists(buf_path):
                try:
                    with open(buf_path, "rb") as f:
                        buf_data = pickle.load(f)
                    replay_buffer.buffer.extend(buf_data.get("buffer", []))
                    replay_buffer.priorities.extend(buf_data.get("priorities", []))
                    print(f"    Restored replay buffer: "
                          f"{len(replay_buffer.buffer)} samples")
                except Exception as e:
                    print(f"    Could not restore buffer: {e}")

            # Restore metrics
            if "metrics" in ckpt:
                m = ckpt["metrics"]
                self.metrics["iterations"] = m.get("iterations", [])
                self.metrics["elo_history"] = m.get("elo_history", [])
                self.metrics["current_elo"] = m.get("current_elo", 1000)
                self.metrics["total_games"] = m.get("total_games", 0)

            # Restore AutoTuner
            if "auto_tuner" in ckpt:
                at = ckpt["auto_tuner"]
                auto_tuner.params = at.get("params", auto_tuner.params)
                auto_tuner.params["lr"] = optimizer.param_groups[0]["lr"]
                auto_tuner.params["mix_normal"] = 1.0
                auto_tuner.params["mix_catalog"] = 0.0
                auto_tuner.params["mix_endgame"] = 0.0
                auto_tuner.params["mix_formation"] = 0.0
                auto_tuner.params["mix_sequence"] = 0.0
                auto_tuner.loss_history = at.get("loss_history", [])
                auto_tuner.elo_history = at.get("elo_history", [])

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Resumed from {resume_path} (iter {start_iteration})")
            print(f"    ELO: {self.metrics['current_elo']:.0f}, "
                  f"Games: {self.metrics['total_games']}, "
                  f"LR: {current_lr:.5f}")
            return start_iteration

        except Exception as e:
            print(f"  Failed to resume: {e} -- starting fresh")
            return 0

    def _load_human_games(self, replay_buffer: ReplayBuffer):
        """Load human games into the replay buffer on first iteration."""
        human_path = os.path.join(os.path.dirname(__file__), "..",
                                  "human_games.jsonl")
        if os.path.exists(human_path):
            t0 = time.perf_counter()
            human_samples = load_human_games(human_path, max_games=500,
                                             min_elo=1000)
            for s in human_samples:
                s.priority = 0.8
                replay_buffer.push(s)
            print(f"  |  Human games: {len(human_samples)} samples loaded "
                  f"({time.perf_counter() - t0:.1f}s)")
        else:
            print(f"  |  Human games: not found")

    def _load_online_games(self, replay_buffer: ReplayBuffer,
                           collected_samples: list):
        """Load new online games (human feedback)."""
        online_path = os.path.join(os.path.dirname(__file__), "..",
                                   "online_games.jsonl")
        if not hasattr(self, "_online_lines_read"):
            self._online_lines_read = 0
        try:
            online_samples, new_pos = load_online_games(
                online_path, start_line=self._online_lines_read)
            if online_samples:
                for s in online_samples:
                    replay_buffer.push(s)
                    collected_samples.append(s)
                self._online_lines_read = new_pos
                print(f"  |  Online games: +{len(online_samples)} samples")
        except Exception:
            pass

    def _build_position_mix(self, current_games: int,
                            params: dict) -> list:
        """Build shuffled position list for self-play workers."""
        from bot import GUIDED_POSITIONS, get_guided_positions_by_level

        ap = params
        mix = (ap["mix_normal"], ap["mix_catalog"], ap["mix_endgame"],
               ap["mix_formation"], ap["mix_sequence"])
        n_normal = int(current_games * mix[0])
        n_catalog = int(current_games * mix[1])
        n_endgame = int(current_games * mix[2])
        n_formation = int(current_games * mix[3])
        n_sequence = current_games - n_normal - n_catalog - n_endgame - n_formation

        print(f"  |  Game mix: {n_normal} normal + {n_catalog} catalog + "
              f"{n_endgame} endgame + {n_formation} formation + "
              f"{n_sequence} sequence")

        catalog_keys = list(POSITION_CATALOG.keys())
        catalog_positions = [
            (POSITION_CATALOG[random.choice(catalog_keys)], None)
            for _ in range(n_catalog)
        ]

        l1 = get_guided_positions_by_level(1)
        l2 = get_guided_positions_by_level(2)
        l3 = get_guided_positions_by_level(3)

        endgame_positions = [random.choice(l1) if l1 else (None, None)
                             for _ in range(n_endgame)]
        formation_positions = [random.choice(l2) if l2 else (None, None)
                               for _ in range(n_formation)]
        sequence_positions = [random.choice(l3) if l3 else (None, None)
                              for _ in range(n_sequence)]

        puzzle_positions = generate_puzzles(n_endgame // 2)
        extra_puzzles = [(p, None) for p in puzzle_positions]

        all_positions = (
            catalog_positions + endgame_positions + formation_positions +
            sequence_positions + extra_puzzles[:n_endgame // 2] +
            [(None, None)] * n_normal
        )
        all_positions = all_positions[:current_games]
        while len(all_positions) < current_games:
            all_positions.append((None, None))
        random.shuffle(all_positions)
        return all_positions

    def _run_self_play(self, use_v2: bool, current_sims: int,
                       current_games: int, all_positions: list,
                       onnx_path: str,
                       replay_buffer: ReplayBuffer) -> dict:
        """Dispatch parallel self-play and collect results."""
        # Distribute positions across workers
        chunks = []
        chunk_positions = []
        remaining = current_games
        pos_offset = 0
        for i in range(self.num_workers):
            c = remaining // (self.num_workers - i)
            chunks.append(c)
            chunk_positions.append(all_positions[pos_offset:pos_offset + c])
            pos_offset += c
            remaining -= c

        active_chunks = [c for c in chunks if c > 0]
        print(f"  |  Dispatching {len(active_chunks)} workers: {active_chunks}")

        total_samples = 0
        total_moves = 0
        wins = [0, 0, 0]
        game_idx = 0
        collected_samples = []
        worker_errors = 0

        if use_v2:
            net_state = {k: v.cpu() for k, v in self.net.state_dict().items()}
            print(f"  |  Self-play: V2 (C engine + MCTS {current_sims} sims)...")

            with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
                futures = []
                GAMES_PER_FUTURE = 2
                flat_positions = []
                for c, pos in zip(chunks, chunk_positions):
                    for gi in range(c):
                        p = pos[gi] if pos and gi < len(pos) else None
                        flat_positions.append(p)
                for i in range(0, len(flat_positions), GAMES_PER_FUTURE):
                    batch_pos = flat_positions[i:i + GAMES_PER_FUTURE]
                    pos_arg = ([p for p in batch_pos]
                               if any(batch_pos) else None)
                    futures.append(
                        pool.submit(_self_play_worker_v2, net_state,
                                    self.net_config, current_sims,
                                    len(batch_pos), pos_arg,
                                    use_alphabeta=False))

                print(f"  |  {len(futures)} futures "
                      f"({GAMES_PER_FUTURE} games each)")

                for future in as_completed(futures):
                    if self.observer.should_stop():
                        break
                    try:
                        batch_results = future.result()
                    except Exception as e:
                        worker_errors += 1
                        print(f"  |  Worker error: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                    for ser, move_hist, result_val, n_samp in batch_results:
                        for sd in ser:
                            sample = TrainingSample(
                                encoded_state=torch.from_numpy(sd["state"]),
                                policy_target=sd["policy"],
                                player=sd["player"],
                                result=sd["result"],
                                threat_label=sd.get("threat"),
                                priority=sd.get("priority", 1.0),
                            )
                            replay_buffer.push(sample)
                            collected_samples.append(sample)
                        total_samples += n_samp
                        total_moves += len(move_hist)
                        if result_val > 0:
                            wins[0] += 1
                        elif result_val < 0:
                            wins[1] += 1
                        else:
                            wins[2] += 1
                        game_idx += 1
                        self.observer.on_game_complete(
                            game_idx, current_games, move_hist,
                            result_val, n_samp)
        else:
            print(f"  |  Self-play: V1 (ONNX Runtime)...")
            with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
                futures = [
                    pool.submit(_self_play_worker_v1, onnx_path,
                                current_sims, c, pos)
                    for c, pos in zip(chunks, chunk_positions) if c > 0
                ]
                for future in as_completed(futures):
                    if self.observer.should_stop():
                        break
                    try:
                        batch_results = future.result()
                    except Exception as e:
                        worker_errors += 1
                        print(f"  |  Worker error: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                    for ser, move_hist, result_val, n_samp in batch_results:
                        for sd in ser:
                            sample = TrainingSample(
                                encoded_state=torch.from_numpy(sd["state"]),
                                policy_target=sd["policy"],
                                player=sd["player"],
                                result=sd["result"],
                                threat_label=sd.get("threat"),
                                priority=sd.get("priority", 1.0),
                            )
                            replay_buffer.push(sample)
                            collected_samples.append(sample)
                        total_samples += n_samp
                        total_moves += len(move_hist)
                        if result_val > 0:
                            wins[0] += 1
                        elif result_val < 0:
                            wins[1] += 1
                        else:
                            wins[2] += 1
                        game_idx += 1
                        self.observer.on_game_complete(
                            game_idx, current_games, move_hist,
                            result_val, n_samp)

        if worker_errors > 0:
            print(f"  |  {worker_errors} worker(s) failed")

        return {
            "game_idx": game_idx,
            "total_samples": total_samples,
            "total_moves": total_moves,
            "wins": wins,
            "collected_samples": collected_samples,
        }

    def _save_checkpoint(self, iteration: int, optimizer, scheduler,
                         replay_buffer: ReplayBuffer, auto_tuner: AutoTuner):
        """Save model checkpoint and replay buffer."""
        ckpt_path = f"hex_checkpoint_{iteration + 1}.pt"
        torch.save({
            "iteration": iteration,
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": {
                "iterations": self.metrics["iterations"],
                "elo_history": self.metrics["elo_history"],
                "current_elo": self.metrics["current_elo"],
                "total_games": self.metrics["total_games"],
            },
            "auto_tuner": {
                "params": auto_tuner.params,
                "loss_history": auto_tuner.loss_history,
                "elo_history": auto_tuner.elo_history,
            },
        }, ckpt_path)

        try:
            buf_path = "replay_buffer.pkl"
            with open(buf_path, "wb") as f:
                pickle.dump({
                    "buffer": list(replay_buffer.buffer),
                    "priorities": list(replay_buffer.priorities),
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"  Buffer save failed: {e}")

        print(f"  Checkpoint saved: {ckpt_path} + buffer "
              f"({len(replay_buffer.buffer)} samples)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Orca Training Pipeline - AlphaZero-style self-play trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m orca.train                          # train with defaults
  python -m orca.train --fresh --iterations 50  # fresh start, 50 iters
  python -m orca.train --lr 0.002 --batch-size 512 --mcts-sims 100
  python -m orca.train --config orca-transformer --device cuda

All parameters default to values in orca/config.py. CLI args override config.
        """)

    # Pipeline
    g = parser.add_argument_group("pipeline")
    g.add_argument("--iterations", type=int, default=999999,
                   help="Training iterations (default: infinite)")
    g.add_argument("--games-per-iter", type=int, default=None,
                   help="Games per iteration (default: from curriculum)")
    g.add_argument("--train-steps", type=int, default=None,
                   help="Gradient steps per iteration (default: 200)")
    g.add_argument("--resume", action="store_true", default=True,
                   help="Resume from latest checkpoint (default)")
    g.add_argument("--fresh", action="store_true",
                   help="Start fresh, ignore existing checkpoints")
    g.add_argument("--workers", type=int, default=None,
                   help="Parallel self-play workers (default: auto)")

    # Network
    g = parser.add_argument_group("network")
    g.add_argument("--config", type=str, default="standard",
                   choices=["fast", "standard", "large", "hybrid", "orca-transformer"],
                   help="Network architecture (default: standard)")
    g.add_argument("--device", type=str, default=None,
                   help="Device: cuda, mps, cpu (default: auto)")

    # Optimizer
    g = parser.add_argument_group("optimizer")
    g.add_argument("--lr", type=float, default=None,
                   help="Learning rate (default: 0.001)")
    g.add_argument("--weight-decay", type=float, default=None,
                   help="L2 regularization (default: 1e-4)")
    g.add_argument("--scheduler-t0", type=int, default=None,
                   help="CosineAnnealing T_0 (default: 50)")
    g.add_argument("--scheduler-tmult", type=int, default=None,
                   help="CosineAnnealing T_mult (default: 2)")
    g.add_argument("--scheduler-eta-min", type=float, default=None,
                   help="CosineAnnealing eta_min (default: 1e-4)")

    # Search
    g = parser.add_argument_group("search")
    g.add_argument("--mcts-sims", type=int, default=None,
                   help="MCTS simulations per move (default: 400)")
    g.add_argument("--mcts-batch", type=int, default=None,
                   help="MCTS batch size for NN eval (default: 64)")

    # Replay buffer
    g = parser.add_argument_group("replay buffer")
    g.add_argument("--buffer-size", type=int, default=None,
                   help="Replay buffer capacity (default: 400000)")
    g.add_argument("--batch-size", type=int, default=None,
                   help="Training batch size (default: 1024)")

    # ELO
    g = parser.add_argument_group("evaluation")
    g.add_argument("--elo-every", type=int, default=None,
                   help="ELO eval frequency in iterations (default: 2)")
    g.add_argument("--elo-games", type=int, default=None,
                   help="Games per ELO evaluation (default: 4)")

    # Plateau detection
    g = parser.add_argument_group("plateau detection")
    g.add_argument("--plateau-threshold", type=float, default=None,
                   help="ELO delta to detect stall (default: 15)")
    g.add_argument("--plateau-iters", type=int, default=None,
                   help="Stall iterations before boosting sims (default: 10)")
    g.add_argument("--plateau-boost", type=int, default=None,
                   help="Extra sims on plateau (default: 50)")

    # Arena
    g = parser.add_argument_group("arena")
    g.add_argument("--elo-sims", type=int, default=None,
                   help="MCTS sims during ELO games (default: 30)")
    g.add_argument("--elo-max-opponents", type=int, default=None,
                   help="Max past versions to play against (default: 6)")

    # Feature toggles
    g = parser.add_argument_group("feature toggles")
    g.add_argument("--no-curriculum", action="store_true",
                   help="Disable adaptive sim/game curriculum (use fixed values)")
    g.add_argument("--no-auto-tuner", action="store_true",
                   help="Disable AutoTuner hyperparameter adjustment")
    g.add_argument("--no-adaptive-lr", action="store_true",
                   help="Disable cosine annealing LR (use fixed LR)")
    g.add_argument("--no-augmentation", action="store_true",
                   help="Disable symmetry data augmentation")

    args = parser.parse_args()

    trainer = OrcaTrainer(
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        train_steps=args.train_steps,
        resume=not args.fresh,
        device=args.device,
        net_config=args.config,
        num_workers=args.workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_T0=args.scheduler_t0,
        scheduler_Tmult=args.scheduler_tmult,
        scheduler_eta_min=args.scheduler_eta_min,
        mcts_sims=args.mcts_sims,
        mcts_batch_size=args.mcts_batch,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        elo_every=args.elo_every,
        elo_games=args.elo_games,
        use_curriculum=not args.no_curriculum,
        use_auto_tuner=not args.no_auto_tuner,
        use_adaptive_lr=not args.no_adaptive_lr,
        use_augmentation=not args.no_augmentation,
        plateau_threshold=args.plateau_threshold,
        plateau_iters=args.plateau_iters,
        plateau_sim_boost=args.plateau_boost,
        elo_sims=args.elo_sims,
        elo_max_opponents=args.elo_max_opponents,
    )

    def _sigint_handler(sig, frame):
        print("\nCtrl+C received, finishing current iteration...")
        trainer.observer.request_stop()

    signal.signal(signal.SIGINT, _sigint_handler)

    trainer.run()


if __name__ == "__main__":
    main()
