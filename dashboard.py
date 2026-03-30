"""
HEX BOT — Training Dashboard

Black-and-white minimalist dashboard with live game visualization,
ELO progression, loss curves, and auto-scaling parallel training.

Usage: python dashboard.py
Then open http://localhost:5001
"""

from __future__ import annotations
import sys
import math
import multiprocessing
import random
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
from flask import Flask, jsonify, request, Response
from flask_socketio import SocketIO

from main import HexGame
from bot import (
    HexNet, MCTS, ReplayBuffer, self_play_game, train_step,
    get_device, BOARD_SIZE, NUM_SIMULATIONS, BATCH_SIZE, LEARNING_RATE, L2_REG,
    TrainingSample, OnnxPredictor, export_onnx,
    POSITION_CATALOG, generate_puzzles, augment_sample,
    load_human_games, load_online_games, find_forced_move,
    encode_state,
)

# ---------------------------------------------------------------------------
# Timestamped print helper
# ---------------------------------------------------------------------------
from datetime import datetime as _dt
_print = print
def print(*args, **kwargs):
    """Override print to prepend timestamp."""
    ts = _dt.now().strftime('%H:%M:%S')
    _print(f'[{ts}]', *args, **kwargs)


# Resource monitor
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """Monitors CPU, GPU (MPS), and RAM usage."""

    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        # Scale workers: C engine releases GIL, so more workers = more throughput
        # Use most cores, leaving 2 for training + system
        self.num_threads = min(12, max(2, self.cpu_count - 2))
        self._lock = threading.Lock()
        self._history: List[dict] = []

    def snapshot(self) -> dict:
        cpu_pct = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        snap = {
            'cpu_pct': round(cpu_pct, 1),
            'ram_pct': round(mem.percent, 1),
            'ram_used_gb': round(mem.used / (1024 ** 3), 1),
            'ram_total_gb': round(mem.total / (1024 ** 3), 1),
            'cpu_count': self.cpu_count,
            'workers': self.num_threads,
            'gpu_available': torch.backends.mps.is_available(),
        }
        with self._lock:
            self._history.append(snap)
            if len(self._history) > 300:
                self._history = self._history[-300:]
        return snap

    def get_history(self) -> List[dict]:
        with self._lock:
            return list(self._history[-60:])

    @property
    def current_workers(self) -> int:
        return self.num_threads


# ---------------------------------------------------------------------------
# Curriculum: MCTS sim count by iteration
# ---------------------------------------------------------------------------

def get_curriculum_sims(iteration: int) -> int:
    """MCTS sims for training with 2.7M param network (fast inference).
    200 sims with small network ≈ same wall time as 30 sims with 14.5M."""
    if iteration < 5:
        return 50
    if iteration < 20:
        return 100
    return 200  # deep search with fast network


def get_curriculum_games(iteration: int, base: int) -> int:
    """Dynamic games per iteration.
    With 14.5M param network + 100 sims, each game takes ~12-15s.
    Target: ~2-3 min self-play per iteration for fast iteration cycles.
    5 workers × 8 games each = 40 games, ~2 min self-play."""
    if iteration < 5:
        return 50   # fast initial ramp
    if iteration < 20:
        return 40   # moderate
    return 40       # steady state: 40 games × 100 sims = quality + speed


# ---------------------------------------------------------------------------
# Parallel self-play worker (subprocess, CPU inference)
# ---------------------------------------------------------------------------

def _self_play_worker(onnx_path: str, num_sims: int,
                      games: int, positions: Optional[list] = None) -> list:
    """Run self-play games in a subprocess using ONNX Runtime.
    positions: list of (position_dict, hint_moves) tuples or plain dicts."""
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
        samples, move_history = self_play_game(predictor, mcts, start_position=pos,
                                                hint_moves=hints)
        serialized = []
        for s in samples:
            serialized.append({
                'state': s.encoded_state.numpy(),
                'policy': s.policy_target,
                'player': s.player,
                'result': s.result,
                'threat': s.threat_label,
                'priority': s.priority,
            })
        result_val = samples[0].result if samples else 0.0
        results.append((serialized, [list(m) for m in move_history],
                        result_val, len(samples)))
    return results


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
    print(f'  │  [Worker {pid}] loaded in {t_load:.1f}s, playing {games} games ({num_sims} sims)')

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
        samples, move_history = self_play_game_v2(net, searcher, start_position=pos,
                                                   hint_moves=hints)
        t_game = _time.perf_counter() - t_game
        winner = 'P0' if (samples and samples[0].result > 0) else 'P1'
        print(f'  │  [W{pid}] game {i+1}/{games}: {winner} {len(move_history)}mv {t_game:.1f}s ({t_game/max(len(move_history),1):.2f}s/mv)')
        serialized = []
        for s in samples:
            serialized.append({
                'state': s.encoded_state.numpy(),
                'policy': s.policy_target,
                'player': s.player,
                'result': s.result,
                'threat': s.threat_label,
                'priority': s.priority,
            })
        result_val = samples[0].result if samples else 0.0
        results.append((serialized, [list(m) for m in move_history],
                        result_val, len(samples)))
    return results


# ---------------------------------------------------------------------------
# Metrics store (thread-safe)
# ---------------------------------------------------------------------------

class MetricsStore:
    def __init__(self):
        self._lock = threading.Lock()
        self.iterations: List[dict] = []
        self.elo_history: List[dict] = [{'iteration': 0, 'elo': 1000.0}]
        self.current_elo: float = 1000.0
        self.total_games: int = 0
        self.current_iteration: int = 0
        self.total_iterations: int = 0
        self.is_training: bool = False

    def add_iteration(self, metrics: dict) -> None:
        with self._lock:
            self.iterations.append(metrics)
            self.total_games += metrics.get('games', 0)

    def update_elo(self, iteration: int, elo: float) -> None:
        with self._lock:
            self.current_elo = elo
            self.elo_history.append({'iteration': iteration, 'elo': round(elo, 1)})

    def get_stats(self) -> dict:
        with self._lock:
            latest = self.iterations[-1] if self.iterations else {}
            return {
                'iteration': self.current_iteration,
                'total_iterations': self.total_iterations,
                'total_games': self.total_games,
                'current_elo': round(self.current_elo, 1),
                'is_training': self.is_training,
                'buffer_size': latest.get('buffer_size', 0),
                'latest_loss': latest.get('loss', {}),
                'latest_wins': latest.get('wins', [0, 0, 0]),
                'self_play_time': latest.get('self_play_time', 0),
                'train_time': latest.get('train_time', 0),
                'avg_game_length': latest.get('avg_game_length', 0),
            }

    def get_elo_history(self) -> List[dict]:
        with self._lock:
            return list(self.elo_history)

    def get_loss_history(self) -> List[dict]:
        with self._lock:
            return [
                {
                    'iteration': m['iteration'],
                    'total': m['loss']['total'],
                    'value': m['loss']['value'],
                    'policy': m['loss']['policy'],
                }
                for m in self.iterations
                if m.get('loss') and isinstance(m['loss'], dict) and 'total' in m['loss']
            ]


# ---------------------------------------------------------------------------
# ELO evaluator
# ---------------------------------------------------------------------------

class ModelVault:
    """Stores compressed weights for every evaluated generation."""

    def __init__(self, max_models: int = 200):
        self.models: list = []  # [(iteration, state_dict_cpu_fp16), ...]
        self.max_models = max_models

    def add(self, iteration: int, state_dict: dict):
        compressed = {k: v.detach().cpu().half() for k, v in state_dict.items()}
        self.models.append((iteration, compressed))
        # Thin old models if too many (keep first, last 20, evenly spaced)
        if len(self.models) > self.max_models:
            n = len(self.models)
            keep = {0, n - 1}  # first + last
            keep.update(range(max(0, n - 20), n))  # last 20
            step = max(1, n // 50)
            keep.update(range(0, n, step))  # evenly spaced
            self.models = [self.models[i] for i in sorted(keep)]

    def get_net(self, idx: int, device) -> HexNet:
        """Load a stored model by index, return on device."""
        _, state = self.models[idx]
        state_fp32 = {k: v.float() for k, v in state.items()}
        # Detect network size from stored weights
        init_w = state_fp32.get('conv_init.weight')
        if init_w is not None:
            nf = init_w.shape[0]
            # Count res blocks
            nb = 0
            while f'res_blocks.{nb}.conv1.weight' in state_fp32:
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


class GenerationalArena:
    """Evaluates current model via mini round-robin against past generations.
    Much more stable than 10-game single-opponent ELO."""

    def __init__(self, device: torch.device, games_per_opponent: int = 4,
                 num_sims: int = 30, max_opponents: int = 6):
        self.device = device
        self.games_per_opponent = games_per_opponent
        self.num_sims = num_sims
        self.max_opponents = max_opponents
        self.matchup_history: list = []  # [{gen_i: {gen_j: {w,l,d}}}]

    def evaluate(self, current_net: HexNet, vault: ModelVault,
                 current_elo: float) -> float:
        """Play mini round-robin against selected past generations."""
        current_net.eval()
        n = len(vault)
        if n == 0:
            return current_elo

        # Select opponents: first + last + evenly spaced
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

            matchups[opp_iter] = {'w': w, 'l': l, 'd': d}
            total_wins += w
            total_losses += l
            total_draws += d

            # Free GPU memory
            del opp_net, mcts_opp

        self.matchup_history.append(matchups)

        # Compute ELO delta from aggregate score
        total_games = total_wins + total_losses + total_draws
        if total_games == 0:
            return current_elo
        score = (total_wins + 0.5 * total_draws) / total_games
        # Use K=16 (lower than before) since we have more games
        new_elo = current_elo + 16 * (score - 0.5) * len(opponent_indices)
        return new_elo

    def _select_opponents(self, n: int) -> list:
        """Pick diverse opponents from vault."""
        if n <= self.max_opponents:
            return list(range(n))
        selected = {0, n - 1}  # always include first and last
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

    def get_matchup_summary(self) -> dict:
        """Return matchup data for dashboard display."""
        return {
            'history': self.matchup_history[-20:],
            'total_evaluations': len(self.matchup_history),
        }


# ---------------------------------------------------------------------------
# AutoTuner — self-improving hyperparameter controller
# ---------------------------------------------------------------------------

class AutoTuner:
    """Observes metrics after each iteration and adjusts hyperparams for the next.
    Rule-based + trend detection. No external ML needed."""

    def __init__(self):
        self.loss_history: list = []
        self.elo_history: list = []
        self.decisions: list = []  # log of (iteration, decision_str)

        # Current tunable params (conservative — AlphaZero-style)
        self.params = {
            'lr': 0.002,           # was 0.01 — too high, caused forgetting
            'sims': 10,
            'mix_normal': 1.00,     # pure self-play — bot generates its own positions
            'mix_catalog': 0.00,
            'mix_endgame': 0.00,
            'mix_formation': 0.00,
            'mix_sequence': 0.00,
            'hint_blend': 0.3,
            'temp_threshold': 20,
            'train_steps': 200,    # was 400 — overfitting early
        }
        self.param_history: list = []  # list of params dicts per iteration

    def _log(self, iteration: int, msg: str):
        self.decisions.append((iteration, msg))
        print(f'  │  🔧 AutoTuner: {msg}')

    def update(self, metrics: dict, iteration: int) -> dict:
        """Observe metrics, return updated params for next iteration."""
        p = self.params.copy()

        # Track histories
        total_loss = metrics.get('total_loss', 0)
        self.loss_history.append(total_loss)

        elo = metrics.get('elo')
        if elo is not None:
            self.elo_history.append(elo)

        changes = []

        # --- Learning Rate: managed by CosineAnnealingWarmRestarts (not AutoTuner) ---
        # LR is set by the scheduler, AutoTuner just reports it
        # No manual decay — cosine annealing handles everything

        # --- MCTS Sims: fixed for training, only ramp slowly ---
        # Keep sims at 50 for training (good balance of quality vs speed)
        # Sims 200 is only for online play
        if p['sims'] > 50:
            p['sims'] = 50
            changes.append(f'sims→50 (training cap)')

        # --- Game Mix: PURE SELF-PLAY (locked, no adjustments) ---
        vloss = metrics.get('value_loss', 0)
        ploss = metrics.get('policy_loss', 0)
        p['mix_normal'] = 1.0
        p['mix_endgame'] = 0.0
        p['mix_catalog'] = 0.0
        p['mix_formation'] = 0.0
        p['mix_sequence'] = 0.0

        # --- Hint Blend: decay over time ---
        p['hint_blend'] = max(0.0, 0.3 - iteration * 0.015)

        # --- Train Steps: only increase if loss is decreasing AND buffer full ---
        buf_fill = metrics.get('buffer_fill', 0)
        loss_decreasing = (len(self.loss_history) >= 3 and
                           self.loss_history[-1] < self.loss_history[-3] * 0.95)
        if buf_fill > 0.9 and loss_decreasing and p['train_steps'] < 600:
            p['train_steps'] = min(600, p['train_steps'] + 50)
            changes.append(f'train_steps→{p["train_steps"]}')

        if changes:
            self._log(iteration, ' | '.join(changes))
        else:
            self._log(iteration, 'no changes')

        # Normalize mix ratios
        total_mix = (p['mix_normal'] + p['mix_catalog'] + p['mix_endgame']
                     + p['mix_formation'] + p['mix_sequence'])
        if abs(total_mix - 1.0) > 0.01:
            for k in ['mix_normal', 'mix_catalog', 'mix_endgame',
                       'mix_formation', 'mix_sequence']:
                p[k] /= total_mix

        self.params = p
        self.param_history.append(p.copy())
        return p

    def get_status(self) -> dict:
        return {
            'params': self.params,
            'history': self.param_history[-20:],
            'decisions': self.decisions[-20:],
        }


# ---------------------------------------------------------------------------
# Dashboard observer
# ---------------------------------------------------------------------------

class DashboardObserver:
    def __init__(self, metrics: MetricsStore, sio: SocketIO):
        self.metrics = metrics
        self.sio = sio
        self._stop_flag = threading.Event()

    def on_iteration_start(self, iteration: int, total: int) -> None:
        self.metrics.current_iteration = iteration + 1
        self.metrics.total_iterations = total
        self.sio.emit('iteration_start', {'iteration': iteration + 1, 'total': total})

    def on_game_complete(self, game_idx: int, total_games: int,
                         move_history: list, result: float,
                         num_samples: int) -> None:
        self.sio.emit('game_complete', {
            'game_idx': game_idx + 1,
            'total_games': total_games,
            'result': result,
            'num_moves': len(move_history),
            'moves': move_history,
        })

    def on_iteration_complete(self, metrics: dict) -> None:
        self.metrics.add_iteration(metrics)
        self.sio.emit('iteration_complete', metrics)

    def on_training_complete(self) -> None:
        self.metrics.is_training = False
        self.sio.emit('training_complete', {})

    def should_stop(self) -> bool:
        return self._stop_flag.is_set()

    def request_stop(self):
        self._stop_flag.set()

    def reset_stop(self):
        self._stop_flag.clear()


# ---------------------------------------------------------------------------
# Training manager (parallel self-play with auto-scaling)
# ---------------------------------------------------------------------------

class TrainingManager:
    def __init__(self, metrics: MetricsStore, observer: DashboardObserver,
                 sio: SocketIO, resource_monitor: ResourceMonitor):
        self.metrics = metrics
        self.observer = observer
        self.sio = sio
        self.resource = resource_monitor
        self.device = get_device()
        self.net: Optional[HexNet] = None
        self.model_vault = ModelVault(max_models=200)
        self.arena = GenerationalArena(self.device)
        self._thread: Optional[threading.Thread] = None

    def start(self, num_iterations=999999, games_per_iter=100,
              train_steps=400) -> bool:
        if self.metrics.is_training:
            return False
        self.observer.reset_stop()
        self.metrics.is_training = True
        self._thread = threading.Thread(
            target=self._run,
            args=(num_iterations, games_per_iter, train_steps),
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self):
        self.observer.request_stop()

    @staticmethod
    def _find_latest_checkpoint() -> Optional[str]:
        """Find the latest hex_checkpoint_N.pt file."""
        import glob
        ckpts = glob.glob('hex_checkpoint_*.pt')
        if not ckpts:
            return None
        # Sort by iteration number
        def get_iter(path):
            try:
                return int(path.replace('hex_checkpoint_', '').replace('.pt', ''))
            except ValueError:
                return -1
        ckpts.sort(key=get_iter)
        return ckpts[-1]

    def _run(self, num_iterations, games_per_iter, train_steps):
        # Use create_network factory — 'large' for maximum strength
        try:
            from bot import create_network
            self.net = create_network('standard').to(self.device)
            self._net_config = 'standard'
        except ImportError:
            self.net = HexNet(num_filters=256, num_res_blocks=12).to(self.device)
            self._net_config = 'standard'
        param_count = sum(p.numel() for p in self.net.parameters())

        print(f'\n{"="*60}')
        print(f'  TRAINING PIPELINE INITIALIZED')
        print(f'{"="*60}')
        print(f'  Network:     {self._net_config} ({param_count:,} params)')
        print(f'  Device:      {self.device}')
        print(f'  Iterations:  ∞ (runs until stopped)')
        print(f'  Games/iter:  {games_per_iter} (base, scaled by curriculum)')
        print(f'  Train steps: {train_steps}/iter')

        # Detect C engine availability
        use_v2 = False
        try:
            from bot import CGameState
            CGameState()
            use_v2 = True
            print(f'  Engine:      C engine (v2) ✓')
        except Exception as e:
            print(f'  Engine:      Python (v1) — C engine unavailable: {e}')

        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=0.001, weight_decay=L2_REG
        )
        # Cosine annealing: cycles LR between 0.001 and 0.0001
        # T_0=50 means first cycle is 50 iterations, then doubles each cycle
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-4
        )
        replay_buffer = ReplayBuffer()
        # Cap workers for power-limited training (90W charger)
        num_workers = min(self.resource.num_threads, 5)
        auto_tuner = AutoTuner()
        self._auto_tuner = auto_tuner

        # --- Resume from checkpoint if available ---
        start_iteration = 0
        resume_path = self._find_latest_checkpoint()
        if resume_path:
            try:
                ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
                # Migrate channels (5→7) and filters (128→256) if needed
                from bot import migrate_checkpoint_5to7, migrate_checkpoint_filters
                old_shape = ckpt['model_state_dict'].get('conv_init.weight', None)
                migrated_sd = migrate_checkpoint_5to7(ckpt['model_state_dict'])
                migrated_sd = migrate_checkpoint_filters(migrated_sd)
                new_shape = migrated_sd.get('conv_init.weight', None)
                arch_changed = (old_shape is not None and new_shape is not None
                                and old_shape.shape != new_shape.shape)
                ckpt['model_state_dict'] = migrated_sd
                self.net.load_state_dict(migrated_sd, strict=False)
                # Skip optimizer restore if architecture changed (Adam momentum buffers wrong shape)
                if arch_changed:
                    print(f'    ⚠ Fresh optimizer (conv_init migrated {old_shape.shape}→{new_shape.shape})')
                else:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_iteration = ckpt.get('iteration', 0) + 1

                # CRITICAL: Force LR back to a healthy value
                # The old training decayed LR to ~7e-6 (frozen). Reset it.
                for pg in optimizer.param_groups:
                    if pg['lr'] < 1e-4:
                        pg['lr'] = 0.001
                        print(f'    ⚠ LR was frozen at {pg["lr"]:.2e}, reset to 0.001')

                # Restore scheduler state if available
                if 'scheduler_state_dict' in ckpt:
                    try:
                        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                    except Exception:
                        pass  # scheduler format changed, start fresh

                # Restore replay buffer from separate file
                import pickle
                buf_path = os.path.join(os.path.dirname(resume_path), 'replay_buffer.pkl')
                if os.path.exists(buf_path):
                    try:
                        with open(buf_path, 'rb') as f:
                            buf_data = pickle.load(f)
                        replay_buffer.buffer.extend(buf_data.get('buffer', []))
                        replay_buffer.priorities.extend(buf_data.get('priorities', []))
                        print(f'    ✓ Restored replay buffer: {len(replay_buffer.buffer)} samples')
                    except Exception as e:
                        print(f'    ⚠ Could not restore buffer: {e}')

                # Restore metrics
                if 'metrics' in ckpt:
                    m = ckpt['metrics']
                    self.metrics.iterations = m.get('iterations', [])
                    self.metrics.elo_history = m.get('elo_history', [])
                    self.metrics.current_elo = m.get('current_elo', 1000)
                    self.metrics.total_games = m.get('total_games', 0)
                    self.metrics.current_iteration = start_iteration
                # Restore AutoTuner (but NOT the LR — that's managed by scheduler now)
                if 'auto_tuner' in ckpt:
                    at = ckpt['auto_tuner']
                    auto_tuner.params = at.get('params', auto_tuner.params)
                    auto_tuner.params['lr'] = optimizer.param_groups[0]['lr']  # sync with optimizer
                    # Force pure self-play (override old mix from checkpoint)
                    auto_tuner.params['mix_normal'] = 1.0
                    auto_tuner.params['mix_catalog'] = 0.0
                    auto_tuner.params['mix_endgame'] = 0.0
                    auto_tuner.params['mix_formation'] = 0.0
                    auto_tuner.params['mix_sequence'] = 0.0
                    auto_tuner.loss_history = at.get('loss_history', [])
                    auto_tuner.elo_history = at.get('elo_history', [])
                current_lr = optimizer.param_groups[0]['lr']
                print(f'  ✓ Resumed from {resume_path} (iter {start_iteration})')
                print(f'    ELO: {self.metrics.current_elo:.0f}, '
                      f'Games: {self.metrics.total_games}, '
                      f'LR: {current_lr:.5f}')
            except Exception as e:
                print(f'  ⚠ Failed to resume: {e} — starting fresh')
                start_iteration = 0
        else:
            print(f'  No checkpoint found — starting fresh')

        print(f'  Workers:     {num_workers}')
        print(f'  LR:          {auto_tuner.params["lr"]} → auto-tuned')
        print(f'  AutoTuner:   ✓ (self-improving)')
        print(f'{"="*60}\n')

        for iteration in range(start_iteration, num_iterations):
            if self.observer.should_stop():
                print(f'\n  ⏹ Training stopped by user at iteration {iteration}')
                break
            self.observer.on_iteration_start(iteration, num_iterations)

            prev_state = {k: v.clone() for k, v in self.net.state_dict().items()}

            # --- Curriculum (AutoTuner can override sims) ---
            current_sims = max(auto_tuner.params['sims'],
                               get_curriculum_sims(iteration))
            current_games = get_curriculum_games(iteration, games_per_iter)
            current_lr = optimizer.param_groups[0]['lr']

            print(f'  ┌─ Iter {iteration+1}/{num_iterations} '
                  f'│ sims={current_sims} │ games={current_games} │ lr={current_lr:.4f}')

            # --- Load human games on first iteration ---
            if iteration == 0:
                human_path = os.path.join(os.path.dirname(__file__), 'human_games.jsonl')
                if os.path.exists(human_path):
                    t_human = time.perf_counter()
                    human_samples = load_human_games(human_path, max_games=500, min_elo=1000)
                    for s in human_samples:
                        s.priority = 0.8  # lower than self-play to avoid dominating
                        replay_buffer.push(s)
                    t_human = time.perf_counter() - t_human
                    print(f'  │  Human games: {len(human_samples)} samples loaded ({t_human:.1f}s)')
                else:
                    print(f'  │  Human games: not found (run scrape_games.py first)')

            # --- Export ONNX for fast CPU workers ---
            t0 = time.perf_counter()
            t_export_start = time.perf_counter()
            self.net.eval()
            onnx_path = f'/tmp/hex_model_{iteration}.onnx'
            export_onnx(self.net, onnx_path)
            self.net.to(self.device)  # move back to GPU after export
            t_export = time.perf_counter() - t_export_start
            print(f'  │  ONNX export: {t_export:.1f}s')

            # --- Adaptive game mix: 5 categories (AutoTuner-driven) ---
            import random as _rng
            ap = auto_tuner.params
            mix = (ap['mix_normal'], ap['mix_catalog'], ap['mix_endgame'],
                   ap['mix_formation'], ap['mix_sequence'])
            n_normal = int(current_games * mix[0])
            n_catalog = int(current_games * mix[1])
            n_endgame = int(current_games * mix[2])
            n_formation = int(current_games * mix[3])
            n_sequence = current_games - n_normal - n_catalog - n_endgame - n_formation
            print(f'  │  Game mix: {n_normal} normal + {n_catalog} catalog + '
                  f'{n_endgame} L1-endgame + {n_formation} L2-formation + {n_sequence} L3-sequence')

            # Generate position lists for workers
            catalog_keys = list(POSITION_CATALOG.keys())
            catalog_positions = [
                (POSITION_CATALOG[_rng.choice(catalog_keys)], None)
                for _ in range(n_catalog)
            ]

            # Guided positions by level
            from bot import GUIDED_POSITIONS, get_guided_positions_by_level
            l1 = get_guided_positions_by_level(1)
            l2 = get_guided_positions_by_level(2)
            l3 = get_guided_positions_by_level(3)

            endgame_positions = [_rng.choice(l1) if l1 else (None, None) for _ in range(n_endgame)]
            formation_positions = [_rng.choice(l2) if l2 else (None, None) for _ in range(n_formation)]
            sequence_positions = [_rng.choice(l3) if l3 else (None, None) for _ in range(n_sequence)]

            # Also generate some random puzzles as fallback
            puzzle_positions = generate_puzzles(n_endgame // 2)
            extra_puzzles = [(p, None) for p in puzzle_positions]

            # All positions: (position_dict_or_None, hint_moves_or_None)
            all_positions = (
                catalog_positions +
                endgame_positions +
                formation_positions +
                sequence_positions +
                extra_puzzles[:n_endgame // 2] +
                [(None, None)] * n_normal
            )
            # Trim to exact game count
            all_positions = all_positions[:current_games]
            while len(all_positions) < current_games:
                all_positions.append((None, None))
            _rng.shuffle(all_positions)

            # Distribute across workers with positions
            chunks = []
            chunk_positions = []
            remaining = current_games
            pos_offset = 0
            for i in range(num_workers):
                c = remaining // (num_workers - i)
                chunks.append(c)
                chunk_positions.append(all_positions[pos_offset:pos_offset + c])
                pos_offset += c
                remaining -= c

            active_chunks = [c for c in chunks if c > 0]
            print(f'  │  Dispatching {len(active_chunks)} workers: '
                  f'{active_chunks}')

            total_samples = 0
            total_moves = 0
            wins = [0, 0, 0]
            game_idx = 0
            collected_samples = []  # for augmentation
            worker_errors = 0

            t_selfplay_start = time.perf_counter()

            if use_v2:
                # V2: ProcessPool with C engine + PyTorch
                net_state = {k: v.cpu() for k, v in self.net.state_dict().items()}

                # Asymmetric self-play: 25% of workers use an older checkpoint as opponent
                n_asymmetric = 0
                if len(self.model_vault) > 1:
                    n_asymmetric = max(1, num_workers // 4)

                print(f'  │  Self-play: V2 (C engine + MCTS {current_sims} sims'
                      f'{f", {n_asymmetric} asymmetric" if n_asymmetric else ""})...')

                with ProcessPoolExecutor(max_workers=num_workers) as pool:
                    futures = []
                    for wi, (c, pos) in enumerate(zip(chunks, chunk_positions)):
                        if c <= 0:
                            continue
                        futures.append(
                            pool.submit(_self_play_worker_v2, net_state,
                                        self._net_config, current_sims, c, pos,
                                        use_alphabeta=False)
                        )
                    for future in as_completed(futures):
                        if self.observer.should_stop():
                            break
                        try:
                            batch_results = future.result()
                        except Exception as e:
                            worker_errors += 1
                            print(f'  │  ⚠ V2 Worker error: {e}')
                            import traceback
                            traceback.print_exc()
                            continue

                        for serialized, move_hist, result_val, n_samples in batch_results:
                            for sd in serialized:
                                sample = TrainingSample(
                                    encoded_state=torch.from_numpy(sd['state']),
                                    policy_target=sd['policy'],
                                    player=sd['player'],
                                    result=sd['result'],
                                    threat_label=sd.get('threat'),
                                    priority=sd.get('priority', 1.0),
                                )
                                replay_buffer.push(sample)
                                collected_samples.append(sample)
                            total_samples += n_samples
                            total_moves += len(move_hist)
                            if result_val > 0:
                                wins[0] += 1
                            elif result_val < 0:
                                wins[1] += 1
                            else:
                                wins[2] += 1
                            game_idx += 1
                            from datetime import datetime
                            winner = 'P0' if result_val > 0 else ('P1' if result_val < 0 else 'draw')
                            print(f'  │  [{datetime.now().strftime("%H:%M:%S")}] Game {game_idx}/{current_games}: {winner} in {len(move_hist)} moves ({n_samples} samples)')
                            self.observer.on_game_complete(
                                game_idx, current_games, move_hist,
                                result_val, n_samples,
                            )
            else:
                # V1 fallback: ONNX workers
                print(f'  │  Self-play: V1 (ONNX Runtime)...')
                with ProcessPoolExecutor(max_workers=num_workers) as pool:
                    futures = [
                        pool.submit(_self_play_worker, onnx_path, current_sims, c, pos)
                        for c, pos in zip(chunks, chunk_positions) if c > 0
                    ]
                    for future in as_completed(futures):
                        if self.observer.should_stop():
                            break
                        try:
                            batch_results = future.result()
                        except Exception as e:
                            worker_errors += 1
                            print(f'  │  ⚠ Worker error: {e}')
                            import traceback
                            traceback.print_exc()
                            continue

                        for serialized, move_hist, result_val, n_samples in batch_results:
                            for sd in serialized:
                                sample = TrainingSample(
                                    encoded_state=torch.from_numpy(sd['state']),
                                    policy_target=sd['policy'],
                                    player=sd['player'],
                                    result=sd['result'],
                                    threat_label=sd.get('threat'),
                                    priority=sd.get('priority', 1.0),
                                )
                                replay_buffer.push(sample)
                                collected_samples.append(sample)
                            total_samples += n_samples
                            total_moves += len(move_hist)
                            if result_val > 0:
                                wins[0] += 1
                            elif result_val < 0:
                                wins[1] += 1
                            else:
                                wins[2] += 1
                            game_idx += 1
                            from datetime import datetime
                            winner = 'P0' if result_val > 0 else ('P1' if result_val < 0 else 'draw')
                            print(f'  │  [{datetime.now().strftime("%H:%M:%S")}] Game {game_idx}/{current_games}: {winner} in {len(move_hist)} moves ({n_samples} samples)')
                            self.observer.on_game_complete(
                                game_idx, current_games, move_hist,
                                result_val, n_samples,
                            )

            t_selfplay = time.perf_counter() - t_selfplay_start
            games_per_sec = game_idx / t_selfplay if t_selfplay > 0 else 0
            print(f'  │  Self-play done: {game_idx} games, {total_samples} samples, '
                  f'{t_selfplay:.1f}s ({games_per_sec:.1f} games/s)')
            print(f'  │  Wins: P0={wins[0]} P1={wins[1]} draw={wins[2]}')
            if worker_errors > 0:
                print(f'  │  ⚠ {worker_errors} worker(s) failed')

            # --- Load new online games (human feedback) ---
            online_path = os.path.join(os.path.dirname(__file__), 'online_games.jsonl')
            if not hasattr(self, '_online_lines_read'):
                self._online_lines_read = 0
            try:
                online_samples, new_pos = load_online_games(
                    online_path, start_line=self._online_lines_read)
                if online_samples:
                    for s in online_samples:
                        replay_buffer.push(s)
                        collected_samples.append(s)  # include in augmentation too
                    self._online_lines_read = new_pos
                    print(f'  │  Online games: +{len(online_samples)} samples (priority 2.0)')
            except Exception as e:
                pass  # online games are optional

            # --- Symmetry augmentation: 4x data multiplier ---
            t_aug_start = time.perf_counter()
            aug_count = 0
            for sample in collected_samples:
                for aug in augment_sample(sample):
                    replay_buffer.push(aug)
                    aug_count += 1
            total_samples += aug_count
            t_aug = time.perf_counter() - t_aug_start
            print(f'  │  Augmentation: +{aug_count} samples ({t_aug:.1f}s) '
                  f'→ buffer={len(replay_buffer)}/{replay_buffer.buffer.maxlen}')

            sp_time = time.perf_counter() - t0

            # --- GPU training (aggressive) ---
            losses = {'total': 0, 'value': 0, 'policy': 0}
            train_time = 0
            if len(replay_buffer) >= BATCH_SIZE:
                print(f'  │  Training: {train_steps} steps on {self.device} '
                      f'(batch={BATCH_SIZE}, buffer={len(replay_buffer)})...')
                t1 = time.perf_counter()
                self.net.train()
                for step in range(train_steps):
                    losses = train_step(
                        self.net, optimizer, replay_buffer, self.device
                    )
                    # Emit progress every 20 steps
                    if step % 20 == 0 or step == train_steps - 1:
                        self.sio.emit('train_progress', {
                            'step': step + 1,
                            'total': train_steps,
                            'loss': round(losses['total'], 4),
                            'pct': round((step + 1) / train_steps * 100),
                        })
                train_time = time.perf_counter() - t1
                steps_per_sec = train_steps / train_time if train_time > 0 else 0
                scheduler.step()
                # Sync LR from scheduler to AutoTuner (for display)
                current_lr = optimizer.param_groups[0]['lr']
                auto_tuner.params['lr'] = current_lr
                print(f'  │  Training done: {train_time:.1f}s ({steps_per_sec:.0f} steps/s) '
                      f'loss={losses["total"]:.4f} (v={losses["value"]:.4f} p={losses["policy"]:.4f}) '
                      f'lr={current_lr:.6f}')
            else:
                print(f'  │  Training skipped: buffer too small ({len(replay_buffer)}<{BATCH_SIZE})')

            # ELO eval (every 3 iterations) — generational tournament
            elo_str = ''
            if len(replay_buffer) >= BATCH_SIZE and (iteration + 1) % 3 == 0:
                # Store current generation in vault
                self.model_vault.add(iteration + 1, self.net.state_dict())
                n_opp = min(self.arena.max_opponents, len(self.model_vault) - 1)
                n_games = n_opp * self.arena.games_per_opponent
                print(f'  │  ELO evaluation ({n_games} games vs {n_opp} generations, '
                      f'vault={len(self.model_vault)})...')
                t_elo = time.perf_counter()
                new_elo = self.arena.evaluate(
                    self.net, self.model_vault, self.metrics.current_elo
                )
                t_elo = time.perf_counter() - t_elo
                delta = new_elo - self.metrics.current_elo
                sign = '+' if delta >= 0 else ''
                self.metrics.update_elo(iteration + 1, new_elo)
                elo_str = f' │ ELO {new_elo:.0f} ({sign}{delta:.0f})'
                print(f'  │  ELO: {new_elo:.0f} ({sign}{delta:.0f}) in {t_elo:.1f}s')

            res_snap = self.resource.snapshot()
            avg_len = total_moves / max(game_idx, 1)
            total_time = time.perf_counter() - t0
            metrics = {
                'iteration': iteration + 1,
                'games': game_idx,
                'samples': total_samples,
                'wins': wins,
                'self_play_time': round(sp_time, 1),
                'train_time': round(train_time, 1),
                'loss': losses,
                'buffer_size': len(replay_buffer),
                'avg_game_length': round(avg_len, 1),
                'elo': round(self.metrics.current_elo, 1),
                'workers': num_workers,
                'cpu_pct': res_snap['cpu_pct'],
                'ram_pct': res_snap['ram_pct'],
                'sims': current_sims,
                'lr': round(optimizer.param_groups[0]['lr'], 5),
            }
            self.observer.on_iteration_complete(metrics)

            # --- AutoTuner: observe and adapt ---
            gps = game_idx / t_selfplay if t_selfplay > 0 else 1.0
            at_metrics = {
                'total_loss': metrics.get('loss', {}).get('total', 0),
                'value_loss': metrics.get('loss', {}).get('value', 0),
                'policy_loss': metrics.get('loss', {}).get('policy', 0),
                'elo': self.metrics.current_elo,
                'games_per_sec': gps,
                'buffer_fill': len(replay_buffer) / replay_buffer.buffer.maxlen,
            }
            new_params = auto_tuner.update(at_metrics, iteration)
            # Apply LR change
            for pg in optimizer.param_groups:
                pg['lr'] = new_params['lr']

            print(f'  └─ Iter {iteration+1} done: {total_time:.1f}s total '
                  f'│ {game_idx} games │ {total_samples} samples '
                  f'│ CPU {res_snap["cpu_pct"]:.0f}% │ RAM {res_snap["ram_pct"]:.0f}%'
                  f'{elo_str}')
            print()

            if (iteration + 1) % 5 == 0:
                ckpt_path = f'hex_checkpoint_{iteration+1}.pt'
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'metrics': {
                        'iterations': self.metrics.iterations,
                        'elo_history': self.metrics.elo_history,
                        'current_elo': self.metrics.current_elo,
                        'total_games': self.metrics.total_games,
                    },
                    'auto_tuner': {
                        'params': auto_tuner.params,
                        'loss_history': auto_tuner.loss_history,
                        'elo_history': auto_tuner.elo_history,
                    },
                }, ckpt_path)
                # Save replay buffer separately (too large for .pt)
                import pickle
                try:
                    buf_path = 'replay_buffer.pkl'
                    with open(buf_path, 'wb') as f:
                        pickle.dump({
                            'buffer': list(replay_buffer.buffer),
                            'priorities': list(replay_buffer.priorities),
                        }, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print(f'  ⚠ Buffer save failed: {e}')
                print(f'  💾 Checkpoint saved: {ckpt_path} + buffer ({len(replay_buffer.buffer)} samples)')

        print(f'\n{"="*60}')
        print(f'  TRAINING COMPLETE — {num_iterations} iterations')
        print(f'  Final ELO: {self.metrics.current_elo:.0f}')
        print(f'  Total games: {self.metrics.total_games}')
        print(f'{"="*60}\n')
        self.observer.on_training_complete()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Background game scraper (gentle — 1 game every 10 seconds)
# ---------------------------------------------------------------------------

class BackgroundScraper:
    """Slowly downloads human games in background without overwhelming the server."""

    def __init__(self, output_path: str = "human_games.jsonl"):
        self.output_path = output_path
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        import requests as _req
        base_ts = int(time.time() * 1000)

        # Load existing IDs
        existing = set()
        if os.path.exists(self.output_path):
            with open(self.output_path, 'r') as f:
                for line in f:
                    try:
                        g = json.loads(line)
                        existing.add(g.get("id", ""))
                    except Exception:
                        pass

        page = 1
        while not self._stop.is_set():
            try:
                resp = _req.get(f"https://hexo.did.science/api/finished-games",
                                params={"page": page, "pageSize": 5, "baseTimestamp": base_ts},
                                timeout=10)
                if resp.status_code != 200:
                    self._stop.wait(30)
                    continue

                data = resp.json()
                games = data.get("games", [])
                if not games:
                    page = 1  # restart from beginning
                    self._stop.wait(60)
                    continue

                for g in games:
                    if self._stop.is_set():
                        return
                    gid = g.get("id", "")
                    if gid in existing:
                        continue

                    # Filter: only six-in-a-row wins, 10+ moves
                    result = g.get("gameResult", {})
                    if result.get("reason") != "six-in-a-row":
                        continue
                    if g.get("moveCount", 0) < 10:
                        continue

                    # Fetch detail
                    try:
                        detail = _req.get(f"https://hexo.did.science/api/finished-games/{gid}",
                                          timeout=10)
                        if detail.status_code == 200:
                            with open(self.output_path, 'a') as f:
                                f.write(json.dumps(detail.json()) + "\n")
                            existing.add(gid)
                    except Exception:
                        pass

                    # Wait 10 seconds between games — very gentle
                    self._stop.wait(10)

                page += 1
            except Exception:
                self._stop.wait(30)


# ---------------------------------------------------------------------------
# Live game broadcaster
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hex'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

resource_monitor = ResourceMonitor()  # auto-detects CPU count, sets threads
metrics_store = MetricsStore()
observer = DashboardObserver(metrics_store, socketio)
training_mgr = TrainingManager(metrics_store, observer, socketio, resource_monitor)

# Start gentle background scraper
bg_scraper = BackgroundScraper(os.path.join(os.path.dirname(__file__), 'human_games.jsonl'))
bg_scraper.start()


@app.route('/')
def index():
    return Response(DASHBOARD_HTML, mimetype='text/html')


@app.route('/api/stats')
def api_stats():
    return jsonify(metrics_store.get_stats())


@app.route('/api/elo')
def api_elo():
    result = {
        'elo_history': metrics_store.get_elo_history(),
        'vault_size': len(training_mgr.model_vault) if training_mgr else 0,
        'matchups': training_mgr.arena.get_matchup_summary() if training_mgr else {},
    }
    return jsonify(result)


@app.route('/api/losses')
def api_losses():
    return jsonify(metrics_store.get_loss_history())


@app.route('/api/resources')
def api_resources():
    snap = resource_monitor.snapshot()
    result = {
        'cpu_pct': snap['cpu_pct'],
        'ram_pct': snap['ram_pct'],
        'ram_used_gb': snap['ram_used_gb'],
        'ram_total_gb': snap['ram_total_gb'],
        'cpu_count': snap['cpu_count'],
        'workers': snap['workers'],
        'gpu_available': snap['gpu_available'],
        'history': resource_monitor.get_history(),
    }
    return jsonify(result)


@app.route('/api/autotuner')
def api_autotuner():
    if training_mgr and hasattr(training_mgr, '_auto_tuner'):
        return jsonify(training_mgr._auto_tuner.get_status())
    return jsonify({'params': {}, 'history': [], 'decisions': []})


@app.route('/api/winrates')
def api_winrates():
    data = []
    for entry in metrics_store.iterations:
        wins = entry.get('wins', [0, 0, 0])
        total = sum(wins) or 1
        data.append({
            'iteration': entry['iteration'],
            'p0': round(wins[0] / total * 100, 1),
            'p1': round(wins[1] / total * 100, 1),
            'draw': round(wins[2] / total * 100, 1),
        })
    return jsonify(data)


@app.route('/api/speed')
def api_speed():
    data = []
    for entry in metrics_store.iterations:
        games = entry.get('games', 0)
        sp_time = entry.get('self_play_time', 1)
        data.append({
            'iteration': entry['iteration'],
            'games_per_sec': round(games / max(sp_time, 0.1), 2),
            'samples': entry.get('samples', 0),
            'avg_game_length': round(entry.get('avg_game_length', 0), 1),
        })
    return jsonify(data)


@app.route('/api/train/start', methods=['POST'])
def train_start():
    data = request.json or {}
    ok = training_mgr.start(
        num_iterations=data.get('iterations', 999999),
        games_per_iter=data.get('games_per_iter', 100),
        train_steps=data.get('train_steps', 400),
    )
    if ok:
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'}), 409


@app.route('/api/train/stop', methods=['POST'])
def train_stop():
    training_mgr.stop()
    return jsonify({'status': 'stopping'})


# ---------------------------------------------------------------------------
# Online Play Manager (runs browser_player.py as subprocess)
# ---------------------------------------------------------------------------

class OnlinePlayManager:
    """Manages the browser bot subprocess from the dashboard."""

    def __init__(self):
        self._proc = None
        self._log_lines: list = []
        self._max_log = 200
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.is_running = False
        self._reader_thread = None

    def start(self, games: int = 9999, fast: bool = False):
        if self._proc and self._proc.poll() is None:
            return False
        import subprocess
        cmd = [
            sys.executable, '-u', 'browser_player.py',
            '--auto', '--games', str(games),
        ]
        if fast:
            cmd.append('--fast')
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=os.path.dirname(__file__),
            env=env,
        )
        self.is_running = True
        self._log_lines = []
        # Reader thread to capture output
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()
        return True

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self.is_running = False

    def _read_output(self):
        try:
            for line in self._proc.stdout:
                line = line.rstrip()
                if line:
                    self._log_lines.append(line)
                    if len(self._log_lines) > self._max_log:
                        self._log_lines = self._log_lines[-self._max_log:]
                    # Parse wins/losses
                    if '🏆' in line or 'WIN' in line:
                        self.wins += 1
                        self.games_played += 1
                    elif '✗' in line or 'Lost' in line:
                        self.losses += 1
                        self.games_played += 1
        except Exception:
            pass
        self.is_running = False

    def get_status(self):
        return {
            'running': self.is_running,
            'games': self.games_played,
            'wins': self.wins,
            'losses': self.losses,
            'winrate': round(self.wins / max(self.games_played, 1) * 100, 1),
            'log': self._log_lines[-30:],
        }


online_mgr = OnlinePlayManager()


@app.route('/api/online/start', methods=['POST'])
def online_start():
    data = request.json or {}
    ok = online_mgr.start(
        games=data.get('games', 9999),
        fast=data.get('fast', False),
    )
    return jsonify({'status': 'started' if ok else 'already_running'})


@app.route('/api/online/stop', methods=['POST'])
def online_stop():
    online_mgr.stop()
    return jsonify({'status': 'stopped'})


@app.route('/api/online/status')
def online_status():
    return jsonify(online_mgr.get_status())


@app.route('/api/network_vis')
def network_vis():
    """Generate hex-grid network visualization on the fly."""
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import RegularPolygon

    try:
        if training_manager.net is None:
            return 'No network loaded', 404

        net = training_manager.net
        net.eval()

        game = HexGame(candidate_radius=2, max_total_stones=200)
        for q, r in [(0,0), (2,0), (2,-1), (1,0), (0,1)]:
            game.place_stone(q, r)

        encoded, oq, orr = encode_state(game)
        x = encoded.unsqueeze(0).to(next(net.parameters()).device)

        with torch.no_grad():
            p_logits, value, threats = net(x)
            policy = torch.softmax(p_logits, dim=1)[0].cpu().numpy().reshape(19, 19)

        def draw_hex(ax, data, title, cmap='hot', vmin=None, vmax=None):
            if vmin is None: vmin = data.min()
            if vmax is None: vmax = data.max()
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cm = plt.get_cmap(cmap)
            sz = 0.55
            for r in range(19):
                for q in range(19):
                    px = sz * (np.sqrt(3) * q + np.sqrt(3)/2 * r)
                    py = sz * (3/2 * r)
                    h = RegularPolygon((px, -py), 6, radius=sz*0.58, orientation=0,
                                       facecolor=cm(norm(data[r, q])),
                                       edgecolor='#555', linewidth=0.2)
                    ax.add_patch(h)
            ax.set_xlim(-1, sz * np.sqrt(3) * 19 + 1)
            ax.set_ylim(-sz * 1.5 * 19 - 1, 1)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(title, fontsize=9, fontweight='bold')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        board = encoded[0].numpy() - encoded[1].numpy()
        draw_hex(ax1, board, 'Board (Red=P0, Blue=P1)', cmap='RdBu_r', vmin=-1, vmax=1)
        draw_hex(ax2, policy, f'Policy (Value: {value[0].item():.3f})', cmap='YlOrRd', vmin=0)
        fig.suptitle('Network Insight', fontsize=12, fontweight='bold')
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue(), 200, {'Content-Type': 'image/png'}
    except Exception as e:
        return f'Error: {e}', 500


@socketio.on('connect')
def on_connect():
    pass


# ---------------------------------------------------------------------------
# Frontend HTML (high-res, responsive canvases, resource monitoring)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HEX BOT</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#fff;color:#000;font-family:'Courier New',monospace;font-size:13px;line-height:1.5}
header{border-bottom:1px solid #000;padding:14px 24px;display:flex;align-items:center;gap:20px}
.title{font-size:15px;font-weight:700;letter-spacing:4px;text-transform:uppercase}
.controls{display:flex;gap:6px}
.controls button{background:#fff;border:1px solid #000;width:34px;height:34px;cursor:pointer;
  font-size:15px;font-family:inherit;transition:all .15s}
.controls button:hover{background:#000;color:#fff}
.controls button.active{background:#000;color:#fff}
.status{margin-left:auto;letter-spacing:3px;font-size:11px;font-weight:700}
main{display:flex;border-bottom:1px solid #000;min-height:calc(100vh - 130px)}
.left{width:50%;border-right:1px solid #000;padding:20px;display:flex;flex-direction:column;align-items:center}
.right{width:50%;display:flex;flex-direction:column;overflow-y:auto}
.chart-box{padding:16px 20px;border-bottom:1px solid #000;min-height:220px;display:flex;flex-direction:column}
.chart-box:last-child{border-bottom:none}
.label{font-weight:700;letter-spacing:2px;font-size:10px;text-transform:uppercase;margin-bottom:8px;flex-shrink:0}
.canvas-wrap{position:relative;min-height:180px;height:180px}
.canvas-wrap canvas{position:absolute;top:0;left:0;width:100%;height:100%}
#hex-canvas-wrap{flex:1;position:relative;width:100%;min-height:0}
#hex-canvas{position:absolute;top:0;left:0;width:100%;height:100%;border:1px solid #eee}
.game-info{font-size:11px;letter-spacing:1px;min-height:18px;margin-top:8px;flex-shrink:0}
footer{padding:12px 24px;display:flex;gap:20px;flex-wrap:wrap;font-size:11px;letter-spacing:.5px;border-top:1px solid #000}
footer b{font-weight:700}
footer span{white-space:nowrap}
.sep{color:#ccc}
.res-bar{display:flex;gap:4px;align-items:center}
.res-meter{width:40px;height:8px;border:1px solid #000;display:inline-block;position:relative;vertical-align:middle}
.res-meter-fill{height:100%;background:#000;transition:width .3s}
</style>
</head>
<body>
<header>
  <span class="title">Hex Bot Training</span>
  <span class="controls">
    <button id="btn-start" onclick="startTraining()" title="Start">&#9654;</button>
    <button id="btn-stop" onclick="stopTraining()" title="Stop">&#9632;</button>
  </span>
  <span class="status" id="status">IDLE</span>
</header>
<main>
  <div class="left">
    <div class="label">Training Game <button id="btn-overlay" onclick="toggleOverlay()" style="font:9px 'Courier New';border:1px solid #000;background:#000;color:#fff;padding:1px 6px;cursor:pointer;margin-left:8px">OVERLAY ON</button></div>
    <div id="hex-canvas-wrap"><canvas id="hex-canvas"></canvas></div>
    <div class="game-info" id="game-info">Waiting for training to start...</div>
    <div id="progress-wrap" style="margin-top:6px;display:none">
      <div style="display:flex;align-items:center;gap:8px">
        <div style="flex:1;height:6px;background:#eee;border-radius:3px;overflow:hidden">
          <div id="progress-fill" style="height:100%;background:#000;width:0%;transition:width .2s"></div>
        </div>
        <span id="progress-label" style="font:700 10px 'Courier New';min-width:80px;text-align:right"></span>
      </div>
    </div>
  </div>
  <div class="right">
    <div class="chart-box">
      <div class="label">Elo Progression</div>
      <div class="canvas-wrap"><canvas id="elo-chart"></canvas></div>
    </div>
    <div class="chart-box">
      <div class="label">Loss Curves</div>
      <div class="canvas-wrap"><canvas id="loss-chart"></canvas></div>
    </div>
    <div class="chart-box" style="max-height:140px">
      <div class="label">Resources</div>
      <div class="canvas-wrap"><canvas id="res-chart"></canvas></div>
    </div>
    <div class="chart-box" style="max-height:120px;min-height:80px">
      <div class="label">Arena Matchups</div>
      <div id="arena-matchups" style="font-size:11px;letter-spacing:.5px;line-height:1.8;overflow-y:auto;flex:1;min-height:0">No matchup data yet</div>
    </div>
    <div class="chart-box">
      <div class="label">Win Rates</div>
      <div class="canvas-wrap"><canvas id="winrate-chart"></canvas></div>
    </div>
    <div class="chart-box">
      <div class="label">Game Length</div>
      <div class="canvas-wrap"><canvas id="gamelength-chart"></canvas></div>
    </div>
    <div class="chart-box">
      <div class="label">Training Speed</div>
      <div class="canvas-wrap"><canvas id="speed-chart"></canvas></div>
    </div>
    <div class="chart-box" style="min-height:180px">
      <div class="label">AutoTuner</div>
      <div id="autotuner-panel" style="flex:1;overflow-y:auto;min-height:0">
        <table id="autotuner-table" style="width:100%;font-size:11px;border-collapse:collapse;margin-bottom:8px">
          <tr><th style="text-align:left;border-bottom:1px solid #000;padding:2px 6px">Param</th><th style="text-align:right;border-bottom:1px solid #000;padding:2px 6px">Value</th></tr>
        </table>
        <div id="autotuner-log" style="font-size:10px;line-height:1.6;color:#444;max-height:100px;overflow-y:auto"></div>
      </div>
    </div>
    <div class="chart-box" style="min-height:220px">
      <div class="label">Network Insight</div>
      <div style="text-align:center;padding:4px">
        <img id="network-vis-img" src="/api/network_vis" style="max-width:100%;height:auto;border:1px solid #ddd" alt="Network visualization" onerror="this.alt='Generating...'">
        <div style="margin-top:4px"><button onclick="document.getElementById('network-vis-img').src='/api/network_vis?t='+Date.now()" style="font:10px 'Courier New';border:1px solid #000;background:#fff;padding:2px 8px;cursor:pointer">Refresh</button></div>
      </div>
    </div>
    <div class="chart-box" style="min-height:200px">
      <div class="label">Online Play</div>
      <div style="padding:4px 8px">
        <div style="display:flex;gap:8px;margin-bottom:8px">
          <button id="online-start-btn" onclick="startOnline()" style="background:#000;color:#fff;border:none;padding:4px 12px;font:11px 'SF Mono',monospace;cursor:pointer">▶ START</button>
          <button id="online-stop-btn" onclick="stopOnline()" style="background:#fff;color:#000;border:1px solid #000;padding:4px 12px;font:11px 'SF Mono',monospace;cursor:pointer">■ STOP</button>
          <span id="online-status" style="font-size:11px;line-height:24px;color:#666">Idle</span>
        </div>
        <div style="font-size:12px;margin-bottom:4px">
          <b id="online-record">0W-0L</b>
          <span id="online-winrate" style="color:#666;margin-left:8px">0%</span>
        </div>
        <div id="online-log" style="font-size:10px;line-height:1.5;color:#444;max-height:120px;overflow-y:auto;font-family:'SF Mono',monospace;white-space:pre-wrap"></div>
      </div>
    </div>
    </div>
  </div>
</main>
<footer>
  <span>Iter <b id="s-iter">0</b>/<b id="s-itertotal">0</b></span>
  <span class="sep">|</span>
  <span>Games <b id="s-games">0</b></span>
  <span class="sep">|</span>
  <span>Buffer <b id="s-buffer">0</b></span>
  <span class="sep">|</span>
  <span>Elo <b id="s-elo">1000</b></span>
  <span class="sep">|</span>
  <span>Win P0:<b id="s-w0">0</b>% P1:<b id="s-w1">0</b>% D:<b id="s-wd">0</b>%</span>
  <span class="sep">|</span>
  <span>Self-play <b id="s-sp">0</b>s</span>
  <span class="sep">|</span>
  <span>Sims <b id="s-sims">-</b></span>
  <span class="sep">|</span>
  <span>LR <b id="s-lr">-</b></span>
  <span class="sep">|</span>
  <span>Threads <b id="s-workers">-</b></span>
  <span class="sep">|</span>
  <span class="res-bar">CPU <div class="res-meter"><div class="res-meter-fill" id="cpu-fill" style="width:0%"></div></div> <b id="s-cpu">0</b>%</span>
  <span class="res-bar">RAM <div class="res-meter"><div class="res-meter-fill" id="ram-fill" style="width:0%"></div></div> <b id="s-ram">0</b>%</span>
  <span class="sep">|</span>
  <span>Games/s <b id="s-gps">-</b></span>
  <span class="sep">|</span>
  <span>Vault <b id="s-vault">0</b></span>
  <span class="sep">|</span>
  <span>AutoTuner <b id="s-at">-</b></span>
</footer>

<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.min.js"></script>
<script>
const DPR = window.devicePixelRatio || 1;
const socket = io();
let stones0=[], stones1=[], candidates=[];

// ─── HiDPI canvas helper ─────────────────────────────────────────────
function sizeCanvas(cv){
  const r = cv.parentElement.getBoundingClientRect();
  const w = Math.round(r.width), h = Math.round(r.height);
  if(cv.width !== w*DPR || cv.height !== h*DPR){
    cv.width = w*DPR; cv.height = h*DPR;
    cv.style.width = w+'px'; cv.style.height = h+'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(DPR,0,0,DPR,0,0);
  }
  return {w, h};
}

// ─── Socket events ───────────────────────────────────────────────────
let topMovesCurrent=[], topMovesOpponent=[], valueBar=0, currentPlayer=0;
let overlayOn=true;
function toggleOverlay(){
  overlayOn=!overlayOn;
  const btn=el('btn-overlay');
  btn.textContent=overlayOn?'OVERLAY ON':'OVERLAY OFF';
  btn.style.background=overlayOn?'#000':'#fff';
  btn.style.color=overlayOn?'#fff':'#000';
  drawHex();
}
socket.on('live_game_start', ()=>{
  stones0=[]; stones1=[]; candidates=[]; topMovesCurrent=[]; topMovesOpponent=[]; valueBar=0;
  drawHex(); setInfo('New game...');
});
socket.on('live_game_move', d=>{
  stones0=d.stones_0; stones1=d.stones_1; candidates=d.candidates;
  topMovesCurrent=d.top_moves_current||[]; topMovesOpponent=d.top_moves_opponent||[];
  valueBar=d.value||0; currentPlayer=d.current_player||0;
  drawHex();
  const vStr = d.value!==undefined ? ' (v='+d.value.toFixed(2)+')' : '';
  setInfo('Move '+d.move_num+' \u2014 P'+d.player+' \u2192 ('+d.move[0]+','+d.move[1]+')'+vStr);
});
socket.on('live_game_end', d=>{
  const r = d.winner!==null ? 'Player '+d.winner+' wins' : 'Draw';
  setInfo('Game over: '+r+' ('+d.total_moves+' moves)');
});
socket.on('iteration_start', d=>{
  el('status').textContent='ITER '+d.iteration+'/'+d.total;
  el('s-itertotal').textContent=d.total;
});
socket.on('iteration_complete', d=>{
  updateStats(d); fetchCharts();
  el('progress-wrap').style.display='none';
  // Auto-refresh network visualization
  const nv=document.getElementById('network-vis-img');
  if(nv) nv.src='/api/network_vis?t='+Date.now();
});
socket.on('training_complete', ()=>{
  el('status').textContent='COMPLETE';
  el('btn-start').classList.remove('active');
});
let replayTimer=null;
socket.on('game_complete', d=>{
  el('status').textContent='ITER '+el('s-iter').textContent+' G'+d.game_idx+'/'+d.total_games;
  const pw=el('progress-wrap'), pf=el('progress-fill'), pl=el('progress-label');
  pw.style.display='';
  const pct=Math.round(d.game_idx/d.total_games*100);
  pf.style.width=pct+'%';
  pl.textContent='Self-play '+d.game_idx+'/'+d.total_games;
  // Replay this training game client-side
  if(d.moves && d.moves.length>0){
    if(replayTimer) clearInterval(replayTimer);
    stones0=[]; stones1=[]; candidates=[];
    let mi=0, pl0=0, stt=0;
    const mv=d.moves;
    setInfo('Training game '+d.game_idx+'...');
    replayTimer=setInterval(()=>{
      if(mi>=mv.length){
        clearInterval(replayTimer); replayTimer=null;
        const w=d.result>0?'P0':'P1';
        setInfo('Game '+d.game_idx+': '+w+' wins ('+mv.length+' moves)');
        return;
      }
      const m=mv[mi];
      if(pl0===0) stones0.push(m); else stones1.push(m);
      mi++; stt++;
      const need=(stones0.length+stones1.length<=1)?1:2;
      if(stt>=need){pl0=1-pl0; stt=0;}
      drawHex();
      setInfo('Game '+d.game_idx+' — Move '+mi+'/'+mv.length);
    },120);
  }
});
socket.on('train_progress', d=>{
  const pw=el('progress-wrap'), pf=el('progress-fill'), pl=el('progress-label');
  pw.style.display='';
  pf.style.width=d.pct+'%';
  pl.textContent='Training '+d.step+'/'+d.total+' (loss '+d.loss+')';
});

function el(id){return document.getElementById(id)}
function setInfo(t){el('game-info').textContent=t}

function updateStats(d){
  el('s-iter').textContent=d.iteration;
  const t=d.wins[0]+d.wins[1]+d.wins[2]||1;
  el('s-w0').textContent=Math.round(d.wins[0]/t*100);
  el('s-w1').textContent=Math.round(d.wins[1]/t*100);
  el('s-wd').textContent=Math.round(d.wins[2]/t*100);
  el('s-sp').textContent=d.self_play_time;
  el('s-buffer').textContent=d.buffer_size.toLocaleString();
  if(d.elo) el('s-elo').textContent=Math.round(d.elo);
  if(d.workers) el('s-workers').textContent=d.workers;
  if(d.sims) el('s-sims').textContent=d.sims;
  if(d.lr) el('s-lr').textContent=d.lr.toFixed(4);
  if(d.cpu_pct!=null){
    el('s-cpu').textContent=Math.round(d.cpu_pct);
    el('cpu-fill').style.width=Math.round(d.cpu_pct)+'%';
  }
  if(d.ram_pct!=null){
    el('s-ram').textContent=Math.round(d.ram_pct);
    el('ram-fill').style.width=Math.round(d.ram_pct)+'%';
  }
  if(d.self_play_time>0 && d.games) el('s-gps').textContent=(d.games/d.self_play_time).toFixed(1);
  fetch('/api/stats').then(r=>r.json()).then(s=>{el('s-games').textContent=s.total_games});
}

function fetchCharts(){
  fetch('/api/elo').then(r=>r.json()).then(data=>{
    drawLineChart(el('elo-chart'),
      [{label:'ELO',data:data.elo_history.map(d=>({x:d.iteration,y:d.elo})),dash:[]}],{});
    // Arena matchups
    const mu=data.matchups||{};
    const hist=mu.history||[];
    const box=el('arena-matchups');
    if(hist.length){
      const last=hist[hist.length-1];
      const parts=Object.entries(last).map(([gen,r])=>'Gen '+gen+': W'+r.w+'-L'+r.l+(r.d?'-D'+r.d:''));
      box.textContent=parts.join(' | ')||'No matchups';
    }
    // Vault size in footer
    if(data.vault_size!=null) el('s-vault').textContent=data.vault_size;
  });
  fetch('/api/losses').then(r=>r.json()).then(data=>{
    if(!data.length) return;
    drawLineChart(el('loss-chart'),[
      {label:'total',data:data.map(d=>({x:d.iteration,y:d.total})),dash:[]},
      {label:'value',data:data.map(d=>({x:d.iteration,y:d.value})),dash:[6,3]},
      {label:'policy',data:data.map(d=>({x:d.iteration,y:d.policy})),dash:[2,2]},
    ],{});
  });
  fetch('/api/resources').then(r=>r.json()).then(data=>{
    const h=data.history||[];
    if(h.length<2) return;
    drawLineChart(el('res-chart'),[
      {label:'CPU%',data:h.map((d,i)=>({x:i,y:d.cpu_pct})),dash:[]},
      {label:'RAM%',data:h.map((d,i)=>({x:i,y:d.ram_pct})),dash:[6,3]},
    ],{yMin:0,yMax:100});
  });
  fetch('/api/winrates').then(r=>r.json()).then(data=>{
    if(!data.length) return;
    drawLineChart(el('winrate-chart'),[
      {label:'P0%',data:data.map(d=>({x:d.iteration,y:d.p0})),dash:[]},
      {label:'P1%',data:data.map(d=>({x:d.iteration,y:d.p1})),dash:[6,3]},
      {label:'Draw%',data:data.map(d=>({x:d.iteration,y:d.draw})),dash:[2,2]},
    ],{yMin:0,yMax:100});
  });
  fetch('/api/speed').then(r=>r.json()).then(data=>{
    if(!data.length) return;
    drawLineChart(el('gamelength-chart'),[
      {label:'Avg Moves',data:data.map(d=>({x:d.iteration,y:d.avg_game_length})),dash:[]},
    ],{});
    drawLineChart(el('speed-chart'),[
      {label:'Games/s',data:data.map(d=>({x:d.iteration,y:d.games_per_sec})),dash:[]},
    ],{});
    // Footer games/sec
    const last=data[data.length-1];
    if(last) el('s-gps').textContent=last.games_per_sec;
  });
  fetch('/api/autotuner').then(r=>r.json()).then(data=>{
    // Table
    const tbl=el('autotuner-table');
    const params=data.params||{};
    let rows='<tr><th style="text-align:left;border-bottom:1px solid #000;padding:2px 6px">Param</th><th style="text-align:right;border-bottom:1px solid #000;padding:2px 6px">Value</th></tr>';
    const show={'lr':1,'sims':1,'mix_normal':1,'mix_catalog':1,'mix_endgame':1,'mix_formation':1,'mix_sequence':1};
    for(const[k,v] of Object.entries(params)){
      if(show[k]) rows+='<tr><td style="padding:2px 6px">'+k+'</td><td style="text-align:right;padding:2px 6px">'+(typeof v==='number'?v.toFixed(4):v)+'</td></tr>';
    }
    tbl.innerHTML=rows;
    // Decision log
    const log=el('autotuner-log');
    const decs=data.decisions||[];
    const last10=decs.slice(-10);
    log.innerHTML=last10.map(d=>'<div>['+d[0]+'] '+d[1]+'</div>').join('');
    log.scrollTop=log.scrollHeight;
    // Footer indicator
    if(decs.length){
      const ld=decs[decs.length-1];
      el('s-at').textContent=ld[1].substring(0,20);
    }
  });
}

// ─── Controls ────────────────────────────────────────────────────────
function startTraining(){
  fetch('/api/train/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({})});
  el('status').textContent='STARTING...';
  el('btn-start').classList.add('active');
}
function stopTraining(){
  fetch('/api/train/stop',{method:'POST'});
  el('status').textContent='STOPPING...';
  el('btn-start').classList.remove('active');
}

// ─── Online Play ────────────────────────────────────────────────────
function startOnline(){
  fetch('/api/online/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({games:9999,fast:false})});
  el('online-status').textContent='Starting...';
}
function stopOnline(){
  fetch('/api/online/stop',{method:'POST'});
  el('online-status').textContent='Stopping...';
}
function pollOnline(){
  fetch('/api/online/status').then(r=>r.json()).then(d=>{
    el('online-status').textContent=d.running?'🟢 Playing':'⚪ Idle';
    el('online-record').textContent=d.wins+'W-'+d.losses+'L';
    el('online-winrate').textContent=d.winrate+'%';
    const log=el('online-log');
    log.textContent=d.log.join('\n');
    log.scrollTop=log.scrollHeight;
  }).catch(()=>{});
}
setInterval(pollOnline,3000);

// ─── Hex Canvas (HiDPI) ─────────────────────────────────────────────
const HEX_SIZE = 18;
const S3 = Math.sqrt(3);

function axToPixel(q,r){
  return [HEX_SIZE*(S3*q+S3/2*r), HEX_SIZE*(1.5*r)];
}

function drawHex(){
  const cv=el('hex-canvas');
  const {w:W,h:H}=sizeCanvas(cv);
  const ctx=cv.getContext('2d');
  ctx.clearRect(0,0,W,H);

  const all=[...stones0,...stones1,...candidates];
  if(!all.length){
    ctx.fillStyle='#aaa'; ctx.font='12px Courier New'; ctx.textAlign='center';
    ctx.fillText('Waiting for training game...',W/2,H/2);
    return;
  }

  let mnX=1e9,mxX=-1e9,mnY=1e9,mxY=-1e9;
  for(const[q,r] of all){
    const[px,py]=axToPixel(q,r);
    if(px<mnX) mnX=px; if(px>mxX) mxX=px;
    if(py<mnY) mnY=py; if(py>mxY) mxY=py;
  }
  const mg=HEX_SIZE*3;
  const spanX=mxX-mnX+mg*2, spanY=mxY-mnY+mg*2;
  const sc=Math.min(W/spanX,H/spanY,2.5);
  const ox=W/2-(mnX+mxX)/2*sc, oy=H/2-(mnY+mxY)/2*sc;

  function toS(q,r){const[px,py]=axToPixel(q,r);return[px*sc+ox,py*sc+oy]}
  function hexPath(cx,cy,sz){
    ctx.beginPath();
    for(let i=0;i<6;i++){
      const a=Math.PI/3*i-Math.PI/6;
      const hx=cx+sz*Math.cos(a), hy=cy+sz*Math.sin(a);
      i===0?ctx.moveTo(hx,hy):ctx.lineTo(hx,hy);
    }
    ctx.closePath();
  }
  const hr=HEX_SIZE*sc*0.88;

  // Candidates
  ctx.fillStyle='#ddd';
  for(const[q,r] of candidates){
    const[sx,sy]=toS(q,r);
    ctx.beginPath(); ctx.arc(sx,sy,Math.max(1,1.5*sc),0,Math.PI*2); ctx.fill();
  }

  // P0: solid black
  for(const[q,r] of stones0){
    const[sx,sy]=toS(q,r);
    hexPath(sx,sy,hr);
    ctx.fillStyle='#000'; ctx.fill();
    ctx.strokeStyle='#000'; ctx.lineWidth=1; ctx.stroke();
  }

  // P1: white + hatching
  for(const[q,r] of stones1){
    const[sx,sy]=toS(q,r);
    hexPath(sx,sy,hr);
    ctx.fillStyle='#fff'; ctx.fill();
    ctx.strokeStyle='#000'; ctx.lineWidth=1.2; ctx.stroke();
    ctx.save();
    hexPath(sx,sy,hr); ctx.clip();
    ctx.strokeStyle='#000'; ctx.lineWidth=0.6;
    const step=Math.max(3,4*sc);
    for(let d=-hr*2;d<=hr*2;d+=step){
      ctx.beginPath();
      ctx.moveTo(sx+d-hr,sy-hr);
      ctx.lineTo(sx+d+hr,sy+hr);
      ctx.stroke();
    }
    ctx.restore();
  }

  // ─── BEST MOVES OVERLAY — Both players, split shared cells ─────
  if(!overlayOn) return;  // skip overlay rendering + value bar when toggled off
  // Build lookup maps
  const curMap={}, oppMap={};
  for(const[m,p] of topMovesCurrent) curMap[m[0]+','+m[1]]=p;
  for(const[m,p] of topMovesOpponent) oppMap[m[0]+','+m[1]]=p;
  const allKeys=new Set([...Object.keys(curMap),...Object.keys(oppMap)]);

  for(const key of allKeys){
    const[mq,mr]=key.split(',').map(Number);
    const[sx,sy]=toS(mq,mr);
    const cProb=curMap[key]||0;
    const oProb=oppMap[key]||0;
    const both=cProb>0 && oProb>0;

    if(both){
      // Split hex: left half orange, right half blue
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(sx,sy-hr); ctx.lineTo(sx,sy+hr); ctx.lineTo(sx-hr,sy+hr);
      ctx.lineTo(sx-hr,sy-hr); ctx.closePath(); ctx.clip();
      ctx.globalAlpha=Math.min(0.55, cProb*1.8+0.1);
      hexPath(sx,sy,hr*0.85); ctx.fillStyle='#ff8800'; ctx.fill();
      ctx.restore();

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(sx,sy-hr); ctx.lineTo(sx,sy+hr); ctx.lineTo(sx+hr,sy+hr);
      ctx.lineTo(sx+hr,sy-hr); ctx.closePath(); ctx.clip();
      ctx.globalAlpha=Math.min(0.45, oProb*1.5+0.08);
      hexPath(sx,sy,hr*0.85); ctx.fillStyle='#2277ff'; ctx.fill();
      ctx.restore();

      // Divider line
      ctx.globalAlpha=0.4; ctx.strokeStyle='#000'; ctx.lineWidth=0.5;
      ctx.beginPath(); ctx.moveTo(sx,sy-hr*0.7); ctx.lineTo(sx,sy+hr*0.7); ctx.stroke();
      ctx.globalAlpha=1.0;

      // Both percentages
      ctx.font='bold '+Math.max(6,7*sc)+'px Courier New'; ctx.textBaseline='middle';
      if(cProb>0.04){ctx.fillStyle='#000';ctx.textAlign='right';ctx.fillText((cProb*100).toFixed(0)+'%',sx-2,sy);}
      if(oProb>0.15){ctx.fillStyle='#fff';ctx.textAlign='left';ctx.fillText((oProb*100).toFixed(0)+'%',sx+2,sy);}
    } else if(cProb>0){
      // Only current player
      ctx.globalAlpha=Math.min(0.55, cProb*1.8+0.1);
      hexPath(sx,sy,hr*0.85); ctx.fillStyle='#ff8800'; ctx.fill();
      ctx.globalAlpha=1.0;
      if(cProb>0.04){
        ctx.fillStyle='#000'; ctx.font='bold '+Math.max(7,9*sc)+'px Courier New';
        ctx.textAlign='center'; ctx.textBaseline='middle';
        ctx.fillText((cProb*100).toFixed(0)+'%',sx,sy);
      }
    } else {
      // Only opponent
      ctx.globalAlpha=Math.min(0.45, oProb*1.5+0.08);
      hexPath(sx,sy,hr*0.85); ctx.fillStyle='#2277ff'; ctx.fill();
      ctx.globalAlpha=1.0;
      if(oProb>0.15){
        ctx.fillStyle='#fff'; ctx.font='bold '+Math.max(7,9*sc)+'px Courier New';
        ctx.textAlign='center'; ctx.textBaseline='middle';
        ctx.fillText((oProb*100).toFixed(0)+'%',sx,sy);
      }
    }
  }

  // ─── VALUE BAR (right edge) ───────────────────────
  if(valueBar!==0 || stones0.length>0){
    const barW=8, barH=H*0.6, barX=W-barW-4, barY=(H-barH)/2;
    // Background
    ctx.fillStyle='#eee';
    ctx.fillRect(barX,barY,barW,barH);
    // Value fill
    const mid=barY+barH/2;
    const fill=valueBar*barH/2;
    if(valueBar>0){
      ctx.fillStyle='#000';
      ctx.fillRect(barX,mid-fill,barW,fill);
    } else {
      ctx.fillStyle='#999';
      ctx.fillRect(barX,mid,barW,-fill);
    }
    // Center line
    ctx.strokeStyle='#666'; ctx.lineWidth=0.5;
    ctx.beginPath(); ctx.moveTo(barX,mid); ctx.lineTo(barX+barW,mid); ctx.stroke();
    // Label
    ctx.fillStyle='#000'; ctx.font='8px Courier New'; ctx.textAlign='center';
    ctx.fillText(valueBar>0?'X':'O',barX+barW/2,valueBar>0?barY-3:barY+barH+10);
    ctx.fillText(Math.abs(valueBar).toFixed(2),barX+barW/2,valueBar>0?barY+barH+10:barY-3);
  }
}

// ─── Line Chart (HiDPI) ─────────────────────────────────────────────
function drawLineChart(cv,datasets,opts){
  const {w:W,h:H}=sizeCanvas(cv);
  const ctx=cv.getContext('2d');
  ctx.clearRect(0,0,W,H);
  const pad={t:14,r:16,b:24,l:52};
  const pW=W-pad.l-pad.r, pH=H-pad.t-pad.b;
  if(pW<10||pH<10) return;

  const pts=datasets.flatMap(d=>d.data);
  if(pts.length<2){
    ctx.fillStyle='#ccc'; ctx.font='11px Courier New'; ctx.textAlign='center';
    ctx.fillText('No data yet',W/2,H/2);
    return;
  }

  let xMn=Math.min(...pts.map(p=>p.x)), xMx=Math.max(...pts.map(p=>p.x));
  let yMn=opts.yMin!=null?opts.yMin:Math.min(...pts.map(p=>p.y));
  let yMx=opts.yMax!=null?opts.yMax:Math.max(...pts.map(p=>p.y));
  if(xMn===xMx) xMx=xMn+1;
  if(opts.yMin==null){const yP=(yMx-yMn)*0.1||1; yMn-=yP; yMx+=yP;}

  function toP(x,y){return[pad.l+(x-xMn)/(xMx-xMn)*pW, pad.t+pH-(y-yMn)/(yMx-yMn)*pH]}

  // Axes
  ctx.strokeStyle='#000'; ctx.lineWidth=1;
  ctx.beginPath();
  ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+pH); ctx.lineTo(pad.l+pW,pad.t+pH);
  ctx.stroke();

  // Y ticks
  ctx.fillStyle='#000'; ctx.font='10px Courier New'; ctx.textAlign='right';
  for(let i=0;i<=4;i++){
    const yV=yMn+(yMx-yMn)*i/4;
    const[,sy]=toP(xMn,yV);
    ctx.fillText(yV.toFixed(1),pad.l-4,sy+3);
    ctx.strokeStyle='#eee'; ctx.lineWidth=0.5;
    ctx.beginPath(); ctx.moveTo(pad.l,sy); ctx.lineTo(pad.l+pW,sy); ctx.stroke();
  }

  // X ticks
  ctx.textAlign='center';
  const xStep=Math.max(1,Math.ceil((xMx-xMn)/6));
  for(let x=Math.ceil(xMn);x<=xMx;x+=xStep){
    const[sx]=toP(x,yMn);
    ctx.fillStyle='#000'; ctx.fillText(x,sx,pad.t+pH+14);
  }

  // Lines
  const defaultDash=[[],[6,3],[2,2]];
  datasets.forEach((ds,di)=>{
    if(ds.data.length<2) return;
    ctx.setLineDash(ds.dash||defaultDash[di]||[]);
    ctx.strokeStyle='#000'; ctx.lineWidth=1.3;
    ctx.beginPath();
    ds.data.forEach((p,i)=>{
      const[sx,sy]=toP(p.x,p.y);
      i===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy);
    });
    ctx.stroke();
  });
  ctx.setLineDash([]);

  // Legend
  ctx.font='10px Courier New'; ctx.textAlign='left';
  datasets.forEach((ds,di)=>{
    const lx=pad.l+8+di*90, ly=pad.t+10;
    ctx.setLineDash(ds.dash||defaultDash[di]||[]);
    ctx.strokeStyle='#000'; ctx.lineWidth=1.3;
    ctx.beginPath(); ctx.moveTo(lx,ly); ctx.lineTo(lx+18,ly); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='#000'; ctx.fillText(ds.label,lx+22,ly+3);
  });
}

// ─── Resize handler ──────────────────────────────────────────────────
let resizeTimer;
window.addEventListener('resize', ()=>{
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(()=>{ drawHex(); fetchCharts(); }, 150);
});

// ─── Resource polling ────────────────────────────────────────────────
setInterval(()=>{
  fetch('/api/resources').then(r=>r.json()).then(d=>{
    el('s-cpu').textContent=Math.round(d.cpu_pct);
    el('cpu-fill').style.width=Math.round(d.cpu_pct)+'%';
    el('s-ram').textContent=Math.round(d.ram_pct);
    el('ram-fill').style.width=Math.round(d.ram_pct)+'%';
    el('s-workers').textContent=d.workers;
  }).catch(()=>{});
}, 3000);

// ─── Init ────────────────────────────────────────────────────────────
setTimeout(()=>{ drawHex(); fetchCharts(); }, 100);
fetch('/api/stats').then(r=>r.json()).then(s=>{
  el('s-games').textContent=s.total_games;
  el('s-elo').textContent=Math.round(s.current_elo);
  if(s.is_training) el('status').textContent='TRAINING';
});
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    port = 5001
    print('HEX BOT Dashboard')
    print(f'Device: {get_device()}')
    print(f'CPU cores: {multiprocessing.cpu_count()}')
    print(f'Workers: {resource_monitor.num_threads}')
    print(f'Curriculum: 20→50→100→200 sims, games scale down as sims increase')
    print(f'LR: 0.01 → decay 0.5 every 20 iters')
    print(f'Open http://localhost:{port}')
    # Suppress Flask/Werkzeug HTTP request spam
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True, log_output=False)
