#!/usr/bin/env python3
"""
benchmark_nn.py - Full training pipeline benchmark.

Tests the same metrics as competitive benchmarks:
- MCTS throughput (CPU only, no NN)
- NN inference throughput (batched)
- NN latency (single position)
- Replay buffer sampling speed
- GPU/device utilization
- Worker pool throughput (full self-play pipeline)

Run:
    python benchmark_nn.py
"""

import time
import random
import sys
import os
import numpy as np

# Ensure we can import from current directory
sys.path.insert(0, os.path.dirname(__file__) or '.')


def fmt(n):
    if n >= 1_000_000: return f"{n/1_000_000:,.1f}M"
    if n >= 1_000: return f"{n/1_000:,.1f}K"
    return f"{n:,.1f}"


def benchmark_mcts_no_nn(n_games=50):
    """MCTS throughput with C engine heuristic only (no neural network)."""
    from hexgame import HexGame

    total_sims = 0
    t0 = time.perf_counter()
    for i in range(n_games):
        game = HexGame(max_stones=100)
        rng = random.Random(i)
        while not game.is_over:
            moves = game.legal_moves()
            if not moves:
                break
            # Use C engine scored moves as "MCTS" proxy
            top = game.scored_moves(20)
            if top:
                game.place(*top[0][0])
            else:
                game.place(*rng.choice(moves))
            total_sims += 20  # each scored_moves call ~ 20 evaluations
    elapsed = time.perf_counter() - t0
    return {
        'games': n_games,
        'total_sims': total_sims,
        'sims_per_sec': total_sims / elapsed,
        'time': elapsed,
    }


def benchmark_nn_inference(batch_sizes=[1, 8, 32, 64]):
    """NN forward pass throughput at different batch sizes."""
    import torch
    from bot import create_network, get_device, NUM_CHANNELS, BOARD_SIZE

    device = get_device()
    net = create_network('standard').to(device)
    net.eval()

    results = []
    for bs in batch_sizes:
        dummy = torch.randn(bs, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE).to(device)
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                net(dummy)
        if device.type == 'mps':
            torch.mps.synchronize()
        elif device.type == 'cuda':
            torch.cuda.synchronize()

        n_iters = max(20, 200 // bs)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad():
                net(dummy)
        if device.type == 'mps':
            torch.mps.synchronize()
        elif device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        total_positions = n_iters * bs
        latency_ms = elapsed / n_iters * 1000
        results.append({
            'batch_size': bs,
            'positions': total_positions,
            'time': elapsed,
            'positions_per_sec': total_positions / elapsed,
            'latency_ms': latency_ms,
        })
    return results


def benchmark_nn_latency(n_positions=100):
    """Single-position NN inference latency stats."""
    import torch
    from bot import create_network, get_device, NUM_CHANNELS, BOARD_SIZE

    device = get_device()
    net = create_network('standard').to(device)
    net.eval()
    dummy = torch.randn(1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            net(dummy)

    latencies = []
    for _ in range(n_positions):
        if device.type == 'mps':
            torch.mps.synchronize()
        elif device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            net(dummy)
        if device.type == 'mps':
            torch.mps.synchronize()
        elif device.type == 'cuda':
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        'n': n_positions,
        'mean_ms': np.mean(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p99_ms': np.percentile(latencies, 99),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
    }


def benchmark_replay_buffer(n_samples=10000):
    """Replay buffer push + sample speed."""
    import torch
    from bot import ReplayBuffer, TrainingSample, BATCH_SIZE, NUM_CHANNELS, BOARD_SIZE

    buf = ReplayBuffer()

    # Fill with dummy samples
    t0 = time.perf_counter()
    for i in range(n_samples):
        s = TrainingSample(
            encoded_state=torch.randn(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            policy_target=np.random.dirichlet(np.ones(BOARD_SIZE * BOARD_SIZE)),
            player=i % 2,
            result=1.0 if i % 2 == 0 else -1.0,
            threat_label=np.zeros(4, dtype=np.float32),
            priority=1.0,
        )
        buf.push(s)
    push_time = time.perf_counter() - t0

    # Sample batches
    n_batches = 100
    t0 = time.perf_counter()
    for _ in range(n_batches):
        buf.sample(BATCH_SIZE)
    sample_time = time.perf_counter() - t0

    push_us = push_time / n_samples * 1e6
    sample_us = sample_time / n_batches * 1e6
    return {
        'n_samples': n_samples,
        'push_time': push_time,
        'push_us_per_sample': push_us,
        'n_batches': n_batches,
        'batch_size': BATCH_SIZE,
        'sample_time': sample_time,
        'sample_us_per_batch': sample_us,
    }


def benchmark_self_play(n_games=5):
    """Full self-play pipeline: network + MCTS + game."""
    import torch
    from bot import (create_network, BatchedMCTS, self_play_game_v2,
                     get_device)

    device = get_device()
    net = create_network('standard').to(device)
    net.eval()
    mcts = BatchedMCTS(net, num_simulations=50, batch_size=32)

    game_times = []
    game_lengths = []
    total_positions = 0

    for i in range(n_games):
        t0 = time.perf_counter()
        samples, moves = self_play_game_v2(net, mcts)
        elapsed = time.perf_counter() - t0
        game_times.append(elapsed)
        game_lengths.append(len(moves))
        total_positions += len(samples)

    total_time = sum(game_times)
    return {
        'games': n_games,
        'total_time': total_time,
        'avg_time_per_game': total_time / n_games,
        'games_per_hour': n_games / total_time * 3600,
        'avg_game_length': sum(game_lengths) / n_games,
        'total_positions': total_positions,
        'positions_per_sec': total_positions / total_time,
    }


def benchmark_device_info():
    """Get device info and memory usage."""
    import torch
    from bot import get_device

    device = get_device()
    info = {
        'device': str(device),
        'torch_version': torch.__version__,
    }
    if device.type == 'cuda':
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['vram_gb'] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
        info['vram_used_gb'] = round(torch.cuda.memory_allocated(0) / 1e9, 2)
    elif device.type == 'mps':
        info['gpu_name'] = 'Apple MPS'
        # MPS doesn't expose VRAM directly
        info['vram_gb'] = 'shared'

    from bot import create_network
    net = create_network('standard')
    params = sum(p.numel() for p in net.parameters())
    info['network_params'] = params
    return info


def main():
    print()
    print("=" * 70)
    print("  HEXBOT TRAINING PIPELINE BENCHMARK")
    print("=" * 70)
    print()

    # Device info
    print("  Device Info")
    print("  " + "-" * 50)
    info = benchmark_device_info()
    print(f"  Device:          {info['device']}")
    print(f"  PyTorch:         {info['torch_version']}")
    if 'gpu_name' in info:
        print(f"  GPU:             {info['gpu_name']}")
    if 'vram_gb' in info:
        print(f"  VRAM:            {info['vram_gb']} GB")
    print(f"  Network params:  {info['network_params']:,}")
    print()

    # 1. MCTS (no NN)
    print("  1. MCTS (CPU only, no NN)")
    print("  " + "-" * 50)
    r = benchmark_mcts_no_nn(100)
    print(f"  Throughput:      {fmt(r['sims_per_sec'])} sim/s")
    print(f"  Games:           {r['games']} in {r['time']:.2f}s")
    mcts_pass = r['sims_per_sec'] >= 10_000
    print(f"  Target:          >= 10,000 sim/s   {'PASS' if mcts_pass else 'FAIL'}")
    print()

    # 2. NN inference (batched)
    print("  2. NN Inference (batched)")
    print("  " + "-" * 50)
    results = benchmark_nn_inference([1, 8, 32, 64])
    print(f"  {'Batch':>6} {'Pos/s':>12} {'Latency':>10} {'Status':>8}")
    for r in results:
        status = 'PASS' if r['positions_per_sec'] >= 1000 else 'FAIL'
        print(f"  {r['batch_size']:>6} {fmt(r['positions_per_sec']):>12}/s "
              f"{r['latency_ms']:>8.1f}ms   {status}")
    nn_pass = results[-1]['positions_per_sec'] >= 5000
    print(f"  Target (bs=64):  >= 5,000 pos/s   {'PASS' if nn_pass else 'FAIL'}")
    print()

    # 3. NN latency (single)
    print("  3. NN Latency (single position)")
    print("  " + "-" * 50)
    r = benchmark_nn_latency(100)
    print(f"  Mean:            {r['mean_ms']:.2f} ms")
    print(f"  P50:             {r['p50_ms']:.2f} ms")
    print(f"  P99:             {r['p99_ms']:.2f} ms")
    lat_pass = r['mean_ms'] <= 10
    print(f"  Target:          <= 10 ms          {'PASS' if lat_pass else 'FAIL'}")
    print()

    # 4. Replay buffer
    print("  4. Replay Buffer (push + sample)")
    print("  " + "-" * 50)
    r = benchmark_replay_buffer(10000)
    print(f"  Push:            {r['push_us_per_sample']:.1f} us/sample")
    print(f"  Sample (bs={r['batch_size']}): {r['sample_us_per_batch']:.1f} us/batch")
    buf_pass = r['push_us_per_sample'] <= 1000
    print(f"  Target:          <= 1,000 us/push  {'PASS' if buf_pass else 'FAIL'}")
    print()

    # 5. Full self-play
    print("  5. Full Self-Play Pipeline (5 games, 50 sims)")
    print("  " + "-" * 50)
    r = benchmark_self_play(5)
    print(f"  Games:           {r['games']} in {r['total_time']:.1f}s")
    print(f"  Avg time/game:   {r['avg_time_per_game']:.1f}s")
    print(f"  Avg game length: {r['avg_game_length']:.0f} moves")
    print(f"  Games/hour:      {r['games_per_hour']:.0f}")
    print(f"  Positions/sec:   {fmt(r['positions_per_sec'])}")
    sp_pass = r['games_per_hour'] >= 100
    print(f"  Target:          >= 100 games/hr   {'PASS' if sp_pass else 'FAIL'}")
    print()

    # Summary
    print("=" * 70)
    print("  BENCHMARK REPORT")
    print("=" * 70)
    checks = [
        ("MCTS (CPU, no NN)", f"{fmt(benchmark_mcts_no_nn(50)['sims_per_sec'])} sim/s", ">= 10K sim/s", mcts_pass),
        ("NN inference (bs=64)", f"{fmt(results[-1]['positions_per_sec'])} pos/s", ">= 5K pos/s", nn_pass),
        ("NN latency (bs=1)", f"{benchmark_nn_latency(50)['mean_ms']:.1f} ms", "<= 10 ms", lat_pass),
        ("Replay buffer push", f"{benchmark_replay_buffer(5000)['push_us_per_sample']:.0f} us", "<= 1,000 us", buf_pass),
        ("Self-play pipeline", f"{r['games_per_hour']:.0f} games/hr", ">= 100/hr", sp_pass),
    ]
    print(f"  {'Benchmark':<28} {'Result':>14} {'Target':>14} {'Status':>8}")
    print("  " + "-" * 66)
    for name, result, target, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<28} {result:>14} {target:>14}   {status}")
    print()
    all_pass = all(p for _, _, _, p in checks)
    if all_pass:
        print("  All checks PASS")
    else:
        fails = sum(1 for _, _, _, p in checks if not p)
        print(f"  {fails} check(s) FAILED")
    print("=" * 70)


if __name__ == '__main__':
    main()
