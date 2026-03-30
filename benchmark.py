#!/usr/bin/env python3
"""
benchmark.py — Performance benchmarks for the Hex Connect-6 C engine.

Run:
    python benchmark.py
"""

import random
import time
from hexgame import HexGame


def benchmark_random_games(n_games: int = 1000) -> dict:
    """Benchmark random game playouts."""
    total_stones = 0
    total_moves = 0
    winners = [0, 0, 0]  # P0, P1, draw

    t0 = time.perf_counter()
    for _ in range(n_games):
        game = HexGame(max_stones=150)
        rng = random.Random(42 + _)
        while not game.is_over:
            moves = game.legal_moves()
            game.place(*rng.choice(moves))
            total_moves += 1
        total_stones += game.total_stones
        if game.winner == 0:
            winners[0] += 1
        elif game.winner == 1:
            winners[1] += 1
        else:
            winners[2] += 1
    elapsed = time.perf_counter() - t0

    return {
        'games': n_games,
        'total_stones': total_stones,
        'avg_game_length': total_stones / n_games,
        'time': elapsed,
        'games_per_sec': n_games / elapsed,
        'stones_per_sec': total_stones / elapsed,
        'p0_wins': winners[0],
        'p1_wins': winners[1],
        'draws': winners[2],
    }


def benchmark_scored_moves(n_positions: int = 10000) -> dict:
    """Benchmark move scoring from random positions."""
    t0 = time.perf_counter()
    total_calls = 0
    for i in range(n_positions):
        game = HexGame()
        rng = random.Random(i)
        # Play 5-15 random moves to get a mid-game position
        n_moves = rng.randint(5, 15)
        for _ in range(n_moves):
            if game.is_over:
                break
            moves = game.legal_moves()
            game.place(*rng.choice(moves))
        if not game.is_over:
            game.scored_moves(20)
            total_calls += 1
    elapsed = time.perf_counter() - t0

    return {
        'positions': total_calls,
        'time': elapsed,
        'positions_per_sec': total_calls / elapsed,
    }


def benchmark_search(depths: list = None) -> list:
    """Benchmark alpha-beta search at various depths."""
    if depths is None:
        depths = [4, 6, 8]

    game = HexGame()
    game.place(0, 0)
    game.place(2, 0)
    game.place(2, -1)
    game.place(1, 0)
    game.place(0, 1)

    results = []
    for depth in depths:
        # Reset game to same position each time
        g = HexGame.from_moves(game.moves)
        t0 = time.perf_counter()
        result = g.search(depth=depth)
        elapsed = time.perf_counter() - t0

        results.append({
            'depth': depth,
            'turns_ahead': depth // 2,
            'nodes': result['nodes'],
            'time': elapsed,
            'nodes_per_sec': result['nodes'] / elapsed if elapsed > 0 else 0,
            'best_move': result['best_move'],
            'value': result['value'],
        })

    return results


def benchmark_clone(n_clones: int = 100000) -> dict:
    """Benchmark game cloning speed."""
    game = HexGame()
    for q, r in [(0,0), (1,0), (1,-1), (0,1), (-1,1)]:
        game.place(q, r)

    t0 = time.perf_counter()
    for _ in range(n_clones):
        _ = game.clone()
    elapsed = time.perf_counter() - t0

    return {
        'clones': n_clones,
        'time': elapsed,
        'clones_per_sec': n_clones / elapsed,
    }


def benchmark_undo(n_ops: int = 100000) -> dict:
    """Benchmark place + undo cycle speed."""
    game = HexGame()
    game.place(0, 0)
    game.place(1, 0)
    game.place(1, -1)

    moves = game.legal_moves()[:10]

    t0 = time.perf_counter()
    for i in range(n_ops):
        m = moves[i % len(moves)]
        game.place(*m)
        game.undo()
    elapsed = time.perf_counter() - t0

    return {
        'operations': n_ops,
        'time': elapsed,
        'ops_per_sec': n_ops / elapsed,
    }


def benchmark_threat_detection(n_checks: int = 50000) -> dict:
    """Benchmark winning move detection speed."""
    game = HexGame()
    # Set up a near-win position
    for q, r in [(0,0), (5,0), (5,-1), (1,0), (2,0), (6,0), (6,-1), (3,0), (4,0)]:
        game.place(q, r)
    # P0 has 5 in a row: (0,0),(1,0),(2,0),(3,0),(4,0) — needs (5,0) but that's P1's

    t0 = time.perf_counter()
    for _ in range(n_checks):
        game.has_winning_move(0)
        game.has_winning_move(1)
        game.count_winning_moves(0)
    elapsed = time.perf_counter() - t0

    return {
        'checks': n_checks * 3,
        'time': elapsed,
        'checks_per_sec': (n_checks * 3) / elapsed,
    }


def format_number(n: float) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:.1f}"


def main():
    print("=" * 65)
    print("  HEX CONNECT-6 ENGINE BENCHMARKS")
    print("=" * 65)
    print()

    # 1. Random games
    print("1. Random Game Playouts (1,000 games)")
    print("-" * 45)
    r = benchmark_random_games(1000)
    print(f"   Games:        {r['games']:,}")
    print(f"   Total stones: {r['total_stones']:,}")
    print(f"   Avg length:   {r['avg_game_length']:.1f} moves/game")
    print(f"   Time:         {r['time']:.2f}s")
    print(f"   Throughput:   {format_number(r['games_per_sec'])} games/sec")
    print(f"   Throughput:   {format_number(r['stones_per_sec'])} stones/sec")
    print(f"   P0 wins:      {r['p0_wins']} ({r['p0_wins']/r['games']*100:.1f}%)")
    print(f"   P1 wins:      {r['p1_wins']} ({r['p1_wins']/r['games']*100:.1f}%)")
    print(f"   Draws:        {r['draws']}")
    print()

    # 2. Move scoring
    print("2. Move Scoring (10,000 positions)")
    print("-" * 45)
    r = benchmark_scored_moves(10000)
    print(f"   Positions:    {r['positions']:,}")
    print(f"   Time:         {r['time']:.2f}s")
    print(f"   Throughput:   {format_number(r['positions_per_sec'])} positions/sec")
    print()

    # 3. Alpha-beta search
    print("3. Alpha-Beta Search (from triangle position)")
    print("-" * 45)
    results = benchmark_search([4, 6, 8, 10])
    print(f"   {'Depth':>5} {'Turns':>5} {'Nodes':>12} {'Time':>8} {'Nodes/s':>12} {'Best':>8} {'Value':>7}")
    for r in results:
        print(f"   {r['depth']:>5} {r['turns_ahead']:>5} {r['nodes']:>12,} "
              f"{r['time']:>7.2f}s {format_number(r['nodes_per_sec']):>12}/s "
              f"{str(r['best_move']):>8} {r['value']:>+6.3f}")
    print()

    # 4. Clone speed
    print("4. Clone Speed (100,000 clones)")
    print("-" * 45)
    r = benchmark_clone(100000)
    print(f"   Clones:       {r['clones']:,}")
    print(f"   Time:         {r['time']:.2f}s")
    print(f"   Throughput:   {format_number(r['clones_per_sec'])} clones/sec")
    print()

    # 5. Place + Undo cycle
    print("5. Place/Undo Cycle (100,000 ops)")
    print("-" * 45)
    r = benchmark_undo(100000)
    print(f"   Operations:   {r['operations']:,}")
    print(f"   Time:         {r['time']:.2f}s")
    print(f"   Throughput:   {format_number(r['ops_per_sec'])} ops/sec")
    print()

    # 6. Threat detection
    print("6. Threat Detection (150,000 checks)")
    print("-" * 45)
    r = benchmark_threat_detection(50000)
    print(f"   Checks:       {r['checks']:,}")
    print(f"   Time:         {r['time']:.2f}s")
    print(f"   Throughput:   {format_number(r['checks_per_sec'])} checks/sec")
    print()

    print("=" * 65)
    print("  DONE")
    print("=" * 65)


if __name__ == '__main__':
    main()
