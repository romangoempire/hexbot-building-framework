"""
Supervised Fine-Tuning (SFT) from game collections.

Train a network by imitating moves from strong games. Much faster than
self-play for bootstrapping - 100K games in ~15 minutes.

Usage:
    # CLI
    python -m orca.sft --data games.jsonl --epochs 10
    python -m orca.sft --data games.jsonl --epochs 5 --then-selfplay 50

    # Python
    from orca.sft import sft_train
    net = sft_train('games.jsonl', epochs=10)

    # Chain: SFT warmup then self-play refinement
    from orca.sft import sft_then_selfplay
    net = sft_then_selfplay('games.jsonl', sft_epochs=5, selfplay_iters=50)
"""

import argparse
import glob
import json
import os
import signal
import sys
import time
from datetime import datetime as _dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Ensure parent is importable
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from bot import (
    HexNet, ReplayBuffer, TrainingSample, get_device, create_network,
    encode_state, compute_threat_label, augment_sample,
    migrate_checkpoint_5to7, migrate_checkpoint_filters,
    BOARD_SIZE, NUM_CHANNELS, BATCH_SIZE, LEARNING_RATE, L2_REG,
)
from main import HexGame


# ---------------------------------------------------------------------------
# Game file parsers
# ---------------------------------------------------------------------------

def parse_jsonl(path: str, min_moves: int = 6, min_elo: int = 0,
                max_games: int = 999999) -> List[dict]:
    """Parse JSONL game file (hexo.did.science format).

    Each line: {"moves": [{"x": q, "y": r, "playerId": ..., "moveNumber": N}],
                "players": [{"elo": N}], "gameResult": {"winningPlayerId": ...}}

    Also supports simple format: {"moves": [[q,r], ...], "result": 1.0}
    """
    games = []
    with open(path, 'r') as f:
        for line in f:
            if len(games) >= max_games:
                break
            try:
                record = json.loads(line.strip())
            except (json.JSONDecodeError, ValueError):
                continue

            # Detect format
            moves = record.get("moves", [])
            if not moves or len(moves) < min_moves:
                continue

            # Simple format: [[q,r], [q,r], ...]
            if isinstance(moves[0], (list, tuple)):
                result = record.get("result", record.get("winner", 0))
                if isinstance(result, int) and result in (0, 1):
                    result = 1.0 if result == 0 else -1.0
                games.append({
                    "moves": [(m[0], m[1]) for m in moves],
                    "result": float(result),
                })
                continue

            # hexo.did.science format
            players = record.get("players", [])
            if min_elo > 0:
                elos = [p.get("elo", 0) or 0 for p in players]
                if elos and max(elos) < min_elo:
                    continue

            sorted_moves = sorted(moves, key=lambda m: m.get("moveNumber", 0))
            player_map = {}
            move_list = []
            for m in sorted_moves:
                pid = m.get("playerId", m.get("player_id", ""))
                if pid not in player_map:
                    player_map[pid] = len(player_map)
                move_list.append((m.get("x", m.get("q", 0)),
                                  m.get("y", m.get("r", 0))))

            game_result = record.get("gameResult", record.get("game_result", {}))
            winner_pid = game_result.get("winningPlayerId",
                                         game_result.get("winning_player_id"))
            winner = player_map.get(winner_pid)
            result = 1.0 if winner == 0 else (-1.0 if winner == 1 else 0.0)

            games.append({"moves": move_list, "result": result})

    return games


def parse_csv(path: str, min_moves: int = 6,
              max_games: int = 999999) -> List[dict]:
    """Parse CSV game file. Expected columns: game_id, move_num, q, r, result."""
    games = {}
    with open(path, 'r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            gid = parts[0]
            q, r = int(parts[2]), int(parts[3])
            result = float(parts[4]) if len(parts) > 4 else 0.0
            if gid not in games:
                games[gid] = {"moves": [], "result": result}
            games[gid]["moves"].append((q, r))

    return [g for g in games.values()
            if len(g["moves"]) >= min_moves][:max_games]


def parse_txt(path: str, min_moves: int = 6,
              max_games: int = 999999) -> List[dict]:
    """Parse plain text: one game per line, moves as 'q,r q,r q,r ...' result."""
    games = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            moves = []
            result = 0.0
            for p in parts:
                if ',' in p:
                    q, r = p.split(',')
                    moves.append((int(q), int(r)))
                else:
                    try:
                        result = float(p)
                    except ValueError:
                        pass
            if len(moves) >= min_moves:
                games.append({"moves": moves, "result": result})
            if len(games) >= max_games:
                break
    return games


def import_games(path: str, min_moves: int = 6, min_elo: int = 0,
                 max_games: int = 999999) -> List[dict]:
    """Auto-detect format and parse games from a file.

    Supports: JSONL, CSV, plain text.
    Returns list of {"moves": [(q,r), ...], "result": float}.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.jsonl', '.json', '.ndjson'):
        return parse_jsonl(path, min_moves, min_elo, max_games)
    elif ext == '.csv':
        return parse_csv(path, min_moves, max_games)
    elif ext in ('.txt', '.dat'):
        return parse_txt(path, min_moves, max_games)
    else:
        # Try JSONL first, fall back to text
        try:
            return parse_jsonl(path, min_moves, min_elo, max_games)
        except Exception:
            return parse_txt(path, min_moves, max_games)


# ---------------------------------------------------------------------------
# Convert games to training samples
# ---------------------------------------------------------------------------

def games_to_samples(games: List[dict],
                     include_threats: bool = True) -> List[TrainingSample]:
    """Convert parsed games to TrainingSample list for training.

    Each position in each game becomes a sample with:
    - encoded_state: the board position
    - policy_target: the move that was actually played (one-hot)
    - result: game outcome from this player's perspective
    - threat_label: threat features (optional)
    """
    samples = []
    for game_data in games:
        moves = game_data["moves"]
        result = game_data.get("result", 0.0)

        game = HexGame(candidate_radius=3, max_total_stones=300)
        game_samples = []
        valid = True

        for q, r in moves:
            player = game.current_player
            try:
                encoded, oq, orr = encode_state(game)
            except Exception:
                valid = False
                break

            # Policy: the played move is the target
            policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
            i, j = q - oq, r - orr
            if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
                policy[i * BOARD_SIZE + j] = 1.0
            else:
                # Move outside encoding window - skip sample, continue game
                try:
                    game.place_stone(q, r)
                except Exception:
                    valid = False
                    break
                continue

            threat = None
            if include_threats:
                try:
                    threat = compute_threat_label(game)
                except Exception:
                    threat = np.zeros(4, dtype=np.float32)

            game_samples.append(TrainingSample(
                encoded_state=encoded,
                policy_target=policy,
                player=player,
                threat_label=threat,
                priority=1.5,
            ))

            try:
                game.place_stone(q, r)
            except Exception:
                valid = False
                break

        if not valid or not game_samples:
            continue

        # Fill results
        for s in game_samples:
            if result > 0:
                s.result = 1.0 if s.player == 0 else -1.0
            elif result < 0:
                s.result = -1.0 if s.player == 0 else 1.0
            else:
                s.result = 0.0

        samples.extend(game_samples)

    return samples


# ---------------------------------------------------------------------------
# SFT training step (policy + value, no MCTS)
# ---------------------------------------------------------------------------

def sft_step(net, optimizer, samples: List[TrainingSample],
             device, batch_size: int = 512) -> Dict[str, float]:
    """One SFT gradient step. Policy cross-entropy + value MSE."""
    net.train()
    n = len(samples)
    indices = np.random.choice(n, size=min(batch_size, n), replace=False)
    batch = [samples[i] for i in indices]

    states = torch.zeros(len(batch), NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    policies = np.empty((len(batch), BOARD_SIZE * BOARD_SIZE), dtype=np.float32)
    values = np.empty(len(batch), dtype=np.float32)
    threats = np.empty((len(batch), 4), dtype=np.float32)

    for i, s in enumerate(batch):
        t = s.encoded_state
        c = t.shape[0]
        states[i, :c] = t
        policies[i] = s.policy_target
        values[i] = s.result
        threats[i] = s.threat_label if s.threat_label is not None else np.zeros(4)

    states = states.to(device, non_blocking=True)
    target_pol = torch.from_numpy(policies).to(device, non_blocking=True)
    target_val = torch.from_numpy(values).to(device, non_blocking=True)
    target_thr = torch.from_numpy(threats).to(device, non_blocking=True)

    policy_logits, value_preds, threat_preds = net(states)
    value_preds = value_preds.squeeze(-1)

    # Losses
    value_loss = F.mse_loss(value_preds, target_val)
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -torch.sum(target_pol * log_probs, dim=1).mean()
    threat_loss = F.binary_cross_entropy_with_logits(threat_preds, target_thr)

    loss = value_loss + policy_loss + 0.5 * threat_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        'total': loss.item(),
        'value': value_loss.item(),
        'policy': policy_loss.item(),
        'threat': threat_loss.item(),
    }


# ---------------------------------------------------------------------------
# Main SFT function
# ---------------------------------------------------------------------------

def sft_train(
    data_path: str,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    net_config: str = 'standard',
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    augment: bool = True,
    min_elo: int = 0,
    max_games: int = 999999,
    save_path: str = 'sft_checkpoint.pt',
    verbose: bool = True,
) -> HexNet:
    """Train a network via supervised fine-tuning on a game collection.

    Args:
        data_path: path to game file (JSONL, CSV, or text)
        epochs: training epochs over the dataset
        batch_size: training batch size
        lr: learning rate
        weight_decay: L2 regularization
        net_config: network architecture ('standard', 'fast', etc.)
        checkpoint_path: resume from checkpoint (None = fresh or auto-detect)
        device: torch device (auto-detect if None)
        augment: apply hex symmetry augmentation (3x data)
        min_elo: minimum ELO filter for games
        max_games: maximum games to load
        save_path: where to save the trained model
        verbose: print progress

    Returns:
        Trained network.
    """
    dev = torch.device(device) if device else get_device()

    # Load or create network
    net = create_network(net_config).to(dev)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        sd = migrate_checkpoint_5to7(sd)
        sd = migrate_checkpoint_filters(sd)
        net.load_state_dict(sd, strict=False)
        if verbose:
            print(f'Loaded checkpoint: {checkpoint_path}')
    elif checkpoint_path is None:
        # Auto-detect: try orca/checkpoint.pt, pretrained.pt, latest hex_checkpoint
        for p in ['orca/checkpoint.pt', 'pretrained.pt']:
            if os.path.exists(p):
                ckpt = torch.load(p, map_location='cpu', weights_only=False)
                sd = ckpt.get('model_state_dict', ckpt)
                sd = migrate_checkpoint_5to7(sd)
                sd = migrate_checkpoint_filters(sd)
                net.load_state_dict(sd, strict=False)
                if verbose:
                    print(f'Loaded checkpoint: {p}')
                break

    params = sum(p.numel() for p in net.parameters())
    if verbose:
        print(f'Network: {params:,} params on {dev}')

    # Load games
    if verbose:
        print(f'Loading games from {data_path}...')
    games = import_games(data_path, min_elo=min_elo, max_games=max_games)
    if not games:
        raise ValueError(f'No games loaded from {data_path}')
    if verbose:
        print(f'Loaded {len(games)} games')

    # Convert to samples
    if verbose:
        print(f'Converting to training samples...')
    t0 = time.perf_counter()
    samples = games_to_samples(games)
    if verbose:
        print(f'  {len(samples)} samples from {len(games)} games ({time.perf_counter()-t0:.1f}s)')

    # Augment
    if augment:
        t0 = time.perf_counter()
        augmented = []
        for s in samples:
            augmented.extend(augment_sample(s))
        samples.extend(augmented)
        if verbose:
            print(f'  +{len(augmented)} augmented = {len(samples)} total ({time.perf_counter()-t0:.1f}s)')

    # Train
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(samples) // batch_size)

    if verbose:
        print(f'\nSFT Training: {epochs} epochs, {steps_per_epoch} steps/epoch, '
              f'batch={batch_size}, lr={lr}')

    for epoch in range(epochs):
        t0 = time.perf_counter()
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            losses = sft_step(net, optimizer, samples, dev, batch_size)
            epoch_loss += losses['total']

        avg_loss = epoch_loss / steps_per_epoch
        elapsed = time.perf_counter() - t0
        if verbose:
            print(f'  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} '
                  f'(v={losses["value"]:.4f} p={losses["policy"]:.4f}) '
                  f'{elapsed:.1f}s')

    # Save
    torch.save({
        'model_state_dict': net.state_dict(),
        'sft_config': {
            'data_path': data_path,
            'epochs': epochs,
            'games': len(games),
            'samples': len(samples),
            'net_config': net_config,
        },
    }, save_path)
    if verbose:
        print(f'\nSaved to {save_path}')

    return net


def sft_then_selfplay(
    data_path: str,
    sft_epochs: int = 5,
    selfplay_iters: int = 50,
    **kwargs,
):
    """Chain SFT warmup with self-play refinement.

    First trains on the game collection, then continues with
    AlphaZero-style self-play for further improvement.
    """
    # Phase 1: SFT
    save_path = kwargs.pop('save_path', 'sft_checkpoint.pt')
    net = sft_train(data_path, epochs=sft_epochs, save_path=save_path, **kwargs)

    # Phase 2: Self-play
    print(f'\n--- Switching to self-play ({selfplay_iters} iterations) ---\n')
    from orca.train import OrcaTrainer
    trainer = OrcaTrainer(
        iterations=selfplay_iters,
        resume=True,  # will pick up sft_checkpoint.pt
    )
    trainer.run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Orca SFT - Supervised Fine-Tuning from game collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m orca.sft --data games.jsonl --epochs 10
  python -m orca.sft --data games.jsonl --epochs 5 --then-selfplay 50
  python -m orca.sft --data games.jsonl --min-elo 1200 --batch-size 1024
  python -m orca.sft --data games.jsonl --config orca-transformer
        """)
    parser.add_argument("--data", required=True, help="Path to game file (JSONL, CSV, text)")
    parser.add_argument("--epochs", type=int, default=10, help="SFT training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size (default: 512)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--config", type=str, default="standard", help="Network config")
    parser.add_argument("--device", type=str, default=None, help="Device (auto if omitted)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--min-elo", type=int, default=0, help="Min ELO filter for games")
    parser.add_argument("--max-games", type=int, default=999999, help="Max games to load")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    parser.add_argument("--save", type=str, default="sft_checkpoint.pt", help="Save path")
    parser.add_argument("--then-selfplay", type=int, default=0,
                        help="Continue with N iterations of self-play after SFT")
    args = parser.parse_args()

    if args.then_selfplay > 0:
        sft_then_selfplay(
            args.data,
            sft_epochs=args.epochs,
            selfplay_iters=args.then_selfplay,
            batch_size=args.batch_size,
            lr=args.lr,
            net_config=args.config,
            device=args.device,
            checkpoint_path=args.checkpoint,
            augment=not args.no_augment,
            min_elo=args.min_elo,
            max_games=args.max_games,
            save_path=args.save,
        )
    else:
        sft_train(
            args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            net_config=args.config,
            device=args.device,
            checkpoint_path=args.checkpoint,
            augment=not args.no_augment,
            min_elo=args.min_elo,
            max_games=args.max_games,
            save_path=args.save,
        )


if __name__ == '__main__':
    main()
