"""
Leaderboard - community bot ranking system.

Submit bots for rating against reference opponents.
View rankings and compare performance.

Usage:
    from orca.leaderboard import Leaderboard

    # Rate a bot locally against references
    lb = Leaderboard()
    lb.rate(my_bot, name='phoenix-v1', games=50)
    lb.show()

    # Compare two bots
    lb.compare(bot_a, bot_b, games=20)
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

LEADERBOARD_FILE = 'leaderboard.json'


class Leaderboard:
    """Local leaderboard for rating bots against reference opponents.

    Stores ratings in a JSON file. Each bot is rated by playing against
    a set of reference bots (random, heuristic, orca) and computing
    ELO from win rates.
    """

    def __init__(self, path: str = LEADERBOARD_FILE):
        self.path = path
        self.entries: List[Dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    self.entries = json.load(f)
            except Exception:
                self.entries = []

    def _save(self):
        with open(self.path, 'w') as f:
            json.dump(self.entries, f, indent=2)

    def rate(self, bot, name: str = 'unnamed', games: int = 50,
             verbose: bool = True) -> Dict:
        """Rate a bot against reference opponents.

        Plays against random, heuristic, and (if available) orca bots.
        Returns dict with ELO estimate and detailed results.

        Example:
            lb = Leaderboard()
            result = lb.rate(my_bot, name='my-bot', games=30)
            print(f"ELO: {result['elo']}")
        """
        from hexbot import Bot, Arena

        import math

        results = {}
        total_wins, total_games_played = 0, 0

        # Reference opponents with expected strength
        refs = [
            ('random', Bot.random(), 500),
            ('heuristic', Bot.heuristic(), 1000),
        ]
        try:
            refs.append(('orca', Bot.orca(sims=50), 1200))
        except Exception:
            pass

        for ref_name, ref_bot, ref_elo in refs:
            games_per = max(4, games // len(refs))
            if verbose:
                print(f"  vs {ref_name} ({games_per} games)...", end=' ', flush=True)
            result = Arena(bot, ref_bot, num_games=games_per).play(verbose=False)
            wins = result.wins[0]
            total = result.total_games
            wr = wins / max(total, 1)
            if verbose:
                print(f"{wins}W-{result.wins[1]}L ({wr:.0%})")
            results[ref_name] = {
                'wins': wins, 'losses': result.wins[1],
                'draws': result.draws, 'total': total, 'win_rate': wr,
                'ref_elo': ref_elo,
            }
            total_wins += wins
            total_games_played += total

        # Compute ELO from weighted results
        elo = 1000  # start at baseline
        for ref_name, ref_data in results.items():
            wr = ref_data['win_rate']
            wr = max(0.01, min(0.99, wr))
            ref_elo = ref_data['ref_elo']
            # ELO from expected score formula
            elo_vs_ref = ref_elo + 400 * math.log10(wr / (1 - wr))
            weight = ref_data['total'] / max(total_games_played, 1)
            elo = elo * (1 - weight) + elo_vs_ref * weight

        elo = round(elo)

        entry = {
            'name': name,
            'elo': elo,
            'results': results,
            'total_wins': total_wins,
            'total_games': total_games_played,
            'overall_win_rate': total_wins / max(total_games_played, 1),
            'rated_at': time.time(),
        }

        # Update or add entry
        existing = [i for i, e in enumerate(self.entries) if e['name'] == name]
        if existing:
            self.entries[existing[0]] = entry
        else:
            self.entries.append(entry)
        self.entries.sort(key=lambda e: e['elo'], reverse=True)
        self._save()

        if verbose:
            print(f"\n  {name}: ELO {elo}")
        return entry

    def compare(self, bot_a, bot_b, name_a: str = 'bot_a',
                name_b: str = 'bot_b', games: int = 20,
                verbose: bool = True) -> Dict:
        """Direct comparison between two bots.

        Returns dict with win/loss/draw and ELO difference estimate.
        """
        from hexbot import Arena
        import math

        result = Arena(bot_a, bot_b, num_games=games).play(verbose=verbose)
        wr = result.wins[0] / max(result.total_games, 1)
        wr = max(0.01, min(0.99, wr))
        elo_diff = round(400 * math.log10(wr / (1 - wr)))

        comparison = {
            'bot_a': name_a, 'bot_b': name_b,
            'wins_a': result.wins[0], 'wins_b': result.wins[1],
            'draws': result.draws, 'total': result.total_games,
            'win_rate_a': wr, 'elo_diff': elo_diff,
        }

        if verbose:
            sign = '+' if elo_diff >= 0 else ''
            print(f"\n{name_a} vs {name_b}: {sign}{elo_diff} ELO")

        return comparison

    def show(self, top_n: int = 20):
        """Display the leaderboard."""
        if not self.entries:
            print("Leaderboard is empty. Rate a bot with: lb.rate(my_bot, 'name')")
            return

        print(f"\n{'Rank':<6} {'Name':<25} {'ELO':>6} {'W/L':>10} {'Win%':>6}")
        print("-" * 55)
        for i, e in enumerate(self.entries[:top_n]):
            wl = f"{e['total_wins']}/{e['total_games'] - e['total_wins']}"
            wr = f"{e['overall_win_rate']:.0%}"
            print(f"{i+1:<6} {e['name']:<25} {e['elo']:>6} {wl:>10} {wr:>6}")
        print()

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return f'Leaderboard({len(self.entries)} bots)'
