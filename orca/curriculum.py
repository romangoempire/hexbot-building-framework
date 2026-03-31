"""
Skill-based training curriculum for Orca.

Instead of just scaling simulations, this curriculum progresses through
skill levels based on demonstrated competence. Auto-advances when the
bot achieves >80% win rate at each level.

Levels:
    1. Basics      - place stones, extend lines
    2. Blocking    - detect and block opponent threats
    3. Tactics     - create 4-in-a-row threats, simple forks
    4. Forks       - double threats, unstoppable patterns
    5. Colony      - distant play, multi-cluster strategy
    6. Endgame     - deep tactical sequences, proof-number situations

Usage:
    from orca.curriculum import SkillCurriculum

    curriculum = SkillCurriculum()
    level, positions = curriculum.get_training_config()

    # After each iteration:
    curriculum.update(win_rate=0.85, avg_game_length=25)
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)


SKILL_LEVELS = {
    1: {
        'name': 'Basics',
        'description': 'Place stones near each other, extend lines',
        'sims': 30,
        'opponent': 'random',
        'advance_win_rate': 0.80,
        'min_iterations': 5,
    },
    2: {
        'name': 'Blocking',
        'description': 'Detect and block opponent 4+ threats',
        'sims': 50,
        'opponent': 'heuristic',
        'advance_win_rate': 0.60,
        'min_iterations': 10,
    },
    3: {
        'name': 'Tactics',
        'description': 'Create 4-in-a-row threats, simple attacking patterns',
        'sims': 100,
        'opponent': 'self',
        'advance_win_rate': 0.55,
        'min_iterations': 15,
    },
    4: {
        'name': 'Forks',
        'description': 'Double threats, unstoppable attack patterns',
        'sims': 150,
        'opponent': 'self',
        'advance_win_rate': 0.55,
        'min_iterations': 20,
    },
    5: {
        'name': 'Colony',
        'description': 'Distant play, multi-cluster strategy',
        'sims': 200,
        'opponent': 'self',
        'advance_win_rate': 0.55,
        'min_iterations': 30,
    },
    6: {
        'name': 'Endgame',
        'description': 'Deep tactical sequences, proof-number situations',
        'sims': 200,
        'opponent': 'self',
        'advance_win_rate': None,  # never auto-advance (top level)
        'min_iterations': None,
    },
}


class SkillCurriculum:
    """Skill-based training curriculum with auto-progression.

    Tracks win rates per level and advances when the bot demonstrates
    competence. Each level increases search depth and opponent difficulty.
    """

    def __init__(self, start_level: int = 1):
        self.current_level = start_level
        self.iterations_at_level = 0
        self.win_rates: List[float] = []
        self.level_history: List[Dict] = []

    def get_config(self) -> Dict:
        """Get current training configuration for this skill level.

        Returns dict with: level, name, sims, opponent, description.
        """
        level_cfg = SKILL_LEVELS[self.current_level]
        return {
            'level': self.current_level,
            'name': level_cfg['name'],
            'sims': level_cfg['sims'],
            'opponent': level_cfg['opponent'],
            'description': level_cfg['description'],
            'iterations_at_level': self.iterations_at_level,
        }

    def get_sims(self) -> int:
        """Get MCTS simulation count for current level."""
        return SKILL_LEVELS[self.current_level]['sims']

    def get_opponent_type(self) -> str:
        """Get opponent type for current level ('random', 'heuristic', 'self')."""
        return SKILL_LEVELS[self.current_level]['opponent']

    def update(self, win_rate: float, avg_game_length: float = 0,
               verbose: bool = True) -> bool:
        """Update curriculum with latest results. Returns True if level advanced.

        Args:
            win_rate: P0 win rate this iteration (0-1)
            avg_game_length: average moves per game
            verbose: print level changes
        """
        self.iterations_at_level += 1
        self.win_rates.append(win_rate)

        level_cfg = SKILL_LEVELS[self.current_level]
        advance_wr = level_cfg['advance_win_rate']
        min_iters = level_cfg['min_iterations']

        # Check for advancement
        if (advance_wr is not None
                and min_iters is not None
                and self.iterations_at_level >= min_iters):
            # Check recent win rate (last 5 iterations)
            recent = self.win_rates[-5:] if len(self.win_rates) >= 5 else self.win_rates
            avg_recent = sum(recent) / len(recent)

            if avg_recent >= advance_wr and self.current_level < max(SKILL_LEVELS.keys()):
                old_level = self.current_level
                self.level_history.append({
                    'from_level': old_level,
                    'to_level': old_level + 1,
                    'iterations': self.iterations_at_level,
                    'final_win_rate': avg_recent,
                })
                self.current_level += 1
                self.iterations_at_level = 0
                self.win_rates = []
                if verbose:
                    new_cfg = SKILL_LEVELS[self.current_level]
                    print(f'  CURRICULUM: Level {old_level} -> {self.current_level} '
                          f'({new_cfg["name"]}): {new_cfg["description"]}')
                return True

        return False

    def get_status(self) -> str:
        """Human-readable status string."""
        cfg = SKILL_LEVELS[self.current_level]
        wr = f'{self.win_rates[-1]:.0%}' if self.win_rates else '---'
        return (f'Level {self.current_level}/{max(SKILL_LEVELS.keys())} '
                f'({cfg["name"]}) iter={self.iterations_at_level} wr={wr}')

    def __repr__(self):
        return f'SkillCurriculum(level={self.current_level}, iters={self.iterations_at_level})'
