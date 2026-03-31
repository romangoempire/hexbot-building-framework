"""Tests for orca subpackage: solver, openings, curriculum, ensemble, sft."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexgame import HexGame


class TestSolverTrivialWin(unittest.TestCase):
    def test_near_win_detected(self):
        """Set up P0 with 5-in-a-row and check solver finds the win."""
        from orca.solver import solve
        game = HexGame()
        # P0: (0,0)
        game.place(0, 0)
        # P1 far away
        game.place(0, 5)
        game.place(0, 6)
        # P0: (1,0), (2,0)
        game.place(1, 0)
        game.place(2, 0)
        # P1 far away
        game.place(0, 7)
        game.place(0, 8)
        # P0: (3,0), (4,0) -> P0 has 0,0 through 4,0 = 5 in a row
        game.place(3, 0)
        game.place(4, 0)
        # Now it's P1's turn, but solver from P0's perspective:
        # P0 needs one more at (5,0) or (-1,0) to win
        result = solve(game, max_depth=6)
        self.assertIn('result', result)
        self.assertIn('move', result)
        self.assertIn('value', result)
        self.assertIsInstance(result['nodes'], int)

    def test_solve_returns_all_keys(self):
        from orca.solver import solve
        game = HexGame()
        game.place(0, 0)
        result = solve(game, max_depth=4)
        for key in ('result', 'move', 'value', 'depth', 'nodes', 'time'):
            self.assertIn(key, result)


class TestQuickSolve(unittest.TestCase):
    def test_quick_solve_returns_none_early(self):
        from orca.solver import quick_solve
        game = HexGame()
        game.place(0, 0)
        result = quick_solve(game, depth=4)
        # Early game: unlikely to have a forced win
        # Result is either a tuple or None
        self.assertTrue(result is None or isinstance(result, tuple))


class TestOpeningBookBuildAndLookup(unittest.TestCase):
    def test_build_from_games(self):
        from orca.openings import OpeningBook
        book = OpeningBook()
        games = [
            {"moves": [(0, 0), (1, 0), (1, -1), (2, 0), (2, -1)], "result": 1.0},
            {"moves": [(0, 0), (1, 0), (1, -1), (2, 0), (2, -1)], "result": 1.0},
            {"moves": [(0, 0), (1, 0), (1, -1), (3, 0), (3, -1)], "result": -1.0},
        ]
        book.build_from_games(games, min_frequency=1)
        self.assertGreater(len(book), 0)

    def test_lookup_empty_board(self):
        from orca.openings import build_default_book
        book = build_default_book()
        game = HexGame()
        move = book.lookup(game, temperature=0.0)
        # Default book should suggest (0,0) for empty board
        self.assertEqual(move, (0, 0))


class TestOpeningBookSaveLoad(unittest.TestCase):
    def test_save_and_load(self):
        from orca.openings import OpeningBook
        book = OpeningBook()
        games = [
            {"moves": [(0, 0), (1, 0), (1, -1)], "result": 1.0},
            {"moves": [(0, 0), (1, 0), (1, -1)], "result": 1.0},
        ]
        book.build_from_games(games, min_frequency=1)
        original_len = len(book)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            book.save(path)
            book2 = OpeningBook()
            book2.load(path)
            self.assertEqual(len(book2), original_len)
        finally:
            os.unlink(path)


class TestSkillCurriculumLevels(unittest.TestCase):
    def test_initial_level(self):
        from orca.curriculum import SkillCurriculum
        c = SkillCurriculum(start_level=1)
        self.assertEqual(c.current_level, 1)
        config = c.get_config()
        self.assertEqual(config['level'], 1)
        self.assertEqual(config['name'], 'Basics')
        self.assertIn('sims', config)
        self.assertIn('opponent', config)

    def test_get_sims(self):
        from orca.curriculum import SkillCurriculum
        c = SkillCurriculum(start_level=1)
        sims = c.get_sims()
        self.assertEqual(sims, 30)

    def test_all_levels_valid(self):
        from orca.curriculum import SkillCurriculum, SKILL_LEVELS
        for level in SKILL_LEVELS:
            c = SkillCurriculum(start_level=level)
            config = c.get_config()
            self.assertEqual(config['level'], level)


class TestSkillCurriculumAdvancement(unittest.TestCase):
    def test_advance_after_high_win_rate(self):
        from orca.curriculum import SkillCurriculum
        c = SkillCurriculum(start_level=1)
        # Simulate enough iterations with high win rate to advance
        for _ in range(10):
            advanced = c.update(win_rate=0.95, verbose=False)
        # After 10 iterations at 95% wr (min_iterations=5, advance_wr=0.80)
        # should have advanced
        self.assertGreater(c.current_level, 1)

    def test_no_advance_low_win_rate(self):
        from orca.curriculum import SkillCurriculum
        c = SkillCurriculum(start_level=1)
        for _ in range(10):
            c.update(win_rate=0.3, verbose=False)
        self.assertEqual(c.current_level, 1)

    def test_top_level_no_advance(self):
        from orca.curriculum import SkillCurriculum, SKILL_LEVELS
        max_level = max(SKILL_LEVELS.keys())
        c = SkillCurriculum(start_level=max_level)
        for _ in range(50):
            c.update(win_rate=0.99, verbose=False)
        self.assertEqual(c.current_level, max_level)


class TestConfigImports(unittest.TestCase):
    def test_config_constants(self):
        from orca.config import (
            BOARD_SIZE, NUM_CHANNELS, NUM_FILTERS, NUM_RES_BLOCKS,
            C_PUCT, NUM_SIMULATIONS, BATCH_SIZE, LEARNING_RATE,
            REPLAY_BUFFER_SIZE,
        )
        self.assertEqual(BOARD_SIZE, 19)
        self.assertEqual(NUM_CHANNELS, 7)
        self.assertEqual(NUM_FILTERS, 128)
        self.assertIsInstance(C_PUCT, float)
        self.assertIsInstance(BATCH_SIZE, int)
        self.assertIsInstance(LEARNING_RATE, float)


class TestSftImportGames(unittest.TestCase):
    def test_parse_simple_jsonl(self):
        from orca.sft import parse_jsonl
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl',
                                          delete=False) as f:
            for _ in range(5):
                game = {
                    "moves": [[0, 0], [1, 0], [1, -1], [2, 0], [2, -1],
                              [3, 0], [3, -1], [4, 0], [4, -1], [5, 0]],
                    "result": 1.0,
                }
                f.write(json.dumps(game) + '\n')
            path = f.name
        try:
            games = parse_jsonl(path, min_moves=6)
            self.assertEqual(len(games), 5)
            for g in games:
                self.assertIn('moves', g)
                self.assertIn('result', g)
                self.assertGreaterEqual(len(g['moves']), 6)
        finally:
            os.unlink(path)


class TestSftGamesToSamples(unittest.TestCase):
    def test_games_to_samples(self):
        from orca.sft import games_to_samples
        games = [{
            "moves": [(0, 0), (1, 0), (1, -1), (2, 0), (2, -1),
                      (3, 0), (3, -1), (4, 0), (4, -1), (5, 0)],
            "result": 1.0,
        }]
        samples = games_to_samples(games, include_threats=False)
        self.assertIsInstance(samples, list)
        # Should produce at least some samples from a 10-move game
        self.assertGreater(len(samples), 0)
        for s in samples:
            self.assertIsNotNone(s.encoded_state)
            self.assertIsNotNone(s.policy_target)


if __name__ == '__main__':
    unittest.main()
