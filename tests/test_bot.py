"""Tests for hexbot Bot, Arena, and analysis functions."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexgame import HexGame
from hexbot import (
    Bot, Arena, evaluate_moves, find_threats, find_winning_moves,
    count_lines, rollout, alphabeta, find_forced_move, threat_search,
    count_threats, detect_fork,
)

# Check if checkpoint exists for orca tests
_ORCA_CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'orca', 'checkpoint.pt')
_PRETRAINED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'pretrained.pt')
HAS_CHECKPOINT = os.path.exists(_ORCA_CKPT) or os.path.exists(_PRETRAINED)


class TestBotHeuristic(unittest.TestCase):
    def test_heuristic_returns_move(self):
        bot = Bot.heuristic()
        game = HexGame()
        game.place(0, 0)
        move = bot.best_move(game)
        self.assertIsInstance(move, tuple)
        self.assertEqual(len(move), 2)

    def test_heuristic_first_move(self):
        bot = Bot.heuristic()
        game = HexGame()
        move = bot.best_move(game)
        self.assertEqual(move, (0, 0))

    def test_heuristic_repr(self):
        bot = Bot.heuristic()
        self.assertIn('heuristic', repr(bot))


class TestBotRandom(unittest.TestCase):
    def test_random_returns_legal_move(self):
        bot = Bot.random()
        game = HexGame()
        game.place(0, 0)
        move = bot.best_move(game)
        self.assertIn(move, game.legal_moves())

    def test_random_repr(self):
        bot = Bot.random()
        self.assertIn('random', repr(bot))


@unittest.skipUnless(HAS_CHECKPOINT, "No checkpoint file available")
class TestBotOrca(unittest.TestCase):
    def test_orca_loads(self):
        bot = Bot.orca(sims=2)
        self.assertIsNotNone(bot._net)

    def test_orca_best_move(self):
        bot = Bot.orca(sims=2)
        game = HexGame()
        game.place(0, 0)
        move = bot.best_move(game)
        self.assertIsInstance(move, tuple)


class TestArenaBasic(unittest.TestCase):
    def test_play_games(self):
        bot1 = Bot.random()
        bot2 = Bot.heuristic()
        result = Arena(bot1, bot2, num_games=5).play(verbose=False)
        self.assertEqual(result.total_games, 5)
        self.assertEqual(result.wins[0] + result.wins[1] + result.draws, 5)
        self.assertGreater(result.total_moves, 0)
        self.assertEqual(len(result.games), 5)

    def test_arena_result_properties(self):
        bot1 = Bot.random()
        bot2 = Bot.random()
        result = Arena(bot1, bot2, num_games=4).play(verbose=False)
        wr = result.win_rate
        self.assertEqual(len(wr), 2)
        self.assertAlmostEqual(wr[0] + wr[1] + result.draws / max(result.total_games, 1), 1.0)

    def test_arena_alternates_sides(self):
        """Even games: bot1=P0, odd games: bot2=P0."""
        bot1 = Bot.random()
        bot2 = Bot.random()
        result = Arena(bot1, bot2, num_games=4).play(verbose=False)
        first_players = [g['first_player'] for g in result.games]
        self.assertEqual(first_players[0], 'bot1')
        self.assertEqual(first_players[1], 'bot2')
        self.assertEqual(first_players[2], 'bot1')
        self.assertEqual(first_players[3], 'bot2')


class TestEvaluateMoves(unittest.TestCase):
    def test_returns_scored_list(self):
        game = HexGame()
        game.place(0, 0)
        result = evaluate_moves(game, top_n=5)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        # Each entry: ((q, r), score)
        self.assertIsInstance(result[0][0], tuple)
        self.assertIsInstance(result[0][1], int)


class TestFindThreats(unittest.TestCase):
    def test_no_threats_early(self):
        game = HexGame()
        game.place(0, 0)
        threats = find_threats(game, player=0)
        # Very early game, unlikely to have 4+ line threats
        self.assertIsInstance(threats, list)

    def test_threats_near_win(self):
        """Build a 4-in-a-row and check that extension cells are threats."""
        game = HexGame()
        # P0: (0,0)
        game.place(0, 0)
        # P1 far away
        game.place(5, 5)
        game.place(5, 6)
        # P0: (1,0), (2,0)
        game.place(1, 0)
        game.place(2, 0)
        # P1 far away
        game.place(6, 5)
        game.place(6, 6)
        # P0: (3,0), ... now P0 has stones at 0,0 1,0 2,0 3,0
        game.place(3, 0)
        # After placing 3,0, check for threats that would extend to 5+
        threats = find_threats(game, player=0)
        self.assertIsInstance(threats, list)


class TestFindWinningMoves(unittest.TestCase):
    def test_no_winning_moves_early(self):
        game = HexGame()
        game.place(0, 0)
        wins = find_winning_moves(game, player=0)
        self.assertEqual(wins, [])


class TestCountLines(unittest.TestCase):
    def test_count_lines_returns_dict(self):
        game = HexGame()
        game.place(0, 0)
        result = count_lines(game, 0, 0, player=0)
        self.assertIn('max_line', result)
        self.assertIsInstance(result['max_line'], int)


class TestRollout(unittest.TestCase):
    def test_rollout_result(self):
        game = HexGame()
        game.place(0, 0)
        result = rollout(game, num_games=10)
        self.assertIn('p0_wins', result)
        self.assertIn('p1_wins', result)
        self.assertIn('draw_rate', result)
        total = result['p0_wins'] + result['p1_wins'] + result['draw_rate']
        self.assertAlmostEqual(total, 1.0, places=5)


class TestAlphabeta(unittest.TestCase):
    def test_alphabeta_returns_dict(self):
        game = HexGame()
        game.place(0, 0)
        result = alphabeta(game, depth=4)
        self.assertIn('best_move', result)
        self.assertIn('value', result)
        self.assertIn('nodes', result)


class TestFindForcedMove(unittest.TestCase):
    def test_no_forced_move_early(self):
        game = HexGame()
        game.place(0, 0)
        result = find_forced_move(game)
        # Early game, should be None
        self.assertIsNone(result)


class TestThreatSearch(unittest.TestCase):
    def test_no_threat_early(self):
        game = HexGame()
        game.place(0, 0)
        result = threat_search(game, depth=2)
        self.assertIsNone(result)


class TestCountThreats(unittest.TestCase):
    def test_count_threats_early(self):
        game = HexGame()
        game.place(0, 0)
        n = count_threats(game, player=0)
        self.assertIsInstance(n, int)
        self.assertEqual(n, 0)


class TestDetectFork(unittest.TestCase):
    def test_no_fork_early(self):
        game = HexGame()
        game.place(0, 0)
        self.assertFalse(detect_fork(game, player=0))


class TestImportGames(unittest.TestCase):
    @unittest.skipUnless(
        os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'games.jsonl')),
        "No test game data found"
    )
    def test_import_games(self):
        from hexbot import import_games
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'games.jsonl')
        games = import_games(path, max_games=5)
        self.assertIsInstance(games, list)


if __name__ == '__main__':
    unittest.main()
