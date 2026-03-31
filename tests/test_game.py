"""Tests for hexgame.HexGame - the core game engine."""

import unittest
from hexgame import HexGame


class TestCreateGame(unittest.TestCase):
    def test_create_game(self):
        game = HexGame()
        self.assertEqual(game.total_stones, 0)
        self.assertEqual(game.current_player, 0)
        self.assertIsNone(game.winner)
        self.assertFalse(game.is_over)
        self.assertEqual(game.moves, [])

    def test_create_game_max_stones(self):
        game = HexGame(max_stones=50)
        self.assertFalse(game.is_over)


class TestPlaceStone(unittest.TestCase):
    def test_place_stone(self):
        game = HexGame()
        game.place(0, 0)
        self.assertEqual(game.total_stones, 1)
        self.assertEqual(game.moves, [(0, 0)])

    def test_place_multiple(self):
        game = HexGame()
        game.place(0, 0)
        game.place(1, 0)
        game.place(1, -1)
        self.assertEqual(game.total_stones, 3)
        self.assertEqual(len(game.moves), 3)


class TestTurnStructure(unittest.TestCase):
    """Player 0 places 1 stone first, then 2 stones per turn."""

    def test_first_turn_one_stone(self):
        game = HexGame()
        self.assertEqual(game.current_player, 0)
        self.assertEqual(game.stones_per_turn, 1)
        game.place(0, 0)
        # After P0's single stone, it should be P1's turn
        self.assertEqual(game.current_player, 1)

    def test_second_turn_two_stones(self):
        game = HexGame()
        game.place(0, 0)       # P0 turn (1 stone)
        self.assertEqual(game.current_player, 1)
        self.assertEqual(game.stones_per_turn, 2)
        game.place(1, 0)       # P1 first stone
        self.assertEqual(game.current_player, 1)  # still P1's turn
        game.place(1, -1)      # P1 second stone
        self.assertEqual(game.current_player, 0)  # now P0's turn

    def test_alternating_two_stones(self):
        game = HexGame()
        game.place(0, 0)   # P0 (1 stone)
        game.place(1, 0)   # P1 stone 1
        game.place(1, -1)  # P1 stone 2
        game.place(2, 0)   # P0 stone 1
        game.place(2, -1)  # P0 stone 2
        self.assertEqual(game.current_player, 1)
        self.assertEqual(game.total_stones, 5)


class TestUndo(unittest.TestCase):
    def test_undo_single(self):
        game = HexGame()
        game.place(0, 0)
        game.undo()
        self.assertEqual(game.total_stones, 0)
        self.assertEqual(game.moves, [])
        self.assertEqual(game.current_player, 0)

    def test_undo_multiple(self):
        game = HexGame()
        game.place(0, 0)
        game.place(1, 0)
        game.undo()
        self.assertEqual(game.total_stones, 1)
        self.assertEqual(game.current_player, 1)


class TestClone(unittest.TestCase):
    def test_clone_independent(self):
        game = HexGame()
        game.place(0, 0)
        clone = game.clone()
        self.assertEqual(clone.total_stones, 1)
        self.assertEqual(clone.moves, [(0, 0)])
        # Modify clone, original should be unchanged
        clone.place(1, 0)
        self.assertEqual(game.total_stones, 1)
        self.assertEqual(clone.total_stones, 2)


class TestLegalMoves(unittest.TestCase):
    def test_empty_board(self):
        game = HexGame()
        moves = game.legal_moves()
        self.assertEqual(moves, [(0, 0)])

    def test_after_first_move(self):
        game = HexGame()
        game.place(0, 0)
        moves = game.legal_moves()
        self.assertGreater(len(moves), 0)
        # (0,0) is occupied, should not be in legal moves
        self.assertNotIn((0, 0), moves)


class TestWinDetection(unittest.TestCase):
    def test_six_in_a_row_horizontal(self):
        """Build 6 in a row on q-axis for P0 and check winner."""
        game = HexGame()
        # P0 places (0,0)
        game.place(0, 0)
        # P1 places far away
        game.place(0, 5)
        game.place(0, 6)
        # P0 places (1,0) and (2,0)
        game.place(1, 0)
        game.place(2, 0)
        # P1 places far away
        game.place(0, 7)
        game.place(0, 8)
        # P0 places (3,0) and (4,0)
        game.place(3, 0)
        game.place(4, 0)
        # P1 places far away
        game.place(0, 9)
        game.place(0, 10)
        # P0 places (5,0) - that's 6 in a row: (0,0) through (5,0)
        game.place(5, 0)
        self.assertEqual(game.winner, 0)
        self.assertTrue(game.is_over)


class TestScoredMoves(unittest.TestCase):
    def test_returns_tuples(self):
        game = HexGame()
        game.place(0, 0)
        scored = game.scored_moves(10)
        self.assertIsInstance(scored, list)
        self.assertGreater(len(scored), 0)
        # Each entry is (q, r, score)
        first = scored[0]
        self.assertEqual(len(first), 3)
        self.assertIsInstance(first[2], int)

    def test_sorted_descending(self):
        game = HexGame()
        game.place(0, 0)
        game.place(1, 0)
        game.place(1, -1)
        scored = game.scored_moves(10)
        scores = [s[2] for s in scored]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestSearch(unittest.TestCase):
    def test_search_returns_dict(self):
        game = HexGame()
        game.place(0, 0)
        result = game.search(depth=4)
        self.assertIn('best_move', result)
        self.assertIn('value', result)
        self.assertIn('nodes', result)
        self.assertIsInstance(result['best_move'], tuple)
        self.assertEqual(len(result['best_move']), 2)
        self.assertIsInstance(result['value'], float)
        self.assertIsInstance(result['nodes'], int)


class TestZobristHash(unittest.TestCase):
    def test_different_positions_different_hashes(self):
        g1 = HexGame()
        g1.place(0, 0)
        g2 = HexGame()
        g2.place(1, 0)
        self.assertNotEqual(g1.zhash, g2.zhash)

    def test_same_position_same_hash(self):
        g1 = HexGame()
        g1.place(0, 0)
        g2 = HexGame()
        g2.place(0, 0)
        self.assertEqual(g1.zhash, g2.zhash)

    def test_empty_board_hash(self):
        g = HexGame()
        self.assertIsInstance(g.zhash, int)


class TestFromMoves(unittest.TestCase):
    def test_from_moves(self):
        moves = [(0, 0), (1, 0), (1, -1)]
        game = HexGame.from_moves(moves)
        self.assertEqual(game.total_stones, 3)
        self.assertEqual(game.moves, moves)


class TestSerialization(unittest.TestCase):
    def test_to_dict_from_dict(self):
        game = HexGame()
        game.place(0, 0)
        game.place(1, 0)
        game.place(1, -1)
        d = game.to_dict()
        self.assertIn('moves', d)
        self.assertIn('max_stones', d)
        restored = HexGame.from_dict(d)
        self.assertEqual(restored.total_stones, game.total_stones)
        self.assertEqual(restored.moves, game.moves)
        self.assertEqual(restored.current_player, game.current_player)


if __name__ == '__main__':
    unittest.main()
