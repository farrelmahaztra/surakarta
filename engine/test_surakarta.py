import unittest
import numpy as np
from engine.surakarta import SurakartaEnv, Player


class TestSurakarta(unittest.TestCase):
    def setUp(self):
        self.env = SurakartaEnv()
        self.env.reset()

    def test_initial_board_setup(self):
        self.assertTrue(np.all(self.env.board[0:2, :] == Player.WHITE.value))
        self.assertTrue(np.all(self.env.board[-2:, :] == Player.BLACK.value))
        self.assertTrue(np.all(self.env.board[2:4, :] == Player.NONE.value))

    def test_valid_basic_moves(self):
        self.env.current_player = Player.BLACK
        self.assertTrue(self.env._is_valid_move(4, 0, 3, 0))
        self.assertTrue(self.env._is_valid_move(4, 0, 3, 1))
        
        self.env.current_player = Player.WHITE
        self.assertTrue(self.env._is_valid_move(1, 0, 2, 0))
        self.assertTrue(self.env._is_valid_move(1, 0, 2, 1))

    def test_invalid_basic_moves(self):
        self.env.current_player = Player.BLACK
        self.assertFalse(self.env._is_valid_move(4, 0, -1, 0))
        self.assertFalse(self.env._is_valid_move(4, 0, 5, 0))
        self.assertFalse(self.env._is_valid_move(4, 0, 2, 0))
        self.assertFalse(self.env._is_valid_move(0, 0, 1, 0)) 

    def test_move_execution(self):
        self.env.current_player = Player.BLACK
        self.env._make_move(4, 0, 3, 0)
        self.assertEqual(self.env.board[3, 0], Player.BLACK.value)
        self.assertEqual(self.env.board[4, 0], Player.NONE.value)

    def test_capture(self):
        self.env.board = np.zeros((6, 6), dtype=np.int8)
        self.env.board[1, 1] = Player.BLACK.value
        self.env.board[1, 5] = Player.WHITE.value
        self.env.current_player = Player.BLACK
        captures = self.env._get_capture_moves((1, 1))
        self.assertIn((1, 5), captures)

    def test_no_capture_without_arc(self):
        self.env.board = np.zeros((6, 6), dtype=np.int8)
        self.env.board[0, 0] = Player.BLACK.value
        self.env.board[0, 1] = Player.WHITE.value
        self.env.current_player = Player.BLACK
        captures = self.env._get_capture_moves((0, 0))
        self.assertEqual(len(captures), 0)

    def test_arc_path_blocked(self):
        self.env.board = np.zeros((6, 6), dtype=np.int8)
        self.env.board[1, 0] = Player.BLACK.value
        self.env.board[2, 1] = Player.BLACK.value
        self.env.board[3, 1] = Player.WHITE.value
        
        self.env.current_player = Player.BLACK
        
        captures = self.env._get_capture_moves((1, 0))
        self.assertEqual(len(captures), 0)

    def test_step_function(self):
        self.env.board = np.zeros((6, 6), dtype=np.int8)
        self.env.board[1, 0] = Player.BLACK.value 
        self.env.board[0, 1] = Player.WHITE.value 
        
        self.env.current_player = Player.BLACK
        
        _, reward, _, _, _ = self.env.step([1, 0, 0, 1])
        
        self.assertEqual(self.env.board[0, 1], Player.BLACK.value) 
        self.assertEqual(self.env.board[1, 0], Player.NONE.value) 
        
        self.assertGreater(reward, 0)

    def test_win_condition(self):
        self.env.board = np.zeros((6, 6), dtype=np.int8)
        self.env.board[4, 0] = Player.BLACK.value

        winner = self.env._check_win()
        self.assertEqual(winner, Player.BLACK)
        
        self.env.board[4, 0] = Player.NONE.value
        self.env.board[0, 0] = Player.WHITE.value 
        
        winner = self.env._check_win()
        self.assertEqual(winner, Player.WHITE)
        
    def test_caching(self):
        self.env.board = np.zeros((6, 6), dtype=np.int8)
        self.env.board[1, 1] = Player.BLACK.value
        self.env.board[4, 4] = Player.WHITE.value
        self.env.current_player = Player.BLACK
        self.env.move_cache = {}
        self.env.capture_cache = {}
        
        pos = (1, 1)
        moves = self.env._get_valid_moves(pos)
        captures = self.env._get_capture_moves(pos)
        
        board_tuple = tuple(map(tuple, self.env.board))
        move_cache_key = (pos, board_tuple, Player.BLACK.value)
        capture_cache_key = (pos, board_tuple, Player.BLACK.value)
        
        self.assertIn(move_cache_key, self.env.move_cache)
        self.assertIn(capture_cache_key, self.env.capture_cache)
        
        self.assertEqual(self.env.move_cache[move_cache_key], moves)
        self.assertEqual(self.env.capture_cache[capture_cache_key], captures)
        
        moves2 = self.env._get_valid_moves(pos)
        captures2 = self.env._get_capture_moves(pos)
        
        self.assertEqual(moves, moves2)
        self.assertEqual(captures, captures2)


if __name__ == "__main__":
    unittest.main()