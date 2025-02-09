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


if __name__ == "__main__":
    unittest.main()