import numpy as np
from typing import Tuple, Optional, List
from engine.surakarta import Player

class MinimaxSurakartaAgent:
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth

    def get_action(self, env) -> Optional[Tuple[int, int, int, int]]:
        board = env.board

        _, best_action = self._minimax(
            env, board, self.max_depth, float("-inf"), float("inf"), True
        )
        return best_action

    def _minimax(
        self, env, board, depth, alpha, beta, is_maximizing
    ) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
        if depth == 0 or env._check_win() is not None:
            return self._evaluate_position(board), None

        best_action = None
        if is_maximizing:
            max_eval = float("-inf")
            for action in self._get_all_valid_actions(env):
                board_copy = board.copy()
                from_row, from_col, to_row, to_col = action

                temp_piece = board_copy[from_row, from_col]
                board_copy[to_row, to_col] = temp_piece
                board_copy[from_row, from_col] = Player.NONE.value

                eval, _ = self._minimax(env, board_copy, depth - 1, alpha, beta, False)

                if eval > max_eval:
                    max_eval = eval
                    best_action = action

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

            return max_eval, best_action
        else:
            min_eval = float("inf")
            for action in self._get_all_valid_actions(env):
                board_copy = board.copy()
                from_row, from_col, to_row, to_col = action

                temp_piece = board_copy[from_row, from_col]
                board_copy[to_row, to_col] = temp_piece
                board_copy[from_row, from_col] = Player.NONE.value

                eval, _ = self._minimax(env, board_copy, depth - 1, alpha, beta, True)

                if eval < min_eval:
                    min_eval = eval
                    best_action = action

                beta = min(beta, eval)
                if beta <= alpha:
                    break

            return min_eval, best_action

    def _get_all_valid_actions(self, env) -> List[Tuple[int, int, int, int]]:
        actions = []
        current_player = env.current_player.value

        for row in range(env.board_size):
            for col in range(env.board_size):
                if env.board[row, col] == current_player:
                    valid_moves = env._get_valid_moves((row, col))
                    actions.extend(
                        [(row, col, move[0], move[1]) for move in valid_moves]
                    )

        return actions

    def _evaluate_position(self, board) -> float:
        black_pieces = np.count_nonzero(board == Player.BLACK.value)
        white_pieces = np.count_nonzero(board == Player.WHITE.value)

        material_score = black_pieces - white_pieces

        center_mask = np.zeros_like(board)
        center_mask[2:4, 2:4] = 1
        black_center = np.sum(board == Player.BLACK.value & center_mask)
        white_center = np.sum(board == Player.WHITE.value & center_mask)
        position_score = (black_center - white_center) * 0.5

        return material_score + position_score