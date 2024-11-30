import numpy as np
from typing import Tuple, Optional, List
from engine.surakarta import Player

class MinimaxSurakartaAgent:
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth

    def get_action(self, env) -> Optional[Tuple[int, int, int, int]]:
        board = env.board

        all_actions = self._get_all_valid_actions(env)
        capture_actions = self._filter_capture_actions(env, all_actions)

        actions_to_consider = capture_actions if capture_actions else all_actions

        best_value = float("-inf")
        best_action = None

        for action in actions_to_consider:
            board_copy = board.copy()
            from_row, from_col, to_row, to_col = action

            temp_piece = board_copy[from_row, from_col]
            board_copy[to_row, to_col] = temp_piece
            board_copy[from_row, from_col] = Player.NONE.value

            value, _ = self._minimax(
                env, board_copy, self.max_depth - 1, float("-inf"), float("inf"), False
            )

            is_capture = env.board[to_row, to_col] != Player.NONE.value
            if is_capture:
                value += 10 

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _filter_capture_actions(self, env, actions) -> List[Tuple[int, int, int, int]]:
        """Filter actions to only those that result in captures."""
        capture_actions = []
        for action in actions:
            _, _, to_row, to_col = action
            if (
                env.board[to_row, to_col] != Player.NONE.value
                and env.board[to_row, to_col] != env.current_player.value
            ):
                capture_actions.append(action)
        return capture_actions

    def _minimax(
        self, env, board, depth, alpha, beta, is_maximizing
    ) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
        if depth == 0 or env._check_win() is not None:
            return self._evaluate_position(board, env.current_player), None

        if is_maximizing:
            max_eval = float("-inf")
            best_action = None

            for action in self._get_all_valid_actions(env):
                board_copy = board.copy()
                from_row, from_col, to_row, to_col = action

                is_capture = board_copy[to_row, to_col] != Player.NONE.value

                temp_piece = board_copy[from_row, from_col]
                board_copy[to_row, to_col] = temp_piece
                board_copy[from_row, from_col] = Player.NONE.value

                eval, _ = self._minimax(env, board_copy, depth - 1, alpha, beta, False)

                if is_capture:
                    eval += 5

                if eval > max_eval:
                    max_eval = eval
                    best_action = action

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

            return max_eval, best_action
        else:
            min_eval = float("inf")
            best_action = None

            for action in self._get_all_valid_actions(env):
                board_copy = board.copy()
                from_row, from_col, to_row, to_col = action

                is_capture = board_copy[to_row, to_col] != Player.NONE.value

                temp_piece = board_copy[from_row, from_col]
                board_copy[to_row, to_col] = temp_piece
                board_copy[from_row, from_col] = Player.NONE.value

                eval, _ = self._minimax(env, board_copy, depth - 1, alpha, beta, True)

                if is_capture:
                    eval -= 5  

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

    def _evaluate_position(self, board, current_player) -> float:
        black_pieces = np.count_nonzero(board == Player.BLACK.value)
        white_pieces = np.count_nonzero(board == Player.WHITE.value)

        material_score = (black_pieces - white_pieces) * 10

        center_mask = np.zeros_like(board)
        center_mask[2:4, 2:4] = 1
        black_center = np.sum((board == Player.BLACK.value) & (center_mask == 1))
        white_center = np.sum((board == Player.WHITE.value) & (center_mask == 1))
        position_score = (black_center - white_center) * 3

        arc_positions = [(0, 1), (0, 4), (1, 0), (1, 5), (4, 0), (4, 5), (5, 1), (5, 4)]
        black_arc = sum(
            1 for pos in arc_positions if board[pos[0], pos[1]] == Player.BLACK.value
        )
        white_arc = sum(
            1 for pos in arc_positions if board[pos[0], pos[1]] == Player.WHITE.value
        )
        arc_score = (black_arc - white_arc) * 2

        total_score = material_score + position_score + arc_score

        if current_player == Player.WHITE:
            total_score = -total_score

        return total_score