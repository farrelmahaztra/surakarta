import numpy as np
from typing import Tuple, Optional, List
import random 
from engine.surakarta import Player

class MinimaxSurakartaAgent:
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth

    def get_action(self, env) -> Optional[Tuple[int, int, int, int]]:
        board = env.board
        
        all_actions = self._get_all_valid_actions(env)
        capture_actions = self._filter_capture_actions(env, all_actions)
        
        actions_to_consider = capture_actions if capture_actions else all_actions
        
        if not actions_to_consider:
            return None
            
        action_values = []
        
        for action in actions_to_consider:
            board_copy = board.copy()
            from_row, from_col, to_row, to_col = action
            
            temp_piece = board_copy[from_row, from_col]
            board_copy[to_row, to_col] = temp_piece
            board_copy[from_row, from_col] = Player.NONE.value
            
            value, _ = self._minimax(
                env, board_copy, self.max_depth - 1, float("-inf"), float("inf"), False
            )
            
            is_capture = (
                env.board[to_row, to_col] != Player.NONE.value
                and env.board[to_row, to_col] != env.current_player.value
            )
            if is_capture:
                value += 10  
                
            noise_factor = 0.5
            value += random.uniform(-noise_factor, noise_factor)
            
            action_values.append((value, action))
        
        action_values.sort(reverse=True, key=lambda x: x[0])
        
        n_best = min(3, len(action_values)) 
        
        weights = [n_best - i for i in range(n_best)]
        selected_index = random.choices(range(n_best), weights=weights, k=1)[0]
        
        best_action = action_values[selected_index][1]
        
        return best_action

    def _filter_capture_actions(self, env, actions) -> List[Tuple[int, int, int, int]]:
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
        black_pieces = np.count_nonzero(board == Player.BLACK.value)
        white_pieces = np.count_nonzero(board == Player.WHITE.value)
        
        if black_pieces == 0:
            return float("-inf") if env.current_player == Player.BLACK else float("inf"), None
        elif white_pieces == 0:
            return float("inf") if env.current_player == Player.BLACK else float("-inf"), None

        if depth == 0:
            return self._evaluate_position(board, env.current_player), None

        current_player = env.current_player
        opponent = Player.WHITE if current_player == Player.BLACK else Player.BLACK

        original_board = env.board.copy()
        original_player = env.current_player
        
        env.board = board.copy()

        if is_maximizing:
            max_eval = float("-inf")
            best_action = None

            for action in self._get_all_valid_actions(env):
                board_copy = board.copy()
                from_row, from_col, to_row, to_col = action

                is_capture = (board_copy[to_row, to_col] != Player.NONE.value and 
                            board_copy[to_row, to_col] == opponent.value)

                temp_piece = board_copy[from_row, from_col]
                board_copy[to_row, to_col] = temp_piece
                board_copy[from_row, from_col] = Player.NONE.value

                env.current_player = opponent
                
                eval, _ = self._minimax(env, board_copy, depth - 1, alpha, beta, False)
                
                env.current_player = current_player

                if is_capture:
                    eval += 5

                if eval > max_eval:
                    max_eval = eval
                    best_action = action

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

            env.board = original_board
            env.current_player = original_player
            
            if best_action is None:
                return self._evaluate_position(board, current_player), None
                
            return max_eval, best_action
        else:
            min_eval = float("inf")
            best_action = None

            env.current_player = opponent
            
            for action in self._get_all_valid_actions(env):
                board_copy = board.copy()
                from_row, from_col, to_row, to_col = action

                is_capture = (board_copy[to_row, to_col] != Player.NONE.value and 
                            board_copy[to_row, to_col] == current_player.value)

                temp_piece = board_copy[from_row, from_col]
                board_copy[to_row, to_col] = temp_piece
                board_copy[from_row, from_col] = Player.NONE.value

                env.current_player = current_player
                
                eval, _ = self._minimax(env, board_copy, depth - 1, alpha, beta, True)
                
                env.current_player = opponent

                if is_capture:
                    eval -= 5

                if eval < min_eval:
                    min_eval = eval
                    best_action = action

                beta = min(beta, eval)
                if beta <= alpha:
                    break

            env.board = original_board
            env.current_player = original_player
            
            if best_action is None:
                return self._evaluate_position(board, current_player), None
                
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
        
        center_mask = np.zeros_like(board)
        center_mask[2:4, 2:4] = 1
        black_center = np.sum((board == Player.BLACK.value) & (center_mask == 1))
        white_center = np.sum((board == Player.WHITE.value) & (center_mask == 1))
        
        arc_positions = [(0, 1), (0, 4), (1, 0), (1, 5), (4, 0), (4, 5), (5, 1), (5, 4)]
        black_arc = sum(
            1 for pos in arc_positions if board[pos[0], pos[1]] == Player.BLACK.value
        )
        white_arc = sum(
            1 for pos in arc_positions if board[pos[0], pos[1]] == Player.WHITE.value
        )
        
        material_diff = (black_pieces - white_pieces) * 10
        position_diff = (black_center - white_center) * 3
        arc_diff = (black_arc - white_arc) * 2
        
        if current_player == Player.BLACK.value:
            material_score = material_diff
            position_score = position_diff
            arc_score = arc_diff
        else: 
            material_score = -material_diff
            position_score = -position_diff
            arc_score = -arc_diff
        
        total_score = material_score + position_score + arc_score
        
        return total_score