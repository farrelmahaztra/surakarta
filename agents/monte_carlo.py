import numpy as np
import random
from typing import Tuple, Optional, List
from engine.surakarta import Player

class MonteCarloSurakartaAgent:
    def __init__(self, num_playouts: int = 10, playout_depth: int = 10, 
                 capture_weight: float = 1.5, random_seed: Optional[int] = None):
        self.num_playouts = max(1, num_playouts) 
        self.playout_depth = max(1, playout_depth)
        self.capture_weight = capture_weight
        
        if random_seed is not None:
            random.seed(random_seed)
        
    def get_action(self, env) -> Optional[Tuple[int, int, int, int]]:
        all_actions = self._get_all_valid_actions(env)
        if not all_actions:
            return None
            
        capture_actions = self._filter_capture_actions(env, all_actions)
        if capture_actions and len(capture_actions) == 1:
            return capture_actions[0]
            
        actions_to_evaluate = capture_actions if capture_actions else all_actions
            
        if len(actions_to_evaluate) > 8:
            actions_to_evaluate = random.sample(actions_to_evaluate, 8)
        
        best_action = None
        best_score = float('-inf')
        
        for action in actions_to_evaluate:
            score = self._evaluate_action(env, action)
            
            if action in capture_actions:
                score *= self.capture_weight
                
            if score > best_score:
                best_score = score
                best_action = action
                
        if best_action is None:
            return random.choice(all_actions)
            
        return best_action
        
    def _evaluate_action(self, env, action: Tuple[int, int, int, int]) -> float:
        wins = 0
        draws = 0
        
        for _ in range(self.num_playouts):
            sim_env = self._clone_environment(env)
            from_row, from_col, to_row, to_col = action
            
            current_player = sim_env.current_player
            self._make_move_in_sim(sim_env, from_row, from_col, to_row, to_col)
            
            sim_env.current_player = (
                Player.WHITE if sim_env.current_player == Player.BLACK else Player.BLACK
            )
            
            result = self._run_playout(sim_env, current_player)
            
            if result == 1: 
                wins += 1
            elif result == 0:  
                draws += 0.5
        
        score = (wins + draws) / self.num_playouts
        return score
        
    def _run_playout(self, sim_env, original_player: Player) -> int:
        for _ in range(self.playout_depth):
            winner = self._check_win_in_sim(sim_env)
            if winner is not None:
                if winner == original_player:
                    return 1
                else:
                    return -1
                    
            actions = self._get_all_valid_actions(sim_env)
            if not actions:
                return 0
                
            capture_actions = self._filter_capture_actions(sim_env, actions)
            if capture_actions:
                action = random.choice(capture_actions)
            else:
                action = random.choice(actions)
                
            from_row, from_col, to_row, to_col = action
            self._make_move_in_sim(sim_env, from_row, from_col, to_row, to_col)
            
            sim_env.current_player = (
                Player.WHITE if sim_env.current_player == Player.BLACK else Player.BLACK
            )
        
        return self._evaluate_terminal_state(sim_env, original_player)
        
    def _evaluate_terminal_state(self, sim_env, original_player: Player) -> int:
        original_player_value = original_player.value
        opponent_value = Player.WHITE.value if original_player == Player.BLACK else Player.BLACK.value
        
        original_pieces = np.count_nonzero(sim_env.board == original_player_value)
        opponent_pieces = np.count_nonzero(sim_env.board == opponent_value)
        
        piece_advantage = original_pieces - opponent_pieces
        
        if piece_advantage > 0:
            return min(piece_advantage / 12, 1.0)
        elif piece_advantage < 0:
            return max(piece_advantage / 12, -1.0)
        else:
            return 0.0
    
    def _clone_environment(self, env):
        class SimpleEnv:
            pass
            
        sim_env = SimpleEnv()
        sim_env.board = env.board.copy()
        sim_env.board_size = env.board_size
        sim_env.current_player = env.current_player
        sim_env._get_valid_moves = env._get_valid_moves
        sim_env._get_capture_moves = env._get_capture_moves
        
        return sim_env
        
    def _make_move_in_sim(self, sim_env, from_row: int, from_col: int, to_row: int, to_col: int):
        sim_env.board[to_row, to_col] = sim_env.board[from_row, from_col]
        sim_env.board[from_row, from_col] = Player.NONE.value
        
    def _check_win_in_sim(self, sim_env) -> Optional[Player]:
        black_pieces = np.count_nonzero(sim_env.board == Player.BLACK.value)
        white_pieces = np.count_nonzero(sim_env.board == Player.WHITE.value)
        
        if black_pieces == 0:
            return Player.WHITE
        elif white_pieces == 0:
            return Player.BLACK
        return None
    
    def _filter_capture_actions(self, env, actions) -> List[Tuple[int, int, int, int]]:
        capture_actions = []
        for action in actions:
            _, _, to_row, to_col = action
            if (env.board[to_row, to_col] != 0 and 
                env.board[to_row, to_col] != env.current_player.value):
                capture_actions.append(action)
        return capture_actions
        
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