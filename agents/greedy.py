import random
from typing import Tuple, Optional, List

class GreedySurakartaAgent:
    def __init__(self, capture_probability=1.0):
        self.capture_probability = capture_probability
        
    def get_action(self, env) -> Optional[Tuple[int, int, int, int]]:
        all_actions = self._get_all_valid_actions(env)
        if not all_actions:
            return None
            
        capture_actions = self._filter_capture_actions(env, all_actions)
        
        if capture_actions and random.random() < self.capture_probability:
            return random.choice(capture_actions)
        
        return random.choice(all_actions)
        
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