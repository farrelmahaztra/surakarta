import random
from typing import Tuple, Optional, List

class RandomSurakartaAgent:
    def __init__(self):
        pass
        
    def get_action(self, env) -> Optional[Tuple[int, int, int, int]]:
        all_actions = self._get_all_valid_actions(env)
        if not all_actions:
            return None
            
        return random.choice(all_actions)
        
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