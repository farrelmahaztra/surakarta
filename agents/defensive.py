import numpy as np
import random
from typing import Tuple, Optional, List, Set
from engine.surakarta import Player

class DefensiveSurakartaAgent:
    def __init__(self, 
                 defense_priority: float = 0.9, 
                 safety_priority: float = 0.7,
                 random_seed: Optional[int] = None):
        self.defense_priority = min(max(0.0, defense_priority), 1.0)
        self.safety_priority = min(max(0.0, safety_priority), 1.0) 
        
        if random_seed is not None:
            random.seed(random_seed)
        
    def get_action(self, env) -> Optional[Tuple[int, int, int, int]]:
        all_actions = self._get_all_valid_actions(env)
        if not all_actions:
            return None
        
        threatened_pieces = self._find_threatened_pieces(env)
        
        if threatened_pieces and random.random() < self.defense_priority:
            defensive_actions = self._get_defensive_actions(env, all_actions, threatened_pieces)
            if defensive_actions:
                return random.choice(defensive_actions)
        
        capture_actions = self._filter_capture_actions(env, all_actions)
        if capture_actions:
            return random.choice(capture_actions)
        
        if random.random() < self.safety_priority:
            safe_actions = self._get_safe_actions(env, all_actions)
            if safe_actions:
                return random.choice(safe_actions)
        
        return random.choice(all_actions)
    
    def _find_threatened_pieces(self, env) -> Set[Tuple[int, int]]:
        threatened_pieces = set()
        current_player = env.current_player.value
        opponent = Player.WHITE.value if current_player == Player.BLACK.value else Player.BLACK.value
        
        opponent_positions = np.argwhere(env.board == opponent)
        
        for o_row, o_col in opponent_positions:
            captures = env._get_capture_moves((o_row, o_col))
            
            for capture_row, capture_col in captures:
                if env.board[capture_row, capture_col] == current_player:
                    threatened_pieces.add((capture_row, capture_col))
        
        return threatened_pieces
    
    def _get_defensive_actions(self, env, all_actions: List[Tuple[int, int, int, int]], 
                              threatened_pieces: Set[Tuple[int, int]]) -> List[Tuple[int, int, int, int]]:
        defensive_actions = []
        
        for action in all_actions:
            from_row, from_col, to_row, to_col = action
            
            if (from_row, from_col) in threatened_pieces:
                if not self._would_be_threatened(env, from_row, from_col, to_row, to_col):
                    defensive_actions.append(action)
        
        return defensive_actions
    
    def _would_be_threatened(self, env, from_row: int, from_col: int, 
                            to_row: int, to_col: int) -> bool:
        board_copy = env.board.copy()
        
        piece = board_copy[from_row, from_col]
        board_copy[to_row, to_col] = piece
        board_copy[from_row, from_col] = Player.NONE.value
        
        current_player = env.current_player.value
        opponent = Player.WHITE.value if current_player == Player.BLACK.value else Player.BLACK.value
        
        opponent_positions = np.argwhere(board_copy == opponent)
        
        original_board = env.board
        env.board = board_copy
        
        is_threatened = False
        for o_row, o_col in opponent_positions:
            captures = env._get_capture_moves((o_row, o_col))
            if (to_row, to_col) in captures:
                is_threatened = True
                break
        
        env.board = original_board
        
        return is_threatened
    
    def _get_safe_actions(self, env, all_actions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        safe_actions = []
        
        for action in all_actions:
            from_row, from_col, to_row, to_col = action
            
            if (from_row, from_col) in self._find_threatened_pieces(env):
                continue
                
            if not self._would_be_threatened(env, from_row, from_col, to_row, to_col):
                safe_actions.append(action)
        
        return safe_actions
        
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