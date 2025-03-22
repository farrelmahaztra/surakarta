from engine.surakarta import (
    SurakartaEnv,
    Player
)
from agents.dqn import SurakartaRLAgent
from agents.minimax import MinimaxSurakartaAgent
from typing import Dict, Union, Optional, List, Tuple, Any
import numpy as np
import random
import torch
import os
from django.utils import timezone
from django.contrib.auth.models import User
from .models import GameRecord, Match
import numpy as np
from uuid import uuid4

AgentType = Union[MinimaxSurakartaAgent, SurakartaRLAgent]
GameState = Dict[str, Union[SurakartaEnv, AgentType, List, User]]

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(
    current_dir,
    "../../../optimal_runs/minimax_20250326_181130/dqn_episode1300.pt",
)

class GameManager:
    _games: Dict[str, GameState] = {}

    @classmethod
    def create_game(cls, game_id, agent_type="rule", user: Optional[User] = None, player_color="black"):
        env = SurakartaEnv(render_mode=None)
        observation, _ = env.reset()

        if agent_type == "rule":
            agent = MinimaxSurakartaAgent(max_depth=1)
        elif agent_type == "rl":
            agent = SurakartaRLAgent(env, device="mps")
            checkpoint = torch.load(model_path, map_location=agent.device)
            agent.q_net.load_state_dict(checkpoint["q_net_state_dict"])
            agent.target_net.load_state_dict(checkpoint["q_net_state_dict"])

        player_is_white = (player_color == "white")
        player_is_random = (player_color == "random")
        
        if player_is_random:
            player_is_white = random.choice([True, False])
            actual_player_color = "white" if player_is_white else "black"
        else:
            actual_player_color = player_color
        
        if player_is_white:
            ai_move = agent.get_action(env)
            if ai_move: 
                observation, _, _, _, _ = env.step(ai_move)

        cls._games[game_id] = {
            "env": env, 
            "agent": agent, 
            "moves": [],
            "user": user,
            "type": "single_player",
            "player_color": actual_player_color
        }
        
        if user:
            GameRecord.objects.create(
                user=user,
                game_id=game_id,
                opponent_type=agent_type,
            )
            
        return observation, actual_player_color

    @classmethod
    def make_move(cls, game_id, move):
        if game_id not in cls._games:
            raise ValueError("No such game")

        game = cls._games[game_id]
        env = game["env"]
        agent = game["agent"]
        player_color = game.get("player_color", "black")
        
        from_row, from_col, to_row, to_col = move

        print(f"Player ({player_color}) move: {from_row}, {from_col}, {to_row}, {to_col}")
        if not env._is_valid_move(from_row, from_col, to_row, to_col):
            raise ValueError("Invalid move")

        game["moves"].append(move)

        observation, reward, terminated, truncated, info = env.step(move)
        
        player_is_black = (player_color == "black")
        
        if not (terminated or truncated):
            ai_move = agent.get_action(env)
            if ai_move: 
                game["moves"].append(ai_move)
                observation, reward, terminated, truncated, info = env.step(ai_move)
            else:
                terminated = True

        winner = env._check_win()
        
        player_won = False
        if winner is not None:
            player_won = (player_is_black and winner == Player.BLACK.value) or \
                         (not player_is_black and winner == Player.WHITE.value)
                
        print(f"Game over: {terminated or truncated}, Player won: {player_won}, " +
              f"winner: {winner}, current player: {env.current_player.value}, " +
              f"player color: {player_color}")
              
        if terminated or truncated:
            user = game.get("user")
            if user:
                cls._update_game_record(game_id, user, info, player_won)

        return {
            "observation": observation,
            "terminated": terminated,
            "truncated": truncated,
            "reward": reward,
            "info": info,
        }
    
    @classmethod
    def create_multiplayer_match(cls, match: Match) -> Dict[str, Any]:
        env = SurakartaEnv(render_mode=None)
        observation, _ = env.reset()
        
        black_player, white_player = cls._assign_player_colors(match)
        
        match.black_player = black_player
        match.white_player = white_player
        
        black_name = black_player.username if black_player else "None"
        white_name = white_player.username if white_player else "None"
        print(f"Debug - Final create_multiplayer_match color assignment - Black: {black_name}, White: {white_name}")
        
        match.current_turn = black_player if black_player else match.creator
        match.current_player_idx = 0 
        
        match.board_state = {
            "board": observation["board"].tolist(),
            "current_player": 0, 
            "game_over": observation.get("game_over", False)
        }
        match.save()
        
        black_player_name = black_player.username if black_player else "Waiting for opponent"
        white_player_name = white_player.username if white_player else "Waiting for opponent"
        current_turn_name = match.current_turn.username if match.current_turn else "Waiting for opponent"
        
        print(f"Created multiplayer match: {match.game.game_id if match.game else 'None'}, black: {black_player_name}, white: {white_player_name}")
        
        return {
            "observation": observation,
            "black_player": black_player_name,
            "white_player": white_player_name,
            "current_turn": current_turn_name,
        }
    
    @classmethod
    def join_match(cls, match: Match, user: User) -> Dict[str, Any]:
        if match.opponent:
            raise ValueError("Match already has an opponent")
        
        if match.creator == user:
            raise ValueError("Cannot join your own match")
        
        match.opponent = user
        match.status = "in_progress"
        
        black_player, white_player = cls._assign_player_colors(match)
        
        match.black_player = black_player
        match.white_player = white_player
        
        black_name = black_player.username if black_player else "None"
        white_name = white_player.username if white_player else "None"
        print(f"Debug - Final join_match color assignment - Black: {black_name}, White: {white_name}")
        
        env = SurakartaEnv(render_mode=None)
        observation, _ = env.reset()
        
        match.board_state = {
            "board": observation["board"].tolist(),
            "current_player": 0,
            "game_over": False
        }
        
        match.current_turn = black_player
        match.current_player_idx = 0
        match.save()
        
        print(f"Joined match: {match.game.game_id if match.game else 'None'}, black: {black_player.username}, white: {white_player.username}")
        
        return {
            "observation": observation,
            "black_player": black_player.username,
            "white_player": white_player.username,
            "current_turn": black_player.username,
        }
        
    @classmethod
    def make_multiplayer_move(cls, game_id, move, user):
        try:
            match = Match.objects.get(game__game_id=game_id)
        except Match.DoesNotExist:
            raise ValueError("Match does not exist")
            
        if match.status != 'in_progress':
            raise ValueError(f"Match is not in progress (current status: {match.status})")
        
        if match.current_turn != user:
            raise ValueError(f"Not your turn. Current turn: {match.current_turn.username}")
            
        is_black_player = match.black_player == user
        is_white_player = match.white_player == user
        
        if not is_black_player and not is_white_player:
            raise ValueError("You are not assigned to any color in this match")
            
        print(f"Move validation - User: {user.username}, Black player: {match.black_player.username if match.black_player else 'None'}, White player: {match.white_player.username if match.white_player else 'None'}")
        print(f"Current player index: {match.current_player_idx} (0=Black's turn, 1=White's turn)")
        
        if (is_black_player and match.current_player_idx != 0) or (is_white_player and match.current_player_idx != 1):
            raise ValueError(f"Inconsistent turn state. Please refresh and try again.")
            
        from_row, from_col, to_row, to_col = move
        
        env = SurakartaEnv(render_mode=None)
        env.reset()
        board_state = match.board_state
        
        stored_board = board_state.get('board', [])
        if isinstance(stored_board, list):
            env.board = np.array(stored_board, dtype=np.int8)
        else:
            env.board = stored_board
            
        if match.current_player_idx == 0:
            env.current_player = Player.BLACK
            print("Debug - Setting environment current_player to BLACK")
        else:
            env.current_player = Player.WHITE
            print("Debug - Setting environment current_player to WHITE")
            
        env.game_over = board_state.get('game_over', False)
        
        
        if env.board is not None:
            source_piece = env.board[from_row][from_col]
            expected_piece = 1 if is_black_player else 2  
            print(f"Debug - Piece at source: {source_piece}, Expected piece: {expected_piece}")
            
            if source_piece != expected_piece:
                raise ValueError(f"You can only move your own pieces. You are {'Black' if is_black_player else 'White'}")
        else:
            print("Warning: Board is None, cannot check piece ownership")
        expected_player_value = 1 if env.current_player == Player.BLACK else 2
        piece_value = env.board[from_row][from_col] if env.board is not None else None
        
        print(f"Debug - Is move from my piece: {piece_value == expected_player_value}")
        print(f"Debug - Board value at position: {piece_value}, Expected current player value: {expected_player_value}")
        
        if not env._is_valid_move(from_row, from_col, to_row, to_col):
            print(f"Invalid move: {move, env.current_player}")
            raise ValueError("Invalid move")
        
        match.moves_history.append(move)
        observation, reward, terminated, truncated, info = env.step(move)
        
        if match.current_player_idx == 0:  
            match.current_turn = match.white_player
            match.current_player_idx = 1
        else: 
            match.current_turn = match.black_player
            match.current_player_idx = 0
        
        print(f"Debug - Observation current_player: {observation['current_player']}")
        
        next_player_value = 0 if match.current_player_idx == 1 else 1 
        
        match.board_state = {
            "board": observation["board"].tolist(),
            "current_player": next_player_value, 
            "game_over": terminated or truncated
        }
        
        print(f"Debug - Saved next player value: {next_player_value}")
        
        if terminated or truncated:
            winner = env._check_win()
            match.status = "completed"
        
            cls._update_multiplayer_results(match, winner, env, info)
        
        match.save()
        
        print(f"Move made in {game_id} by {user.username}: {move}")
        print(f"New current_turn: {match.current_turn.username if match.current_turn else 'None'}")
        
        return {
            "observation": observation,
            "terminated": terminated,
            "truncated": truncated,
            "reward": reward,
            "info": info,
            "current_turn": match.current_turn.username if match.current_turn else None
        }
    
    @classmethod
    def get_match_state(cls, match: Match, current_user: Optional[User] = None) -> Dict[str, Any]:
        board_state = match.board_state
        
        current_player_from_db = board_state.get("current_player", 0)
        print(f"Debug - get_match_state current_player from DB: {current_player_from_db}")
        
        observation = {
            "board": board_state.get("board", []),
            "current_player": current_player_from_db,
            "game_over": board_state.get("game_over", False)
        }
        
        black_player = match.black_player
        white_player = match.white_player
        
        if match.status == "open":
            black_player_name = black_player.username if black_player else "Waiting for opponent"
            white_player_name = white_player.username if white_player else "Waiting for opponent"
        else:
            black_player_name = black_player.username if black_player else "Unknown"
            white_player_name = white_player.username if white_player else "Unknown"
        
        current_turn_name = match.current_turn.username if match.current_turn else "Waiting for opponent"
        
        is_user_turn = False
        user_color = "Spectator"
        turn_message = ""
        
        if current_user:
            is_black = current_user == black_player
            is_white = current_user == white_player
            
            if is_black:
                user_color = "Black"
                is_user_turn = match.current_turn == current_user
            elif is_white:
                user_color = "White" 
                is_user_turn = match.current_turn == current_user
                
            if match.status == "in_progress":
                if is_user_turn:
                    turn_message = f"Your turn to move ({user_color})"
                else:
                    opponent_color = "White" if user_color == "Black" else "Black"
                    opponent_name = white_player_name if user_color == "Black" else black_player_name
                    turn_message = f"Waiting for {opponent_name} ({opponent_color})"
            elif match.status == "open":
                turn_message = "Waiting for opponent to join"
            elif match.status == "completed":
                turn_message = "Game completed"
        
        return {
            "observation": observation,
            "black_player": black_player_name,
            "white_player": white_player_name,
            "current_turn": current_turn_name,
            "status": match.status,
            "moves_history": match.moves_history,
            "turn_message": turn_message,
            "user_color": user_color,
            "is_user_turn": is_user_turn
        }
    
    @classmethod
    def _assign_player_colors(cls, match: Match) -> Tuple[Optional[User], Optional[User]]:
        creator = match.creator
        opponent = match.opponent 
        
        if opponent is None:
            if match.creator_color == "black":
                return creator, None  
            elif match.creator_color == "white":
                return None, creator 
            else: 
                return creator, None
        
        print(f"Debug - Color Assignment - Creator: {creator.username}, Opponent: {opponent.username}, Creator Color Preference: {match.creator_color}")
        
        if match.creator_color == "black":
            print(f"Debug - Assigning creator as BLACK, opponent as WHITE")
            return creator, opponent
        elif match.creator_color == "white":
            print(f"Debug - Assigning creator as WHITE, opponent as BLACK")
            return opponent, creator
        else:  
            is_creator_black = random.choice([True, False])
            if is_creator_black:
                print(f"Debug - Random assigned creator as BLACK, opponent as WHITE")
                return creator, opponent
            else:
                print(f"Debug - Random assigned creator as WHITE, opponent as BLACK")
                return opponent, creator
    
    @classmethod
    def _update_multiplayer_results(cls, match: Match, winner, env, info):
        black_player = match.black_player
        white_player = match.white_player
        
        if not black_player or not white_player:
            print(f"Warning: Missing players in completed match {match.game.game_id if match.game else 'None'}")
            return
        
        try:
            black_profile = black_player.profile
            white_profile = white_player.profile
            
            black_profile.games_played += 1
            white_profile.games_played += 1
            
            black_result = None
            white_result = None
            
            if winner == Player.BLACK.value: 
                black_profile.wins += 1
                white_profile.losses += 1
                black_result = 'win'
                white_result = 'loss'
                print(f"Black ({black_player.username}) won against White ({white_player.username})")
            elif winner == Player.WHITE.value:
                black_profile.losses += 1
                white_result = 'win'
                black_result = 'loss'
                print(f"White ({white_player.username}) won against Black ({black_player.username})")
            else: 
                black_profile.draws += 1
                white_profile.draws += 1
                black_result = 'draw'
                white_result = 'draw'
                print(f"Draw between Black ({black_player.username}) and White ({white_player.username})")
            
            black_score = 12 - info.get('white_pieces', 0) if info else 0
            white_score = 12 - info.get('black_pieces', 0) if info else 0
            
            black_profile.save()
            white_profile.save()
            
            if not match.game:
                new_game_id = f"{str(uuid4())}"
                
                black_game_record = GameRecord.objects.create(
                    user=black_player,
                    game_id=new_game_id,
                    opponent_type='multiplayer',
                    start_time=match.created_at,
                    moves=match.moves_history,
                    final_score=black_score,
                    result=black_result
                )
                
                if black_player.profile.analytics_consent:
                    black_game_record.end_time = timezone.now()
                    black_game_record.save()
                
                match.game = black_game_record
                match.save()
                
                white_game_record = GameRecord.objects.create(
                    user=white_player,
                    game_id=f"{new_game_id}_w", 
                    opponent_type='multiplayer',
                    start_time=match.created_at,
                    moves=match.moves_history,
                    final_score=white_score,
                    result=white_result
                )
                
                if white_player.profile.analytics_consent:
                    white_game_record.end_time = timezone.now()
                    white_game_record.save()

            match.game.result = black_result if winner == Player.BLACK.value else white_result
            match.game.final_score = black_score if winner == Player.BLACK.value else white_score
            
            if black_player.profile.analytics_consent and match.game.user == black_player:
                match.game.end_time = timezone.now()
            
            match.game.save()
            
        except Exception as e:
            import traceback
            print(f"Error updating player profiles: {str(e)}")
            print(traceback.format_exc())
    
    @classmethod
    def _update_game_record(cls, game_id, user, info, player_won):
        try:
            game_record = GameRecord.objects.get(game_id=game_id)
            
            player_color = cls._games[game_id].get("player_color", "black")
            player_is_black = (player_color == "black")
            
            if player_is_black:
                score = 12 - info.get('white_pieces', 0) if info else 0
            else:
                score = 12 - info.get('black_pieces', 0) if info else 0
                
            game_record.final_score = score
            game_record.result = 'win' if player_won else 'loss'
            game_record.moves = cls._games[game_id].get("moves", [])
            
            if user.profile.analytics_consent:
                game_record.end_time = timezone.now()
            
            game_record.save()
            
            profile = user.profile
            profile.games_played += 1
            
            if player_won:
                profile.wins += 1
            else:
                profile.losses += 1
       
            profile.save()
        except GameRecord.DoesNotExist:
            pass
