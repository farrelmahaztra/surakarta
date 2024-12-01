from engine.surakarta import (
    SurakartaEnv,
)
from agents.dqn import SurakartaRLAgent
from agents.minimax import MinimaxSurakartaAgent
from typing import Dict, Union
import torch
import os

AgentType = Union[MinimaxSurakartaAgent, SurakartaRLAgent]
GameState = Dict[str, Union[SurakartaEnv, AgentType]]

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(
    current_dir,
    "../../../training_runs/random/surakarta_agent_vs_random_episode_500.pth",
)
class GameManager:
    _games: Dict[str, GameState] = {}

    @classmethod
    def create_game(cls, game_id, agent_type="rule"):
        env = SurakartaEnv(render_mode=None)
        observation, _ = env.reset()

        if agent_type == "rule":
            agent = MinimaxSurakartaAgent(max_depth=1)
        elif agent_type == "rl":
            agent = SurakartaRLAgent(env=env)
            checkpoint = torch.load(model_path, map_location=agent.device)
            agent.q_net.load_state_dict(checkpoint["q_net_state_dict"])
            agent.target_net.load_state_dict(checkpoint["q_net_state_dict"])

        cls._games[game_id] = {"env": env, "agent": agent}
        return observation

    @classmethod
    def make_move(cls, game_id, move):
        if game_id not in cls._games:
            raise ValueError("No such game")

        game = cls._games[game_id]
        env = game["env"]
        agent = game["agent"]

        observation, reward, terminated, truncated, info = env.step(move)

        if not (terminated or truncated):
            ai_move = agent.get_action(env)
            observation, reward, terminated, truncated, info = env.step(ai_move)

        return {
            "observation": observation,
            "terminated": terminated,
            "truncated": truncated,
            "reward": reward,
            "info": info,
        }
