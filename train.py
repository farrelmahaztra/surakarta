from engine.surakarta import SurakartaEnv
from agents.dqn import SurakartaRLAgent
from agents.minimax import MinimaxSurakartaAgent
from agents.monte_carlo import MonteCarloSurakartaAgent
from agents.greedy import GreedySurakartaAgent
from agents.defensive import DefensiveSurakartaAgent
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from functools import partial
import argparse

def train_against_opponent(opponent_name, opponent_func, episodes=10_000, eval_frequency=100, model_path=None, save_dir='surakarta_checkpoints'):
    env = SurakartaEnv()
    rl_agent = SurakartaRLAgent(env)

    if model_path:
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path)
        rl_agent.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        rl_agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        rl_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        rl_agent.steps_done = checkpoint.get('steps_done', 0)
        print(f"Loaded model with {rl_agent.steps_done} steps done")

    rewards_history = []
    win_rates = []
    wins = 0
    games = 0
    reward_window = []
    window_size = 10

    pbar = tqdm(range(episodes), dynamic_ncols=True)

    for episode in pbar:
        state, _ = env.reset()
        episode_reward = 0
        episode_captures = 0
        done = False
        steps = 0
        max_steps = 500

        games += 1

        while not done and steps < max_steps:
            action = rl_agent.get_action(state, training=True)
            if action is None:
                break

            _, _, to_row, to_col = action
            if env.board[to_row, to_col] != 0: 
                episode_captures += 1

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            if reward == 10.0: 
                wins += 1

            if not done:
                opponent_action = opponent_func(env)
                
                if opponent_action is None:
                    break

                after_opponent_state, reward, terminated, truncated, _ = env.step(
                    opponent_action
                )
                done = terminated or truncated
                steps += 1

                rl_agent.memory.append(
                    (state, action, reward, after_opponent_state, done)
                )
                episode_reward += reward
                state = after_opponent_state
            else:
                rl_agent.memory.append((state, action, reward, next_state, done))
                episode_reward += reward

            if (
                len(rl_agent.memory) >= rl_agent.batch_size
                and len(rl_agent.memory) % 4 == 0
            ):
                rl_agent.update_model()

        rewards_history.append(episode_reward)
        reward_window.append(episode_reward)
        if len(reward_window) > window_size:
            reward_window.pop(0)

        if (episode + 1) % window_size == 0:
            avg_reward = sum(reward_window) / len(reward_window)
            pbar.set_description(
                f"Avg Reward: {avg_reward:.2f} | Captures: {episode_captures} | ε: {rl_agent.epsilon:.3f}"
            )

        rl_agent.epsilon = max(0.01, rl_agent.epsilon * rl_agent.epsilon_decay)

        if (episode + 1) % eval_frequency == 0:
            print(f"\nEpisode {episode + 1}")
            print(f"Epsilon: {rl_agent.epsilon:.3f}")

            win_rate = wins / games
            win_rates.append(win_rate)
            print(f"Win Rate: {win_rate:.2f} ({wins}/{games})")

            base_model_name = ""
            if model_path:
                base_model_name = os.path.basename(model_path).replace('.pth', '') + "_continued_"
            
            save_path = os.path.join(save_dir, f"surakarta_agent_{base_model_name}vs_{opponent_name}_episode_{episode + 1}.pth")
            
            torch.save(
                {
                    "q_net_state_dict": rl_agent.q_net.state_dict(),
                    "target_net_state_dict": rl_agent.target_net.state_dict(),
                    "optimizer_state_dict": rl_agent.optimizer.state_dict(),
                    "steps_done": rl_agent.steps_done,
                },
                save_path,
            )
            print(f"Saved model to {save_path}")

            plot_training_progress(rewards_history, win_rates, eval_frequency, opponent_name, save_dir)


def plot_training_progress(rewards_history, win_rates, eval_frequency, label, save_dir='surakarta_checkpoints'):
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(rewards_history)
    ax1.set_title("Episode Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")

    eval_episodes = np.arange(eval_frequency, len(rewards_history) + 1, eval_frequency)
    ax2.plot(eval_episodes, win_rates, label=label)
    ax2.set_title(f"Win Rates vs {label}")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Win Rate")
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"training_progress_vs_{label}.png")
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.close()


def get_random_opponent(env):
    valid_actions = []
    for row in range(env.board_size):
        for col in range(env.board_size):
            if env.board[row, col] == env.current_player.value:
                moves = env._get_valid_moves((row, col))
                valid_actions.extend(
                    [(row, col, move[0], move[1]) for move in moves]
                )
    
    return random.choice(valid_actions) if valid_actions else None

def get_greedy_opponent(env):
    greedy_agent = GreedySurakartaAgent()
    return greedy_agent.get_action(env)

def get_defensive_opponent(env):
    defensive_agent = DefensiveSurakartaAgent()
    return defensive_agent.get_action(env)

def get_monte_carlo_opponent(env):
    monte_carlo_agent = MonteCarloSurakartaAgent()
    return monte_carlo_agent.get_action(env)

def get_minimax_opponent(env, depth=2):
    minimax_agent = MinimaxSurakartaAgent(max_depth=depth)
    return minimax_agent.get_action(env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Surakarta RL agent')
    parser.add_argument('--opponent', type=str, default='random', 
                        choices=['random', 'minimax'], 
                        help='Opponent type (random or minimax)')
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Number of episodes to train')
    parser.add_argument('--eval-freq', type=int, default=100,
                        help='Evaluation frequency')
    parser.add_argument('--minimax-depth', type=int, default=2,
                        help='Depth for minimax search (only used with minimax opponent)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to a saved model to continue training from')
    parser.add_argument('--save-dir', type=str, default='surakarta_checkpoints',
                        help='Directory to save models in')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Created directory: {args.save_dir}")
    
    if args.opponent == 'random':
        train_against_opponent('Random', get_random_opponent, 
                               episodes=args.episodes, 
                               eval_frequency=args.eval_freq,
                               model_path=args.model_path,
                               save_dir=args.save_dir)
    elif args.opponent == 'greedy':
        train_against_opponent('Greedy', get_greedy_opponent, 
                               episodes=args.episodes, 
                               eval_frequency=args.eval_freq,
                               model_path=args.model_path,
                               save_dir=args.save_dir)
    elif args.opponent == 'defensive':
        train_against_opponent('Defensive', get_defensive_opponent, 
                               episodes=args.episodes, 
                               eval_frequency=args.eval_freq,
                               model_path=args.model_path,
                               save_dir=args.save_dir)
    elif args.opponent == 'monte_carlo':
        train_against_opponent('Monte Carlo', get_monte_carlo_opponent, 
                               episodes=args.episodes, 
                               eval_frequency=args.eval_freq,
                               model_path=args.model_path,
                               save_dir=args.save_dir)
    elif args.opponent == 'minimax':
        minimax_func = partial(get_minimax_opponent, depth=args.minimax_depth)
        train_against_opponent(f'Minimax-{args.minimax_depth}', minimax_func, 
                               episodes=args.episodes, 
                               eval_frequency=args.eval_freq,
                               model_path=args.model_path,
                               save_dir=args.save_dir)
    