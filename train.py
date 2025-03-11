from engine.surakarta import SurakartaEnv
from agents.dqn import SurakartaRLAgent
from agents.minimax import MinimaxSurakartaAgent
from agents.monte_carlo import MonteCarloSurakartaAgent
from agents.greedy import GreedySurakartaAgent
from agents.defensive import DefensiveSurakartaAgent
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import argparse
from datetime import datetime
import json
from engine.surakarta import Player

def get_opponent(env, opponent_type, **opponent_kwargs):
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

    if opponent_type == "random":
        return get_random_opponent(env)
    elif opponent_type == "greedy":
        return GreedySurakartaAgent()
    elif opponent_type == "defensive":
        return DefensiveSurakartaAgent()
    elif opponent_type == "monte_carlo":
        return MonteCarloSurakartaAgent()
    elif opponent_type == "minimax":
        return MinimaxSurakartaAgent(max_depth=opponent_kwargs.get('depth', 2))
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")

def train_against_opponent(
    config_name,
    episodes=10000,
    eval_interval=500,
    save_dir="models",
    device="cpu",
    opponent_type="random",
    **opponent_kwargs
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config_name}_{timestamp}"
    run_dir = os.path.join(save_dir, run_name)
    
    os.makedirs(run_dir, exist_ok=True)
    
    config = {
        "config_name": config_name,
        "episodes": episodes,
        "eval_interval": eval_interval,
        "opponent_type": opponent_type
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    env = SurakartaEnv()
    dqn_agent = SurakartaRLAgent(env, device=device)
    
    all_rewards = []
    eval_rewards = []
    win_rates = []
    black_win_rates = []
    white_win_rates = []
    episode_lengths = []
    losses = []

    opponent = get_opponent(env, opponent_type, **opponent_kwargs)
  
    for episode in tqdm(range(episodes), desc=f"Training {config_name} vs {opponent_type}"):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        moves = 0
        
        dqn_plays_black = random.choice([True, False])
        
        while not done and moves < 200: 
            current_is_black = (env.current_player == Player.BLACK)
            is_dqn_turn = (current_is_black == dqn_plays_black)
            
            if is_dqn_turn: 
                action = dqn_agent.get_action(state, training=True)
                if action is None: 
                    break
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                dqn_agent.memory.append((state, action, reward, next_state, done))
                if dqn_agent.use_prioritized_replay:
                    if len(dqn_agent.priorities) < len(dqn_agent.memory):
                        dqn_agent.priorities.append(1.0)
                
                episode_reward += reward
                state = next_state
                moves += 1
                
                if done:
                    break
            else: 
                opponent_action = opponent.get_action(env)
                if opponent_action is None: 
                    break
                next_state, reward, terminated, truncated, _ = env.step(opponent_action)
                done = terminated or truncated
                
                state = next_state
                moves += 1
        
        if len(dqn_agent.memory) >= dqn_agent.batch_size:
            updates_per_episode = min(10, len(dqn_agent.memory) // dqn_agent.batch_size)
            for _ in range(updates_per_episode):
                loss = dqn_agent.update_model()
                if loss is not None:
                    losses.append(loss)
        
        all_rewards.append(episode_reward)
        episode_lengths.append(moves)
        
        if (episode + 1) % eval_interval == 0 or episode == episodes - 1:
            eval_reward, win_rate, black_win, white_win = evaluate_agent(
                dqn_agent, env, opponent, n_games=50
            )
            eval_rewards.append(eval_reward)
            win_rates.append(win_rate)
            black_win_rates.append(black_win)
            white_win_rates.append(white_win)
            
            print(f"\nEpisode {episode+1}/{episodes}")
            print(f"Average Reward (last 100): {np.mean(all_rewards[-100:]):.2f}")
            print(f"Evaluation Reward vs {opponent_type}: {eval_reward:.2f}")
            print(f"Overall Win Rate vs {opponent_type}: {win_rate:.2f}")
            print(f"Win Rate as Black vs {opponent_type}: {black_win:.2f}")
            print(f"Win Rate as White vs {opponent_type}: {white_win:.2f}")
            print(f"Epsilon: {dqn_agent.get_epsilon():.2f}")
            print(f"Average Loss: {np.mean(losses[-100:]) if losses else 'N/A'}")
            
            dqn_agent.save(f"{run_dir}/dqn_episode{episode+1}.pt")
            
            plot_metrics(all_rewards, eval_rewards, win_rates, episode_lengths, losses, 
                        f"{run_dir}/metrics_episode{episode+1}.png", opponent_type, 
                        black_win_rates, white_win_rates)
            
            metrics = {
                'rewards': np.array(all_rewards),
                'eval_rewards': np.array(eval_rewards),
                'win_rates': np.array(win_rates),
                'black_win_rates': np.array(black_win_rates),
                'white_win_rates': np.array(white_win_rates),
                'episode_lengths': np.array(episode_lengths),
                'losses': np.array(losses) if losses else np.array([])
            }
            np.savez(f"{run_dir}/metrics_episode{episode+1}.npz", **metrics)
    
    print(f"\nFinal Evaluation for {config_name}")
    eval_reward, win_rate, black_win, white_win = evaluate_agent(
        dqn_agent, env, opponent, n_games=100
    )
    print(f"Final Evaluation Reward: {eval_reward:.2f}")
    print(f"Final Overall Win Rate vs {opponent_type}: {win_rate:.2f}")
    print(f"Final Win Rate as Black vs {opponent_type}: {black_win:.2f}")
    print(f"Final Win Rate as White vs {opponent_type}: {white_win:.2f}")
    
    dqn_agent.save(f"{run_dir}/dqn_final.pt")
    
    return dqn_agent, all_rewards, eval_rewards, win_rates, run_dir

def evaluate_agent(dqn_agent, env, opponent, n_games=50):
    rewards = []
    black_wins = 0
    white_wins = 0
    
    for i in range(n_games):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        moves = 0
        
        as_black = i % 2 == 0
        while not done and moves < 200:
            if as_black:
                action = dqn_agent.get_action(state, training=False)
            else:
                action = opponent.get_action(env)

            if action is None:
                break

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if as_black:
                episode_reward += reward
           
            state = next_state
            moves += 1
            
            if done:
                winner = env._check_win()
                if as_black and winner == Player.BLACK:
                    black_wins += 1
                elif not as_black and winner == Player.WHITE:
                    white_wins += 1
                break
            
            if as_black :
                action = opponent.get_action(env)
            else:
                action = dqn_agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if not as_black:
                episode_reward += reward
            
            state = next_state
            moves += 1
            
            if done: 
                winner = env._check_win()
                if as_black and winner == Player.BLACK:
                    black_wins += 1
                elif not as_black and winner == Player.WHITE:
                    white_wins += 1
                break
        
        rewards.append(episode_reward)
    
    wins = black_wins + white_wins
    win_rate = wins / n_games 
    black_win_rate = black_wins / n_games 
    white_win_rate = white_wins / n_games 

    return np.mean(rewards), win_rate, black_win_rate, white_win_rate


def plot_metrics(rewards, eval_rewards, win_rates, episode_lengths, losses, filename, 
               opponent_type="Random", black_win_rates=None, white_win_rates=None):
    _, axs = plt.subplots(3, 2, figsize=(15, 15))
    
    axs[0, 0].plot(rewards)
    axs[0, 0].set_title('Episode Rewards')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    
    window_size = min(100, len(rewards))
    if window_size > 0:
        reward_moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        axs[0, 1].plot(reward_moving_avg)
        axs[0, 1].set_title(f'Reward Moving Average (Window {window_size})')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Average Reward')
    
    if len(eval_rewards) > 0:
        eval_episodes = np.arange(0, len(rewards), max(1, len(rewards)//(len(eval_rewards) or 1)))[:len(eval_rewards)]
        if len(eval_episodes) == len(eval_rewards):
            axs[1, 0].plot(eval_episodes, eval_rewards, marker='o')
        else:
            axs[1, 0].plot(eval_rewards, marker='o')
        axs[1, 0].set_title('Evaluation Rewards')
        axs[1, 0].set_xlabel('Episode')
        axs[1, 0].set_ylabel('Reward')
    
    if len(win_rates) > 0:
        if len(eval_episodes) == len(win_rates):
            axs[1, 1].plot(eval_episodes, win_rates, marker='o', label='Overall')
            
            if black_win_rates is not None and white_win_rates is not None:
                if len(eval_episodes) == len(black_win_rates):
                    axs[1, 1].plot(eval_episodes, black_win_rates, marker='s', label='As Black')
                    axs[1, 1].plot(eval_episodes, white_win_rates, marker='^', label='As White')
                else:
                    axs[1, 1].plot(black_win_rates, marker='s', label='As Black')
                    axs[1, 1].plot(white_win_rates, marker='^', label='As White')
        else:
            axs[1, 1].plot(win_rates, marker='o', label='Overall')
            
            if black_win_rates is not None and white_win_rates is not None:
                axs[1, 1].plot(black_win_rates, marker='s', label='As Black')
                axs[1, 1].plot(white_win_rates, marker='^', label='As White')
        
        axs[1, 1].set_title(f"Win Rate vs {opponent_type}")
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('Win Rate')
        axs[1, 1].set_ylim([0, 1])
        axs[1, 1].legend()
    
    axs[2, 0].plot(episode_lengths)
    axs[2, 0].set_title('Episode Lengths')
    axs[2, 0].set_xlabel('Episode')
    axs[2, 0].set_ylabel('Steps')
    
    if losses:
        window_size = min(100, len(losses))
        if window_size > 0:
            loss_moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            axs[2, 1].plot(loss_moving_avg)
            axs[2, 1].set_title('Loss Moving Average')
            axs[2, 1].set_xlabel('Update Step')
            axs[2, 1].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Surakarta RL agent')
    parser.add_argument('--opponent', type=str, default='random', 
                        choices=['random', 'minimax'], 
                        help='Opponent type (random or minimax)')
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Number of episodes to train')
    parser.add_argument('--eval-interval', type=int, default=100,
                        help='Evaluation frequency')
    parser.add_argument('--minimax-depth', type=int, default=1,
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
        train_against_opponent('Random',
                               opponent_type="random",
                               episodes=args.episodes, 
                               eval_interval=args.eval_interval,
                               save_dir=args.save_dir)
    elif args.opponent == 'greedy':
        train_against_opponent('Greedy', 
                               opponent_type="greedy",
                               episodes=args.episodes, 
                               eval_interval=args.eval_interval,
                               save_dir=args.save_dir)
    elif args.opponent == 'defensive':
        train_against_opponent('Defensive',
                               opponent_type="defensive",
                               episodes=args.episodes, 
                               eval_interval=args.eval_interval,
                               save_dir=args.save_dir)
    elif args.opponent == 'monte_carlo':
        train_against_opponent('Monte Carlo',
                               opponent_type="monte_carlo",
                               episodes=args.episodes, 
                               eval_interval=args.eval_interval,
                               save_dir=args.save_dir)
    elif args.opponent == 'minimax':
        train_against_opponent(f'Minimax-{args.minimax_depth}',
                               opponent_type="minimax",
                               depth=args.minimax_depth,
                               episodes=args.episodes, 
                               eval_interval=args.eval_interval,
                               save_dir=args.save_dir)
    