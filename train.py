from engine.surakarta import SurakartaEnv
from agents.dqn import SurakartaRLAgent
from agents.minimax import MinimaxSurakartaAgent
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random


def get_intermediate_reward(env, action):
    _, _, to_row, to_col = action
    reward = 0

    if env.board[to_row, to_col] != 0: 
        reward += 0.5

    if 2 <= to_row <= 3 and 2 <= to_col <= 3:
        reward += 0.1

    arc_positions = {(0, 1), (0, 4), (1, 0), (1, 5), (4, 0), (4, 5), (5, 1), (5, 4)}
    if (to_row, to_col) in arc_positions:
        reward += 0.1

    return reward


def train(episodes=10000, eval_frequency=100):
    env = SurakartaEnv()
    rl_agent = SurakartaRLAgent(env)

    rl_agent.epsilon = 1.0 
    rl_agent.epsilon_decay = 0.998 
    rl_agent.batch_size = 64 

    minimax_agents = {
        "easy": MinimaxSurakartaAgent(max_depth=1),
        "medium": MinimaxSurakartaAgent(max_depth=2),
        "hard": MinimaxSurakartaAgent(max_depth=3),
    }

    rewards_history = []
    win_rates = {"easy": [], "medium": [], "hard": []}

    random_move_prob = 0.8

    reward_window = []
    window_size = 10

    pbar = tqdm(range(episodes), dynamic_ncols=True)

    for episode in pbar:
        state, _ = env.reset()
        episode_reward = 0
        episode_captures = 0
        done = False
        steps = 0
        max_steps = 200

        if episode < episodes * 0.3:
            current_minimax = minimax_agents["easy"]
        elif episode < episodes * 0.6: 
            current_minimax = minimax_agents["medium"]
        else:
            current_minimax = minimax_agents["hard"]

        while not done and steps < max_steps:
            action = rl_agent.get_action(state, training=True)
            if action is None:
                break

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            intermediate_reward = get_intermediate_reward(env, action)
            total_reward = reward + intermediate_reward

            _, _, to_row, to_col = action

            if env.board[to_row, to_col] != 0:
                episode_captures += 1

            if not done:
                if random.random() < random_move_prob:
                    valid_actions = []
                    for row in range(env.board_size):
                        for col in range(env.board_size):
                            if env.board[row, col] == env.current_player.value:
                                moves = env._get_valid_moves((row, col))
                                valid_actions.extend(
                                    [(row, col, move[0], move[1]) for move in moves]
                                )
                    minimax_action = (
                        random.choice(valid_actions) if valid_actions else None
                    )
                else:
                    minimax_action = current_minimax.get_action(env)

                if minimax_action is None:
                    break

                after_minimax_state, reward, terminated, truncated, _ = env.step(
                    minimax_action
                )
                done = terminated or truncated
                steps += 1

                rl_agent.memory.append(
                    (state, action, total_reward, after_minimax_state, done)
                )
                episode_reward += total_reward
                state = after_minimax_state
            else:
                rl_agent.memory.append((state, action, total_reward, next_state, done))
                episode_reward += total_reward

            if len(rl_agent.memory) >= rl_agent.batch_size:
                for _ in range(4): 
                    rl_agent.update_model()

        rewards_history.append(episode_reward)
        reward_window.append(episode_reward)
        if len(reward_window) > window_size:
            reward_window.pop(0)

        if (episode + 1) % window_size == 0:
            avg_reward = sum(reward_window) / len(reward_window)
            pbar.set_description(
                f"Avg Reward: {avg_reward:.2f} | Captures: {episode_captures} | ε: {rl_agent.epsilon:.3f} | Random Move Prob: {random_move_prob:.3f}"
            )

        random_move_prob = max(0.1, random_move_prob * 0.998)

        if (episode + 1) % eval_frequency == 0:
            print(f"\nEpisode {episode+1}")
            print(f"Random move probability: {random_move_prob:.2f}")
            print(f"Epsilon: {rl_agent.epsilon:.3f}")

            for difficulty in ["easy", "medium", "hard"]:
                win_rate = evaluate_against_minimax(
                    rl_agent, minimax_agents[difficulty], num_games=20
                )
                win_rates[difficulty].append(win_rate)
                print(f"Win Rate vs {difficulty} minimax: {win_rate:.2f}")

            torch.save(
                {
                    "model_state_dict": rl_agent.model.state_dict(),
                    "optimizer_state_dict": rl_agent.optimizer.state_dict(),
                    "epsilon": rl_agent.epsilon,
                },
                f"surakarta_agent_vs_minimax_episode_{episode+1}.pth",
            )

            plot_training_progress(rewards_history, win_rates, eval_frequency)


def evaluate_against_minimax(rl_agent, minimax_agent, num_games=20):
    env = SurakartaEnv()
    wins = 0

    for _ in range(num_games):
        state, _ = env.reset()
        done = False
        steps = 0
        max_steps = 200

        while not done and steps < max_steps:
            action = rl_agent.get_action(state, training=False)
            if action is None:
                break

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            if done:
                if reward > 0:
                    wins += 1
                continue

            action = minimax_agent.get_action(env)
            if action is None:
                break

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            if done and reward < 0:
                wins += 1

    return wins / num_games


def plot_training_progress(rewards_history, win_rates, eval_frequency):
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(rewards_history)
    ax1.set_title("Episode Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")

    eval_episodes = np.arange(eval_frequency, len(rewards_history) + 1, eval_frequency)
    for difficulty in win_rates:
        ax2.plot(eval_episodes, win_rates[difficulty], label=f"vs {difficulty}")
    ax2.set_title("Win Rates vs Different Minimax Depths")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Win Rate")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_progress_vs_minimax.png")
    plt.close()


if __name__ == "__main__":
    train()