from engine.surakarta import SurakartaEnv
from agents.dqn import SurakartaRLAgent
from agents.minimax import MinimaxSurakartaAgent
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random


def train(episodes=10_000, eval_frequency=100):
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

    random_move_prob = 1

    reward_window = []
    window_size = 10

    pbar = tqdm(range(episodes), dynamic_ncols=True)

    wins_per_difficulty = {"easy": 0, "medium": 0, "hard": 0}
    games_per_difficulty = {"easy": 0, "medium": 0, "hard": 0}

    for episode in pbar:
        state, _ = env.reset()
        episode_reward = 0
        episode_captures = 0
        done = False
        steps = 0
        max_steps = 500

        if episode < episodes * 0.4:
            current_difficulty = "easy"
        elif episode < episodes * 0.7:
            current_difficulty = "medium"
        else:
            current_difficulty = "hard"

        current_minimax = minimax_agents[current_difficulty]
        games_per_difficulty[current_difficulty] += 1

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

            if reward == 3.0:
                wins_per_difficulty[current_difficulty] += 1

            _, _, to_row, to_col = action

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
                    (state, action, reward, after_minimax_state, done)
                )
                episode_reward += reward
                state = after_minimax_state
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
                f"Avg Reward: {avg_reward:.2f} | Captures: {episode_captures} | ε: {rl_agent.epsilon:.3f} | Random Move Prob: {random_move_prob:.3f}"
            )

        random_move_prob = max(0.1, random_move_prob * 0.999)

        rl_agent.epsilon = max(
            0.01, rl_agent.epsilon * rl_agent.epsilon_decay
        ) 

        if (episode + 1) % eval_frequency == 0:
            print(f"\nEpisode {episode+1}")
            print(f"Random move probability: {random_move_prob:.2f}")
            print(f"Epsilon: {rl_agent.epsilon:.3f}")

            for difficulty in ["easy", "medium", "hard"]:
                if games_per_difficulty[difficulty] > 0:
                    win_rate = (
                        wins_per_difficulty[difficulty]
                        / games_per_difficulty[difficulty]
                    )
                    win_rates[difficulty].append(win_rate)
                    print(
                        f"Win Rate vs {difficulty} minimax: {win_rate:.2f} ({wins_per_difficulty[difficulty]}/{games_per_difficulty[difficulty]})"
                    )

            torch.save(
                {
                    "q_net_state_dict": rl_agent.q_net.state_dict(),
                    "target_net_state_dict": rl_agent.target_net.state_dict(), 
                    "optimizer_state_dict": rl_agent.optimizer.state_dict(),
                    "steps_done": rl_agent.steps_done, 
                },
                f"surakarta_agent_vs_minimax_episode_{episode+1}.pth",
            )

            plot_training_progress(rewards_history, win_rates, eval_frequency)


def plot_training_progress(rewards_history, win_rates, eval_frequency):
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(rewards_history)
    ax1.set_title("Episode Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")

    eval_episodes = np.arange(eval_frequency, len(rewards_history) + 1, eval_frequency)
    for difficulty in win_rates:
        if len(win_rates[difficulty]) > 0: 
            ax2.plot(
                eval_episodes[: len(win_rates[difficulty])],
                win_rates[difficulty],
                label=f"vs {difficulty}",
            )

    ax2.set_title("Win Rates vs Different Minimax Depths")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Win Rate")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_progress_vs_minimax.png")
    plt.close()


def train_against_random(episodes=10_000, eval_frequency=100):
    env = SurakartaEnv()
    rl_agent = SurakartaRLAgent(env)

    rl_agent.epsilon = 1.0
    rl_agent.epsilon_decay = 0.998
    rl_agent.batch_size = 64

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

            if reward == 3.0: 
                wins += 1

            if not done:
                valid_actions = []
                for row in range(env.board_size):
                    for col in range(env.board_size):
                        if env.board[row, col] == env.current_player.value:
                            moves = env._get_valid_moves((row, col))
                            valid_actions.extend(
                                [(row, col, move[0], move[1]) for move in moves]
                            )

                random_action = random.choice(valid_actions) if valid_actions else None

                if random_action is None:
                    break

                after_random_state, reward, terminated, truncated, _ = env.step(
                    random_action
                )
                done = terminated or truncated
                steps += 1

                rl_agent.memory.append(
                    (state, action, reward, after_random_state, done)
                )
                episode_reward += reward
                state = after_random_state
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
            print(f"\nEpisode {episode+1}")
            print(f"Epsilon: {rl_agent.epsilon:.3f}")

            win_rate = wins / games
            win_rates.append(win_rate)
            print(f"Win Rate: {win_rate:.2f} ({wins}/{games})")

            torch.save(
                {
                    "q_net_state_dict": rl_agent.q_net.state_dict(),
                    "target_net_state_dict": rl_agent.target_net.state_dict(),
                    "optimizer_state_dict": rl_agent.optimizer.state_dict(),
                    "steps_done": rl_agent.steps_done,
                },
                f"surakarta_agent_vs_random_episode_{episode+1}.pth",
            )

            plot_training_progress_random(rewards_history, win_rates, eval_frequency)


def plot_training_progress_random(rewards_history, win_rates, eval_frequency):
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(rewards_history)
    ax1.set_title("Episode Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")

    eval_episodes = np.arange(eval_frequency, len(rewards_history) + 1, eval_frequency)
    ax2.plot(eval_episodes, win_rates, label="vs Random")
    ax2.set_title("Win Rates vs Random Opponent")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Win Rate")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_progress_vs_random.png")
    plt.close()


if __name__ == "__main__":
    train_against_random()
