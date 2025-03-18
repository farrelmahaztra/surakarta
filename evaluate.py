from engine.surakarta import SurakartaEnv, Player
from agents.dqn import SurakartaRLAgent
import numpy as np
import torch
import os
import time
import argparse
import json
from datetime import datetime
from train import get_opponent, evaluate_agent
from tabulate import tabulate

def evaluate(model_path, device="cpu", n_games=100, save_results=True):
    opponents = [
        {"type": "random", "name": "Random"},
        {"type": "greedy", "name": "Greedy"},
        {"type": "defensive", "name": "Defensive"},
        {"type": "monte_carlo", "name": "Monte Carlo"},
        {"type": "minimax", "name": "Minimax-1", "depth": 1},
        {"type": "minimax", "name": "Minimax-2", "depth": 2},
        {"type": "minimax", "name": "Minimax-3", "depth": 3}
    ]
    
    env = SurakartaEnv()
    agent = SurakartaRLAgent(env, device=device)
    
    try:
        checkpoint = torch.load(model_path, map_location=agent.device)
        agent.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        agent.target_net.load_state_dict(checkpoint["q_net_state_dict"])
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    results = {}
    start_time = time.time()
    table_data = []
    
    print(f"Evaluating {model_path} for {n_games} games per opponent")
    
    for opp in opponents:
        opp_name = opp["name"]
        opp_type = opp["type"]
        opp_kwargs = {k: v for k, v in opp.items() if k not in ["type", "name"]}
        
        print(f"Evaluating against {opp_name}...")
        opponent = get_opponent(opp_type, **opp_kwargs)
        
        stats = {"w": 0, "l": 0, "d": 0, "bw": 0, "bl": 0, "bd": 0, "ww": 0, "wl": 0, "wd": 0}
        rewards = []
        
        for i in range(n_games):
            agent_is_black = i % 2 == 0
            state, _ = env.reset()
            episode_reward = 0
            done = False
            moves = 0
            
            while not done and moves < 200:
                current_is_black = (env.current_player == Player.BLACK)
                agent_turn = (current_is_black and agent_is_black) or (not current_is_black and not agent_is_black)
                
                action = agent.get_action(state, training=False) if agent_turn else opponent.get_action(env)
                
                if action is None:
                    if agent_turn:
                        stats["l"] += 1
                        stats["bl" if agent_is_black else "wl"] += 1
                    else:
                        stats["w"] += 1
                        stats["bw" if agent_is_black else "ww"] += 1
                    done = True
                    break
                
                try:
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    if (agent_is_black and current_is_black) or (not agent_is_black and not current_is_black):
                        episode_reward += reward
                    
                    state = next_state
                    moves += 1
                except Exception as e:
                    print(f"Error in step: {e}")
                    break
            
            if done:
                winner = env._check_win()
                if (winner == Player.BLACK and agent_is_black) or (winner == Player.WHITE and not agent_is_black):
                    stats["w"] += 1
                    stats["bw" if agent_is_black else "ww"] += 1
                elif (winner == Player.WHITE and agent_is_black) or (winner == Player.BLACK and not agent_is_black):
                    stats["l"] += 1
                    stats["bl" if agent_is_black else "wl"] += 1
                else:
                    stats["d"] += 1
                    stats["bd" if agent_is_black else "wd"] += 1
            elif moves >= 200:
                stats["d"] += 1
                stats["bd" if agent_is_black else "wd"] += 1
            
            rewards.append(episode_reward)
            
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{n_games}, W:{stats['w']}, L:{stats['l']}, D:{stats['d']}")
        
        total = stats["w"] + stats["l"] + stats["d"]
        black_total = stats["bw"] + stats["bl"] + stats["bd"]
        white_total = stats["ww"] + stats["wl"] + stats["wd"]
        
        win_rate = stats["w"] / total if total > 0 else 0
        black_win_rate = stats["bw"] / black_total if black_total > 0 else 0
        white_win_rate = stats["ww"] / white_total if white_total > 0 else 0
        draw_rate = stats["d"] / total if total > 0 else 0
        
        results[opp_name] = {
            "win_rate": win_rate,
            "black_win_rate": black_win_rate,
            "white_win_rate": white_win_rate,
            "avg_reward": np.mean(rewards),
            "draws": draw_rate,
            "stats": {
                "total_games": total,
                "wins": stats["w"],
                "losses": stats["l"],
                "draws": stats["d"],
                "black_wins": stats["bw"],
                "black_losses": stats["bl"],
                "black_draws": stats["bd"],
                "white_wins": stats["ww"],
                "white_losses": stats["wl"],
                "white_draws": stats["wd"]
            }
        }
        
        table_data.append([
            opp_name,
            f"{win_rate:.2f}",
            f"{black_win_rate:.2f}",
            f"{white_win_rate:.2f}",
            f"{np.mean(rewards):.2f}",
            f"{draw_rate:.2f}"
        ])
    
    print("\nEvaluation Results:")
    print(tabulate(table_data, 
                   headers=["Opponent", "Win Rate", "As Black", "As White", "Avg. Reward", "Draw Rate"], 
                   tablefmt="grid"))
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f} seconds")
    
    if save_results:
        model_name = os.path.basename(model_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.dirname(model_path) or "."
        results_file = os.path.join(results_dir, f"eval_{model_name}_{timestamp}.json")
        
        with open(results_file, "w") as f:
            json.dump({
                "model_path": model_path,
                "timestamp": timestamp,
                "n_games_per_opponent": n_games,
                "results": results
            }, f, indent=4)
        
        print(f"Results saved to {results_file}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a Surakarta RL agent')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the saved model file')
    parser.add_argument('--device', type=str, default='cpu',
                      choices=['cpu', 'cuda', 'mps'],
                      help='Device to run evaluation on')
    parser.add_argument('--n-games', type=int, default=100,
                      help='Number of games to play against each opponent')
    parser.add_argument('--opponent', type=str, default=None,
                      choices=['random', 'greedy', 'defensive', 'monte_carlo', 'minimax-1', 'minimax-2', 'minimax-3'],
                      help='Specific opponent to evaluate against (evaluates against all if not specified)')
    parser.add_argument('--minimax-depth', type=int, default=1,
                      help='Depth for minimax search (only used with minimax opponent)')
    
    args = parser.parse_args()
    
    if args.opponent:
        env = SurakartaEnv()
        agent = SurakartaRLAgent(env, device=args.device)
        
        try:
            checkpoint = torch.load(args.model_path, map_location=agent.device)
            agent.q_net.load_state_dict(checkpoint["q_net_state_dict"])
            agent.target_net.load_state_dict(checkpoint["q_net_state_dict"])
            print(f"Loaded model from {args.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)
        
        if args.opponent.startswith("minimax"):
            opponent_type = "minimax"
            depth = int(args.opponent.split("-")[1]) if "-" in args.opponent else args.minimax_depth
            opponent = get_opponent(opponent_type, depth=depth)
            opponent_name = f"Minimax-{depth}"
        else:
            opponent = get_opponent(args.opponent)
            opponent_name = args.opponent.capitalize()
        
        print(f"Evaluating against {opponent_name} for {args.n_games} games...")
        
        reward, win_rate, black_win_rate, white_win_rate = evaluate_agent(
            agent, env, opponent, n_games=args.n_games
        )
        
        print(f"Results vs {opponent_name}: WR={win_rate:.2f}, Black={black_win_rate:.2f}, White={white_win_rate:.2f}, Reward={reward:.2f}")
    else:
        evaluate(args.model_path, device=args.device, n_games=args.n_games)