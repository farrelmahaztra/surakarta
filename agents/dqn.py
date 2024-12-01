import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class SurakartaNet(nn.Module):
    def __init__(self, board_size=6):
        super().__init__()
        self.board_size = board_size

        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.q_net = nn.Sequential(
            nn.Linear(128 * board_size * board_size, 512),
            nn.ReLU(),
            nn.Linear(512, board_size * board_size * board_size * board_size),
        )

    def forward(self, x):
        features = self.features(x)
        return self.q_net(features)


class SurakartaRLAgent:
    def __init__(self, env, device="mps"):
        self.env = env
        self.device = device

        self.q_net = SurakartaNet().to(device)
        self.target_net = SurakartaNet().to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.memory = deque(maxlen=100000)
        self.batch_size = 32
        self.gamma = 0.99

        self.exploration_fraction = 0.1
        self.exploration_initial_eps = 1.0
        self.exploration_final_eps = 0.05
        self.max_steps = 10000 
        self.exploration_schedule = lambda steps_done: self.exploration_final_eps + (
            self.exploration_initial_eps - self.exploration_final_eps
        ) * max(0, 1 - steps_done / (self.max_steps * self.exploration_fraction))

        self.steps_done = 0
        self.target_update_interval = 1000
        self.tau = 0.005  
        self.max_grad_norm = 10.0

    def get_state_tensor(self, state):
        board = state["board"]

        black_channel = (board == 1).astype(np.float32)
        white_channel = (board == 2).astype(np.float32)

        state_tensor = np.stack([black_channel, white_channel])
        return torch.FloatTensor(state_tensor).unsqueeze(0).to(self.device)

    def get_action(self, state, training=True):
        epsilon = self.exploration_schedule(self.steps_done) if training else 0.0

        if training and random.random() < epsilon:
            valid_actions = []
            for row in range(self.env.board_size):
                for col in range(self.env.board_size):
                    if self.env.board[row, col] == self.env.current_player.value:
                        moves = self.env._get_valid_moves((row, col))
                        valid_actions.extend(
                            [(row, col, move[0], move[1]) for move in moves]
                        )
            return random.choice(valid_actions) if valid_actions else None

        state_tensor = self.get_state_tensor(state)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)

        valid_actions_mask = self.get_valid_actions_mask()
        q_values = q_values * valid_actions_mask

        if q_values.sum() == 0:
            return None

        action_idx = torch.argmax(q_values).item()
        return self.idx_to_action(action_idx)

    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.get_action(state)
                if action is None:
                    break

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                if len(self.memory) >= self.batch_size:
                    self.update_model()

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if episode % 100 == 0:
                print(
                    f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.2f}"
                )

    def update_model(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_batch = torch.cat([self.get_state_tensor(s) for s, _, _, _, _ in batch])
        action_batch = torch.tensor(
            [self.action_to_idx(a) for _, a, _, _, _ in batch], device=self.device
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            [r for _, _, r, _, _ in batch], device=self.device
        ).unsqueeze(1)
        next_state_batch = torch.cat(
            [self.get_state_tensor(s) for _, _, _, s, _ in batch]
        )
        done_batch = torch.tensor(
            [float(d) for _, _, _, _, d in batch],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(1)

        current_q_values = self.q_net(state_batch)
        current_q_values = current_q_values.gather(1, action_batch)

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            next_q_values = next_q_values.max(1, keepdim=True)[0]
            target_q_values = (
                reward_batch + (1 - done_batch) * self.gamma * next_q_values
            )

        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        for target_param, param in zip(
            self.target_net.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        self.steps_done += 1

    def action_to_idx(self, action):
        from_row, from_col, to_row, to_col = action
        return (
            from_row * self.env.board_size * self.env.board_size * self.env.board_size
            + from_col * self.env.board_size * self.env.board_size
            + to_row * self.env.board_size
            + to_col
        )

    def idx_to_action(self, idx):
        board_size = self.env.board_size
        from_row = idx // (board_size * board_size * board_size)
        idx %= board_size * board_size * board_size
        from_col = idx // (board_size * board_size)
        idx %= board_size * board_size
        to_row = idx // board_size
        to_col = idx % board_size
        return (from_row, from_col, to_row, to_col)

    def get_valid_actions_mask(self):
        mask = torch.zeros((1, self.env.board_size**4), device=self.device)
        for row in range(self.env.board_size):
            for col in range(self.env.board_size):
                if self.env.board[row, col] == self.env.current_player.value:
                    moves = self.env._get_valid_moves((row, col))
                    for move_row, move_col in moves:
                        idx = self.action_to_idx((row, col, move_row, move_col))
                        mask[0, idx] = 1
        return mask