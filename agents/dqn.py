import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class SurakartaNet(nn.Module):
    def __init__(self, board_size=6):
        super().__init__()
        self.board_size = board_size

        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.action_head = nn.Sequential(
            nn.Linear(128 * board_size * board_size, 512),
            nn.ReLU(),
            nn.Linear(512, board_size * board_size * board_size * board_size),
        )

        self.value_head = nn.Sequential(
            nn.Linear(128 * board_size * board_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x = x.view(-1, 128 * self.board_size * self.board_size)

        action_logits = self.action_head(x)
        value = self.value_head(x)

        return action_logits, value


class SurakartaRLAgent:
    def __init__(self, env, device="mps"):
        self.env = env
        self.device = device
        self.model = SurakartaNet().to(device)
        self.target_model = SurakartaNet().to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)

        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.update_counter = 0

    def get_state_tensor(self, state):
        board = state["board"]

        black_channel = (board == 1).astype(np.float32)
        white_channel = (board == 2).astype(np.float32)

        state_tensor = np.stack([black_channel, white_channel])
        return torch.FloatTensor(state_tensor).unsqueeze(0).to(self.device)

    def get_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            valid_actions = []
            for row in range(self.env.board_size):
                for col in range(self.env.board_size):
                    if self.env.board[row, col] == self.env.current_player.value:
                        moves = self.env._get_valid_moves((row, col))
                        valid_actions.extend(
                            [(row, col, move[0], move[1]) for move in moves]
                        )

            if not valid_actions:
                return None
            return random.choice(valid_actions)

        state_tensor = self.get_state_tensor(state)
        action_logits, _ = self.model(state_tensor)
        action_probs = torch.softmax(action_logits, dim=1)

        valid_actions_mask = self.get_valid_actions_mask()
        action_probs = action_probs * valid_actions_mask

        if action_probs.sum() == 0:
            return None

        action_idx = torch.argmax(action_probs).item()
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
        batch = random.sample(self.memory, self.batch_size)
        state_batch = torch.cat([self.get_state_tensor(s) for s, _, _, _, _ in batch])

        current_q_values, _ = self.model(state_batch)

        next_state_batch = torch.cat(
            [self.get_state_tensor(s) for _, _, _, s, _ in batch]
        )
        with torch.no_grad():
            next_q_values, _ = self.target_model(next_state_batch)

        target_q_values = current_q_values.clone()
        for i, (_, action, reward, _, done) in enumerate(batch):
            if done:
                target = reward
            else:
                target = reward + self.gamma * torch.max(next_q_values[i])

            action_idx = self.action_to_idx(action)
            target_q_values[i, action_idx] = target

        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

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