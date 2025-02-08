import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class SurakartaNet(nn.Module):
    def __init__(self, board_size=6):
        super().__init__()
        self.board_size = board_size

        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.flat_size = 64 * board_size * board_size

        self.fc1 = nn.Linear(self.flat_size, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, board_size * board_size * board_size * board_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, self.flat_size)
        x = F.relu(self.ln1(self.fc1(x)))
        return self.fc2(x)


class SurakartaRLAgent:
    def __init__(self, env, device="cpu"):
        self.env = env
        self.device = device

        self.q_net = SurakartaNet().to(device)
        self.target_net = SurakartaNet().to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.batch_size = 128  
        self.gamma = 0.95 
        self.optimizer = torch.optim.AdamW(
            self.q_net.parameters(),
            lr=3e-4,
            weight_decay=1e-4, 
        )

        self.memory = deque(maxlen=50000)

        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = 20000 
        self.steps_done = 0

        self.target_update_interval = 1000
        self.tau = 0.005 

    def get_epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -self.steps_done / self.eps_decay
        )

    def get_action(self, state, training=True):
        epsilon = self.get_epsilon() if training else 0.05

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

        with torch.no_grad():
            state_tensor = self.get_state_tensor(state)
            q_values = self.q_net(state_tensor)
            valid_mask = self.get_valid_actions_mask()
            q_values = q_values * valid_mask

            if q_values.sum() == 0:
                return None

            action_idx = q_values.max(1)[1].item()
            return self.idx_to_action(action_idx)

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

        current_q_values = self.q_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1, keepdim=True)[0]
            target_q_values = (
                reward_batch + (1 - done_batch) * self.gamma * next_q_values
            )

        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10) 
        self.optimizer.step()

        if self.steps_done % self.target_update_interval == 0:
            for target_param, param in zip(
                self.target_net.parameters(), self.q_net.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )

        self.steps_done += 1

        return loss.item()

    def get_state_tensor(self, state):
        board = state["board"]
        black_channel = (board == 1).astype(np.float32)
        white_channel = (board == 2).astype(np.float32)
        state_tensor = np.stack([black_channel, white_channel])
        return torch.FloatTensor(state_tensor).unsqueeze(0).to(self.device)

    def action_to_idx(self, action):
        from_row, from_col, to_row, to_col = action
        board_size = self.env.board_size
        return (
            from_row * board_size * board_size * board_size
            + from_col * board_size * board_size
            + to_row * board_size
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