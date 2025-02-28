import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class SurakartaConvFeatures(nn.Module):
    def __init__(self, board_size=6):
        super().__init__()
        self.board_size = board_size
        
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.flat_size = 128 * board_size * board_size
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x.view(-1, self.flat_size)


class DuelingCNNSurakartaNet(nn.Module):
    def __init__(self, board_size=6):
        super().__init__()
        self.board_size = board_size
        
        self.features = SurakartaConvFeatures(board_size)
        self.flat_size = self.features.flat_size
        
        self.piece_fc1 = nn.Linear(self.flat_size, 256)
        self.piece_ln1 = nn.LayerNorm(256)
        self.piece_fc2 = nn.Linear(256, board_size * board_size)
        
        self.move_fc1 = nn.Linear(self.flat_size + board_size * board_size, 256) 
        self.move_ln1 = nn.LayerNorm(256)
        self.move_fc2 = nn.Linear(256, board_size * board_size)
        
        self.val_fc1 = nn.Linear(self.flat_size, 256)
        self.val_ln1 = nn.LayerNorm(256)
        self.val_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        features = self.features(x)
        
        piece_adv = F.relu(self.piece_ln1(self.piece_fc1(features)))
        piece_logits = self.piece_fc2(piece_adv)
        
        if self.training:
            selected_piece = piece_logits
        else:
            selected_piece_idx = piece_logits.argmax(dim=1)
            selected_piece = F.one_hot(selected_piece_idx, self.board_size * self.board_size).float()
            selected_piece = selected_piece.view(-1, self.board_size * self.board_size)
        
        move_input = torch.cat([features, selected_piece], dim=1)
        move_adv = F.relu(self.move_ln1(self.move_fc1(move_input)))
        move_logits = self.move_fc2(move_adv)
        
        val = F.relu(self.val_ln1(self.val_fc1(features)))
        val = self.val_fc2(val)
        
        piece_advantage = piece_logits - piece_logits.mean(dim=1, keepdim=True)
        move_advantage = move_logits - move_logits.mean(dim=1, keepdim=True)
        
        batch_size = x.size(0)
        q_values = torch.zeros(batch_size, self.board_size**4, device=x.device)
        
        for i in range(self.board_size * self.board_size):
            piece_row = i // self.board_size
            piece_col = i % self.board_size
            
            for j in range(self.board_size * self.board_size):
                move_row = j // self.board_size
                move_col = j % self.board_size
                
                action_idx = (piece_row * self.board_size**3 + 
                              piece_col * self.board_size**2 + 
                              move_row * self.board_size + 
                              move_col)
                
                q_values[:, action_idx] = val.squeeze() + piece_advantage[:, i] + move_advantage[:, j]
        
        return q_values


class SurakartaRLAgent:
    def __init__(self, env, device="cpu"):
        self.env = env
        self.device = device
        self.board_size = env.board_size
        
        self.q_net = DuelingCNNSurakartaNet(self.board_size).to(device)
        self.target_net = DuelingCNNSurakartaNet(self.board_size).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.use_double_dqn = True
        
        self.batch_size = 128
        self.gamma = 0.99
        self.optimizer = torch.optim.RMSprop(
            self.q_net.parameters(),
            lr=0.0001,
            weight_decay=1e-5,
        )

        self.use_prioritized_replay = True
        self.memory = deque(maxlen=100000) 
        self.priorities = deque(maxlen=100000)
        self.alpha = 0.6 
        self.beta = 0.4 
        self.beta_increment = 0.001 

        self.eps_start = 1.0
        self.eps_end = 0.02
        self.eps_decay = 10000 
        self.steps_done = 0

        self.target_update_interval = 5000
        self.tau = 0.1 
        
        self._init_arc_positions()

    def _init_arc_positions(self):
        self.inner_loop_positions = set()
        self.outer_loop_positions = set()
        
        for arc_name, points in self.env.arcs.items():
            if "inner" in arc_name:
                self.inner_loop_positions.update(points)
            elif "outer" in arc_name:
                self.outer_loop_positions.update(points)

    def get_epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -self.steps_done / self.eps_decay
        )

    def get_action(self, state, training=True):
        epsilon = self.get_epsilon() if training else 0.01

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

        if self.use_prioritized_replay:
            self.beta = min(1.0, self.beta + self.beta_increment) 
            priorities = np.array(self.priorities)
            probs = priorities ** self.alpha
            probs /= probs.sum()
            
            indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
            batch = [self.memory[idx] for idx in indices]
            
            weights = (len(self.memory) * probs[indices]) ** (-self.beta)
            weights /= weights.max()
            weights = torch.tensor(weights, device=self.device, dtype=torch.float32).unsqueeze(1)
        else:
            batch = random.sample(self.memory, self.batch_size)
            indices = None
            weights = torch.ones((self.batch_size, 1), device=self.device)

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
            if self.use_double_dqn:
                next_action_indices = self.q_net(next_state_batch).max(1, keepdim=True)[1]
                next_q_values = self.target_net(next_state_batch).gather(1, next_action_indices)
            else:
                next_q_values = self.target_net(next_state_batch).max(1, keepdim=True)[0]
                
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()

        loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
        loss = (loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10) 
        self.optimizer.step()

        if self.use_prioritized_replay and indices is not None:
            for idx, error in zip(indices, td_errors):
                self.priorities[idx] = error[0] + 1e-5 

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
        
        inner_loop_channel = np.zeros_like(black_channel)
        outer_loop_channel = np.zeros_like(black_channel)
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                if (row, col) in self.inner_loop_positions:
                    inner_loop_channel[row, col] = 1.0
                if (row, col) in self.outer_loop_positions:
                    outer_loop_channel[row, col] = 1.0
        
        state_tensor = np.stack([black_channel, white_channel, inner_loop_channel, outer_loop_channel])
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
    
    def save(self, path):
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']