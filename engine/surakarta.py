import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import math
from enum import Enum
from typing import Optional, Dict, Tuple, List

class Player(Enum):
    NONE = 0
    BLACK = 1
    WHITE = 2


class SurakartaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None, window_size: int = 800):
        super().__init__()

        self.window_size = window_size
        self.board_size = 6
        self.cell_size = (window_size - 400) // (self.board_size - 1)
        self.piece_radius = self.cell_size // 3
        self.board_offset = 200

        self.move_cache = {}
        self.capture_cache = {}
        self.colors = {
            "BACKGROUND": (1, 50, 32),
            "LINES": (255, 255, 255),
            "BLACK_PIECE": (0, 0, 0),
            "WHITE_PIECE": (255, 245, 208),
            "SELECTED": (109, 113, 46),
            "VALID_MOVE": (109, 113, 46, 128),
        }

        self.action_space = spaces.MultiDiscrete([6, 6, 6, 6])

        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=2, shape=(6, 6), dtype=np.int8),
                "current_player": spaces.Discrete(2), 
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.board: Optional[np.ndarray[tuple[int, int], np.dtype[np.int8]]] = None
        self.current_player: Optional[Player] = None
        self.selected_piece: Optional[Tuple[int, int]] = None
        self.valid_moves: List[Tuple[int, int]] = []

        self.arcs = {
            "top_left_inner": [(1, 0), (0, 1)],
            "top_left_outer": [(2, 0), (0, 2)],
            "top_right_inner": [(1, 5), (0, 4)],
            "top_right_outer": [(2, 5), (0, 3)],
            "bottom_left_inner": [(4, 0), (5, 1)],
            "bottom_left_outer": [(3, 0), (5, 2)],
            "bottom_right_inner": [(4, 5), (5, 4)],
            "bottom_right_outer": [(3, 5), (5, 3)],
        }

    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)

        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.board[0:2, :] = Player.WHITE.value
        self.board[-2:, :] = Player.BLACK.value

        self.current_player = Player.BLACK
        self.selected_piece = None
        self.valid_moves = []

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()
    
    def get_intermediate_reward(self, action):
        _, _, to_row, to_col = action

        reward = 0

        if self.board[to_row, to_col] != 0: 
            reward += 0.5

        if 2 <= to_row <= 3 and 2 <= to_col <= 3:
            reward += 0.1

        arc_positions = {(0, 1), (0, 4), (1, 0), (1, 5), (4, 0), (4, 5), (5, 1), (5, 4)}

        if (to_row, to_col) in arc_positions:
            reward += 0.1

        return reward

    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        from_row, from_col, to_row, to_col = action

        if not self._is_valid_move(from_row, from_col, to_row, to_col):
            return self._get_obs(), -1.0, False, False, self._get_info()

        reward = self.get_intermediate_reward(action)

        self._make_move(from_row, from_col, to_row, to_col)

        winner = self._check_win()
        terminated = winner is not None

        if terminated:
            reward = 10.0 if winner == self.current_player else -10.0

        self.current_player = (
            Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
        )

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _get_obs(self) -> Dict:
        if self.board is None:
            return {
                "board": np.zeros((self.board_size, self.board_size)),
                "current_player": 0,
            }
        return {
            "board": self.board.copy(),
            "current_player": 0 if self.current_player == Player.BLACK else 1,
        }

    def _get_info(self) -> Dict:
        return {
            "black_pieces": np.count_nonzero(self.board == Player.BLACK.value),
            "white_pieces": np.count_nonzero(self.board == Player.WHITE.value),
        }

    def _is_valid_move(
        self, from_row: int, from_col: int, to_row: int, to_col: int
    ) -> bool:
        if (
            self.board is not None
            and self.current_player is not None
            and self.board[from_row, from_col] != self.current_player.value
        ):
            return False

        valid_moves = self._get_valid_moves((from_row, from_col))
        return (to_row, to_col) in valid_moves

    def _make_move(self, from_row: int, from_col: int, to_row: int, to_col: int):
        if self.board is None:
            return

        self.board[to_row, to_col] = self.board[from_row, from_col]
        self.board[from_row, from_col] = Player.NONE.value

    def _get_valid_moves(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        row, col = pos

        if self.board is None:
            return []

        board_tuple = tuple(map(tuple, self.board))
        current_player_value = self.current_player.value if self.current_player else 0
        cache_key = (pos, board_tuple, current_player_value)

        if cache_key in self.move_cache:
            return self.move_cache[cache_key]

        moves: List[Tuple[int, int]] = []

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                new_row, new_col = row + dr, col + dc
                if (
                    0 <= new_row < self.board_size
                    and 0 <= new_col < self.board_size
                    and self.board[new_row, new_col] == Player.NONE.value
                ):
                    moves.append((new_row, new_col))

        moves.extend(self._get_capture_moves(pos))
        
        self.move_cache[cache_key] = moves
        return moves

    def _get_capture_moves(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        if not self.current_player:
            return []
            
        board_tuple = tuple(map(tuple, self.board))
        current_player_value = self.current_player.value
        cache_key = (pos, board_tuple, current_player_value)
        
        if cache_key in self.capture_cache:
            return self.capture_cache[cache_key]
            
        captures = set()
        current_player = self.current_player.value
        opponent = (
            Player.WHITE.value
            if current_player == Player.BLACK.value
            else Player.BLACK.value
        )

        def follow_path(self, direction: str) -> List[Tuple[int, int]]:
            arc_used = False
            path_clear = True
            curr_row, curr_col = pos
            used_arcs = set()

            while (
                curr_row >= 0
                and curr_row < self.board_size
                and curr_col >= 0
                and curr_col < self.board_size
                and path_clear 
            ):
                curr_piece = self.board[curr_row, curr_col]

                if curr_piece == current_player and (curr_row, curr_col) != pos:
                    return []

                if curr_piece == opponent:
                    if arc_used:
                        return [(curr_row, curr_col)]
                    else:
                        return []

                current_pos = (curr_row, curr_col)
                found_arc = False

                for arc_name, points in self.arcs.items():
                    if current_pos in points and arc_name not in used_arcs:

                        current_point = (
                            points[0] if current_pos == points[0] else points[1]
                        )

                        if (
                            (current_point[0] == 0 and direction != "up")
                            or (current_point[0] == 5 and direction != "down")
                            or (current_point[1] == 0 and direction != "left")
                            or (current_point[1] == 5 and direction != "right")
                        ):
                            return []

                        other_point = (
                            points[1] if current_pos == points[0] else points[0]
                        )

                        other_point_piece = self.board[other_point[0], other_point[1]]

                        if (
                            other_point_piece == Player.NONE.value
                            or other_point_piece == opponent
                        ):
                            arc_used = True
                            used_arcs.add(arc_name)
                            curr_row, curr_col = other_point

                            if other_point_piece == opponent:
                                return [other_point]

                            if current_point[0] == 0:
                                direction = "right" if "left" in arc_name else "left"
                            elif current_point[0] == 5:
                                direction = "right" if "left" in arc_name else "left"
                            elif current_point[1] == 0:
                                direction = "down" if "top" in arc_name else "up"
                            elif current_point[1] == 5:
                                direction = "down" if "top" in arc_name else "up"

                            found_arc = True
                            break
                        else:
                            return []

                if found_arc:
                    continue

                if direction == "up":
                    curr_row -= 1
                elif direction == "down":
                    curr_row += 1
                elif direction == "left":
                    curr_col -= 1
                elif direction == "right":
                    curr_col += 1

            return []

        captures.update(follow_path(self, "up"))
        captures.update(follow_path(self, "down"))
        captures.update(follow_path(self, "left"))
        captures.update(follow_path(self, "right"))
        
        result = list(captures)
        self.capture_cache[cache_key] = result
        
        return result

    def _check_win(self) -> Optional[Player]:
        black_pieces = np.count_nonzero(self.board == Player.BLACK.value)
        white_pieces = np.count_nonzero(self.board == Player.WHITE.value)

        if black_pieces == 0:
            return Player.WHITE
        elif white_pieces == 0:
            return Player.BLACK
        return None

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Surakarta")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.colors["BACKGROUND"])

        for i in range(self.board_size):
            pos = i * self.cell_size + self.board_offset
            pygame.draw.line(
                canvas,
                self.colors["LINES"],
                (pos, self.board_offset),
                (pos, self.board_offset + (self.board_size - 1) * self.cell_size),
            )
            pygame.draw.line(
                canvas,
                self.colors["LINES"],
                (self.board_offset, pos),
                (self.board_offset + (self.board_size - 1) * self.cell_size, pos),
            )

        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.board[row, col]
                if piece != Player.NONE.value:
                    center = (
                        col * self.cell_size + self.board_offset,
                        row * self.cell_size + self.board_offset,
                    )
                    color = (
                        self.colors["BLACK_PIECE"]
                        if piece == Player.BLACK.value
                        else self.colors["WHITE_PIECE"]
                    )
                    pygame.draw.circle(canvas, color, center, self.piece_radius)

        arc_radii = [self.cell_size, self.cell_size * 2]

        for corner in [(0, 0), (0, 5), (5, 0), (5, 5)]:
            row, col = corner
            center_x = self.board_offset + col * self.cell_size
            center_y = self.board_offset + row * self.cell_size

            if row == 0 and col == 0: 
                start_angle = 0
                end_angle = 270
            elif row == 0 and col == 5: 
                start_angle = -90
                end_angle = 180
            elif row == 5 and col == 0: 
                start_angle = 90
                end_angle = 360
            else: 
                start_angle = -180
                end_angle = 90

            for radius in arc_radii:
                rect = pygame.Rect(
                    center_x - radius, center_y - radius, radius * 2, radius * 2
                )
                pygame.draw.arc(
                    canvas,
                    self.colors["LINES"],
                    rect,
                    math.radians(start_angle),
                    math.radians(end_angle),
                    2,
                )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()