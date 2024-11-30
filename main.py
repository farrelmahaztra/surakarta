import time
import pygame
from engine.surakarta import SurakartaEnv
from agents.minimax import MinimaxSurakartaAgent


def get_board_position(pos, cell_size, board_offset):
    x, y = pos
    col = round((x - board_offset) / cell_size)
    row = round((y - board_offset) / cell_size)
    return row, col


def play_against_minimax():
    env = SurakartaEnv(render_mode="human")
    agent = MinimaxSurakartaAgent(max_depth=3)
    _, _ = env.reset()

    selected_piece = None
    human_player = 1
    done = False

    while not done:
        current_player = env.current_player.value

        if current_player == human_player:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    row, col = get_board_position(
                        mouse_pos, env.cell_size, env.board_offset
                    )

                    if 0 <= row < env.board_size and 0 <= col < env.board_size:
                        if selected_piece is None:
                            if env.board[row, col] == current_player:
                                selected_piece = (row, col)
                                env.selected_piece = (row, col)
                                env.valid_moves = env._get_valid_moves((row, col))
                        else:
                            if (row, col) in env.valid_moves:
                                from_row, from_col = selected_piece
                                _, _, terminated, truncated, _ = env.step(
                                    (from_row, from_col, row, col)
                                )
                                done = terminated or truncated

                            selected_piece = None
                            env.selected_piece = None
                            env.valid_moves = []
        else:
            action = agent.get_action(env)
            if action is None:
                print("AI has no valid moves!")
                break

            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            time.sleep(0.5)

        env._render_frame()

        if done:
            winner = "Human" if env.current_player.value != human_player else "AI"
            print(f"{winner} wins!")

    env.close()


if __name__ == "__main__":
    play_against_minimax()
