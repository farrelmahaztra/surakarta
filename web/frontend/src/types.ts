export enum AgentType {
    RL = 'rl',
    Rule = 'rule',
}

// This is from_row, from_col, to_row, to_col from the surakarta env
export type Move = [number, number, number, number]
export type Board = BoardState[][];

export enum BoardState {
    Empty = 0,
    Black = 1,
    White = 2,
}

export enum Player {
    Black = 0,
    White = 1,
}

export interface GameState {
    board: Board;
    current_player: Player;
    game_over: boolean;
    score?: number;
}
