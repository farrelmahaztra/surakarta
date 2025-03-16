export enum AgentType {
    RL = 'rl',
    Rule = 'rule',
}

export enum GameMode {
    SinglePlayer = 'single_player',
    Multiplayer = 'multiplayer',
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

export interface User {
    id: number;
    username: string;
    email: string;
}

export interface UserCredentials {
    username: string;
    password: string;
}

export interface UserRegistration extends UserCredentials {
    email?: string;
    analytics_consent?: boolean;
}

export interface UserProfile {
    username: string;
    email?: string;
    games_played: number;
    wins: number;
    losses: number;
    draws: number;
    analytics_consent?: boolean;
}

export interface GameRecord {
    id: number;
    game_id: string;
    username: string;
    opponent_type: string;
    opponent_name?: string | null;
    start_time: string;
    end_time: string | null;
    final_score: number | null;
    result: 'win' | 'loss' | 'draw' | null;
}

export enum MatchStatus {
    Open = 'open',
    Matched = 'matched',
    InProgress = 'in_progress',
    Completed = 'completed',
    Abandoned = 'abandoned',
}

export enum PlayerColor {
    Black = 'black',
    White = 'white',
    Random = 'random',
}

export interface Match {
    id: number;
    game_id: string;
    creator_username: string;
    opponent_username: string | null;
    creator_color: PlayerColor;
    status: MatchStatus;
    created_at: string;
    updated_at: string;
    current_turn_username: string | null;
    last_activity: string;
    final_score: number | null;
    result: 'win' | 'loss' | 'draw' | null;
}

export interface MatchDetail extends Match {
    board_state: any;
    moves_history: Move[];
}

export interface MultiplayerGameState {
    observation: GameState;
    black_player: string;
    white_player: string | null;
    current_turn: string | null;
    status?: MatchStatus;
    moves_history?: Move[];
}

export interface CreateMatchOptions {
    creator_color: PlayerColor;
}
