import { AgentType, Move } from "../types";

const API_URL = 'http://localhost:8000/api';

export const gameApi = {
    createGame: async (agentType: AgentType) => {
        const response = await fetch(`${API_URL}/game/create_game/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ agent_type: agentType }),
        });
        return response.json();
    },

    makeMove: async (move: Move, gameId: string) => {
        const response = await fetch(`${API_URL}/game/make_move/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ move, game_id: gameId }),
        });
        return response.json();
    },
};