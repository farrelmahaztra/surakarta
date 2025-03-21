import {
    AgentType,
    Move,
    UserCredentials,
    UserRegistration,
    CreateMatchOptions,
    Match,
    PlayerColor,
    MultiplayerGameState
} from "../types";

// @ts-ignore
const API_URL = process.env.VITE_API_URL || '/api';

const getAuthToken = () => localStorage.getItem('authToken');

const getAuthHeaders = () => {
    const headers: Record<string, string> = {
        'Content-Type': 'application/json',
    };

    const token = getAuthToken();
    if (token) {
        headers['Authorization'] = `Token ${token}`;
    }

    return headers;
};

export const gameApi = {
    createGame: async (agentType: AgentType, playerColor: PlayerColor = PlayerColor.Black) => {
        const response = await fetch(`${API_URL}/game/create_game/`, {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify({
                agent_type: agentType,
                player_color: playerColor
            }),
        });
        const data = await response.json();

        if (data.player_color) {
            return {
                ...data,
                player_color: data.player_color
            };
        }

        return data;
    },

    makeMove: async (move: Move, gameId: string) => {
        const response = await fetch(`${API_URL}/game/make_move/`, {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify({ move, game_id: gameId }),
        });
        return response.json();
    },
};

export const matchApi = {
    createMatch: async (options: CreateMatchOptions) => {
        const response = await fetch(`${API_URL}/matches/create_match/`, {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify(options),
        });

        if (!response.ok) {
            throw new Error('Failed to create match');
        }

        return response.json();
    },

    listOpenMatches: async () => {
        const url = `${API_URL}/matches/list_open_matches/`;

        const response = await fetch(url, {
            method: 'GET',
            headers: getAuthHeaders(),
        });

        if (!response.ok) {
            throw new Error('Failed to list open matches');
        }

        return response.json();
    },

    getMyMatches: async () => {
        const response = await fetch(`${API_URL}/matches/my_matches/`, {
            method: 'GET',
            headers: getAuthHeaders(),
        });

        if (!response.ok) {
            throw new Error('Failed to get your matches');
        }

        return response.json();
    },

    joinMatch: async (matchId: number) => {
        const response = await fetch(`${API_URL}/matches/${matchId}/join_match/`, {
            method: 'POST',
            headers: getAuthHeaders(),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to join match');
        }

        return response.json();
    },

    getMatchState: async (matchId: number): Promise<{ match: Match, game_state: MultiplayerGameState }> => {
        const response = await fetch(`${API_URL}/matches/${matchId}/match_state/`, {
            method: 'GET',
            headers: getAuthHeaders(),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to get match state');
        }

        return response.json();
    },

    forfeitMatch: async (matchId: number) => {
        const response = await fetch(`${API_URL}/matches/${matchId}/forfeit_match/`, {
            method: 'POST',
            headers: getAuthHeaders(),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to forfeit match');
        }

        return response.json();
    },
};

export const userApi = {
    register: async (userData: UserRegistration) => {
        const response = await fetch(`${API_URL}/users/register/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(userData),
        });

        const data = await response.json();

        if (response.ok) {
            localStorage.setItem('authToken', data.token);
            localStorage.setItem('user', JSON.stringify(data.user));
        }

        return { data, success: response.ok };
    },

    login: async (credentials: UserCredentials) => {
        const response = await fetch(`${API_URL}/users/login/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(credentials),
        });

        const data = await response.json();

        if (response.ok) {
            localStorage.setItem('authToken', data.token);
            localStorage.setItem('user', JSON.stringify(data.user));
        }

        return { data, success: response.ok };
    },

    logout: () => {
        localStorage.removeItem('authToken');
        localStorage.removeItem('user');
    },

    updateProfile: async (profileData: any) => {
        const response = await fetch(`${API_URL}/users/profile/`, {
            method: 'PUT',
            headers: getAuthHeaders(),
            body: JSON.stringify(profileData),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to update profile');
        }

        return data;
    },

    updatePassword: async (currentPassword: string, newPassword: string) => {
        const response = await fetch(`${API_URL}/users/update_password/`, {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify({
                current_password: currentPassword,
                new_password: newPassword
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to update password');
        }

        const data = await response.json();
        if (data.token) {
            localStorage.setItem('authToken', data.token);
        }

        return data;
    },

    deleteAccount: async () => {
        const response = await fetch(`${API_URL}/users/profile/`, {
            method: 'DELETE',
            headers: getAuthHeaders(),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to delete account');
        }

        localStorage.removeItem('authToken');
        localStorage.removeItem('user');

        return true;
    },

    getUserProfile: async () => {
        const response = await fetch(`${API_URL}/users/profile/`, {
            method: 'GET',
            headers: getAuthHeaders(),
        });

        return response.json();
    },

    getGameHistory: async () => {
        const response = await fetch(`${API_URL}/users/game_history/`, {
            method: 'GET',
            headers: getAuthHeaders(),
        });

        return response.json();
    },

    isAuthenticated: () => {
        return !!localStorage.getItem('authToken');
    },

    getCurrentUser: () => {
        const userStr = localStorage.getItem('user');
        return userStr ? JSON.parse(userStr) : null;
    }
};