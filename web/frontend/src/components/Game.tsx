import { useState } from 'react';
import Board from './Board';
import Controls from './Controls';
import { gameApi } from '../api/game'
import { AgentType, Move, GameState } from '../types';

const Game = () => {
    const [gameState, setGameState] = useState<GameState | null>(null);
    const [gameId, setGameId] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);

    const startGame = async (agentType: AgentType) => {
        setLoading(true);
        try {
            const { observation, game_id } = await gameApi.createGame(agentType);
            setGameState(observation);
            setGameId(game_id);
            console.log("Game created with ID:", game_id);
        } catch (error) {
            console.error('Error starting game:', error);
        } finally {
            setLoading(false);
        }
    };

    const makeMove = async (move: Move) => {
        if (!gameState || loading || !gameId) return;

        setLoading(true);
        try {
            const { observation } = await gameApi.makeMove(move, gameId);
            setGameState(observation);
        } catch (error) {
            console.error('Error making move:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex items-center justify-center flex-col h-full">
            {!gameState && <Controls onStartGame={startGame} />}
            {gameState && <Board gameState={gameState} onMove={makeMove} />}
        </div>
    );
};

export default Game;