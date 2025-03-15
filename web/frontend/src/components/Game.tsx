import { useState, useEffect } from 'react';
import Board from './Board';
import Controls from './Controls';
import { gameApi } from '../api/game'
import { AgentType, Move, GameState, Player, PlayerColor } from '../types';

interface GameProps {
    onReturnToMenu?: () => void;
    mode?: 'single' | 'multiplayer';
}

const Game = ({ onReturnToMenu, mode = 'single' }: GameProps) => {
    const [gameState, setGameState] = useState<GameState | null>(null);
    const [gameId, setGameId] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [gameResult, setGameResult] = useState<'win' | 'lose' | null>(null);
    const [showGameEnd, setShowGameEnd] = useState(false);
    const [userColor, setUserColor] = useState<PlayerColor>(PlayerColor.Black);

    const startGame = async (agentType: AgentType, playerColor: PlayerColor) => {
        setLoading(true);
        setGameResult(null);
        setShowGameEnd(false);
        setUserColor(playerColor);

        try {
            const { observation, game_id, player_color } = await gameApi.createGame(agentType, playerColor);
            setGameState(observation);
            setGameId(game_id);

            if (player_color) {
                const actualColor = player_color === "black" ? PlayerColor.Black : PlayerColor.White;
                setUserColor(actualColor);
                console.log("Game created with ID:", game_id, "Player color:", player_color);
            } else {
                console.log("Game created with ID:", game_id, "Player color:", playerColor);
            }
        } catch (error) {
            console.error('Error starting game:', error);
        } finally {
            setLoading(false);
        }
    };

    const makeMove = async (move: Move) => {
        if (!gameState || loading || !gameId || showGameEnd) return;

        setLoading(true);
        try {
            const { observation, reward, terminated, truncated, info } = await gameApi.makeMove(move, gameId);

            if (!observation) {
                throw new Error("No observation returned from makeMove");
            }

            let score = 0;
            if (userColor === PlayerColor.Black) {
                score = info?.white_pieces !== undefined ? 12 - info.white_pieces : 0;
            } else {
                score = info?.black_pieces !== undefined ? 12 - info.black_pieces : 0;
            }

            const newGameState = {
                ...observation,
                score: score
            };
            setGameState(newGameState);

            if (terminated || truncated) {
                let result: 'win' | 'lose' | null = null;

                if (userColor === PlayerColor.Black) {
                    if (newGameState.current_player === Player.White && info.white_pieces === 0) {
                        result = 'win';
                    } else if (newGameState.current_player === Player.Black && info.black_pieces === 0) {
                        result = 'lose';
                    } else {
                        result = reward > 0 ? 'win' : 'lose';
                    }
                } else {
                    if (newGameState.current_player === Player.Black && info.black_pieces === 0) {
                        result = 'win';
                    } else if (newGameState.current_player === Player.White && info.white_pieces === 0) {
                        result = 'lose';
                    } else {
                        result = reward < 0 ? 'win' : 'lose';
                    }
                }

                setGameResult(result);
                setShowGameEnd(true);
            }
        } catch (error) {
            console.error('Error making move:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        let timeoutId: number | undefined;

        if (showGameEnd && onReturnToMenu) {
            timeoutId = window.setTimeout(() => {
                onReturnToMenu();
            }, 5000);
        }

        return () => {
            if (timeoutId) {
                clearTimeout(timeoutId);
            }
        };
    }, [showGameEnd, onReturnToMenu]);

    const handleReturnToMenu = () => {
        if (onReturnToMenu) {
            onReturnToMenu();
        }
    };

    return (
        <div className="flex items-center justify-center flex-col h-full relative">
            {!gameState && <Controls onStartGame={startGame} />}
            {gameState && (
                <Board
                    gameState={gameState}
                    onMove={makeMove}
                    playerColor={userColor === PlayerColor.Black ? 'Black' : 'White'}
                />
            )}

            {showGameEnd && (
                <div className="absolute inset-0 bg-black bg-opacity-70 flex flex-col items-center justify-center z-10">
                    <div className="bg-white p-8 rounded-lg shadow-lg text-center max-w-md">
                        <h2 className={`text-4xl font-bold mb-4 ${gameResult === 'win' ? 'text-green-700' : 'text-red-600'}`}>
                            {gameResult === 'win' ? 'You Win!' : 'You Lose!'}
                        </h2>
                        {gameState?.score !== undefined && (
                            <p className="text-xl mb-6">
                                Final Score: <span className="font-bold">{gameState.score}</span>
                            </p>
                        )}
                        <div className="flex justify-center space-x-4">
                            <button
                                onClick={handleReturnToMenu}
                                className="bg-green-900 hover:bg-green-800 text-white py-2 px-6 rounded-lg"
                            >
                                Return to Menu
                            </button>
                            <button
                                onClick={() => {
                                    setShowGameEnd(false);
                                    startGame(AgentType.Rule, userColor);
                                }}
                                className="bg-green-900 hover:bg-green-800 text-white py-2 px-6 rounded-lg"
                            >
                                Play Again
                            </button>
                        </div>
                        <p className="mt-4 text-gray-500">
                            Automatically returning to menu in a few seconds...
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Game;