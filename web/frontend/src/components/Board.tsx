import { useState } from 'react';
import { BoardState, GameState, Move, MultiplayerGameState, Player } from '../types';
import { userApi } from '../api/game';

interface BoardProps {
    gameState: GameState;
    onMove: (move: Move) => void;
    disabled?: boolean;
    playerColor?: 'Black' | 'White';
    multiplayerState?: MultiplayerGameState;
}

export default function Board({ gameState, onMove, disabled, playerColor = 'Black', multiplayerState }: BoardProps) {
    const [selectedCell, setSelectedCell] = useState<[number, number] | null>(null);
    const username = userApi.getCurrentUser()?.username || '';

    const handleCellClick = (row: number, col: number) => {
        if (disabled || !gameState) return;

        const myPieceValue = playerColor === 'Black' ? BoardState.Black : BoardState.White;

        console.log("Game state:", gameState);
        const isMyTurn =
            multiplayerState?.current_turn === username ||
            (playerColor === 'Black' && gameState.current_player === Player.Black) ||
            (playerColor === 'White' && gameState.current_player === Player.White);

        if (!isMyTurn) {
            console.log("Not your turn!", gameState.current_player, playerColor);
            return;
        }

        if (selectedCell) {
            if (selectedCell[0] === row && selectedCell[1] === col) {
                setSelectedCell(null);
            } else {
                onMove([selectedCell[0], selectedCell[1], row, col]);
                setSelectedCell(null);
            }
        } else {
            if (gameState.board[row][col] === myPieceValue) {
                setSelectedCell([row, col]);
            } else {
                console.log("Not your piece!", gameState.board[row][col], myPieceValue);
            }
        }
    };

    const shouldRotateBoard = playerColor === 'White';

    const renderCell = (value: BoardState, row: number, col: number) => {
        const isSelected = selectedCell?.[0] === row && selectedCell?.[1] === col;
        const cellSize = 60;
        const offset = 150;

        const displayRow = shouldRotateBoard ? 5 - row : row;
        const displayCol = shouldRotateBoard ? 5 - col : col;

        return (
            <div
                key={`${row}-${col}`}
                onClick={() => handleCellClick(row, col)}
                className="absolute flex items-center justify-center"
                style={{
                    width: `${cellSize}px`,
                    height: `${cellSize}px`,
                    left: `${offset + displayCol * cellSize - cellSize / 2}px`,
                    top: `${offset + displayRow * cellSize - cellSize / 2}px`,
                }}
            >
                {value !== 0 && (
                    <div
                        className={`
                            w-12 h-12 rounded-full transition-transform
                            ${value === 1 ? 'bg-black' : 'bg-white'}
                            ${isSelected ? 'scale-110 ring-4 ring-yellow-400' : ''}
                            ${!disabled && 'hover:ring-2 hover:ring-yellow-400'}
                        `}
                    />
                )}
            </div>
        );
    };

    return (
        <div className="flex flex-col items-center gap-6 p-12">
            <div className="text-center">
                <div className="mb-2 text-lg font-semibold">
                    <span className={`mr-2 inline-block w-4 h-4 rounded-full ${shouldRotateBoard ? 'bg-black' : 'bg-white border border-black'}`}></span>
                    {shouldRotateBoard ? 'Black' : 'White'} (Opponent)
                </div>
            </div>
            <div
                className="relative w-[600px] h-[600px] rounded-xl overflow-hidden"
                style={{
                    backgroundImage: 'url("/board_template.png")',
                    backgroundSize: 'cover'
                }}
            >
                {gameState.board.map((row, i) => (
                    row.map((cell, j) => renderCell(cell, i, j))
                ))}
            </div>
            <div className="text-center">
                <div className="mt-2 text-lg font-semibold">
                    <span className={`mr-2 inline-block w-4 h-4 rounded-full ${playerColor === 'Black' ? 'bg-black' : 'bg-white border border-black'}`}></span>
                    {playerColor} (You)
                </div>

                {gameState.game_over ? (
                    <div className="text-2xl font-bold mt-4">
                        Game Over! {gameState.current_player === 0 ? "You won!" : "AI won!"}
                    </div>
                ) : null}
            </div>
        </div>
    );
}