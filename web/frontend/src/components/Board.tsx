import { useState } from 'react';
import { BoardState, GameState, Move } from '../types';

interface BoardProps {
    gameState: GameState;
    onMove: (move: Move) => void;
    disabled?: boolean;
}

export default function Board({ gameState, onMove, disabled }: BoardProps) {
    const [selectedCell, setSelectedCell] = useState<[number, number] | null>(null);

    const handleCellClick = (row: number, col: number) => {
        if (disabled || !gameState || gameState.current_player !== 0) return;

        if (selectedCell) {
            if (selectedCell[0] === row && selectedCell[1] === col) {
                setSelectedCell(null);
            } else {
                onMove([selectedCell[0], selectedCell[1], row, col]);
                setSelectedCell(null);
            }
        } else {
            if (gameState.board[row][col] === 1) {
                setSelectedCell([row, col]);
            }
        }
    };

    const renderCell = (value: BoardState, row: number, col: number) => {
        const isSelected = selectedCell?.[0] === row && selectedCell?.[1] === col;
        const cellSize = 60;
        const offset = 150;

        return (
            <div
                key={`${row}-${col}`}
                onClick={() => handleCellClick(row, col)}
                className="absolute flex items-center justify-center"
                style={{
                    width: `${cellSize}px`,
                    height: `${cellSize}px`,
                    left: `${offset + col * cellSize - cellSize / 2}px`,
                    top: `${offset + row * cellSize - cellSize / 2}px`,
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
                {gameState.game_over && (
                    <div className="text-2xl font-bold">
                        Game Over! {gameState.current_player === 0 ? "You won!" : "AI won!"}
                    </div>
                )}
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
                {gameState.game_over ? (
                    <div className="text-2xl font-bold">
                        Game Over! {gameState.current_player === 0 ? "You won!" : "AI won!"}
                    </div>
                ) : <div className="text-2xl font-bold">
                    {'Score: ' + (gameState.score ?? 0)}
                </div>}
            </div>
        </div>
    );
}