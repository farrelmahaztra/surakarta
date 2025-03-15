import { AgentType, PlayerColor } from "../types";
import { useState } from "react";

interface ControlsProps {
    onStartGame: (agentType: AgentType, playerColor: PlayerColor) => void;
    disabled?: boolean;
}

export default function Controls({ onStartGame, disabled }: ControlsProps) {
    const [playerColor, setPlayerColor] = useState<PlayerColor>(PlayerColor.Black);

    return (
        <div className="flex flex-col items-center gap-6">
            <h1 className="text-3xl font-bold">Surakarta</h1>

            <div className="flex flex-col items-center gap-2">
                <label className="text-lg font-semibold">Choose your color:</label>
                <div className="flex gap-3">
                    <button
                        onClick={() => setPlayerColor(PlayerColor.Black)}
                        className={`
                            px-4 py-2 
                            rounded-lg shadow 
                            transition-colors
                            font-medium
                            ${playerColor === PlayerColor.Black
                                ? 'bg-gray-900 text-white border-2 border-green-600'
                                : ''}
                        `}
                    >
                        Black (First)
                    </button>
                    <button
                        onClick={() => setPlayerColor(PlayerColor.White)}
                        className={`
                            px-4 py-2 
                            rounded-lg shadow 
                            transition-colors
                            font-medium
                            ${playerColor === PlayerColor.White
                                ? 'bg-gray-900 text-white border-2 border-green-600'
                                : ''}
                        `}
                    >
                        White (Second)
                    </button>
                    <button
                        onClick={() => setPlayerColor(PlayerColor.Random)}
                        className={`
                            px-4 py-2 
                            rounded-lg shadow 
                            transition-colors
                            font-medium
                            ${playerColor === PlayerColor.Random
                                ? 'bg-gray-900 text-white border-2 border-green-600'
                                : ''}
                        `}
                    >
                        Random
                    </button>
                </div>
            </div>

            <div className="flex gap-4 mt-2">
                <button
                    onClick={() => onStartGame(AgentType.Rule, playerColor)}
                    disabled={disabled}
                    className="
                        px-6 py-3 
                        bg-green-900 text-white 
                        rounded-lg shadow 
                        hover:bg-green-800 
                        disabled:opacity-50
                        disabled:cursor-not-allowed
                        transition-colors
                        font-medium
                    "
                >
                    Play vs Rule Agent
                </button>
                <button
                    onClick={() => onStartGame(AgentType.RL, playerColor)}
                    disabled={disabled}
                    className="
                        px-6 py-3 
                        bg-green-900 text-white 
                        rounded-lg shadow 
                        hover:bg-green-800  
                        disabled:opacity-50
                        disabled:cursor-not-allowed
                        transition-colors
                        font-medium
                    "
                >
                    Play vs RL Agent
                </button>
            </div>
        </div>
    );
}