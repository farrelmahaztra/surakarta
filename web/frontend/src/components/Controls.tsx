import { AgentType } from "../types";

interface ControlsProps {
    onStartGame: (agentType: AgentType) => void;
    disabled?: boolean;
}

export default function Controls({ onStartGame, disabled }: ControlsProps) {
    return (
        <div className="flex flex-col items-center gap-6">
            <h1 className="text-3xl font-bold">Surakarta</h1>
            <div className="flex gap-4">
                <button
                    onClick={() => onStartGame(AgentType.Rule)}
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
                    onClick={() => onStartGame(AgentType.RL)}
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