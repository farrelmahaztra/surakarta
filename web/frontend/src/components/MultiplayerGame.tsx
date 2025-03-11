import { useState, useEffect, useCallback } from 'react';
import Board from './Board';
import { matchApi, gameApi } from '../api/game';
import { userApi } from '../api/game';
import { Move, GameState, Match, MultiplayerGameState, MatchStatus } from '../types';

interface MultiplayerGameProps {
  matchId: number;
  onReturnToMenu: () => void;
  onReturnToMatchmaking: () => void;
}

const MultiplayerGame = ({ matchId, onReturnToMenu, onReturnToMatchmaking }: MultiplayerGameProps) => {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [match, setMatch] = useState<Match | null>(null);
  const [multiplayerState, setMultiplayerState] = useState<MultiplayerGameState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showGameEnd, setShowGameEnd] = useState(false);
  const [gameResult, setGameResult] = useState<'win' | 'lose' | 'draw' | null>(null);
  const [refreshTimer, setRefreshTimer] = useState<number>(0);
  const [isMyTurn, setIsMyTurn] = useState(false);
  const [userColor, setUserColor] = useState<'Black' | 'White'>('Black');

  const currentUsername = userApi.getCurrentUser()?.username || '';

  const fetchMatchState = useCallback(async () => {
    try {
      console.log('Fetching match state for match ID:', matchId);
      const response = await matchApi.getMatchState(matchId);

      console.log('Match state response:', response);

      if (!response) {
        throw new Error('No response from server');
      }

      const { match: matchData, game_state } = response;

      if (!matchData || !game_state || !game_state.observation) {
        throw new Error('Invalid match state data');
      }

      setMatch(matchData);
      setMultiplayerState(game_state);

      const gameStateData: GameState = {
        board: game_state.observation.board || [],
        current_player: game_state.observation.current_player || 0,
        game_over: game_state.observation.game_over || false
      };

      console.log("Game state data:", gameStateData, game_state);

      setGameState(gameStateData);

      const isTurn = game_state.current_turn === currentUsername;
      console.log('Is my turn?', isTurn, game_state.current_turn, currentUsername);
      setIsMyTurn(isTurn);

      const isBlackPlayer = game_state.black_player === currentUsername;
      const isWhitePlayer = game_state.white_player === currentUsername;
      const playerColor = isBlackPlayer ? 'Black' : (isWhitePlayer ? 'White' : 'Black');
      setUserColor(playerColor);
      console.log('Player color:', playerColor, game_state.black_player, game_state.white_player, currentUsername);

      if (matchData.status === MatchStatus.Completed) {
        const playerWon = determineWinner(game_state, currentUsername);
        setGameResult(playerWon);
        setShowGameEnd(true);
      }

      setError(null);
    } catch (err) {
      console.error('Error fetching match state:', err);
      setError(`Failed to load match state: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
    }
  }, [matchId, currentUsername]);

  useEffect(() => {
    fetchMatchState();

    const intervalId = setInterval(() => {
      if (!isMyTurn && !showGameEnd) {
        setRefreshTimer(prev => prev + 1);
        fetchMatchState();
      }
    }, 3000);

    return () => clearInterval(intervalId);
  }, [fetchMatchState, isMyTurn, refreshTimer, showGameEnd]);

  const makeMove = async (move: Move) => {
    if (!gameState || loading || !matchId || showGameEnd || !isMyTurn) {
      console.log('Skipping move:', {
        hasGameState: !!gameState,
        isLoading: loading,
        hasMatchId: !!matchId,
        isGameOver: showGameEnd,
        isMyTurn
      });
      return;
    }

    setLoading(true);
    try {
      console.log('Making move:', move, 'for game:', match?.game_id);

      if (!match?.game_id) {
        throw new Error('No game ID available');
      }

      const result = await gameApi.makeMove(move, match.game_id);
      console.log('Move result:', result);

      await fetchMatchState();
    } catch (error) {
      console.error('Error making move:', error);
      setError(`Failed to make move: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setLoading(false);
    }
  };

  const forfeitMatch = async () => {
    if (loading || !matchId) return;

    if (window.confirm('Are you sure you want to forfeit this match?')) {
      try {
        setLoading(true);
        await matchApi.forfeitMatch(matchId);
        setGameResult('lose');
        setShowGameEnd(true);
      } catch (err) {
        console.error('Error forfeiting match:', err);
        setError('Failed to forfeit match. Please try again.');
      } finally {
        setLoading(false);
      }
    }
  };

  const determineWinner = (
    gameState: MultiplayerGameState,
    username: string
  ): 'win' | 'lose' | 'draw' => {
    if (!gameState.observation.game_over) {
      return 'draw';
    }

    const isBlackPlayer = gameState.black_player === username;
    const isWhitePlayer = gameState.white_player === username;

    const board = gameState.observation.board;
    let blackPieces = 0;
    let whitePieces = 0;

    for (let row = 0; row < board.length; row++) {
      for (let col = 0; col < board[row].length; col++) {
        if (board[row][col] === 1) {  // 1 = BLACK
          blackPieces++;
        } else if (board[row][col] === 2) {  // 2 = WHITE
          whitePieces++;
        }
      }
    }

    console.log(`Game over! Black pieces: ${blackPieces}, White pieces: ${whitePieces}`);

    if (blackPieces === 0) {
      return isWhitePlayer ? 'win' : 'lose';
    } else if (whitePieces === 0) {
      return isBlackPlayer ? 'win' : 'lose';
    }

    return 'draw';
  };

  if (loading && !gameState) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p>Loading match...</p>
        </div>
      </div>
    );
  }

  if (!gameState || !multiplayerState) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center p-6 bg-white rounded-lg shadow-md max-w-md">
          <h2 className="text-xl font-bold mb-4">Match {error ? 'Error' : 'Not Ready'}</h2>
          {error ? (
            <p className="text-red-600 mb-4">{error}</p>
          ) : (
            <p className="mb-4">Waiting for opponent to join...</p>
          )}
          <div className="flex justify-center space-x-4">
            <button
              onClick={onReturnToMatchmaking}
              className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-md"
            >
              Back to Matchmaking
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center flex-col h-full relative">
      <div className="w-full max-w-3xl bg-white p-4 rounded-lg shadow-md mb-4">
        <div className="flex justify-between items-center">
          <div>
            <p className="font-semibold">
              Black: {multiplayerState.black_player}
              {multiplayerState.black_player === currentUsername && " (You)"}
            </p>
            <p className="font-semibold">
              White: {multiplayerState.white_player || "Waiting..."}
              {multiplayerState.white_player === currentUsername && " (You)"}
            </p>
          </div>
          <div>
            <p className="text-center">
              <span className="font-bold">Current Turn: </span>
              {multiplayerState.current_turn || "Loading..."}
              {multiplayerState.current_turn === currentUsername && " (Your Turn)"}
            </p>
            <p className={`text-center mt-2 font-semibold ${isMyTurn ? 'text-green-600' : 'text-gray-600'}`}>
              {multiplayerState.black_player === currentUsername ? '(Black) ' : '(White) '}
              {isMyTurn ? 'Your turn to move' : 'Waiting for opponent...'}
            </p>
          </div>
          <div>
            <button
              onClick={forfeitMatch}
              disabled={loading || showGameEnd}
              className="px-3 py-1 bg-red-500 hover:bg-red-600 text-white text-sm rounded-md"
            >
              Forfeit Match
            </button>
          </div>
        </div>
      </div>

      <Board
        gameState={gameState}
        multiplayerState={multiplayerState}
        onMove={makeMove}
        disabled={!isMyTurn || loading || showGameEnd}
        playerColor={userColor}
      />

      {error && (
        <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-md max-w-md">
          {error}
        </div>
      )}

      {showGameEnd && (
        <div className="absolute inset-0 bg-black bg-opacity-70 flex flex-col items-center justify-center z-10">
          <div className="bg-white p-8 rounded-lg shadow-lg text-center max-w-md">
            <h2 className={`text-4xl font-bold mb-4 ${gameResult === 'win'
              ? 'text-green-600'
              : gameResult === 'lose'
                ? 'text-red-600'
                : 'text-blue-600'
              }`}>
              {gameResult === 'win'
                ? 'You Win!'
                : gameResult === 'lose'
                  ? 'You Lose!'
                  : 'Game Draw!'}
            </h2>
            <div className="mb-6">
              <p className="text-gray-700">
                Match against {match?.creator_username === currentUsername
                  ? match?.opponent_username
                  : match?.creator_username} has ended.
              </p>
            </div>
            <div className="flex justify-center space-x-4">
              <button
                onClick={onReturnToMenu}
                className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-md"
              >
                Return to Menu
              </button>
              <button
                onClick={onReturnToMatchmaking}
                className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-md"
              >
                Back to Matchmaking
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MultiplayerGame;