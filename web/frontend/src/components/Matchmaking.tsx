import { useState, useEffect } from 'react';
import { matchApi } from '../api/game';
import { Match, PlayerColor, MatchStatus } from '../types';

interface MatchmakingProps {
  onStartMultiplayerGame: (matchId: number) => void;
  onBackToMenu: () => void;
}

const Matchmaking = ({ onStartMultiplayerGame, onBackToMenu }: MatchmakingProps) => {
  const [openMatches, setOpenMatches] = useState<Match[]>([]);
  const [myMatches, setMyMatches] = useState<Match[]>([]);
  const [createMatchColor, setCreateMatchColor] = useState<PlayerColor>(PlayerColor.Random);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshTimer, setRefreshTimer] = useState<number>(0);

  useEffect(() => {
    const loadMatches = async () => {
      try {
        setLoading(true);

        const openMatchesResp = await matchApi.listOpenMatches();
        console.log('Open matches response:', openMatchesResp);

        const myMatchesResp = await matchApi.getMyMatches();
        console.log('My matches response:', myMatchesResp);

        const openMatchList = Array.isArray(openMatchesResp) ? openMatchesResp : [];
        const myMatchList = Array.isArray(myMatchesResp) ? myMatchesResp : [];

        setOpenMatches(openMatchList);
        setMyMatches(myMatchList);
        setError(null);
      } catch (err) {
        console.error('Error loading matches:', err);
        setError('Failed to load matches. Please try again.');
      } finally {
        setLoading(false);
      }
    };

    loadMatches();

    const intervalId = setInterval(() => {
      setRefreshTimer(prev => prev + 1);
    }, 10000);

    return () => clearInterval(intervalId);
  }, [refreshTimer]);

  const handleCreateMatch = async () => {
    try {
      setLoading(true);
      const result = await matchApi.createMatch({ creator_color: createMatchColor });
      console.log('Create match result:', result);

      if (result && result.match && result.match.id) {
        setRefreshTimer(prev => prev + 1);

        setError(null);
        alert('Match created successfully! Waiting for an opponent to join.');
      }
    } catch (err) {
      console.error('Error creating match:', err);
      setError('Failed to create match. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleJoinMatch = async (matchId: number) => {
    try {
      setLoading(true);
      await matchApi.joinMatch(matchId);

      onStartMultiplayerGame(matchId);
    } catch (err) {
      console.error('Error joining match:', err);
      setError('Failed to join match. Please try again.');
      setLoading(false);
    }
  };

  const handleContinueMatch = (matchId: number) => {
    onStartMultiplayerGame(matchId);
  };

  console.log({ openMatches, myMatches });

  return (
    <div className="max-w-4xl mx-auto p-4">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">Multiplayer Match</h2>
        <button
          onClick={onBackToMenu}
          className="px-4 py-2 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-md"
        >
          Back
        </button>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-md">
          {error}
        </div>
      )}

      <div className="mb-8 p-4 bg-white rounded-lg shadow-md">
        <h3 className="text-lg font-semibold mb-4">Create a New Match</h3>
        <div className="flex items-center mb-4">
          <label className="mr-4">Choose your color preference:</label>
          <select
            value={createMatchColor}
            onChange={(e) => setCreateMatchColor(e.target.value as PlayerColor)}
            className="px-3 py-2 border border-gray-300 rounded-md"
            disabled={loading}
          >
            <option value={PlayerColor.Random}>Random</option>
            <option value={PlayerColor.Black}>Black (First Move)</option>
            <option value={PlayerColor.White}>White (Second Move)</option>
          </select>
        </div>
        <button
          onClick={handleCreateMatch}
          disabled={loading}
          className="px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-md"
        >
          {loading ? 'Creating...' : 'Create Match'}
        </button>
      </div>

      <div className="mb-8">
        <h3 className="text-lg font-semibold mb-4">My Active Matches</h3>
        {myMatches.length === 0 ? (
          <p className="text-gray-500">You don't have any active matches.</p>
        ) : (
          <div className="grid gap-4">
            {myMatches.map(match => (
              <div key={match.id} className="p-4 bg-white rounded-lg shadow-md">
                <div className="flex justify-between mb-2">
                  <div>
                    <span className="font-semibold">Match with: </span>
                    {match.opponent_username || 'Waiting for opponent'}
                  </div>
                  <div className={`px-2 py-1 text-sm rounded-full ${match.status === MatchStatus.Open
                    ? 'bg-yellow-100 text-yellow-800'
                    : match.status === MatchStatus.InProgress
                      ? 'bg-green-100 text-green-800'
                      : 'bg-blue-100 text-blue-800'
                    }`}>
                    {match.status}
                  </div>
                </div>
                <div className="mb-2">
                  <span className="text-sm text-gray-600">Created: {new Date(match.created_at).toLocaleString()}</span>
                </div>
                <div className="mb-2">
                  <span className="text-sm text-gray-600">
                    Current turn: {match.current_turn_username || 'N/A'}
                  </span>
                </div>
                {match.status === MatchStatus.InProgress && (
                  <button
                    onClick={() => handleContinueMatch(match.id)}
                    className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-md w-full"
                  >
                    Continue Match
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-4">Open Matches to Join</h3>
        {openMatches.length === 0 ? (
          <p className="text-gray-500">No open matches available to join.</p>
        ) : (
          <div className="grid gap-4">
            {openMatches.map(match => (
              <div key={match.id} className="p-4 bg-white rounded-lg shadow-md">
                <div className="flex justify-between mb-2">
                  <div>
                    <span className="font-semibold">Creator: </span>
                    {match.creator_username}
                  </div>
                  <div className="text-sm text-gray-600">
                    {match.creator_color === PlayerColor.Random
                      ? 'Random colors'
                      : `Creator plays as ${match.creator_color}`
                    }
                  </div>
                </div>
                <div className="mb-4">
                  <span className="text-sm text-gray-600">Created: {new Date(match.created_at).toLocaleString()}</span>
                </div>
                <button
                  onClick={() => handleJoinMatch(match.id)}
                  disabled={loading}
                  className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-md w-full"
                >
                  {loading ? 'Joining...' : 'Join Match'}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Matchmaking;