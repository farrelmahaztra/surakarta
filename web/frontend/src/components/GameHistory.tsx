import { useState, useEffect } from 'react';
import { userApi } from '../api/game';
import { GameRecord } from '../types';

const GameHistory = () => {
  const [gameHistory, setGameHistory] = useState<GameRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [profile, setProfile] = useState<any>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const profileData = await userApi.getUserProfile();
        setProfile(profileData);

        const historyData = await userApi.getGameHistory();
        const gameHistoryRaw = historyData.filter((game: GameRecord) => game.result !== null);
        const gameHistory = gameHistoryRaw.sort((a: GameRecord, b: GameRecord) =>
          new Date(b.start_time).getTime() - new Date(a.start_time).getTime()
        );

        setGameHistory(gameHistory);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError('Failed to load data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return <div className="text-center py-8">Loading game history...</div>;
  }

  if (error) {
    return <div className="text-center py-8 text-red-500">{error}</div>;
  }

  return (
    <div className="w-full max-w-4xl p-6 bg-white rounded-md shadow-md">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-black">Game History</h2>
      </div>

      {profile && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="bg-gray-100 p-4 rounded-md text-center">
            <div className="text-2xl font-bold text-black">{profile.games_played}</div>
            <div className="text-sm text-black">Games Played</div>
          </div>
          <div className="bg-green-100 p-4 rounded-md text-center">
            <div className="text-2xl font-bold text-black">{profile.wins}</div>
            <div className="text-sm text-black">Wins</div>
          </div>
          <div className="bg-red-100 p-4 rounded-md text-center">
            <div className="text-2xl font-bold text-black">{profile.losses}</div>
            <div className="text-sm text-black">Losses</div>
          </div>
        </div>
      )}

      <h3 className="text-xl font-bold mb-4">Recent Games</h3>

      {gameHistory.length === 0 ? (
        <p className="text-gray-500 text-center py-4">No games played yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white">
            <thead>
              <tr className="bg-gray-200 text-black text-sm leading-normal">
                <th className="py-3 px-4 text-left">Date</th>
                <th className="py-3 px-4 text-left">Opponent</th>
                <th className="py-3 px-4 text-left">Result</th>
                <th className="py-3 px-4 text-left">Score</th>
              </tr>
            </thead>
            <tbody className="text-black text-sm">
              {gameHistory
                .filter((game: GameRecord) => game.result !== null)
                .map((game) => (
                  <tr key={game.id} className="border-b border-gray-200 hover:bg-gray-100">
                    <td className="py-3 px-4">
                      {new Date(game.start_time).toLocaleDateString()}
                    </td>
                    <td className="py-3 px-4 capitalize">
                      {game.opponent_type === 'multiplayer'
                        ? `${game.opponent_name || 'Unknown'} (Multiplayer)`
                        : `${game.opponent_type} Agent`}
                    </td>
                    <td className="py-3 px-4">
                      <span className={`font-bold ${game.result === 'win'
                        ? 'text-green-600'
                        : game.result === 'loss'
                          ? 'text-red-500'
                          : 'text-gray-500'
                        }`}>
                        {game.result?.toUpperCase() ?? 'Unknown'}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      {game.final_score !== null ? game.final_score : '-'}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default GameHistory;