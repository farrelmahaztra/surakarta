import { useState, useEffect } from 'react';
import { userApi, matchApi } from '../api/game';
import { UserProfile as UserProfileType, GameRecord } from '../types';

interface UserProfileProps {
  onLogout: () => void;
}

const UserProfile = ({ onLogout }: UserProfileProps) => {
  const [profile, setProfile] = useState<UserProfileType | null>(null);
  const [gameHistory, setGameHistory] = useState<GameRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchUserData = async () => {
      setLoading(true);
      try {
        const profileData = await userApi.getUserProfile();
        setProfile(profileData);

        const historyData = await userApi.getGameHistory();
        const completedSinglePlayerGames = historyData.filter(game => game.result !== null);

        const myMatches = await matchApi.getMyMatches();
        console.log("completedSinglePlayerGames:", completedSinglePlayerGames);

        const completedMatches = myMatches.filter(match =>
          match.status === 'completed'
        );
        console.log("Completed matches:", completedMatches);

        const gameHistory = [...completedSinglePlayerGames, ...completedMatches].sort((a, b) => new Date(b.start_time).getTime() - new Date(a.start_time).getTime());

        setGameHistory(gameHistory);

      } catch (err) {
        console.error('Error fetching user data:', err);
        setError('Failed to load profile data');
      } finally {
        setLoading(false);
      }
    };

    fetchUserData();
  }, []);

  const handleLogout = () => {
    userApi.logout();
    onLogout();
  };

  if (loading) {
    return <div className="text-center py-8">Loading profile...</div>;
  }

  if (error) {
    return <div className="text-center py-8 text-red-500">{error}</div>;
  }

  return (
    <div className="w-full max-w-4xl p-6 bg-white rounded-md shadow-md">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">Profile: {profile?.username}</h2>
        <button
          onClick={handleLogout}
          className="px-4 py-2 bg-red-500 text-white rounded-md hover:bg-red-600"
        >
          Logout
        </button>
      </div>

      {profile && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-gray-100 p-4 rounded-md text-center">
            <div className="text-2xl font-bold">{profile.games_played}</div>
            <div className="text-sm text-gray-600">Games Played</div>
          </div>
          <div className="bg-green-100 p-4 rounded-md text-center">
            <div className="text-2xl font-bold">{profile.wins}</div>
            <div className="text-sm text-gray-600">Wins</div>
          </div>
          <div className="bg-red-100 p-4 rounded-md text-center">
            <div className="text-2xl font-bold">{profile.losses}</div>
            <div className="text-sm text-gray-600">Losses</div>
          </div>
          <div className="bg-blue-100 p-4 rounded-md text-center">
            <div className="text-2xl font-bold">{profile.highest_score}</div>
            <div className="text-sm text-gray-600">Best Score</div>
          </div>
        </div>
      )}

      <h3 className="text-xl font-bold mb-4">Game History</h3>

      {gameHistory.length === 0 ? (
        <p className="text-gray-500 text-center py-4">No games played yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white">
            <thead>
              <tr className="bg-gray-200 text-gray-600 text-sm leading-normal">
                <th className="py-3 px-4 text-left">Date</th>
                <th className="py-3 px-4 text-left">Opponent</th>
                <th className="py-3 px-4 text-left">Result</th>
                <th className="py-3 px-4 text-left">Score</th>
              </tr>
            </thead>
            <tbody className="text-gray-600 text-sm">
              {gameHistory
                .filter(game => game.result !== null)
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
                        ? 'text-green-500'
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

export default UserProfile;