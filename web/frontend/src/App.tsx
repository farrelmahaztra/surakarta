import { useState, useEffect } from 'react'
import Game from './components/Game'
import Auth from './components/Auth'
import UserProfile from './components/UserProfile'
import Matchmaking from './components/Matchmaking'
import MultiplayerGame from './components/MultiplayerGame'
import { userApi } from './api/game'
import { GameMode } from './types'

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  const [showGame, setShowGame] = useState(false);
  const [showMatchmaking, setShowMatchmaking] = useState(false);
  const [showMultiplayerGame, setShowMultiplayerGame] = useState(false);
  const [activeMatchId, setActiveMatchId] = useState<number | null>(null);
  const [_, setGameMode] = useState<GameMode>(GameMode.SinglePlayer);

  useEffect(() => {
    const authStatus = userApi.isAuthenticated();
    setIsAuthenticated(authStatus);
  }, []);

  const handleAuthSuccess = () => {
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    userApi.logout();
    setIsAuthenticated(false);
    setShowProfile(false);
    setShowGame(false);
    setShowMatchmaking(false);
    setShowMultiplayerGame(false);
  };

  const handlePlaySingleplayer = () => {
    setGameMode(GameMode.SinglePlayer);
    setShowGame(true);
    setShowProfile(false);
    setShowMatchmaking(false);
    setShowMultiplayerGame(false);
  };

  const handlePlayMultiplayer = () => {
    setGameMode(GameMode.Multiplayer);
    setShowMatchmaking(true);
    setShowProfile(false);
    setShowGame(false);
    setShowMultiplayerGame(false);
  };

  const handleViewProfile = () => {
    setShowProfile(true);
    setShowGame(false);
    setShowMatchmaking(false);
    setShowMultiplayerGame(false);
  };

  const handleStartMultiplayerGame = (matchId: number) => {
    setActiveMatchId(matchId);
    setShowMultiplayerGame(true);
    setShowMatchmaking(false);
    setShowGame(false);
    setShowProfile(false);
  };

  const handleReturnToMatchmaking = () => {
    setShowMultiplayerGame(false);
    setShowMatchmaking(true);
    setActiveMatchId(null);
  };

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-stone-900">
        <Auth onAuthSuccess={handleAuthSuccess} />
      </div>
    );
  }

  if (!showGame && !showProfile && !showMatchmaking && !showMultiplayerGame) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-stone-900">
        <div className="max-w-md w-full p-6 bg-white rounded-md shadow-md text-center">
          <h2 className="text-2xl font-bold mb-6 text-black">Surakarta</h2>
          <div className="space-y-4">
            <button
              onClick={handlePlaySingleplayer}
              className="w-full bg-green-900 hover:bg-green-800 text-white font-bold py-3 px-4 rounded-md"
            >
              Play vs AI
            </button>
            <button
              onClick={handlePlayMultiplayer}
              className="w-full bg-green-900 hover:bg-green-8000 text-white font-bold py-3 px-4 rounded-md"
            >
              Multiplayer
            </button>
            <button
              onClick={handleViewProfile}
              className="w-full bg-slate-500 hover:bg-slate-6000 text-white font-bold py-3 px-4 rounded-md"
            >
              View Profile & Game History
            </button>
            <button
              onClick={handleLogout}
              className="w-full bg-red-500 hover:bg-red-600 text-white font-bold py-3 px-4 rounded-md"
            >
              Logout
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (showProfile) {
    return (
      <div className="min-h-screen flex flex-col items-center p-4 bg-stone-900">
        <div className="w-full max-w-6xl">
          <div className="flex justify-end mb-4">
            <button
              onClick={() => {
                setShowProfile(false);
              }}
              className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-bold py-2 px-4 rounded-md"
            >
              Back
            </button>
          </div>
          <UserProfile onLogout={handleLogout} />
        </div>
      </div>
    );
  }

  if (showMatchmaking) {
    return (
      <div className="min-h-screen flex flex-col bg-stone-900">
        <div className="flex-grow">
          <Matchmaking
            onStartMultiplayerGame={handleStartMultiplayerGame}
            onBackToMenu={() => setShowMatchmaking(false)}
          />
        </div>
      </div>
    );
  }

  if (showMultiplayerGame && activeMatchId !== null) {
    return (
      <div className="min-h-screen flex flex-col bg-stone-900">
        <div className="flex justify-end p-4">
          <button
            onClick={handleReturnToMatchmaking}
            className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-bold py-2 px-4 rounded-md mr-2"
          >
            Back to Matches
          </button>
          <button
            onClick={handleViewProfile}
            className="bg-green-900 hover:bg-green-800 text-white font-bold py-2 px-4 rounded-md"
          >
            Profile
          </button>
        </div>
        <div className="flex-grow flex items-center justify-center">
          <MultiplayerGame
            matchId={activeMatchId}
            onReturnToMenu={() => {
              setShowMultiplayerGame(false);
              setActiveMatchId(null);
            }}
            onReturnToMatchmaking={handleReturnToMatchmaking}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-stone-900">
      <div className="flex justify-end p-4">
        <button
          onClick={() => {
            setShowGame(false);
          }}
          className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-bold py-2 px-4 rounded-md mr-2"
        >
          Back
        </button>
      </div>
      <div className="flex-grow flex items-center justify-center">
        <Game onReturnToMenu={() => {
          setShowGame(false);
        }} />
      </div>
    </div>
  );
}

export default App
