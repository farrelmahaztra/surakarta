import { useState, useEffect } from 'react';
import { userApi } from '../api/game';
import { UserProfile as UserProfileType } from '../types';

interface UserProfileProps {
  onLogout: () => void;
}

const UserProfile = ({ onLogout }: UserProfileProps) => {
  const [profile, setProfile] = useState<UserProfileType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const [formData, setFormData] = useState({
    username: '',
    email: '',
    current_password: '',
    new_password: '',
    analytics_consent: false
  });

  useEffect(() => {
    const fetchUserData = async () => {
      setLoading(true);
      try {
        const profileData = await userApi.getUserProfile();
        setProfile(profileData);
        setFormData({
          username: profileData.username,
          email: profileData.email || '',
          current_password: '',
          new_password: '',
          analytics_consent: profileData.analytics_consent || false
        });
      } catch (err) {
        console.error('Error fetching user data:', err);
        setError('Failed to load profile data');
      } finally {
        setLoading(false);
      }
    };

    fetchUserData();
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === 'checkbox' ? checked : value,
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccessMessage(null);
    setLoading(true);

    try {
      const updateData: any = {
        username: formData.username,
        email: formData.email,
        analytics_consent: formData.analytics_consent
      };

      if (formData.current_password && formData.new_password) {
        try {
          await userApi.updatePassword(formData.current_password, formData.new_password);
          setSuccessMessage('Profile and password updated successfully');
        } catch (err: any) {
          setError(err.message || 'Failed to update password');
          setLoading(false);
          return;
        }
      }

      await userApi.updateProfile(updateData);

      const updatedProfile = await userApi.getUserProfile();
      setProfile(updatedProfile);

      setFormData({
        ...formData,
        username: updatedProfile.username,
        email: updatedProfile.email || '',
        current_password: '',
        new_password: '',
        analytics_consent: updatedProfile.analytics_consent
      });

      if (!successMessage) {
        setSuccessMessage('Profile updated successfully');
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred while updating the profile');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteAccount = async () => {
    if (window.confirm('Are you sure you want to delete your account? This action cannot be undone.')) {
      try {
        await userApi.deleteAccount();
        onLogout();
      } catch (err: any) {
        setError(err.message || 'An error occurred while deleting the account');
      }
    }
  };

  if (loading && !profile) {
    return <div className="text-center py-8">Loading profile...</div>;
  }

  if (error && !profile) {
    return <div className="text-center py-8 text-red-500">{error}</div>;
  }

  return (
    <div className="w-full max-w-4xl p-6 bg-white rounded-md shadow-md">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-black">User Profile</h2>
      </div>

      <h3 className="text-xl font-bold mb-4 text-black">Edit Profile</h3>

      {error && (
        <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-md">
          {error}
        </div>
      )}

      {successMessage && (
        <div className="mb-4 p-3 bg-green-100 text-green-700 rounded-md">
          {successMessage}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="username">
            Username
          </label>
          <input
            type="text"
            id="username"
            name="username"
            className="w-full px-3 py-2 border border-gray-300 bg-stone-100 text-black rounded-md focus:outline-none focus:ring-2 focus:ring-green-600"
            value={formData.username}
            onChange={handleInputChange}
            required
          />
        </div>

        <div className="mb-4">
          <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="email">
            Email (optional)
          </label>
          <input
            type="email"
            id="email"
            name="email"
            className="w-full px-3 py-2 border border-gray-300 bg-stone-100 text-black rounded-md focus:outline-none focus:ring-2 focus:ring-green-600"
            value={formData.email}
            onChange={handleInputChange}
          />
        </div>

        <div className="mb-4">
          <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="current_password">
            Current Password (required to change password)
          </label>
          <input
            type="password"
            id="current_password"
            name="current_password"
            className="w-full px-3 py-2 border border-gray-300 bg-stone-100 text-black rounded-md focus:outline-none focus:ring-2 focus:ring-green-600"
            value={formData.current_password}
            onChange={handleInputChange}
          />
        </div>

        <div className="mb-6">
          <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="new_password">
            New Password
          </label>
          <input
            type="password"
            id="new_password"
            name="new_password"
            className="w-full px-3 py-2 border border-gray-300 bg-stone-100 text-black rounded-md focus:outline-none focus:ring-2 focus:ring-green-600"
            value={formData.new_password}
            onChange={handleInputChange}
          />
        </div>

        <div className="mb-6">
          <label className="flex items-center">
            <input
              type="checkbox"
              id="analytics_consent"
              name="analytics_consent"
              className="mr-2"
              checked={formData.analytics_consent}
              onChange={handleInputChange}
            />
            <span className="text-gray-700 text-sm">Allow anonymous gameplay data collection to help improve the game</span>
          </label>
        </div>

        <div className="flex items-center justify-between">
          <button
            type="submit"
            className="bg-green-900 hover:bg-green-800 text-white font-bold py-2 px-4 rounded-md focus:outline-none focus:ring-2 focus:ring-green-600 focus:ring-opacity-50"
            disabled={loading}
          >
            {loading ? 'Saving...' : 'Save Changes'}
          </button>

          <button
            type="button"
            className="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded-md"
            onClick={handleDeleteAccount}
          >
            Delete Account
          </button>
        </div>
      </form>
    </div>
  );
};

export default UserProfile;