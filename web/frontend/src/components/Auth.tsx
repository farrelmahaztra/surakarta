import { useState } from 'react';
import { userApi } from '../api/game';
import { UserCredentials, UserRegistration } from '../types';

interface AuthProps {
  onAuthSuccess: () => void;
}

const Auth = ({ onAuthSuccess }: AuthProps) => {
  const [isLogin, setIsLogin] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  
  const [formData, setFormData] = useState<UserRegistration>({
    username: '',
    password: '',
    email: '',
  });
  
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    
    try {
      if (isLogin) {
        const credentials: UserCredentials = {
          username: formData.username,
          password: formData.password,
        };
        
        const { success, data } = await userApi.login(credentials);
        
        if (success) {
          onAuthSuccess();
        } else {
          setError(data.error || 'Login failed');
        }
      } else {
        const { success, data } = await userApi.register(formData);
        
        if (success) {
          onAuthSuccess();
        } else {
          if (typeof data.error === 'string') {
            setError(data.error);
          } else if (data.username) {
            setError(`Username error: ${data.username.join(', ')}`);
          } else if (data.password) {
            setError(`Password error: ${data.password.join(', ')}`);
          } else if (data.email) {
            setError(`Email error: ${data.email.join(', ')}`);
          } else {
            setError('Registration failed');
          }
        }
      }
    } catch (error) {
      setError('An error occurred. Please try again.');
      console.error('Auth error:', error);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="w-full max-w-md p-6 bg-white rounded-md shadow-md">
      <h2 className="text-2xl font-bold text-center mb-6">
        {isLogin ? 'Login' : 'Create Account'}
      </h2>
      
      {error && (
        <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-md">
          {error}
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
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={formData.username}
            onChange={handleInputChange}
            required
          />
        </div>
        
        {!isLogin && (
          <div className="mb-4">
            <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="email">
              Email (optional)
            </label>
            <input
              type="email"
              id="email"
              name="email"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={formData.email}
              onChange={handleInputChange}
            />
          </div>
        )}
        
        <div className="mb-6">
          <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="password">
            Password
          </label>
          <input
            type="password"
            id="password"
            name="password"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={formData.password}
            onChange={handleInputChange}
            required
          />
        </div>
        
        <div className="flex items-center justify-between">
          <button
            type="submit"
            className="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
            disabled={loading}
          >
            {loading ? 'Loading...' : isLogin ? 'Login' : 'Register'}
          </button>
        </div>
      </form>
      
      <div className="mt-4 text-center">
        <button
          className="text-blue-500 hover:text-blue-700 text-sm"
          onClick={() => setIsLogin(!isLogin)}
        >
          {isLogin ? 'Need an account? Register' : 'Already have an account? Login'}
        </button>
      </div>
    </div>
  );
};

export default Auth;