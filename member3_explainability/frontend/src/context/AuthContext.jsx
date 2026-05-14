/**
 * src/context/AuthContext.jsx
 * Global auth state — user info, login/logout, token management.
 */

import { createContext, useContext, useState, useCallback, useEffect } from 'react';
import axios from 'axios';
import { authAPI, setTokens, clearTokens, getRefreshToken } from '../api/client';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const AuthContext = createContext(null);

function decodeUser(accessToken) {
  const payload = JSON.parse(atob(accessToken.split('.')[1]));
  return { user_id: payload.sub, email: payload.email, display_name: payload.email.split('@')[0] };
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);  // true on mount while restoring session
  const [error,   setError  ] = useState('');

  // Restore session from localStorage on mount
  useEffect(() => {
    const stored = getRefreshToken();
    if (!stored) { setLoading(false); return; }
    axios.post(`${BASE_URL}/auth/refresh`, { refresh_token: stored })
      .then(({ data }) => {
        setTokens(data.access_token, data.refresh_token);
        setUser(decodeUser(data.access_token));
      })
      .catch(() => clearTokens())
      .finally(() => setLoading(false));
  }, []);

  const login = useCallback(async (email, password) => {
    setLoading(true); setError('');
    try {
      const { data } = await authAPI.login(email, password);
      setTokens(data.access_token, data.refresh_token);
      setUser(decodeUser(data.access_token));
      return true;
    } catch (e) {
      setError(e.response?.data?.detail || 'Login failed.');
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const register = useCallback(async (email, password, display_name) => {
    setLoading(true); setError('');
    try {
      await authAPI.register(email, password, display_name);
      return await login(email, password);
    } catch (e) {
      setError(e.response?.data?.detail || 'Registration failed.');
      return false;
    } finally {
      setLoading(false);
    }
  }, [login]);

  const logout = useCallback(() => {
    clearTokens();
    setUser(null);
  }, []);

  // Blank screen while checking stored token — avoids login flash on refresh
  if (loading) return null;

  return (
    <AuthContext.Provider value={{ user, loading, error, login, register, logout, setError }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
