/**
 * src/context/AuthContext.jsx
 * Global auth state — user info, login/logout, token management.
 */

import { createContext, useContext, useState, useCallback } from 'react';
import { authAPI, setTokens, clearTokens } from '../api/client';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);   // { user_id, email, display_name }
  const [loading, setLoading] = useState(false);
  const [error,   setError  ] = useState('');

  const login = useCallback(async (email, password) => {
    setLoading(true); setError('');
    try {
      const { data } = await authAPI.login(email, password);
      setTokens(data.access_token, data.refresh_token);
      // Decode display name from token payload (base64 middle segment)
      const payload = JSON.parse(atob(data.access_token.split('.')[1]));
      setUser({ user_id: payload.sub, email: payload.email, display_name: payload.email.split('@')[0] });
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

  return (
    <AuthContext.Provider value={{ user, loading, error, login, register, logout, setError }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
