/**
 * src/App.jsx
 * Root component — switches between auth pages and dashboard.
 * No react-router needed: simple state-based routing (JWT in memory, not URL).
 */

import { useState } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import LoginPage    from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import Dashboard    from './pages/Dashboard';

function AppInner() {
  const { user } = useAuth();
  const [showRegister, setShowRegister] = useState(false);

  if (!user) {
    return showRegister
      ? <RegisterPage onSwitch={() => setShowRegister(false)} />
      : <LoginPage    onSwitch={() => setShowRegister(true)}  />;
  }

  return <Dashboard />;
}

export default function App() {
  return (
    <AuthProvider>
      <AppInner />
    </AuthProvider>
  );
}
