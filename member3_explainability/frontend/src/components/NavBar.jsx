/**
 * src/components/NavBar.jsx
 */
import { useAuth } from '../context/AuthContext';

export default function NavBar() {
  const { user, logout } = useAuth();
  const initials = (user?.display_name || user?.email || 'U')
    .split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2);

  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <div className="brand-icon">
          <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="var(--color-bg)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M9.5 2a4.5 4.5 0 0 0-4.5 4.5v.4A3.5 3.5 0 0 0 3 10.2a3.5 3.5 0 0 0 1.3 6.6 3.6 3.6 0 0 0 3.5 3.7c.5 1 1.5 1.5 2.7 1.5 1.8 0 3-1.5 3-3.3V6.5A4.5 4.5 0 0 0 9.5 2Z" />
            <path d="M14.5 2a4.5 4.5 0 0 1 4.5 4.5v.4A3.5 3.5 0 0 1 21 10.2a3.5 3.5 0 0 1-1.3 6.6 3.6 3.6 0 0 1-3.5 3.7c-.5 1-1.5 1.5-2.7 1.5" />
          </svg>
        </div>
        <span>MedOracle</span>
      </div>

      <div style={{ flex: 1 }} />

      <div className="navbar-user">
        <div className="user-avatar">{initials}</div>
        <span style={{ color: 'var(--color-text)' }}>
          {user?.display_name || user?.email}
        </span>
        <button id="logout-btn" className="btn btn-ghost" onClick={logout}>
          Sign out
        </button>
      </div>
    </nav>
  );
}
