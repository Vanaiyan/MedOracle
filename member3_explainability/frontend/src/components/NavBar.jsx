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
        <div className="brand-icon">🧠</div>
        <span>MedOracle</span>
      </div>

      <div style={{ flex: 1 }} />

      <div className="navbar-user">
        <div className="user-avatar">{initials}</div>
        <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
          {user?.display_name || user?.email}
        </span>
        <button id="logout-btn" className="btn btn-ghost" onClick={logout}>
          Sign out
        </button>
      </div>
    </nav>
  );
}
