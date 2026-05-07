/**
 * src/pages/LoginPage.jsx
 */

import { useState } from 'react';
import { useAuth } from '../context/AuthContext';

export default function LoginPage({ onSwitch }) {
  const { login, loading, error, setError } = useAuth();
  const [email, setEmail]       = useState('');
  const [pass,  setPass ]       = useState('');
  const [showPass, setShowPass] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    await login(email, pass);
  };

  return (
    <div className="auth-page">
      <div className="auth-card animate-fadeinup">
        <div className="auth-logo">
          <div className="logo-icon">🧠</div>
          <h1>MedOracle</h1>
          <p>Multimodal Emotion Recognition System</p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="label" htmlFor="login-email">Email</label>
            <input
              id="login-email"
              className="input"
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={e => { setEmail(e.target.value); setError(''); }}
              required
            />
          </div>

          <div className="form-group">
            <label className="label" htmlFor="login-password">Password</label>
            <div style={{ position: 'relative' }}>
              <input
                id="login-password"
                className="input"
                type={showPass ? 'text' : 'password'}
                placeholder="••••••••"
                value={pass}
                onChange={e => { setPass(e.target.value); setError(''); }}
                required
                style={{ paddingRight: '2.5rem' }}
              />
              <button
                type="button"
                onClick={() => setShowPass(v => !v)}
                style={{
                  position: 'absolute', right: '0.75rem', top: '50%',
                  transform: 'translateY(-50%)', background: 'none',
                  border: 'none', cursor: 'pointer', padding: 0,
                  color: 'var(--text-muted, #888)', fontSize: '1.1rem',
                  lineHeight: 1,
                }}
                aria-label={showPass ? 'Hide password' : 'Show password'}
              >
                {showPass ? '🙈' : '👁️'}
              </button>
            </div>
          </div>

          {error && <p className="error-msg" style={{ marginBottom: '0.75rem' }}>{error}</p>}

          <button id="login-submit-btn" className="btn btn-primary" style={{ width: '100%', justifyContent: 'center' }} disabled={loading}>
            {loading ? <span className="spinner" /> : 'Sign In'}
          </button>
        </form>

        <p className="auth-switch">
          Don&apos;t have an account?{' '}
          <a id="switch-to-register" onClick={onSwitch}>Create one</a>
        </p>
      </div>
    </div>
  );
}