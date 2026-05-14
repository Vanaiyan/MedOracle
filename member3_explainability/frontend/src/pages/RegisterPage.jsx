/**
 * src/pages/RegisterPage.jsx
 */

import { useState } from 'react';
import { useAuth } from '../context/AuthContext';

export default function RegisterPage({ onSwitch }) {
  const { register, loading, error, setError } = useAuth();
  const [name,  setName ]       = useState('');
  const [email, setEmail]       = useState('');
  const [pass,  setPass ]       = useState('');
  const [showPass, setShowPass] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    await register(email, pass, name);
  };

  return (
    <div className="auth-page">
      <div className="auth-card animate-fadeinup">
        <div className="auth-logo">
          <div className="logo-icon">🧠</div>
          <h1>MedOracle</h1>
          <p>Create your account</p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="label" htmlFor="reg-name">Display Name</label>
            <input
              id="reg-name"
              className="input"
              type="text"
              placeholder="Dr. Smith"
              value={name}
              onChange={e => { setName(e.target.value); setError(''); }}
            />
          </div>

          <div className="form-group">
            <label className="label" htmlFor="reg-email">Email</label>
            <input
              id="reg-email"
              className="input"
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={e => { setEmail(e.target.value); setError(''); }}
              required
            />
          </div>

          <div className="form-group">
            <label className="label" htmlFor="reg-password">Password</label>
            <div style={{ position: 'relative' }}>
              <input
                id="reg-password"
                className="input"
                type={showPass ? 'text' : 'password'}
                placeholder="Min. 6 characters"
                value={pass}
                onChange={e => { setPass(e.target.value); setError(''); }}
                required minLength={6}
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

          <button id="register-submit-btn" className="btn btn-primary" style={{ width: '100%', justifyContent: 'center' }} disabled={loading}>
            {loading ? <span className="spinner" /> : 'Create Account'}
          </button>
        </form>

        <p className="auth-switch">
          Already have an account?{' '}
          <a id="switch-to-login" onClick={onSwitch}>Sign in</a>
        </p>
      </div>
    </div>
  );
}