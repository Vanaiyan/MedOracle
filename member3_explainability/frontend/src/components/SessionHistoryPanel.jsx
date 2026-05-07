/**
 * src/components/SessionHistoryPanel.jsx
 * Scrollable list of past sessions. Clicking loads session detail.
 */

import { useState, useEffect } from 'react';
import { sessionsAPI } from '../api/client';

const EMOTION_EMOJI = {
  stress: '😰', calm: '😌', happy: '😊', sad: '😢', angry: '😠',
};

function formatDate(ts) {
  // Append Z if missing so JS treats it as UTC, then converts to local time
  const utcTs = ts.endsWith('Z') ? ts : ts + 'Z';
  const d = new Date(utcTs);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
    + ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

export default function SessionHistoryPanel({ activeSessionId, onSelectSession }) {
  const [sessions, setSessions] = useState([]);
  const [loading,  setLoading ] = useState(true);

  useEffect(() => {
    sessionsAPI.list()
      .then(r => setSessions(r.data.sessions || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [activeSessionId]); // re-fetch when new session is created

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ padding: '1rem 1.25rem', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
        <h3 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
          Session History
        </h3>
        <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
          {sessions.length} sessions total
        </p>
      </div>

      <div className="scroll-panel" style={{ flex: 1, padding: '0.5rem' }}>
        {loading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '2rem' }}>
            <span className="spinner" />
          </div>
        ) : sessions.length === 0 ? (
          <div style={{ padding: '2rem 1rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
            No sessions yet.<br />Run a prediction to get started.
          </div>
        ) : (
          sessions.map(s => (
            <div
              key={s.session_id}
              id={`session-item-${s.session_id}`}
              className={`session-item ${s.session_id === activeSessionId ? 'active' : ''}`}
              onClick={() => onSelectSession(s.session_id)}
            >
              <div style={{ fontSize: 20, flexShrink: 0 }}>
                {EMOTION_EMOJI[s.predicted_emotion] || '🎭'}
              </div>
              <div className="session-item-info">
                <div className="session-item-title">
                  {s.predicted_emotion.charAt(0).toUpperCase() + s.predicted_emotion.slice(1)}
                  <span style={{ marginLeft: 6, fontSize: 11, color: 'var(--accent-blue)', fontWeight: 500 }}>
                    {(s.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="session-item-time">{formatDate(s.timestamp)}</div>
              </div>
              <div
                className={`badge badge-${s.predicted_emotion}`}
                style={{ fontSize: 10, flexShrink: 0 }}
              >
                {(s.confidence * 100).toFixed(0)}%
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
