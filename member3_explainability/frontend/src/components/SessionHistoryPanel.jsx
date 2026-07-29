/**
 * src/components/SessionHistoryPanel.jsx
 * Scrollable list of past sessions. Clicking loads session detail.
 */

import { useState, useEffect } from 'react';
import { sessionsAPI } from '../api/client';

// Emotion → tonal treatment, mirrors the design's tagTone().
const TONE = {
  angry:  { dot: 'var(--color-accent-600)',   badge: 'angry' },
  stress: { dot: 'var(--color-accent-500)',   badge: 'stress' },
  sad:    { dot: 'var(--color-neutral-600)',  badge: 'sad' },
  calm:   { dot: 'var(--color-accent-2-500)', badge: 'calm' },
  happy:  { dot: 'var(--color-accent-2-700)', badge: 'happy' },
};
const toneFor = (e) => TONE[e] || { dot: 'var(--color-neutral-600)', badge: 'sad' };

function formatDate(ts) {
  // Append Z if missing so JS treats it as UTC, then converts to local time
  const utcTs = ts.endsWith('Z') ? ts : ts + 'Z';
  const d = new Date(utcTs);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
    + ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

export default function SessionHistoryPanel({ activeSessionId, onSelectSession }) {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    sessionsAPI.list()
      .then(r => setSessions(r.data.sessions || []))
      .catch(() => { })
      .finally(() => setLoading(false));
  }, [activeSessionId]); // re-fetch when new session is created

  return (
    <>
      <div className="card" style={{ gap: 2 }}>
        <h6 style={{ margin: 0 }}>Session history</h6>
        <div style={{ fontSize: 12, color: 'var(--color-neutral-600)' }}>
          {sessions.length} sessions total
        </div>
      </div>

      <div
        className="scroll-panel"
        style={{ display: 'flex', flexDirection: 'column', gap: 10, maxHeight: 'calc(100vh - 180px)', paddingRight: 2 }}
      >
        {loading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '2rem' }}>
            <span className="spinner" />
          </div>
        ) : sessions.length === 0 ? (
          <div style={{ padding: '2rem 1rem', textAlign: 'center', color: 'var(--color-neutral-600)', fontSize: 13 }}>
            No sessions yet.<br />Run a prediction to get started.
          </div>
        ) : (
          sessions.map(s => {
            const tone = toneFor(s.predicted_emotion);
            const pct = (s.confidence * 100).toFixed(0);
            return (
              <div
                key={s.session_id}
                id={`session-item-${s.session_id}`}
                className={`session-item ${s.session_id === activeSessionId ? 'active' : ''}`}
                onClick={() => onSelectSession(s.session_id)}
              >
                <span className="session-dot" style={{ background: tone.dot }} />
                <div className="session-item-info">
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
                    <span className="session-item-title">
                      {s.predicted_emotion.charAt(0).toUpperCase() + s.predicted_emotion.slice(1)}
                    </span>
                    <span style={{ fontSize: 12.5, color: 'var(--color-accent-700)' }}>{pct}%</span>
                  </div>
                  <div className="session-item-time">{formatDate(s.timestamp)}</div>
                </div>
                <span className={`badge badge-${tone.badge}`} style={{ flexShrink: 0 }}>{pct}%</span>
              </div>
            );
          })
        )}
      </div>
    </>
  );
}
