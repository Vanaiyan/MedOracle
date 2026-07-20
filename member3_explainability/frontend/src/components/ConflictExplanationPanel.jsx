/**
 * src/components/ConflictExplanationPanel.jsx
 * Member 3 novel contribution — Trust-Aware Modality-Conflict Explanation.
 *
 * Two modes:
 *   - "session" (default): explains the CURRENTLY SELECTED session (GET
 *     /explain/conflict/{id}) so it is consistent with the Session Detail card.
 *   - "what-if": manual dropdowns build a hypothetical conflict (POST
 *     /explain/conflict) for interactive counterfactual exploration.
 */

import { useState, useEffect } from 'react';
import { conflictAPI } from '../api/client';

const EMOTIONS = ['stress', 'calm', 'happy', 'sad', 'angry'];
const QUALITIES = ['good', 'degraded', 'poor'];

function makeConflictPayload(physioE, videoE, q) {
  const probs = (dom) => {
    const rest = (1 - 0.72) / 4;
    const p = {};
    EMOTIONS.forEach((e) => (p[e] = e === dom ? 0.72 : rest));
    return p;
  };
  return {
    predicted_emotion: physioE,
    confidence: 0.5,
    class_probabilities: probs(physioE),
    modality_weights: { physio: 0.5, video: 0.5 },
    signal_quality: q,
    per_modality_predictions: {
      physio: { predicted_emotion: physioE, confidence: 0.72, class_probabilities: probs(physioE) },
      video: { predicted_emotion: videoE, confidence: 0.72, class_probabilities: probs(videoE) },
    },
  };
}

const sel = {
  background: '#060404f4', border: '1px solid var(--border)', color: 'var(--text-primary)',
  padding: '0.4rem 0.6rem', borderRadius: 8, fontFamily: 'var(--font)', fontSize: 13, cursor: 'pointer',
};

function Bar({ label, value, color }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-secondary)' }}>
        <span>{label}</span><span style={{ color: 'var(--text-primary)', fontWeight: 700 }}>{(value * 100).toFixed(0)}%</span>
      </div>
      <div style={{ height: 8, background: 'rgba(255,255,255,0.06)', borderRadius: 6, overflow: 'hidden' }}>
        <div style={{ width: `${Math.max(2, value * 100)}%`, height: '100%', background: color }} />
      </div>
    </div>
  );
}

export default function ConflictExplanationPanel({ sessionId }) {
  const [mode, setMode] = useState('session');   // 'session' | 'whatif'
  const [physioE, setPhysioE] = useState('stress');
  const [videoE, setVideoE] = useState('happy');
  const [q, setQ] = useState({ eeg: 'good', gsr: 'good', video: 'poor' });
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  // In session mode, auto-explain the selected session whenever it changes.
  useEffect(() => {
    if (mode !== 'session') return;
    if (!sessionId) { setData(null); setErr(null); return; }
    let cancelled = false;
    setLoading(true); setErr(null); setData(null);
    conflictAPI.explainSession(sessionId)
      .then(({ data }) => { if (!cancelled) setData(data); })
      .catch((e) => {
        if (cancelled) return;
        setErr(e.response?.status === 422
          ? "This session has no separate per-modality data (e.g. a video-only run), so there is no conflict to explain."
          : (e.response?.data?.detail || 'Could not load the conflict explanation for this session.'));
      })
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, [sessionId, mode]);

  const runWhatIf = async () => {
    setLoading(true); setErr(null);
    try {
      const { data } = await conflictAPI.explain(makeConflictPayload(physioE, videoE, q));
      setData(data);
    } catch (e) {
      setErr(e.response?.data?.detail || 'Request failed. Is the API running and are you logged in?');
    } finally {
      setLoading(false);
    }
  };

  const g = data?.gate;
  const ex = data?.explanation;
  const rec = data?.losing_modality_recovery;

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <p className="card-title" style={{ margin: 0 }}>Modality-Conflict Explanation</p>
        <div style={{ display: 'flex', gap: 6 }}>
          <button
            className={`btn ${mode === 'session' ? 'btn-primary' : ''}`}
            style={mode !== 'session' ? { background: 'rgba(255,255,255,0.06)' } : {}}
            onClick={() => { setMode('session'); setData(null); setErr(null); }}
          >Selected session</button>
          <button
            className={`btn ${mode === 'whatif' ? 'btn-primary' : ''}`}
            style={mode !== 'whatif' ? { background: 'rgba(255,255,255,0.06)' } : {}}
            onClick={() => { setMode('whatif'); setData(null); setErr(null); }}
          >What-if</button>
        </div>
      </div>

      {/* Session mode: hint when nothing selected */}
      {mode === 'session' && !sessionId && !loading && (
        <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 12 }}>
          Select a session, or click <strong>New conflict session</strong> above to generate one, and its
          conflict explanation will appear here.
        </p>
      )}

      {/* What-if controls */}
      {mode === 'whatif' && (
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'flex-end', marginTop: 12 }}>
          <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Physiology says<br />
            <select style={sel} value={physioE} onChange={(e) => setPhysioE(e.target.value)}>
              {EMOTIONS.map((e) => <option key={e} value={e} style={{ background: '#060404f4' }}>{e}</option>)}
            </select>
          </label>
          <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Video says<br />
            <select style={sel} value={videoE} onChange={(e) => setVideoE(e.target.value)}>
              {EMOTIONS.map((e) => <option key={e} value={e} style={{ background: '#060404f4' }}>{e}</option>)}
            </select>
          </label>
          {['eeg', 'gsr', 'video'].map((k) => (
            <label key={k} style={{ fontSize: 11, color: 'var(--text-muted)' }}>{k.toUpperCase()} quality<br />
              <select style={sel} value={q[k]} onChange={(e) => setQ({ ...q, [k]: e.target.value })}>
                {QUALITIES.map((v) => <option key={v} value={v} style={{ background: '#060404f4' }}>{v}</option>)}
              </select>
            </label>
          ))}
          <button className="btn btn-primary" onClick={runWhatIf} disabled={loading}>
            {loading ? 'Explaining…' : 'Explain Conflict'}
          </button>
        </div>
      )}

      {loading && <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 12 }}>Explaining…</p>}
      {err && <p className="error-msg" style={{ marginTop: 12 }}>⚠ {err}</p>}

      {data && (
        <div style={{ marginTop: 16 }}>
          <div style={{
            padding: '0.6rem 0.9rem', borderRadius: 10, marginBottom: 12,
            background: data.is_conflict ? 'rgba(246,173,85,0.12)' : 'rgba(104,211,145,0.12)',
            border: `1px solid ${data.is_conflict ? 'var(--accent-orange, #f6ad55)' : 'var(--accent-green, #68d391)'}`,
            fontSize: 13,
          }}>
            {data.is_conflict
              ? <>⚠ <strong>Conflict</strong>: physiology → <strong>{data.physio_emotion}</strong> vs video → <strong>{data.video_emotion}</strong>. Gate trusted <strong>{g.dominant_modality}</strong> → <strong>{data.fused_emotion}</strong>.</>
              : <>✓ Both modalities agree on <strong>{data.fused_emotion}</strong>.</>}
          </div>

          <div className="grid-2" style={{ gap: 16 }}>
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>Gate resolution (weight × quality)</div>
              <Bar label={`Physiology (${g.physio_quality}, α=${g.physio_alpha})`} value={g.w_physio} color="#63b3ed" />
              <Bar label={`Video (${g.video_quality}, α=${g.video_alpha})`} value={g.w_video} color="#68d391" />
              <div style={{ marginTop: 8, fontSize: 12 }}>
                Resolution trust:{' '}
                <span className={`badge badge-${data.trust.band === 'high' ? 'good' : data.trust.band === 'moderate' ? 'degraded' : 'poor'}`}>
                  {data.trust.band} ({data.trust.trust_score})
                </span>
              </div>
            </div>

            <div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
                Shapley contribution to “{data.fused_emotion}”
              </div>
              {Object.entries(data.modality_shapley).map(([k, v]) => (
                <div key={k} style={{ marginBottom: 6, fontSize: 12 }}>
                  <span style={{ color: 'var(--text-secondary)' }}>{k}</span>
                  <span style={{ float: 'right', fontWeight: 700, color: v >= 0 ? '#68d391' : '#fc8181' }}>
                    {v >= 0 ? '+' : ''}{v.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {rec && (
            <div style={{ marginTop: 12, fontSize: 13, color: 'var(--text-secondary)' }}>
              <strong>Counterfactual:</strong>{' '}
              {rec.would_flip
                ? <>if the <strong>{rec.losing_modality}</strong> signal quality were good, the decision would flip to <strong>{rec.new_fused_emotion}</strong>.</>
                : <>even with good <strong>{rec.losing_modality}</strong> signal quality, the decision would remain <strong>{rec.new_fused_emotion}</strong>.</>}
            </div>
          )}

          {ex && (
            <div style={{ marginTop: 14, padding: '0.8rem 1rem', borderRadius: 10, background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', gap: 6, marginBottom: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                <span className={`badge badge-${ex.verified ? 'good' : 'poor'}`}>{ex.verified ? '✓ verified' : 'unverified'}</span>
                <span className="badge" style={{ background: 'rgba(255,255,255,0.06)' }}>source: {ex.source}</span>
                <span className="badge badge-good">faithfulness {ex.faithfulness}</span>
                <span className={`badge badge-${ex.hallucination_rate <= 0.05 ? 'good' : ex.hallucination_rate <= 0.2 ? 'degraded' : 'poor'}`}>
                  hallucination {ex.hallucination_rate}
                </span>
              </div>
              <p style={{ fontSize: 13, lineHeight: 1.5, color: 'var(--text-primary)', margin: 0 }}>{ex.text}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
