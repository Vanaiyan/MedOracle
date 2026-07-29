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
  background: 'var(--color-surface)', border: '1px solid var(--color-divider)', color: 'var(--color-text)',
  padding: '7px 12px', borderRadius: 999, fontFamily: 'var(--font-body)', fontSize: 13, cursor: 'pointer',
};

function Bar({ label, value, color }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13.5, marginBottom: 5, color: 'var(--color-neutral-800)' }}>
        <span>{label}</span><span style={{ fontWeight: 700 }}>{(value * 100).toFixed(0)}%</span>
      </div>
      <div style={{ height: 8, background: 'var(--color-neutral-200)', borderRadius: 999, overflow: 'hidden' }}>
        <div style={{ width: `${Math.max(2, value * 100)}%`, height: '100%', background: color, borderRadius: 999 }} />
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
    <div className="card elev-md" style={{ border: '1.5px solid var(--color-accent-300)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
        <h6 style={{ margin: 0 }}>Modality-conflict explanation</h6>
        <div className="seg">
          <label
            className={`seg-opt ${mode === 'session' ? 'is-active' : ''}`}
            onClick={() => { setMode('session'); setData(null); setErr(null); }}
          >Selected session</label>
          <label
            className={`seg-opt ${mode === 'whatif' ? 'is-active' : ''}`}
            onClick={() => { setMode('whatif'); setData(null); setErr(null); }}
          >What-if</label>
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
              {EMOTIONS.map((e) => <option key={e} value={e} style={{ background: 'var(--color-surface)' }}>{e}</option>)}
            </select>
          </label>
          <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Video says<br />
            <select style={sel} value={videoE} onChange={(e) => setVideoE(e.target.value)}>
              {EMOTIONS.map((e) => <option key={e} value={e} style={{ background: 'var(--color-surface)' }}>{e}</option>)}
            </select>
          </label>
          {['eeg', 'gsr', 'video'].map((k) => (
            <label key={k} style={{ fontSize: 11, color: 'var(--text-muted)' }}>{k.toUpperCase()} quality<br />
              <select style={sel} value={q[k]} onChange={(e) => setQ({ ...q, [k]: e.target.value })}>
                {QUALITIES.map((v) => <option key={v} value={v} style={{ background: 'var(--color-surface)' }}>{v}</option>)}
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
            padding: '16px 18px', borderRadius: 'var(--radius-md)', marginBottom: 18,
            background: data.is_conflict ? 'var(--color-accent-100)' : 'var(--color-accent-2-100)',
            border: `1px solid ${data.is_conflict ? 'var(--color-accent-300)' : 'var(--color-accent-2-300)'}`,
            fontSize: 14.5,
          }}>
            {data.is_conflict
              ? <><strong>Conflict:</strong> physiology → <strong>{data.physio_emotion}</strong> vs video → <strong>{data.video_emotion}</strong>. Gate trusted <strong>{g.dominant_modality} → {data.fused_emotion}</strong>.</>
              : <><strong>Aligned:</strong> physiology and video both point to <strong>{data.fused_emotion}</strong> — no gating conflict for this session.</>}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 28 }}>
            <div>
              <div style={{ fontSize: 12, color: 'var(--color-neutral-600)', marginBottom: 10 }}>Gate resolution (weight × quality)</div>
              <Bar label={`Physiology (${g.physio_quality}, α=${g.physio_alpha})`} value={g.w_physio} color="var(--color-accent-400)" />
              <Bar label={`Video (${g.video_quality}, α=${g.video_alpha})`} value={g.w_video} color="var(--color-accent-2-600)" />
              <div style={{ marginTop: 8, fontSize: 14, display: 'flex', alignItems: 'center', gap: 10 }}>
                Resolution trust:{' '}
                <span className="tag tag-accent-2" style={{ fontFamily: 'var(--font-heading)' }}>
                  {data.trust.band} ({data.trust.trust_score})
                </span>
              </div>
            </div>

            <div>
              <div style={{ fontSize: 12, color: 'var(--color-neutral-600)', marginBottom: 10 }}>
                Shapley contribution to “{data.fused_emotion}”
              </div>
              {Object.entries(data.modality_shapley).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 14, padding: '9px 0', borderBottom: '1px solid var(--color-divider)' }}>
                  <span>{k}</span>
                  <strong style={{ color: v >= 0 ? 'var(--color-accent-2-700)' : 'var(--color-accent-700)' }}>
                    {v >= 0 ? '+' : ''}{v.toFixed(3)}
                  </strong>
                </div>
              ))}
            </div>
          </div>

          {rec && (
            <p style={{ marginTop: 18, fontSize: 14.5 }}>
              <strong>Counterfactual:</strong>{' '}
              {rec.would_flip
                ? <>if the <strong>{rec.losing_modality}</strong> signal quality were good, the decision would flip to <strong>{rec.new_fused_emotion}</strong>.</>
                : <>even with good <strong>{rec.losing_modality}</strong> signal quality, the decision would remain <strong>{rec.new_fused_emotion}</strong>.</>}
            </p>
          )}

          {ex && (
            <div style={{ marginTop: 14, padding: '16px 18px', borderRadius: 'var(--radius-md)', background: 'var(--color-neutral-100)', display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                <span className={`tag ${ex.verified ? 'tag-accent-2' : 'tag-outline'}`}>{ex.verified ? '✓ Verified' : 'Unverified'}</span>
                <span className="tag tag-neutral">Source: {ex.source}</span>
                <span className="tag tag-outline">Faithfulness {ex.faithfulness}</span>
                <span className="tag tag-outline">Hallucination {ex.hallucination_rate}</span>
              </div>
              <p style={{ fontSize: 13.5, lineHeight: 1.6, color: 'var(--color-neutral-700)', margin: 0 }}>{ex.text}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
