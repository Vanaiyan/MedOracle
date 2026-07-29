/**
 * src/pages/Dashboard.jsx
 * Main dashboard — assembles all components.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import NavBar from '../components/NavBar';
import StatCards from '../components/StatCards';
import EmotionTrendChart from '../components/EmotionTrendChart';
import SHAPBarChart from '../components/SHAPBarChart';
import IGAttributionPanel from '../components/IGAttributionPanel';
import ConflictExplanationPanel from '../components/ConflictExplanationPanel';
import SessionHistoryPanel from '../components/SessionHistoryPanel';
import ChatbotPanel from '../components/ChatbotPanel';
import { dashboardAPI, sessionsAPI, predictAPI, conflictAPI } from '../api/client';
import { chatAPI } from '../api/client';

const QUALITY_OPTIONS = ['good', 'degraded', 'poor'];
const EMOTIONS = ['stress', 'calm', 'happy', 'sad', 'angry'];

// Demo prediction_output for quick testing
function makeDemoPayload(emotion) {
  // Emotion-specific probability profiles
  const profiles = {
    stress: { stress: 0.72, calm: 0.08, happy: 0.04, sad: 0.10, angry: 0.06 },
    calm: { stress: 0.05, calm: 0.75, happy: 0.12, sad: 0.05, angry: 0.03 },
    happy: { stress: 0.04, calm: 0.10, happy: 0.78, sad: 0.04, angry: 0.04 },
    sad: { stress: 0.08, calm: 0.06, happy: 0.03, sad: 0.74, angry: 0.09 },
    angry: { stress: 0.10, calm: 0.03, happy: 0.04, sad: 0.07, angry: 0.76 },
  };

  // Emotion-specific modality weights (some emotions show more in physio, some in video)
  const modalityProfiles = {
    stress: { physio: 0.78, video: 0.22 },
    calm: { physio: 0.55, video: 0.45 },
    happy: { physio: 0.38, video: 0.62 },
    sad: { physio: 0.60, video: 0.40 },
    angry: { physio: 0.45, video: 0.55 },
  };

  // Emotion-specific signal quality
  const qualityProfiles = {
    stress: { eeg: 'good', gsr: 'good', video: 'degraded' },
    calm: { eeg: 'good', gsr: 'good', video: 'good' },
    happy: { eeg: 'degraded', gsr: 'good', video: 'good' },
    sad: { eeg: 'good', gsr: 'degraded', video: 'good' },
    angry: { eeg: 'good', gsr: 'good', video: 'good' },
  };

  const probs = profiles[emotion];
  const weights = modalityProfiles[emotion];
  const quality = qualityProfiles[emotion];
  const confidence = probs[emotion];

  // Slightly different per-modality predictions
  const physioProbs = { ...probs };
  physioProbs[emotion] = Math.min(1, probs[emotion] + 0.05);

  const videoProbs = { ...probs };
  videoProbs[emotion] = Math.max(0, probs[emotion] - 0.10);

  return {
    predicted_emotion: emotion,
    confidence: confidence,
    class_probabilities: probs,
    modality_weights: weights,
    signal_quality: quality,
    per_modality_predictions: {
      physio: {
        predicted_emotion: emotion,
        confidence: physioProbs[emotion],
        class_probabilities: physioProbs,
      },
      video: {
        predicted_emotion: emotion,
        confidence: videoProbs[emotion],
        class_probabilities: videoProbs,
      },
    },
  };
}

export default function Dashboard() {
  const [summary, setSummary] = useState(null);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [sessionDetail, setSessionDetail] = useState(null);
  const [lastResult, setLastResult] = useState(null);   // most recent prediction result
  const [autoExplanation, setAutoExplanation] = useState(null);
  const [runningPrediction, setRunningPrediction] = useState(false);
  const [demoEmotion, setDemoEmotion] = useState('stress');
  const [showDemoPanel, setShowDemoPanel] = useState(false);
  const [videoFile, setVideoFile] = useState(null);
  const [eegFile, setEegFile] = useState(null);   // EEG .npy/.csv (32,512)
  const [gsrFile, setGsrFile] = useState(null);   // GSR .npy/.csv (512,)
  const [videoError, setVideoError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);
  const [chatOpen, setChatOpen] = useState(false);

  // Load dashboard summary
  const loadSummary = useCallback(() => {
    dashboardAPI.summary().then(r => setSummary(r.data)).catch(() => { });
  }, []);

  useEffect(() => { loadSummary(); }, [loadSummary]);

  // Load session detail when selected
  useEffect(() => {
    if (!activeSessionId) { setSessionDetail(null); return; }
    sessionsAPI.detail(activeSessionId).then(r => setSessionDetail(r.data)).catch(() => { });
  }, [activeSessionId]);

  // Cards follow the active session (selected from history) or the latest
  // prediction; fall back to the all-time aggregate only when nothing is active.
  const activeSource = sessionDetail || lastResult;
  const sessionStats = activeSource
    ? {
      ...summary,
      session_count: summary?.session_count ?? null,
      dominant_emotion: activeSource.predicted_emotion,
      // Top modality = signal with highest SHAP importance for this prediction.
      dominant_modality: (() => {
        const fi = activeSource.feature_importance;
        if (fi) {
          return Object.entries(fi).reduce((a, b) => b[1] > a[1] ? b : a)[0];
        }
        return 'video';
      })(),
      avg_confidence: activeSource.confidence,
    }
    : null;

  const runVideoPrediction = async () => {
    if (!videoFile) return;
    setVideoError(null);
    setRunningPrediction(true);
    try {
      const { data } = await predictAPI.predictVideo(videoFile);
      setLastResult(data);
      setActiveSessionId(data.session_id);
      setVideoFile(null);
      try {
        const chatResp = await chatAPI.send(data.session_id, 'Please explain these results for me.');
        setAutoExplanation(chatResp.data.response);
      } catch {
        setAutoExplanation('Analysis complete. Ask me any questions about this result.');
      }
      loadSummary();
    } catch (e) {
      setVideoError(e.response?.data?.detail || 'Video prediction failed. Check the file and try again.');
    } finally {
      setRunningPrediction(false);
    }
  };

  // Real MULTIMODAL fusion: uploaded video + EEG + GSR files.
  // Runs both models (M1 physio + M2 video) and fuses them via the gated fusion.
  const runMultimodalPrediction = async () => {
    if (!videoFile || !eegFile || !gsrFile) return;
    setVideoError(null);
    setRunningPrediction(true);
    try {
      const { data } = await predictAPI.predictMultimodal(videoFile, eegFile, gsrFile);
      setLastResult(data);
      setActiveSessionId(data.session_id);
      setVideoFile(null); setEegFile(null); setGsrFile(null);
      try {
        const chatResp = await chatAPI.send(data.session_id, 'Please explain these results for me.');
        setAutoExplanation(chatResp.data.response);
      } catch {
        setAutoExplanation('Multimodal analysis complete. Ask me any questions about this result.');
      }
      loadSummary();
    } catch (e) {
      setVideoError(e.response?.data?.detail || 'Multimodal prediction failed. Check the file and try again.');
    } finally {
      setRunningPrediction(false);
    }
  };

  // Generate a CONFLICT session (physio != video) and select it so the
  // Modality-Conflict panel explains a real disagreement.
  const runConflictPrediction = async () => {
    setRunningPrediction(true);
    try {
      const { data } = await conflictAPI.generate();
      setLastResult(data);
      setActiveSessionId(data.session_id);
      setAutoExplanation(null);
      loadSummary();
    } catch (e) {
      console.error('Conflict prediction failed:', e);
    } finally {
      setRunningPrediction(false);
    }
  };

  // Run synthetic prediction (backend generates randomised fused output)
  const runDemoPrediction = async () => {
    setRunningPrediction(true);
    try {
      const { data } = await predictAPI.predictSynthetic(demoEmotion);
      setLastResult(data);
      setActiveSessionId(data.session_id);
      // Get auto-explanation from LLM
      try {
        const chatResp = await chatAPI.send(data.session_id, 'Please explain these results for me.');
        setAutoExplanation(chatResp.data.response);
      } catch {
        setAutoExplanation('Analysis complete. Ask me any questions about this result.');
      }
      loadSummary();
    } catch (e) {
      console.error('Prediction failed:', e);
    } finally {
      setRunningPrediction(false);
    }
  };

  const shap = sessionDetail?.shap_values
    ? { shap_values: sessionDetail.shap_values, feature_importance: sessionDetail.feature_importance, faithfulness_score: sessionDetail.faithfulness_score }
    : null;

  return (
    <div style={{ minHeight: '100vh', background: 'var(--color-bg)', color: 'var(--color-text)' }}>
      <NavBar />

      <div className="dashboard-layout">
        {/* ── Main content ───────────────────────────────────── */}
        <div className="dashboard-main">

          {/* Top bar */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', flexWrap: 'wrap', gap: 16 }}>
            <div>
              <h1 style={{ fontSize: 34, margin: '0 0 6px' }}>Emotion Analysis Dashboard</h1>
              <p style={{ margin: 0, fontSize: 14.5, color: 'var(--color-neutral-700)' }}>
                Multimodal emotion recognition · SHAP explainability
              </p>
            </div>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
              {/* Demo emotion picker */}
              <select
                id="demo-emotion-select"
                className="input"
                value={demoEmotion}
                onChange={e => setDemoEmotion(e.target.value)}
                style={{ width: 'auto', minWidth: 130, cursor: 'pointer' }}
              >
                {EMOTIONS.map(e => (
                  <option key={e} value={e}>
                    {e.charAt(0).toUpperCase() + e.slice(1)}
                  </option>
                ))}
              </select>
              <button
                id="run-prediction-btn"
                className="btn btn-secondary"
                onClick={runDemoPrediction}
                disabled={runningPrediction}
              >
                {runningPrediction
                  ? <><span className="spinner" style={{ width: 14, height: 14 }} /> Running…</>
                  : <>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.75" strokeLinecap="round" strokeLinejoin="round"><polygon points="6 3 20 12 6 21 6 3" /></svg>
                    Run prediction
                  </>}
              </button>
              <button
                id="run-conflict-btn"
                className="btn btn-primary"
                onClick={runConflictPrediction}
                disabled={runningPrediction}
                title="Generate a session where physiology and video disagree"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.75" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg>
                New conflict session
              </button>
            </div>
          </div>

          {/* Video upload panel */}
          <section className="card" style={{ gap: 18 }}>
            <h6 style={{ margin: 0 }}>Analyse video</h6>

            <input
              ref={fileInputRef}
              type="file"
              accept=".mp4,.avi,.mov,.mkv,.flv,.webm,video/mp4,video/avi,video/x-msvideo,video/quicktime,video/x-matroska,video/x-flv,video/webm"
              style={{ display: 'none' }}
              onChange={e => { setVideoFile(e.target.files[0] || null); setVideoError(null); }}
            />

            <div
              className={`video-drop-zone${dragOver ? ' drag-over' : ''}${videoFile ? ' has-file' : ''}${runningPrediction ? ' loading' : ''}`}
              onClick={() => !runningPrediction && fileInputRef.current?.click()}
              onDragOver={e => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={e => {
                e.preventDefault();
                setDragOver(false);
                const f = e.dataTransfer.files[0];
                if (f) { setVideoFile(f); setVideoError(null); }
              }}
            >
              {runningPrediction ? (
                <>
                  <span className="drop-zone-icon">⏳</span>
                  <p className="drop-zone-title">Analysing video…</p>
                  <p className="drop-zone-sub">Running emotion recognition, please wait</p>
                  <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'center' }}>
                    <span className="spinner" />
                  </div>
                </>
              ) : videoFile ? (
                <>
                  <span className="drop-zone-icon">🎬</span>
                  <p className="drop-zone-title">Ready to analyse</p>
                  <div className="drop-zone-file-info">
                    <span>📄</span>
                    <span style={{ maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {videoFile.name}
                    </span>
                    <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>
                      {(videoFile.size / 1024 / 1024).toFixed(1)} MB
                    </span>
                    <button
                      className="drop-zone-clear"
                      onClick={e => { e.stopPropagation(); setVideoFile(null); setVideoError(null); }}
                      title="Remove file"
                    >✕</button>
                  </div>
                </>
              ) : (
                <>
                  <div className="drop-zone-icon-wrap">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent-700)" strokeWidth="2.75" strokeLinecap="round" strokeLinejoin="round"><path d="M23 7l-7 5 7 5V7z" /><rect x="1" y="5" width="15" height="14" rx="3" /></svg>
                  </div>
                  <p className="drop-zone-title">Drop a video here or click to browse</p>
                  <p className="drop-zone-sub">Supports MP4 · MOV · AVI · MKV · FLV · WebM</p>
                </>
              )}
            </div>

            {videoError && (
              <p className="error-msg" style={{ marginTop: '0.6rem' }}>⚠ {videoError}</p>
            )}

            {/* Physiological uploads (EEG + GSR) for real multimodal fusion */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
              <div className="field">
                <label title="EEG window file — shape (32, 512), .npy or .csv">EEG file (.npy)</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                  <input
                    type="file" accept=".npy,.csv,.txt"
                    disabled={runningPrediction}
                    onChange={e => setEegFile(e.target.files[0] || null)}
                    style={{ fontSize: '0.8rem' }}
                  />
                  {eegFile && <span style={{ fontSize: 13, color: 'var(--color-accent-2-700)' }}>✓ {eegFile.name}</span>}
                </div>
              </div>
              <div className="field">
                <label title="GSR window file — shape (512,), .npy or .csv">GSR file (.npy)</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                  <input
                    type="file" accept=".npy,.csv,.txt"
                    disabled={runningPrediction}
                    onChange={e => setGsrFile(e.target.files[0] || null)}
                    style={{ fontSize: '0.8rem' }}
                  />
                  {gsrFile && <span style={{ fontSize: 13, color: 'var(--color-accent-2-700)' }}>✓ {gsrFile.name}</span>}
                </div>
              </div>
            </div>
            <p style={{ margin: 0, fontSize: 12.5, color: 'var(--color-neutral-600)' }}>
              Tip: ready-made subjects (video + eeg.npy + gsr.npy) are in <code>data/synced_samples/</code>.
            </p>

            <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'flex-end', gap: 10 }}>
              <button
                className="btn btn-secondary"
                onClick={runVideoPrediction}
                disabled={!videoFile || runningPrediction}
                title="Video only (no physiological input → graceful degradation)"
              >
                {runningPrediction
                  ? <><span className="spinner" style={{ width: 14, height: 14 }} /> Analysing…</>
                  : 'Video only'}
              </button>

              <button
                className="btn btn-primary"
                onClick={runMultimodalPrediction}
                disabled={!videoFile || !eegFile || !gsrFile || runningPrediction}
                title="Fuse video + EEG + GSR (real gated multimodal fusion)"
              >
                {runningPrediction
                  ? <><span className="spinner" style={{ width: 14, height: 14 }} /> Fusing…</>
                  : 'Run multimodal fusion'}
              </button>
            </div>
          </section>

          {/* Stat cards */}
          <StatCards summary={sessionStats ?? summary} />

          {/* Charts row */}
          <div className="grid-2">
            <EmotionTrendChart trend={summary?.emotion_trend || []} />
            <SHAPBarChart
              shapValues={shap?.shap_values}
              featureImportance={shap?.feature_importance}
              faithfulness={shap?.faithfulness_score}
            />
          </div>

          {/* Real, fused Integrated Gradients — direct comparison against SHAP above */}
          <IGAttributionPanel igAttribution={sessionDetail?.ig_attribution} />

          {/* Modality-Conflict Explanation (Member 3 novel contribution) */}
          <ConflictExplanationPanel sessionId={activeSessionId} />

          {/* Session detail card */}
          {sessionDetail && (
            <section className="card animate-fadeinup" style={{ gap: 16 }}>
              <h6 style={{ margin: 0 }}>Session detail</h6>
              <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr 1fr', gap: 32, alignItems: 'start' }}>
                <div>
                  <div style={{ fontSize: 12, color: 'var(--color-neutral-600)', marginBottom: 6 }}>Predicted emotion</div>
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
                    <span style={{ fontFamily: 'var(--font-heading)', fontSize: 28 }}>
                      {sessionDetail.predicted_emotion?.charAt(0).toUpperCase() + sessionDetail.predicted_emotion?.slice(1)}
                    </span>
                    <span className={`badge badge-${sessionDetail.predicted_emotion}`}>
                      {(sessionDetail.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 12, color: 'var(--color-neutral-600)', marginBottom: 6 }}>Modality weights</div>
                  <div style={{ fontSize: 14 }}>
                    {(() => {
                      const sq = sessionDetail.signal_quality || {};
                      const qw = { good: 1.0, degraded: 0.5, poor: 0.1 };
                      const wEEG = qw[sq.eeg] ?? 1.0;
                      const wGSR = qw[sq.gsr] ?? 1.0;
                      const total = wEEG + wGSR || 1;
                      const physio = sessionDetail.modality_weights?.physio ?? 0;
                      const eegW = (physio * (wEEG / total) * 100).toFixed(0);
                      const gsrW = (physio * (wGSR / total) * 100).toFixed(0);
                      const vidW = ((sessionDetail.modality_weights?.video ?? 0) * 100).toFixed(0);
                      return <>
                        EEG <strong>{eegW}%</strong>{' · '}GSR <strong>{gsrW}%</strong>{' · '}Video <strong>{vidW}%</strong>
                      </>;
                    })()}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 12, color: 'var(--color-neutral-600)', marginBottom: 6 }}>Signal quality</div>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {Object.entries(sessionDetail.signal_quality || {}).map(([k, v]) => (
                      <span key={k} className={`tag ${v === 'good' ? 'tag-accent-2' : 'tag-outline'}`}>
                        {k.toUpperCase()} · {v}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Class probabilities */}
              <div>
                <div style={{ fontSize: 12, color: 'var(--color-neutral-600)', marginBottom: 10 }}>Class probabilities</div>
                <div className="prob-grid">
                  {Object.entries(sessionDetail.class_probabilities || {}).map(([emotion, prob]) => (
                    <div key={emotion} className={`prob-cell${emotion === sessionDetail.predicted_emotion ? ' active' : ''}`}>
                      <div className="prob-cell-label">{emotion}</div>
                      <div className="prob-cell-value">{(prob * 100).toFixed(1)}%</div>
                    </div>
                  ))}
                </div>
              </div>
            </section>
          )}
        </div>

        {/* ── Right sidebar — session history ─────────────────── */}
        <aside className="dashboard-sidebar">
          <SessionHistoryPanel
            activeSessionId={activeSessionId}
            onSelectSession={(id) => { setActiveSessionId(id); setAutoExplanation(null); setChatOpen(true); }}
          />
        </aside>
      </div>

      {/* ── Floating chat panel ─────────────────────────────────── */}
      {chatOpen && (
        <div className="chat-float-panel">
          <ChatbotPanel
            sessionId={activeSessionId}
            autoExplanation={autoExplanation}
            onClose={() => setChatOpen(false)}
          />
        </div>
      )}

      {/* ── Floating chat button ────────────────────────────────── */}
      <button
        className="chat-fab"
        onClick={() => setChatOpen(o => !o)}
        title={chatOpen ? 'Close chat' : 'Open AI assistant'}
      >
        {chatOpen ? '✕' : '💬'}
        {!chatOpen && activeSessionId && <span className="chat-fab-badge" />}
      </button>
    </div>
  );
}
