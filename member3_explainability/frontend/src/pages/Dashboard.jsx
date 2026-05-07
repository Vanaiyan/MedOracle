/**
 * src/pages/Dashboard.jsx
 * Main dashboard — assembles all components.
 */

import { useState, useEffect, useCallback } from 'react';
import NavBar from '../components/NavBar';
import StatCards from '../components/StatCards';
import EmotionTrendChart from '../components/EmotionTrendChart';
import SHAPBarChart from '../components/SHAPBarChart';
import SessionHistoryPanel from '../components/SessionHistoryPanel';
import ChatbotPanel from '../components/ChatbotPanel';
import { dashboardAPI, sessionsAPI, predictAPI } from '../api/client';
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
  const [autoExplanation, setAutoExplanation] = useState(null);
  const [runningPrediction, setRunningPrediction] = useState(false);
  const [demoEmotion, setDemoEmotion] = useState('stress');
  const [showDemoPanel, setShowDemoPanel] = useState(false);

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

  const sessionStats = sessionDetail
    ? {
      ...summary,
      session_count: summary?.session_count ?? null,
      dominant_emotion: sessionDetail.predicted_emotion,
      dominant_modality: sessionDetail.modality_weights?.physio >= sessionDetail.modality_weights?.video
        ? 'physio' : 'video',
      avg_confidence: sessionDetail.confidence,
    }
    : null;

  // Run demo prediction
  const runDemoPrediction = async () => {
    setRunningPrediction(true);
    try {
      const payload = makeDemoPayload(demoEmotion);
      const { data } = await predictAPI.predict(payload);
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
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <NavBar />

      <div className="dashboard-layout">
        {/* ── Main content ───────────────────────────────────── */}
        <div className="dashboard-main">

          {/* Top bar */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <h2 style={{ fontSize: '1.3rem', fontWeight: 800 }}>Emotion Analysis Dashboard</h2>
              <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 2 }}>
                Multimodal Emotion Recognition · SHAP Explainability
              </p>
            </div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              {/* Demo emotion picker */}
              <select
                id="demo-emotion-select"
                value={demoEmotion}
                onChange={e => setDemoEmotion(e.target.value)}
                style={{
                  background: '#060404f4', border: '1px solid var(--border)',
                  color: 'var(--text-primary)', padding: '0.5rem 0.75rem',
                  borderRadius: 8, fontFamily: 'var(--font)', fontSize: 13, cursor: 'pointer',
                }}
              >
                {EMOTIONS.map(e => (
                  <option key={e} value={e} style={{ background: '#060404f4', color: '#ffffff' }}>
                    {e.charAt(0).toUpperCase() + e.slice(1)}
                  </option>
                ))}
              </select>
              <button
                id="run-prediction-btn"
                className="btn btn-primary"
                onClick={runDemoPrediction}
                disabled={runningPrediction}
              >
                {runningPrediction ? <><span className="spinner" style={{ width: 14, height: 14 }} /> Running…</> : '▶ Run Prediction'}
              </button>
            </div>
          </div>

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

          {/* Session detail card */}
          {sessionDetail && (
            <div className="card animate-fadeinup">
              <p className="card-title">Session Detail</p>
              <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
                <div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Predicted Emotion</div>
                  <div style={{ fontSize: 22, fontWeight: 800, marginTop: 2 }}>
                    {sessionDetail.predicted_emotion?.charAt(0).toUpperCase() + sessionDetail.predicted_emotion?.slice(1)}
                    <span className={`badge badge-${sessionDetail.predicted_emotion}`} style={{ marginLeft: 8, fontSize: 11 }}>
                      {(sessionDetail.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Modality Weights</div>
                  <div style={{ fontSize: 13, marginTop: 4, color: 'var(--text-primary)' }}>
                    Physio <strong>{((sessionDetail.modality_weights?.physio ?? 0) * 100).toFixed(0)}%</strong>
                    {' · '}
                    Video <strong>{((sessionDetail.modality_weights?.video ?? 0) * 100).toFixed(0)}%</strong>
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Signal Quality</div>
                  <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
                    {Object.entries(sessionDetail.signal_quality || {}).map(([k, v]) => (
                      <span key={k} className={`badge badge-${v}`} style={{ fontSize: 10 }}>
                        {k.toUpperCase()} · {v}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Class probabilities */}
              <div style={{ marginTop: '1rem' }}>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>Class Probabilities</div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  {Object.entries(sessionDetail.class_probabilities || {}).map(([emotion, prob]) => (
                    <div key={emotion} style={{
                      background: 'rgba(255,255,255,0.04)',
                      borderRadius: 8, padding: '0.5rem 0.75rem',
                      border: `1px solid ${emotion === sessionDetail.predicted_emotion ? 'var(--accent-blue)' : 'var(--border)'}`,
                      minWidth: 80, textAlign: 'center',
                    }}>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{emotion}</div>
                      <div style={{ fontSize: 15, fontWeight: 700, marginTop: 2 }}>{(prob * 100).toFixed(1)}%</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ── Right sidebar ──────────────────────────────────── */}
        <div className="dashboard-sidebar">
          {/* Session History top half */}
          <div style={{ flex: 1, borderBottom: '1px solid var(--border)', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            <SessionHistoryPanel
              activeSessionId={activeSessionId}
              onSelectSession={(id) => { setActiveSessionId(id); setAutoExplanation(null); }}
            />
          </div>

          {/* Chatbot bottom half */}
          <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            <ChatbotPanel
              sessionId={activeSessionId}
              autoExplanation={autoExplanation}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
