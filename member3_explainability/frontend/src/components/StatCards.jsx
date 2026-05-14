/**
 * src/components/StatCards.jsx
 * Top-row summary stat cards from /dashboard/summary
 */

const EMOTION_EMOJI = {
  stress: '😰', calm: '😌', happy: '😊', sad: '😢', angry: '😠',
};

function StatCard({ icon, label, value, sub, glowColor }) {
  return (
    <div className="card stat-card" style={{ '--glow-color': glowColor }}>
      <div className="stat-icon">{icon}</div>
      <p className="card-title">{label}</p>
      <div className="stat-value">{value ?? '—'}</div>
      {sub && <div className="stat-delta">{sub}</div>}
    </div>
  );
}

export default function StatCards({ summary }) {
  if (!summary) {
    return (
      <div className="grid-4">
        {[0,1,2,3].map(i => (
          <div key={i} className="card" style={{ height: 110, background: 'rgba(255,255,255,0.02)' }} />
        ))}
      </div>
    );
  }

  const { session_count, dominant_emotion, dominant_modality, avg_confidence } = summary;

  const MODALITY_LABELS = {
    EEG:   'EEG (Brain)',
    GSR:   'GSR (Skin)',
    video: 'Video',
  };
  const dominantLabel = MODALITY_LABELS[dominant_modality] ?? dominant_modality ?? '—';

  return (
    <div className="grid-4">
      <StatCard
        icon="📊"
        label="Total Sessions"
        value={session_count}
        sub="this account"
        glowColor="rgba(99,179,237,0.2)"
      />
      <StatCard
        icon={EMOTION_EMOJI[dominant_emotion] || '🎭'}
        label="Dominant Emotion"
        value={dominant_emotion ? dominant_emotion.charAt(0).toUpperCase() + dominant_emotion.slice(1) : '—'}
        sub="most frequent"
        glowColor="rgba(183,148,244,0.2)"
      />
      <StatCard
        icon="📡"
        label="Top Modality"
        value={dominantLabel}
        sub="by avg SHAP"
        glowColor="rgba(118,228,247,0.2)"
      />
      <StatCard
        icon="🎯"
        label="Avg Confidence"
        value={avg_confidence != null ? `${(avg_confidence * 100).toFixed(1)}%` : '—'}
        sub="fused prediction"
        glowColor="rgba(104,211,145,0.2)"
      />
    </div>
  );
}
