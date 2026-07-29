/**
 * src/components/StatCards.jsx
 * Top-row summary stat cards from /dashboard/summary
 */

const ICONS = {
  bars: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--color-neutral-800)" strokeWidth="2.75" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="20" x2="12" y2="10" /><line x1="18" y1="20" x2="18" y2="4" /><line x1="6" y1="20" x2="6" y2="16" />
    </svg>
  ),
  face: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent-700)" strokeWidth="2.75" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="9" /><line x1="9" y1="9" x2="9.01" y2="9" /><line x1="15" y1="9" x2="15.01" y2="9" /><path d="M8 16s1.5-2 4-2 4 2 4 2" />
    </svg>
  ),
  video: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent-2-800)" strokeWidth="2.75" strokeLinecap="round" strokeLinejoin="round">
      <path d="M23 7l-7 5 7 5V7z" /><rect x="1" y="5" width="15" height="14" rx="3" />
    </svg>
  ),
  target: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--color-neutral-800)" strokeWidth="2.75" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" /><circle cx="12" cy="12" r="6" /><circle cx="12" cy="12" r="2" />
    </svg>
  ),
};

function StatCard({ icon, iconBg, label, value, sub }) {
  return (
    <div className="card stat-card">
      <div className="stat-icon-wrap" style={{ background: iconBg }}>{icon}</div>
      <div className="card-kicker">{label}</div>
      <div className="stat-value">{value ?? '—'}</div>
      {sub && <div className="stat-delta">{sub}</div>}
    </div>
  );
}

export default function StatCards({ summary }) {
  if (!summary) {
    return (
      <div className="grid-4">
        {[0, 1, 2, 3].map(i => (
          <div key={i} className="card" style={{ height: 128 }} />
        ))}
      </div>
    );
  }

  const { session_count, dominant_emotion, dominant_modality, avg_confidence } = summary;

  const MODALITY_LABELS = { EEG: 'EEG', GSR: 'GSR', video: 'Video' };
  const dominantLabel = MODALITY_LABELS[dominant_modality] ?? dominant_modality ?? '—';

  return (
    <div className="grid-4">
      <StatCard
        icon={ICONS.bars}
        iconBg="var(--color-neutral-200)"
        label="Total sessions"
        value={session_count}
        sub="this account"
      />
      <StatCard
        icon={ICONS.face}
        iconBg="var(--color-accent-100)"
        label="Dominant emotion"
        value={dominant_emotion ? dominant_emotion.charAt(0).toUpperCase() + dominant_emotion.slice(1) : '—'}
        sub="most frequent"
      />
      <StatCard
        icon={ICONS.video}
        iconBg="var(--color-accent-2-200)"
        label="Top modality"
        value={dominantLabel}
        sub="by avg SHAP"
      />
      <StatCard
        icon={ICONS.target}
        iconBg="var(--color-neutral-200)"
        label="Avg confidence"
        value={avg_confidence != null ? `${(avg_confidence * 100).toFixed(1)}%` : '—'}
        sub="fused prediction"
      />
    </div>
  );
}
