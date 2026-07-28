/**
 * src/components/EmotionTrendChart.jsx
 * Recharts LineChart — fused confidence + physio-only confidence per session.
 */

import { useState } from 'react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend,
} from 'recharts';

// How many of the most-recent sessions to plot. 0 = all.
const WINDOW_OPTIONS = [
  { label: 'Last 10', value: 10 },
  { label: 'Last 25', value: 25 },
  { label: 'Last 50', value: 50 },
  { label: 'All', value: 0 },
];

// Organic palette: EEG=sage-500, GSR=terracotta-400, Video=sage-700, Fused=terracotta-700
const SERIES_COLORS = {
  Fused: '#8c491a', EEG: '#8fa073', GSR: '#f6a06b', Video: '#56633f',
};

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'var(--color-surface)',
      border: '1px solid var(--color-divider)',
      borderRadius: 12,
      padding: '0.75rem 1rem',
      fontSize: 13,
      boxShadow: 'var(--shadow-md)',
    }}>
      <p style={{ color: 'var(--text-secondary)', marginBottom: 6, fontSize: 11 }}>{label}</p>
      {payload.map(p => p.value != null && (
        <div key={p.name} style={{ color: p.color, fontWeight: 600, marginBottom: 2 }}>
          {p.name}: {(p.value * 100).toFixed(1)}%
        </div>
      ))}
      {payload[0]?.payload?.predicted_emotion && (
        <div style={{ color: 'var(--text-muted)', fontSize: 11, marginTop: 4 }}>
          Emotion: {payload[0].payload.predicted_emotion}
        </div>
      )}
    </div>
  );
}

export default function EmotionTrendChart({ trend = [] }) {
  // Default to the 10 most recent sessions; user can widen the window.
  const [limit, setLimit] = useState(10);

  // Number against the full history first, then keep only the last N so the
  // x-axis labels stay the true session index (e.g. S38…S47, not S1…S10).
  const allPoints = trend.map((t, i) => ({
    name: `S${i + 1}`,
    'Fused': t.fused_confidence,
    'EEG':   t.eeg_confidence,
    'GSR':   t.gsr_confidence,
    'Video': t.video_confidence,
    predicted_emotion: t.predicted_emotion,
    timestamp: new Date(t.timestamp).toLocaleDateString(),
  }));

  const data = limit > 0 ? allPoints.slice(-limit) : allPoints;

  if (allPoints.length === 0) {
    return (
      <div className="card">
        <h6 style={{ margin: 0 }}>Emotion confidence trend</h6>
        <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-neutral-600)', fontSize: 14 }}>
          No sessions yet — run a prediction to see the trend.
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <h6 style={{ margin: 0 }}>Emotion confidence trend</h6>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 11.5, color: 'var(--color-neutral-600)' }}>
            Showing {data.length} of {allPoints.length}
          </span>
          <select
            className="input"
            aria-label="Number of sessions to show"
            value={limit}
            onChange={e => setLimit(Number(e.target.value))}
            style={{ width: 'auto', minHeight: 32, padding: '4px 12px', fontSize: 12.5, cursor: 'pointer' }}
          >
            {WINDOW_OPTIONS.map(o => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
      </div>
      <div className="chart-container" style={{ height: 220, marginTop: 8 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 20, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 4" stroke="var(--color-divider)" />
            <XAxis
              dataKey="name"
              tick={{ fill: '#82796a', fontSize: 11 }}
              axisLine={{ stroke: 'var(--color-neutral-400)' }}
              tickLine={false}
            />
            <YAxis
              domain={[0, 1]}
              tickFormatter={v => `${(v * 100).toFixed(0)}%`}
              tick={{ fill: '#82796a', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, color: '#645c50', paddingTop: 8 }} />
            <Line
              type="monotone" dataKey="Fused" stroke={SERIES_COLORS.Fused}
              strokeWidth={2.5} dot={{ fill: SERIES_COLORS.Fused, r: 4, strokeWidth: 0 }}
              activeDot={{ r: 6, fill: SERIES_COLORS.Fused }}
            />
            <Line
              type="monotone" dataKey="EEG" stroke={SERIES_COLORS.EEG}
              strokeWidth={2} strokeDasharray="3 3" dot={{ fill: SERIES_COLORS.EEG, r: 3, strokeWidth: 0 }}
              activeDot={{ r: 5, fill: SERIES_COLORS.EEG }}
            />
            <Line
              type="monotone" dataKey="GSR" stroke={SERIES_COLORS.GSR}
              strokeWidth={2} strokeDasharray="3 3" dot={{ fill: SERIES_COLORS.GSR, r: 3, strokeWidth: 0 }}
              activeDot={{ r: 5, fill: SERIES_COLORS.GSR }}
            />
            <Line
              type="monotone" dataKey="Video" stroke={SERIES_COLORS.Video}
              strokeWidth={2.5} dot={{ fill: SERIES_COLORS.Video, r: 3, strokeWidth: 0 }}
              activeDot={{ r: 5, fill: SERIES_COLORS.Video }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
