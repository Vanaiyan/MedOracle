/**
 * src/components/EmotionTrendChart.jsx
 * Recharts LineChart — fused confidence + physio-only confidence per session.
 */

import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend,
} from 'recharts';

const EMOTION_COLORS = {
  stress: '#fc8181', calm: '#68d391', happy: '#f6ad55',
  sad: '#63b3ed', angry: '#ff6b6b',
};

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'rgba(10,13,22,0.95)',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: 10,
      padding: '0.75rem 1rem',
      fontSize: 13,
    }}>
      <p style={{ color: 'var(--text-secondary)', marginBottom: 6, fontSize: 11 }}>{label}</p>
      {payload.map(p => (
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
  const data = trend.map((t, i) => ({
    name: `S${i + 1}`,
    'Fused': t.fused_confidence,
    'Physio': t.physio_confidence,
    predicted_emotion: t.predicted_emotion,
    timestamp: new Date(t.timestamp).toLocaleDateString(),
  }));

  if (data.length === 0) {
    return (
      <div className="card">
        <p className="card-title">Emotion Confidence Trend</p>
        <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: 14 }}>
          No sessions yet — run a prediction to see the trend.
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <p className="card-title">Emotion Confidence Trend</p>
      <div className="chart-container" style={{ height: 220, marginTop: 8 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 20, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis
              dataKey="name"
              tick={{ fill: '#8892a4', fontSize: 11 }}
              axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
              tickLine={false}
            />
            <YAxis
              domain={[0, 1]}
              tickFormatter={v => `${(v * 100).toFixed(0)}%`}
              tick={{ fill: '#8892a4', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 12, color: '#8892a4', paddingTop: 8 }}
            />
            <Line
              type="monotone"
              dataKey="Fused"
              stroke="#63b3ed"
              strokeWidth={2.5}
              dot={{ fill: '#63b3ed', r: 4, strokeWidth: 0 }}
              activeDot={{ r: 6, fill: '#63b3ed' }}
            />
            <Line
              type="monotone"
              dataKey="Physio"
              stroke="#b794f4"
              strokeWidth={2}
              strokeDasharray="5 3"
              dot={{ fill: '#b794f4', r: 3, strokeWidth: 0 }}
              activeDot={{ r: 5, fill: '#b794f4' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
