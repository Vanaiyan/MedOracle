/**
 * src/components/SHAPBarChart.jsx
 * Recharts BarChart — EEG / GSR / Video SHAP contributions for current session.
 */

import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Cell, ReferenceLine,
} from 'recharts';

const FEATURE_COLORS = {
  EEG:   '#63b3ed',
  GSR:   '#b794f4',
  video: '#68d391',
};

const FEATURE_LABELS = {
  EEG:   'EEG (Brain Activity)',
  GSR:   'GSR (Skin Response)',
  video: 'Video (Facial)',
};

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0];
  return (
    <div style={{
      background: 'rgba(10,13,22,0.95)',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: 10,
      padding: '0.75rem 1rem',
      fontSize: 13,
    }}>
      <div style={{ color: d.fill, fontWeight: 700, marginBottom: 4 }}>{FEATURE_LABELS[d.payload.feature]}</div>
      <div style={{ color: 'var(--text-secondary)' }}>
        SHAP: <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
          {d.payload.value >= 0 ? '+' : ''}{d.payload.value.toFixed(4)}
        </span>
      </div>
      <div style={{ color: 'var(--text-secondary)', marginTop: 2 }}>
        |SHAP|: <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
          {Math.abs(d.payload.value).toFixed(4)}
        </span>
      </div>
    </div>
  );
}

export default function SHAPBarChart({ shapValues, featureImportance, faithfulness }) {
  if (!shapValues) {
    return (
      <div className="card">
        <p className="card-title">SHAP Modality Contributions</p>
        <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: 14 }}>
          Select a session to view SHAP explanations.
        </div>
      </div>
    );
  }

  const data = Object.entries(shapValues).map(([feature, value]) => ({
    feature,
    label: feature,
    value,
    importance: featureImportance?.[feature] ?? Math.abs(value),
  }));

  const faithColor = faithfulness >= 0.7 ? 'var(--accent-green)'
    : faithfulness >= 0.4 ? 'var(--accent-orange)' : 'var(--accent-red)';

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
        <p className="card-title" style={{ margin: 0 }}>SHAP Modality Contributions</p>
        {faithfulness != null && (
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 3 }}>Faithfulness</div>
            <div style={{ fontSize: 13, fontWeight: 700, color: faithColor }}>
              {(faithfulness * 100).toFixed(0)}%
            </div>
            <div className="faithfulness-bar" style={{ width: 80 }}>
              <div
                className="faithfulness-fill"
                style={{
                  width: `${faithfulness * 100}%`,
                  background: faithColor,
                }}
              />
            </div>
          </div>
        )}
      </div>

      <div className="chart-container" style={{ height: 200 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
            <XAxis
              dataKey="label"
              tick={{ fill: '#8892a4', fontSize: 12 }}
              axisLine={false} tickLine={false}
            />
            <YAxis
              tick={{ fill: '#8892a4', fontSize: 11 }}
              axisLine={false} tickLine={false}
              tickFormatter={v => v.toFixed(2)}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" />
            <Bar dataKey="value" radius={[6, 6, 0, 0]} maxBarSize={60}>
              {data.map((entry) => (
                <Cell key={entry.feature} fill={FEATURE_COLORS[entry.feature]} fillOpacity={0.85} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Feature importance legend */}
      <div style={{ display: 'flex', gap: '1rem', marginTop: 12, flexWrap: 'wrap' }}>
        {data.map(d => (
          <div key={d.feature} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
            <div style={{ width: 8, height: 8, borderRadius: 2, background: FEATURE_COLORS[d.feature] }} />
            <span style={{ color: 'var(--text-secondary)' }}>{FEATURE_LABELS[d.feature]}</span>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
              {d.value >= 0 ? '+' : ''}{d.value.toFixed(3)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
