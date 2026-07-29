/**
 * src/components/IGAttributionPanel.jsx
 * Real, fused Integrated Gradients (attribution/fused_ig.py) for the current
 * session — one gradient path through BOTH real trained models (Member 1's
 * PhysiologicalNet + Member 2's VideoEmotionModel) and the real gated-fusion
 * formula, explaining the fused decision directly from raw EEG/GSR/video.
 *
 * Modality totals sit next to the SHAP bar chart for direct comparison
 * ("SHAP says X, IG says Y — same fused decision, two different attribution
 * methods"). Expandable sections show the raw per-EEG-channel and
 * per-video-frame breakdown IG actually computed.
 *
 * Only populated for sessions created via a full multimodal prediction
 * (video + EEG + GSR all supplied) — real IG needs the raw signal tensors,
 * which aren't persisted, so older / video-only / synthetic sessions won't
 * have this data. See backend/routers/predict_router.py.
 */

import { useState } from 'react';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Cell, ReferenceLine,
} from 'recharts';

const MODALITY_COLORS = { physio: '#aebf92', video: '#56633f' };
const MODALITY_LABELS = { physio: 'Physiological (EEG+GSR)', video: 'Video (Facial)' };

function MiniBar({ label, value, max, color }) {
  const pct = max > 0 ? Math.min(100, (Math.abs(value) / max) * 100) : 0;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
      <span style={{ fontSize: 11, width: 56, color: 'var(--text-secondary)' }}>{label}</span>
      <div style={{ flex: 1, height: 6, background: 'var(--color-neutral-200)', borderRadius: 999, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 999 }} />
      </div>
      <span style={{ fontSize: 11, width: 56, textAlign: 'right', color: 'var(--text-primary)', fontWeight: 600 }}>
        {value.toFixed(4)}
      </span>
    </div>
  );
}

export default function IGAttributionPanel({ igAttribution }) {
  const [expanded, setExpanded] = useState(false);

  if (!igAttribution) {
    return (
      <div className="card">
        <h6 style={{ margin: 0 }}>Integrated Gradients (fused)</h6>
        <div style={{
          height: 120, display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: 'var(--color-neutral-600)', fontSize: 13, textAlign: 'center', padding: '0 1rem',
        }}>
          Not available for this session. Real IG needs raw EEG + GSR + video —
          run <strong>Multimodal fusion</strong> with all three files to see it here.
        </div>
      </div>
    );
  }

  const {
    modality_totals = {}, eeg_channel_importance = {},
    video_frame_importance = {}, completeness_check, target_emotion,
  } = igAttribution;

  const modalityData = Object.entries(modality_totals).map(([modality, value]) => ({ modality, value }));
  const maxModality = Math.max(...modalityData.map(d => Math.abs(d.value)), 1e-9);

  const eegEntries = Object.entries(eeg_channel_importance)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const videoEntries = Object.entries(video_frame_importance)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxEeg = Math.max(...eegEntries.map(([, v]) => Math.abs(v)), 1e-9);
  const maxVideo = Math.max(...videoEntries.map(([, v]) => Math.abs(v)), 1e-9);

  const gap = completeness_check?.gap;
  const gapOk = gap != null && gap < 0.05;

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12, flexWrap: 'wrap', gap: 8 }}>
        <div>
          <h6 style={{ margin: 0 }}>Integrated Gradients (fused)</h6>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', margin: '2px 0 0' }}>
            Real gradients through both trained models + the fusion formula, explaining "{target_emotion}"
          </p>
        </div>
        {gap != null && (
          <div
            className="tag tag-outline"
            title="Completeness axiom: how closely the summed per-input attributions match the actual change in fused probability from baseline to this input. Lower is better."
          >
            completeness gap {gap.toFixed(4)} {gapOk ? '✓' : ''}
          </div>
        )}
      </div>

      <div className="chart-container" style={{ height: 130 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={modalityData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 4" stroke="var(--color-divider)" vertical={false} />
            <XAxis
              dataKey="modality"
              tickFormatter={(m) => MODALITY_LABELS[m] || m}
              tick={{ fill: '#82796a', fontSize: 12 }}
              axisLine={false} tickLine={false}
            />
            <YAxis
              tick={{ fill: '#82796a', fontSize: 11 }}
              axisLine={false} tickLine={false}
              tickFormatter={(v) => v.toFixed(2)}
            />
            <Tooltip formatter={(v) => Number(v).toFixed(4)} labelFormatter={(m) => MODALITY_LABELS[m] || m} />
            <ReferenceLine y={0} stroke="var(--color-neutral-400)" />
            <Bar dataKey="value" radius={[6, 6, 0, 0]} maxBarSize={60}>
              {modalityData.map((d) => (
                <Cell key={d.modality} fill={MODALITY_COLORS[d.modality]} fillOpacity={0.85} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <button
        className="btn btn-ghost"
        style={{ marginTop: 10, fontSize: 12, padding: '4px 10px' }}
        onClick={() => setExpanded((e) => !e)}
      >
        {expanded ? 'Hide' : 'Show'} per-channel / per-frame breakdown
      </button>

      {expanded && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginTop: 14 }}>
          <div>
            <div style={{ fontSize: 12, color: 'var(--color-neutral-600)', marginBottom: 8 }}>
              EEG channels ({eegEntries.length})
            </div>
            <div style={{ maxHeight: 220, overflowY: 'auto', paddingRight: 4 }}>
              {eegEntries.map(([ch, v]) => (
                <MiniBar key={ch} label={ch} value={v} max={maxEeg} color={MODALITY_COLORS.physio} />
              ))}
            </div>
          </div>
          <div>
            <div style={{ fontSize: 12, color: 'var(--color-neutral-600)', marginBottom: 8 }}>
              Video frames ({videoEntries.length})
            </div>
            <div style={{ maxHeight: 220, overflowY: 'auto', paddingRight: 4 }}>
              {videoEntries.map(([fr, v]) => (
                <MiniBar key={fr} label={fr} value={v} max={maxVideo} color={MODALITY_COLORS.video} />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
