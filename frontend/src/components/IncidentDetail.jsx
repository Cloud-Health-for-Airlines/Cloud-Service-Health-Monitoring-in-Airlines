import React from 'react';

export default function IncidentDetail({ incident, onClose }) {
  if (!incident) return null;

  const isCritical = incident.severity === 'CRITICAL';
  const color = isCritical ? 'var(--status-crit)' : 'var(--status-warn)';

  return (
    <div
      className="panel-subtle"
      style={{
        border: `1px solid ${color}`,
        borderLeft: `4px solid ${color}`,
        background: 'var(--bg-surface)',
        marginBottom: 'var(--space-4)',
        boxShadow: '0 4px 16px -2px rgba(0, 0, 0, 0.4)',
      }}
    >
      <div className="sre-card-header" style={{ borderColor: 'var(--border-quiet)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '15px' }}>📋</span>
          <h3 className="sre-card-title" style={{ fontSize: '14px', color }}>
            Incident diagnosis: cross-generation cascade brief
          </h3>
        </div>
        <button
          onClick={onClose}
          className="sre-btn sre-btn-secondary"
          style={{ padding: '2px 8px', fontSize: '11px' }}
        >
          ✕ Close brief
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 'var(--space-3)', marginBottom: 'var(--space-3)' }}>
        {/* Incident Summary */}
        <div style={{ background: 'var(--bg-canvas)', padding: '12px', borderRadius: '6px', border: '1px solid var(--border-quiet)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontWeight: 600, marginBottom: '4px' }}>
            What happened (AIOps incident brief)
          </div>
          <div style={{ fontSize: '12.5px', color: 'var(--text-primary)', lineHeight: 1.5 }}>
            {incident.explanation}
          </div>
        </div>

        {/* Origin & Affected Path */}
        <div style={{ background: 'var(--bg-canvas)', padding: '12px', borderRadius: '6px', border: '1px solid var(--border-quiet)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontWeight: 600, marginBottom: '4px' }}>
            Likely origin & propagation path
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-primary)', marginBottom: '6px' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Originating failure node: </span>
            <code>{incident.rootCause}</code>
          </div>
          <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)' }}>
            <span style={{ color: 'var(--text-tertiary)' }}>Propagation path: </span>
            <code>{incident.affectedDependency}</code>
          </div>
        </div>
      </div>

      {/* Relevant Telemetry & Predicted Risk (Monospace metrics) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '8px', marginBottom: 'var(--space-3)' }}>
        <div style={{ background: 'var(--bg-canvas)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-quiet)' }}>
          <div style={{ fontSize: '10.5px', color: 'var(--text-tertiary)' }}>Cascade probability</div>
          <div style={{ fontSize: '16px', fontWeight: 700, fontFamily: 'monospace', color }}>
            {(incident.cascadeProbability * 100).toFixed(1)}%
          </div>
        </div>

        <div style={{ background: 'var(--bg-canvas)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-quiet)' }}>
          <div style={{ fontSize: '10.5px', color: 'var(--text-tertiary)' }}>Estimated lead time</div>
          <div style={{ fontSize: '16px', fontWeight: 700, fontFamily: 'monospace', color: 'var(--status-warn)' }}>
            ~{incident.estimatedLeadTimeSeconds.toFixed(0)}s
          </div>
        </div>

        <div style={{ background: 'var(--bg-canvas)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-quiet)' }}>
          <div style={{ fontSize: '10.5px', color: 'var(--text-tertiary)' }}>Conformal confidence</div>
          <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)', marginTop: '2px' }}>
            {incident.confidence}
          </div>
        </div>

        <div style={{ background: 'var(--bg-canvas)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-quiet)' }}>
          <div style={{ fontSize: '10.5px', color: 'var(--text-tertiary)' }}>Detection timestamp</div>
          <div style={{ fontSize: '12.5px', fontWeight: 500, fontFamily: 'monospace', color: 'var(--text-primary)', marginTop: '2px' }}>
            {new Date(incident.timestamp).toLocaleTimeString()}
          </div>
        </div>
      </div>

      {/* Recommended Action Box */}
      <div
        style={{
          background: isCritical ? 'var(--status-crit-bg)' : 'var(--status-warn-bg)',
          border: `1px solid ${color}`,
          padding: '10px 14px',
          borderRadius: '6px',
        }}
      >
        <div style={{ fontWeight: 600, fontSize: '11.5px', color, marginBottom: '2px' }}>
          Recommended mitigation action:
        </div>
        <div style={{ fontSize: '12.5px', color: 'var(--text-primary)' }}>
          {incident.recommendedAction}
        </div>
      </div>
    </div>
  );
}
