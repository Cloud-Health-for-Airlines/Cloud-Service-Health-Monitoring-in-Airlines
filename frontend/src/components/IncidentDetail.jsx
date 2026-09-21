import React from 'react';

export default function IncidentDetail({ incident, onClose }) {
  if (!incident) return null;

  const isCritical = incident.severity === 'CRITICAL';
  const color = isCritical ? '#ef4444' : '#f59e0b';

  return (
    <div className="sre-card" style={{ border: `1px solid ${color}`, background: '#0d1322' }}>
      <div className="sre-card-header" style={{ borderBottom: `1px solid ${color}40` }}>
        <span className="sre-card-title" style={{ color }}>
          <span>📋 Incident Detail: Cross-Generation Cascade Diagnosis</span>
        </span>
        <button onClick={onClose} className="sre-btn" style={{ padding: '0.2rem 0.6rem', fontSize: '0.75rem' }}>
          ✕ Close Detail
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
        {/* Incident Summary */}
        <div style={{ background: '#0a0f1d', padding: '1rem', borderRadius: '6px', border: '1px solid #1e293b' }}>
          <div style={{ fontSize: '0.75rem', color: '#9ca3af', textTransform: 'uppercase', marginBottom: '0.3rem' }}>
            What Happened (AIOps Incident Brief)
          </div>
          <div style={{ fontSize: '0.85rem', color: '#f3f4f6', lineHeight: 1.5 }}>
            {incident.explanation}
          </div>
        </div>

        {/* Origin & Affected Path */}
        <div style={{ background: '#0a0f1d', padding: '1rem', borderRadius: '6px', border: '1px solid #1e293b' }}>
          <div style={{ fontSize: '0.75rem', color: '#9ca3af', textTransform: 'uppercase', marginBottom: '0.3rem' }}>
            Likely Origin & Dependency Path
          </div>
          <div style={{ fontSize: '0.85rem', color: '#f3f4f6', marginBottom: '0.5rem' }}>
            <strong>Originating Failure Node:</strong> <code>{incident.rootCause}</code>
          </div>
          <div style={{ fontSize: '0.82rem', color: '#cbd5e1' }}>
            <strong>Propagation Path:</strong><br />
            <code>{incident.affectedDependency}</code>
          </div>
        </div>
      </div>

      {/* Relevant Telemetry & Predicted Risk */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.8rem', marginBottom: '1.25rem' }}>
        <div style={{ background: '#0a0f1d', padding: '0.6rem 0.8rem', borderRadius: '6px', border: '1px solid #1e293b' }}>
          <div style={{ fontSize: '0.7rem', color: '#9ca3af' }}>Cascade Probability</div>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, color }}>{(incident.cascadeProbability * 100).toFixed(1)}%</div>
        </div>

        <div style={{ background: '#0a0f1d', padding: '0.6rem 0.8rem', borderRadius: '6px', border: '1px solid #1e293b' }}>
          <div style={{ fontSize: '0.7rem', color: '#9ca3af' }}>Estimated Lead Time</div>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f59e0b' }}>~{incident.estimatedLeadTimeSeconds.toFixed(0)}s</div>
        </div>

        <div style={{ background: '#0a0f1d', padding: '0.6rem 0.8rem', borderRadius: '6px', border: '1px solid #1e293b' }}>
          <div style={{ fontSize: '0.7rem', color: '#9ca3af' }}>Conformal Confidence</div>
          <div style={{ fontSize: '0.88rem', fontWeight: 600, color: '#93c5fd', marginTop: '0.2rem' }}>{incident.confidence}</div>
        </div>

        <div style={{ background: '#0a0f1d', padding: '0.6rem 0.8rem', borderRadius: '6px', border: '1px solid #1e293b' }}>
          <div style={{ fontSize: '0.7rem', color: '#9ca3af' }}>Detection Timestamp</div>
          <div style={{ fontSize: '0.85rem', fontWeight: 500, color: '#e2e8f0', marginTop: '0.2rem' }}>{new Date(incident.timestamp).toLocaleTimeString()}</div>
        </div>
      </div>

      {/* Recommended Action */}
      <div style={{ background: `${color}15`, border: `1px solid ${color}50`, padding: '0.9rem 1.1rem', borderRadius: '6px' }}>
        <div style={{ fontWeight: 600, fontSize: '0.82rem', color, marginBottom: '0.2rem' }}>
          RECOMMENDED MITIGATION ACTION:
        </div>
        <div style={{ fontSize: '0.82rem', color: '#f3f4f6' }}>
          {incident.recommendedAction}
        </div>
      </div>
    </div>
  );
}
