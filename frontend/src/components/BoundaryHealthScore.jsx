import React, { useState } from 'react';

export default function BoundaryHealthScore({ health }) {
  const [showFormula, setShowFormula] = useState(false);
  const drift = health?.sync_drift_score ?? 14.2;
  const threshold = health?.threshold ?? 45.0;
  const status = health?.status ?? 'HEALTHY';

  const circumference = 2 * Math.PI * 65;
  const dashoffset = circumference - circumference * (Math.min(100, drift) / 100);

  let statusColor = 'var(--status-ok)';
  let badgeClass = 'sre-badge-ok';
  if (drift >= 70.0) {
    statusColor = 'var(--status-crit)';
    badgeClass = 'sre-badge-crit';
  } else if (drift >= threshold) {
    statusColor = 'var(--status-warn)';
    badgeClass = 'sre-badge-warn';
  }

  return (
    <div className="panel-hero">
      <div className="sre-card-header">
        <div>
          <h2 className="sre-card-title">
            <span>🛡️ Digital-twin boundary health</span>
          </h2>
          <span style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>
            Real-time synchronization divergence tracker
          </span>
        </div>
        <span className={`sre-badge ${badgeClass}`}>{status}</span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: 'var(--space-3) 0' }}>
        <div style={{ position: 'relative', width: '160px', height: '160px' }}>
          <svg width="160" height="160" style={{ transform: 'rotate(-90deg)' }}>
            <circle cx="80" cy="80" r="65" fill="none" stroke="var(--border-quiet)" strokeWidth="10" />
            <circle
              cx="80"
              cy="80"
              r="65"
              fill="none"
              stroke={statusColor}
              strokeWidth="10"
              strokeDasharray={circumference}
              strokeDashoffset={dashoffset}
              strokeLinecap="round"
              style={{ transition: 'stroke-dashoffset 0.5s ease, stroke 0.5s ease' }}
            />
          </svg>
          <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center' }}>
            <div style={{ fontSize: '26px', fontWeight: 700, fontFamily: 'monospace', color: statusColor }}>
              {drift.toFixed(1)}%
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Sync drift ε(t)</div>
          </div>
        </div>
      </div>

      <div style={{ textAlign: 'center', marginTop: 'var(--space-2)' }}>
        <button
          onClick={() => setShowFormula(!showFormula)}
          className="sre-btn sre-btn-secondary"
          style={{ padding: '3px 8px', fontSize: '11px' }}
        >
          {showFormula ? 'Hide formula' : 'ℹ️ How drift is calculated'}
        </button>
        {showFormula && (
          <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '8px' }}>
            <code>ε(t) = ||Φ(t) - Ψ(t)|| / ||Φ(t)|| * 100</code><br />
            Baseline threshold: <strong>{threshold}%</strong> | Critical: <strong>70.0%</strong>
          </div>
        )}
      </div>
    </div>
  );
}
