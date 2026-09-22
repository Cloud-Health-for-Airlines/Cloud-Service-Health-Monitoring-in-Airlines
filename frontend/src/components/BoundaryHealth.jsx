import React, { useState } from 'react';

export default function BoundaryHealth({ boundaryHealth }) {
  const [showFormula, setShowFormula] = useState(false);

  const score = boundaryHealth?.score ?? 12.4;
  const threshold = boundaryHealth?.threshold ?? 45.0;
  const legacyLatency = boundaryHealth?.legacyLatencyMs ?? 68.2;
  const gatewayLatency = boundaryHealth?.gatewayLatencyMs ?? 18.6;
  const latencyDelta = Math.abs(legacyLatency - gatewayLatency);
  const reqResHealth = boundaryHealth?.requestResponseHealth ?? 'HEALTHY';
  const genGapStatus = boundaryHealth?.generationGapStatus ?? 'Active protocol translation (HTTP:8080 → TCP:9090)';

  const circumference = 2 * Math.PI * 65; // ~408.4
  const dashoffset = circumference - circumference * (Math.min(100, score) / 100);

  let statusColor = 'var(--status-ok)';
  let badgeClass = 'sre-badge-ok';
  if (score >= 70.0) {
    statusColor = 'var(--status-crit)';
    badgeClass = 'sre-badge-crit';
  } else if (score >= threshold) {
    statusColor = 'var(--status-warn)';
    badgeClass = 'sre-badge-warn';
  }

  return (
    <div className="panel-hero">
      <div className="sre-card-header">
        <div>
          <h2 className="sre-card-title">
            <span>🛡️ Boundary health & synchronization</span>
          </h2>
          <span style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>
            Digital-twin cross-generational divergence monitor
          </span>
        </div>
        <span className={`sre-badge ${badgeClass}`}>{reqResHealth}</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '170px 1fr', gap: 'var(--space-4)', alignItems: 'center' }}>
        {/* Radial Health Gauge */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ position: 'relative', width: '146px', height: '146px' }}>
            <svg width="146" height="146" style={{ transform: 'rotate(-90deg)' }}>
              <circle cx="73" cy="73" r="65" fill="none" stroke="var(--border-quiet)" strokeWidth="10" />
              <circle
                cx="73"
                cy="73"
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
            <div
              style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                textAlign: 'center',
              }}
            >
              <div
                style={{
                  fontSize: '26px',
                  fontWeight: 700,
                  fontFamily: 'JetBrains Mono, monospace',
                  color: statusColor,
                  lineHeight: 1.1,
                }}
              >
                {score.toFixed(1)}%
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                Sync drift ε(t)
              </div>
            </div>
          </div>
        </div>

        {/* Boundary Metrics & Generation-Gap Status */}
        <div>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: 'var(--space-3)', lineHeight: 1.4 }}>
            Tracks real-time digital-twin synchronization divergence between cloud microservices and legacy mainframe state.
          </p>

          {/* Unified Comparative Latency Meter (Side-by-Side) */}
          <div
            style={{
              background: 'var(--bg-canvas)',
              borderRadius: '6px',
              border: '1px solid var(--border-quiet)',
              padding: '10px 12px',
              marginBottom: 'var(--space-3)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
              <span style={{ fontSize: '11px', color: 'var(--text-secondary)', fontWeight: 500 }}>
                Boundary latency comparison
              </span>
              <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>
                Δ {latencyDelta.toFixed(1)} ms across boundary
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
              {/* Cloud Ingress Side */}
              <div style={{ borderRight: '1px solid var(--border-quiet)', paddingRight: '8px' }}>
                <div style={{ fontSize: '10.5px', color: 'var(--text-secondary)' }}>Cloud ingress (HTTP :8080)</div>
                <div style={{ fontSize: '15px', fontWeight: 600, fontFamily: 'monospace', color: '#93c5fd' }}>
                  {gatewayLatency.toFixed(1)} ms
                </div>
              </div>

              {/* Legacy Core Side */}
              <div style={{ paddingLeft: '4px' }}>
                <div style={{ fontSize: '10.5px', color: 'var(--text-secondary)' }}>Legacy core (TCP :9090)</div>
                <div
                  style={{
                    fontSize: '15px',
                    fontWeight: 600,
                    fontFamily: 'monospace',
                    color: legacyLatency > 500 ? 'var(--status-crit)' : 'var(--status-warn)',
                  }}
                >
                  {legacyLatency.toFixed(1)} ms
                </div>
              </div>
            </div>
          </div>

          <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)', marginBottom: '6px' }}>
            <strong style={{ color: 'var(--text-primary)' }}>Protocol boundary:</strong> {genGapStatus}
          </div>

          {/* Collapsible Methodology & Formula Disclosure */}
          <div>
            <button
              onClick={() => setShowFormula(!showFormula)}
              className="sre-btn sre-btn-secondary"
              style={{ padding: '3px 8px', fontSize: '11px' }}
              title="Click to view mathematical drift formula"
            >
              {showFormula ? 'Hide methodology details' : 'ℹ️ How drift is calculated'}
            </button>

            {showFormula && (
              <div
                style={{
                  marginTop: '8px',
                  background: 'var(--bg-canvas)',
                  border: '1px solid var(--border-quiet)',
                  borderRadius: '6px',
                  padding: '8px 12px',
                  fontSize: '11px',
                  color: 'var(--text-secondary)',
                  lineHeight: 1.5,
                }}
              >
                <div>
                  Divergence formula: <code style={{ color: 'var(--accent-brand)' }}>ε(t) = ||Φ(t) - Ψ(t)|| / ||Φ(t)|| * 100</code>
                </div>
                <div style={{ marginTop: '2px', color: 'var(--text-tertiary)' }}>
                  Where <code>Φ(t)</code> is the ground-truth mainframe state vector and <code>Ψ(t)</code> is the cloud shadow state.
                  Warning threshold: <strong>{threshold.toFixed(1)}%</strong> | Critical threshold: <strong>70.0%</strong>.
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
