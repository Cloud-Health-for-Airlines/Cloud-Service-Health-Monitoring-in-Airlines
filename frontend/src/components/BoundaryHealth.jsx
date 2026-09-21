import React from 'react';

export default function BoundaryHealth({ boundaryHealth }) {
  const score = boundaryHealth?.score ?? 12.4;
  const threshold = boundaryHealth?.threshold ?? 45.0;
  const legacyLatency = boundaryHealth?.legacyLatencyMs ?? 68.2;
  const gatewayLatency = boundaryHealth?.gatewayLatencyMs ?? 18.6;
  const syncDrift = boundaryHealth?.synchronizationDrift ?? 12.4;
  const reqResHealth = boundaryHealth?.requestResponseHealth ?? 'HEALTHY';
  const genGapStatus = boundaryHealth?.generationGapStatus ?? 'Active Protocol Translation (HTTP:8080 -> TCP:9090)';

  const circumference = 2 * Math.PI * 70; // ~440
  const dashoffset = circumference - (circumference * (Math.min(100, score) / 100));

  let color = '#10b981';
  if (score >= 70.0) color = '#ef4444';
  else if (score >= threshold) color = '#f59e0b';

  return (
    <div className="sre-card">
      <div className="sre-card-header">
        <span className="sre-card-title">
          <span>🛡️ Boundary Health & State Synchronization</span>
        </span>
        <span className={`sre-badge ${score >= 70 ? 'sre-badge-crit' : (score >= threshold ? 'sre-badge-warn' : 'sre-badge-ok')}`}>
          {reqResHealth}
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '200px 1fr', gap: '1.5rem', alignItems: 'center' }}>
        {/* Radial Health Gauge */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ position: 'relative', width: '160px', height: '160px' }}>
            <svg width="160" height="160" style={{ transform: 'rotate(-90deg)' }}>
              <circle cx="80" cy="80" r="70" fill="none" stroke="#1f293d" strokeWidth="12" />
              <circle
                cx="80"
                cy="80"
                r="70"
                fill="none"
                stroke={color}
                strokeWidth="12"
                strokeDasharray={circumference}
                strokeDashoffset={dashoffset}
                strokeLinecap="round"
                style={{ transition: 'stroke-dashoffset 0.5s ease, stroke 0.5s ease' }}
              />
            </svg>
            <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center' }}>
              <div style={{ fontSize: '1.8rem', fontWeight: 700, color }}>{score.toFixed(1)}%</div>
              <div style={{ fontSize: '0.72rem', color: '#9ca3af' }}>Sync Drift ε(t)</div>
            </div>
          </div>
        </div>

        {/* Boundary Metrics & Generation-Gap Status */}
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.8rem', marginBottom: '0.9rem' }}>
            <div style={{ background: '#0f172a', padding: '0.7rem 0.9rem', borderRadius: '6px', border: '1px solid #1e293b' }}>
              <div style={{ fontSize: '0.72rem', color: '#9ca3af', textTransform: 'uppercase' }}>Legacy Core Latency</div>
              <div style={{ fontSize: '1.15rem', fontWeight: 700, color: legacyLatency > 500 ? '#ef4444' : '#f59e0b' }}>
                {legacyLatency.toFixed(1)} ms
              </div>
              <div style={{ fontSize: '0.7rem', color: '#64748b' }}>Mainframe TCP Socket :9090</div>
            </div>

            <div style={{ background: '#0f172a', padding: '0.7rem 0.9rem', borderRadius: '6px', border: '1px solid #1e293b' }}>
              <div style={{ fontSize: '0.72rem', color: '#9ca3af', textTransform: 'uppercase' }}>Gateway Ingress Latency</div>
              <div style={{ fontSize: '1.15rem', fontWeight: 700, color: '#93c5fd' }}>
                {gatewayLatency.toFixed(1)} ms
              </div>
              <div style={{ fontSize: '0.7rem', color: '#64748b' }}>Cloud HTTP Ingress :8080</div>
            </div>
          </div>

          <div style={{ fontSize: '0.8rem', color: '#cbd5e1', marginBottom: '0.4rem' }}>
            <strong>Generation-Gap / Boundary Status:</strong> {genGapStatus}
          </div>

          <div style={{ fontSize: '0.75rem', color: '#9ca3af', lineHeight: 1.4 }}>
            Digital-Twin Synchronization Divergence Formula:
            <br />
            <code>ε(t) = ||Φ(t) - Ψ(t)|| / ||Φ(t)|| * 100</code> (Warning Threshold: {threshold.toFixed(1)}%)
          </div>
        </div>
      </div>
    </div>
  );
}
