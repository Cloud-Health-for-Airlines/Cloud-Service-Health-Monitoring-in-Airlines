import React from 'react';

export default function BoundaryHealthScore({ health }) {
  const drift = health?.sync_drift_score ?? 14.2;
  const threshold = health?.threshold ?? 45.0;
  const status = health?.status ?? 'HEALTHY';

  const circumference = 2 * Math.PI * 75; // ~471
  const dashoffset = circumference - (circumference * (drift / 100));

  let color = '#10b981';
  let badgeClass = 'badge-ok';
  if (drift >= 70.0) {
    color = '#ef4444';
    badgeClass = 'badge-crit';
  } else if (drift >= threshold) {
    color = '#f59e0b';
    badgeClass = 'badge-legacy';
  }

  return (
    <div style={{ background: '#111827', border: '1px solid #374151', borderRadius: '10px', padding: '1.25rem', marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', borderBottom: '1px solid #374151', paddingBottom: '0.5rem' }}>
        <span style={{ fontWeight: 600 }}>Digital-Twin Boundary Health</span>
        <span style={{ padding: '0.2rem 0.5rem', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600, background: `${color}20`, color: color, border: `1px solid ${color}60` }}>
          {status}
        </span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '1rem 0' }}>
        <div style={{ position: 'relative', width: '190px', height: '190px' }}>
          <svg width="190" height="190" style={{ transform: 'rotate(-90deg)' }}>
            <circle cx="95" cy="95" r="75" fill="none" stroke="#1f2937" strokeWidth="14" />
            <circle cx="95" cy="95" r="75" fill="none" stroke={color} strokeWidth="14"
              strokeDasharray={circumference} strokeDashoffset={dashoffset} strokeLinecap="round"
              style={{ transition: 'stroke-dashoffset 0.5s ease, stroke 0.5s ease' }} />
          </svg>
          <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center' }}>
            <div style={{ fontSize: '2.2rem', fontWeight: 700 }}>{drift.toFixed(1)}%</div>
            <div style={{ fontSize: '0.8rem', color: '#9ca3af' }}>Sync Drift ε(t)</div>
          </div>
        </div>
      </div>

      <div style={{ fontSize: '0.78rem', color: '#9ca3af', textAlign: 'center', lineHeight: 1.4 }}>
        Formal State Divergence: <code>ε(t) = ||Φ(t) - Ψ(t)|| / ||Φ(t)|| * 100</code><br />
        Baseline Threshold: <strong>{threshold}%</strong> | Critical: <strong>70.0%</strong>
      </div>
    </div>
  );
}
