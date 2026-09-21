import React, { useState } from 'react';

export default function CircuitBreakerStatus({ circuitBreaker, onApplyMitigation }) {
  const [sliderRate, setSliderRate] = useState(50);

  const state = circuitBreaker?.state || 'CLOSED';
  const target = circuitBreaker?.targetService || 'boundary-gateway';
  const throttleRate = circuitBreaker?.throttleRate ?? 0.0;
  const reason = circuitBreaker?.reason || 'Nominal healthy envelope';
  const timestamp = circuitBreaker?.timestamp || 'Just now';
  const history = circuitBreaker?.recentActions || [];

  let stateColor = '#10b981';
  let badgeClass = 'sre-badge-ok';
  if (state === 'OPEN') {
    stateColor = '#ef4444';
    badgeClass = 'sre-badge-crit';
  } else if (state === 'THROTTLED') {
    stateColor = '#f59e0b';
    badgeClass = 'sre-badge-warn';
  }

  return (
    <div className="sre-card">
      <div className="sre-card-header">
        <span className="sre-card-title">
          <span>⚡ Boundary Circuit Breaker Status</span>
          <span style={{ fontSize: '0.75rem', color: '#9ca3af', fontWeight: 400 }}>
            (Automated Mitigation Layer)
          </span>
        </span>
        <span className={`sre-badge ${badgeClass}`}>
          STATE: {state} ({Math.round(throttleRate * 100)}%)
        </span>
      </div>

      {/* Prominent Prototype / Simulated Action Notice */}
      <div style={{ background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)', borderRadius: '6px', padding: '0.6rem 0.9rem', marginBottom: '1rem', fontSize: '0.78rem', color: '#93c5fd', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <span>ℹ️</span>
        <div>
          <strong>Operational Notice:</strong> Prototype / simulated action (RL circuit breaker awaiting full online policy training in <code>ai-models/</code>).
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.8rem', marginBottom: '1.2rem' }}>
        <div style={{ background: '#0f172a', padding: '0.7rem 0.9rem', borderRadius: '6px', border: '1px solid #1e293b' }}>
          <div style={{ fontSize: '0.72rem', color: '#9ca3af', textTransform: 'uppercase' }}>Current Breaker State</div>
          <div style={{ fontSize: '1.2rem', fontWeight: 700, color: stateColor }}>{state}</div>
          <div style={{ fontSize: '0.7rem', color: '#64748b' }}>Rate Limit: {Math.round(throttleRate * 100)}%</div>
        </div>

        <div style={{ background: '#0f172a', padding: '0.7rem 0.9rem', borderRadius: '6px', border: '1px solid #1e293b' }}>
          <div style={{ fontSize: '0.72rem', color: '#9ca3af', textTransform: 'uppercase' }}>Target Gateway Node</div>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#c4b5fd' }}><code>{target}</code></div>
          <div style={{ fontSize: '0.7rem', color: '#64748b' }}>Scoped strictly to boundary</div>
        </div>

        <div style={{ background: '#0f172a', padding: '0.7rem 0.9rem', borderRadius: '6px', border: '1px solid #1e293b' }}>
          <div style={{ fontSize: '0.72rem', color: '#9ca3af', textTransform: 'uppercase' }}>Last Action Timestamp</div>
          <div style={{ fontSize: '1.05rem', fontWeight: 600, color: '#e2e8f0' }}>{timestamp}</div>
          <div style={{ fontSize: '0.7rem', color: '#64748b' }}>Automated SRE trigger</div>
        </div>
      </div>

      <div style={{ fontSize: '0.82rem', color: '#cbd5e1', marginBottom: '1rem', background: '#0a0f1d', padding: '0.8rem', borderRadius: '6px', border: '1px solid #1e293b' }}>
        <strong>Current Actuation Reason:</strong> {reason}
      </div>

      {/* Operator Manual Control Actions */}
      <div style={{ borderTop: '1px solid rgba(55, 65, 81, 0.4)', paddingTop: '1rem', marginTop: '1rem' }}>
        <div style={{ fontSize: '0.82rem', fontWeight: 600, color: '#9ca3af', marginBottom: '0.6rem' }}>
          OPERATOR MITIGATION OVERRIDE:
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem', alignItems: 'center' }}>
          <button
            className="sre-btn"
            onClick={() => onApplyMitigation('THROTTLE', sliderRate / 100.0, `Manual throttle: ${sliderRate}%`)}
          >
            Apply {sliderRate}% Throttle
          </button>
          <button
            className="sre-btn sre-btn-danger"
            onClick={() => onApplyMitigation('ISOLATE', 1.0, 'Emergency operator isolation')}
          >
            🚨 Emergency Isolate Gateway
          </button>
          <button
            className="sre-btn sre-btn-primary"
            onClick={() => onApplyMitigation('RESET', 0.0, 'Operator reset to nominal')}
          >
            🔄 Reset Breaker to Closed
          </button>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginLeft: 'auto' }}>
            <span style={{ fontSize: '0.78rem', color: '#9ca3af' }}>Rate:</span>
            <input
              type="range"
              min="10"
              max="90"
              value={sliderRate}
              onChange={(e) => setSliderRate(parseInt(e.target.value))}
              style={{ width: '90px' }}
            />
            <span style={{ fontSize: '0.8rem', fontWeight: 600, minWidth: '35px' }}>{sliderRate}%</span>
          </div>
        </div>
      </div>

      {/* Action History */}
      {history.length > 0 && (
        <div style={{ marginTop: '1.2rem' }}>
          <div style={{ fontSize: '0.78rem', color: '#9ca3af', fontWeight: 600, marginBottom: '0.4rem' }}>
            Recent Mitigation Actuations:
          </div>
          <table className="sre-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Action</th>
                <th>Throttle</th>
                <th>Reason</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {history.slice(-4).reverse().map((item, idx) => (
                <tr key={idx}>
                  <td>{new Date(item.timestamp * 1000).toLocaleTimeString()}</td>
                  <td><strong>{item.action}</strong></td>
                  <td>{Math.round(item.throttle_rate * 100)}%</td>
                  <td style={{ fontSize: '0.75rem', color: '#cbd5e1' }}>{item.reason}</td>
                  <td style={{ fontSize: '0.72rem', color: '#9ca3af' }}><code>{item.invoked_by}</code></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
