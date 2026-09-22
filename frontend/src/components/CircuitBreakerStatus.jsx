import React, { useState } from 'react';

export default function CircuitBreakerStatus({ circuitBreaker, onApplyMitigation }) {
  const [sliderRate, setSliderRate] = useState(50);
  const [confirmIsolate, setConfirmIsolate] = useState(false);

  const state = circuitBreaker?.state || 'CLOSED';
  const target = circuitBreaker?.targetService || 'boundary-gateway';
  const throttleRate = circuitBreaker?.throttleRate ?? 0.0;
  const reason = circuitBreaker?.reason || 'Nominal healthy operating envelope';
  const timestamp = circuitBreaker?.timestamp || 'Live';
  const history = circuitBreaker?.recentActions || [];

  let stateColor = 'var(--status-ok)';
  let badgeClass = 'sre-badge-ok';
  if (state === 'OPEN') {
    stateColor = 'var(--status-crit)';
    badgeClass = 'sre-badge-crit';
  } else if (state === 'THROTTLED') {
    stateColor = 'var(--status-warn)';
    badgeClass = 'sre-badge-warn';
  }

  const handleIsolateClick = () => {
    if (!confirmIsolate) {
      setConfirmIsolate(true);
      return;
    }
    setConfirmIsolate(false);
    onApplyMitigation('ISOLATE', 1.0, 'Emergency operator isolation invoked');
  };

  return (
    <div className="sre-card">
      <div className="sre-card-header">
        <div>
          <h2 className="sre-card-title">
            <span>⚡ Boundary circuit breaker & mitigation</span>
          </h2>
          <span style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>
            Automated boundary actuator & operator manual overrides
          </span>
        </div>
        <span className={`sre-badge ${badgeClass}`}>
          {state} ({Math.round(throttleRate * 100)}%)
        </span>
      </div>

      {/* Quiet Prototype Notice */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          fontSize: '11px',
          color: 'var(--text-secondary)',
          background: 'var(--bg-canvas)',
          padding: '6px 12px',
          borderRadius: '6px',
          marginBottom: 'var(--space-3)',
          border: '1px solid var(--border-quiet)',
        }}
      >
        <span style={{ color: 'var(--accent-brand)' }}>ℹ️</span>
        <span>
          <strong>Operational note:</strong> Prototype policy running in standalone simulated mode (trained via PPO agent in <code>ai-models/circuit-breaker</code>).
        </span>
      </div>

      {/* State Breakdown (Flat Grid, No Nested Boxes) */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
          gap: '10px',
          marginBottom: 'var(--space-3)',
        }}
      >
        <div
          style={{
            background: 'var(--bg-canvas)',
            padding: '10px 14px',
            borderRadius: '6px',
            border: '1px solid var(--border-quiet)',
            borderLeft: `3px solid ${stateColor}`,
          }}
        >
          <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Current breaker state</div>
          <div style={{ fontSize: '16px', fontWeight: 700, color: stateColor, fontFamily: 'monospace' }}>
            {state}
          </div>
          <div style={{ fontSize: '10.5px', color: 'var(--text-tertiary)' }}>
            Rate limit: {Math.round(throttleRate * 100)}%
          </div>
        </div>

        <div
          style={{
            background: 'var(--bg-canvas)',
            padding: '10px 14px',
            borderRadius: '6px',
            border: '1px solid var(--border-quiet)',
          }}
        >
          <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Target boundary node</div>
          <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'monospace' }}>
            {target}
          </div>
          <div style={{ fontSize: '10.5px', color: 'var(--text-tertiary)' }}>
            Scoped strictly to ingress
          </div>
        </div>

        <div
          style={{
            background: 'var(--bg-canvas)',
            padding: '10px 14px',
            borderRadius: '6px',
            border: '1px solid var(--border-quiet)',
          }}
        >
          <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Last actuation</div>
          <div style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-primary)', fontFamily: 'monospace' }}>
            {timestamp}
          </div>
          <div style={{ fontSize: '10.5px', color: 'var(--text-tertiary)' }}>
            Automated SRE policy
          </div>
        </div>
      </div>

      {/* Actuation Reason Callout */}
      <div
        style={{
          fontSize: '12px',
          color: 'var(--text-secondary)',
          background: 'var(--bg-canvas)',
          padding: '8px 12px',
          borderRadius: '6px',
          border: '1px solid var(--border-quiet)',
          marginBottom: 'var(--space-3)',
        }}
      >
        <strong style={{ color: 'var(--text-primary)' }}>Actuation reason:</strong> {reason}
      </div>

      {/* Operator Manual Overrides with Proper Button Hierarchy */}
      <div
        style={{
          borderTop: '1px solid var(--border-quiet)',
          paddingTop: 'var(--space-3)',
          marginTop: 'var(--space-2)',
        }}
      >
        <div style={{ fontSize: '11.5px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>
          Operator manual mitigation overrides:
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center' }}>
          {/* Routine Action: Apply Throttle */}
          <button
            className="sre-btn sre-btn-primary"
            onClick={() =>
              onApplyMitigation('THROTTLE', sliderRate / 100.0, `Manual operator throttle: ${sliderRate}%`)
            }
          >
            Apply {sliderRate}% throttle
          </button>

          {/* Destructive / High-Consequence Action: Emergency Isolate */}
          <button
            className="sre-btn sre-btn-destructive"
            onClick={handleIsolateClick}
            title="Disconnect boundary gateway from legacy mainframe"
          >
            {confirmIsolate ? '⚠️ Confirm emergency isolation?' : '🚨 Emergency isolate gateway'}
          </button>

          {/* Recovery Action: Reset Breaker */}
          <button
            className="sre-btn sre-btn-secondary"
            onClick={() => {
              setConfirmIsolate(false);
              onApplyMitigation('RESET', 0.0, 'Operator reset to nominal');
            }}
          >
            🔄 Reset breaker to closed
          </button>

          {/* Throttle Slider */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              marginLeft: 'auto',
              background: 'var(--bg-canvas)',
              padding: '4px 10px',
              borderRadius: '6px',
              border: '1px solid var(--border-quiet)',
            }}
          >
            <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Throttle rate:</span>
            <input
              type="range"
              min="10"
              max="90"
              value={sliderRate}
              onChange={(e) => setSliderRate(parseInt(e.target.value))}
              style={{ width: '80px', accentColor: 'var(--accent-brand)' }}
            />
            <span style={{ fontSize: '12px', fontWeight: 600, fontFamily: 'monospace', minWidth: '32px' }}>
              {sliderRate}%
            </span>
          </div>
        </div>
      </div>

      {/* Recent Actuation Audit History Table */}
      {history.length > 0 && (
        <div style={{ marginTop: 'var(--space-3)' }}>
          <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)', fontWeight: 600, marginBottom: '6px' }}>
            Recent mitigation audit log:
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table className="sre-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Action</th>
                  <th>Throttle rate</th>
                  <th>Actuation reason</th>
                  <th>Invoked by</th>
                </tr>
              </thead>
              <tbody>
                {history
                  .slice(-4)
                  .reverse()
                  .map((item, idx) => (
                    <tr key={idx}>
                      <td style={{ fontFamily: 'monospace', fontSize: '11.5px' }}>
                        {new Date(item.timestamp * 1000).toLocaleTimeString()}
                      </td>
                      <td>
                        <strong style={{ color: 'var(--text-primary)' }}>{item.action}</strong>
                      </td>
                      <td style={{ fontFamily: 'monospace' }}>{Math.round(item.throttle_rate * 100)}%</td>
                      <td style={{ fontSize: '11.5px', color: 'var(--text-secondary)' }}>{item.reason}</td>
                      <td>
                        <code style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>{item.invoked_by}</code>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
