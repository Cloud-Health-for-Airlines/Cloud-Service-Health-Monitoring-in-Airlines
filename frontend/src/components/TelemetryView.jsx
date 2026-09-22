import React from 'react';

export default function TelemetryView({ onTriggerChaos }) {
  return (
    <div className="panel-subtle">
      <div className="sre-card-header">
        <div>
          <h3 className="sre-card-title">
            <span>🧪 Testbed chaos engineering playground</span>
          </h3>
          <span style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>
            Deterministic NetEm / IPTables fault injection
          </span>
        </div>
        <span className="sre-badge sre-badge-warn">Synthetic faults</span>
      </div>

      <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: 'var(--space-3)' }}>
        Inject deterministic faults on the <code>boundary-gateway → legacy-core:9090</code> interface to evaluate real-time cascade prediction and RL circuit-breaker actuation:
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
        <button
          className="sre-btn sre-btn-danger"
          onClick={() => onTriggerChaos('network-delay', 'high')}
        >
          ⚡ Network delay (1500ms)
        </button>
        <button
          className="sre-btn sre-btn-danger"
          onClick={() => onTriggerChaos('connection-drop', 'high')}
        >
          🚫 Connection drop (100% SYN)
        </button>
        <button
          className="sre-btn sre-btn-danger"
          onClick={() => onTriggerChaos('batch-job-stall', 'high')}
        >
          ⏸️ Batch stall (15s pause)
        </button>
        <button
          className="sre-btn sre-btn-primary"
          onClick={() => onTriggerChaos('clear')}
        >
          🔄 Clear faults (reset baseline)
        </button>
      </div>
    </div>
  );
}
