import React from 'react';

export default function ChaosControls({ onTriggerChaos }) {
  return (
    <div className="sre-card">
      <div className="sre-card-header">
        <span className="sre-card-title">
          <span>🧪 Testbed Chaos Fault Injection Playground</span>
        </span>
        <span className="sre-badge sre-badge-warn">DETERMINISTIC NETEM / IPTABLES</span>
      </div>

      <div style={{ fontSize: '0.8rem', color: '#9ca3af', marginBottom: '0.8rem' }}>
        Inject deterministic faults on the <code>boundary-gateway -&gt; legacy-core:9090</code> interface to evaluate real-time cascade prediction and RL circuit-breaker mitigation:
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem' }}>
        <button
          className="sre-btn sre-btn-danger"
          onClick={() => onTriggerChaos('network-delay', 'high')}
        >
          ⚡ Network Delay (1500ms)
        </button>

        <button
          className="sre-btn sre-btn-danger"
          onClick={() => onTriggerChaos('connection-drop', 'high')}
        >
          🚫 Connection Drop (100% SYN)
        </button>

        <button
          className="sre-btn sre-btn-danger"
          onClick={() => onTriggerChaos('batch-job-stall', 'high')}
        >
          ⏸️ Batch Stall (15s Pause)
        </button>

        <button
          className="sre-btn sre-btn-primary"
          onClick={() => onTriggerChaos('clear', 'none')}
        >
          🔄 Clear Faults (Reset Baseline)
        </button>
      </div>
    </div>
  );
}
