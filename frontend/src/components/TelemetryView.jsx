import React from 'react';

export default function TelemetryView({ onTriggerChaos }) {
  return (
    <div style={{ background: '#111827', border: '1px solid #374151', borderRadius: '10px', padding: '1.25rem', marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', borderBottom: '1px solid #374151', paddingBottom: '0.5rem' }}>
        <span style={{ fontWeight: 600 }}>Testbed Chaos Engineering Playground</span>
        <span style={{ padding: '0.2rem 0.5rem', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600, background: 'rgba(59, 130, 246, 0.2)', color: '#93c5fd' }}>
          DETERMINISTIC FAULTS
        </span>
      </div>

      <div style={{ fontSize: '0.8rem', color: '#9ca3af', marginBottom: '0.8rem' }}>
        Inject deterministic faults on the <code>boundary-gateway -&gt; legacy-core:9090</code> interface to observe real-time cascade prediction and RL circuit-breaker actuation:
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem' }}>
        <button
          onClick={() => onTriggerChaos('network-delay', 'high')}
          style={{ backgroundColor: '#92400e', color: '#fef3c7', border: '1px solid #d97706', padding: '0.5rem 0.8rem', borderRadius: '6px', fontSize: '0.8rem', cursor: 'pointer' }}
        >
          ⚡ Network Delay (1500ms)
        </button>
        <button
          onClick={() => onTriggerChaos('connection-drop', 'high')}
          style={{ backgroundColor: '#991b1b', color: '#fee2e2', border: '1px solid #dc2626', padding: '0.5rem 0.8rem', borderRadius: '6px', fontSize: '0.8rem', cursor: 'pointer' }}
        >
          🚫 Connection Drop (100% SYN)
        </button>
        <button
          onClick={() => onTriggerChaos('batch-job-stall', 'high')}
          style={{ backgroundColor: '#92400e', color: '#fef3c7', border: '1px solid #d97706', padding: '0.5rem 0.8rem', borderRadius: '6px', fontSize: '0.8rem', cursor: 'pointer' }}
        >
          ⏸️ Batch Stall (15s Pause)
        </button>
        <button
          onClick={() => onTriggerChaos('clear')}
          style={{ backgroundColor: '#2563eb', color: '#fff', border: '1px solid #3b82f6', padding: '0.5rem 0.8rem', borderRadius: '6px', fontSize: '0.8rem', cursor: 'pointer' }}
        >
          🔄 Clear Faults (Reset Baseline)
        </button>
      </div>
    </div>
  );
}
