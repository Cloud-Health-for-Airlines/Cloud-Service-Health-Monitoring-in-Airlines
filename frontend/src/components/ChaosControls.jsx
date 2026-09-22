import React, { useState } from 'react';

export default function ChaosControls({ onTriggerChaos, activeFault }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div
      className="panel-subtle"
      style={{
        border: activeFault ? '1px solid var(--status-crit-border)' : '1px solid var(--border-quiet)',
        background: activeFault ? 'rgba(239, 68, 68, 0.05)' : 'var(--bg-surface-subtle)',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          cursor: 'pointer',
          userSelect: 'none',
        }}
        onClick={() => setIsOpen(!isOpen)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '14px' }}>🧪</span>
          <div>
            <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
              Testbed chaos fault injection playground
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>
              Deterministic NetEm / IPTables emulation on <code>boundary-gateway → legacy-core:9090</code>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {activeFault ? (
            <span className="sre-badge sre-badge-crit">Fault active: {activeFault}</span>
          ) : (
            <span className="sre-badge sre-badge-ok">Nominal baseline</span>
          )}
          <button
            className="sre-btn sre-btn-secondary"
            style={{ padding: '2px 8px', fontSize: '11px' }}
            onClick={(e) => {
              e.stopPropagation();
              setIsOpen(!isOpen);
            }}
          >
            {isOpen ? '▲ Hide testbed' : '▼ Expand testbed'}
          </button>
        </div>
      </div>

      {isOpen && (
        <div style={{ marginTop: 'var(--space-3)', paddingTop: 'var(--space-3)', borderTop: '1px solid var(--border-quiet)' }}>
          <p style={{ fontSize: '11.5px', color: 'var(--text-secondary)', marginBottom: 'var(--space-2)' }}>
            Inject synthetic boundary faults to evaluate RGCN cascade prediction lead time and automated RL circuit breaker actuation:
          </p>

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
              onClick={() => onTriggerChaos('clear', 'none')}
            >
              🔄 Clear faults (reset baseline)
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
