import React from 'react';

export default function PredictiveAlerts({ alerts, onSelectAlert, selectedAlertId }) {
  if (!alerts || alerts.length === 0) {
    return (
      <div
        className="panel-subtle"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '12px 18px',
          borderLeft: '3px solid var(--status-ok)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '14px', color: 'var(--status-ok)' }}>●</span>
          <div>
            <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
              Nominal operating envelope
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
              No cascade predicted. Boundary latency and synchronization drift are within statistical limits.
            </div>
          </div>
        </div>
        <span className="sre-badge sre-badge-ok">0 active alarms</span>
      </div>
    );
  }

  return (
    <div
      style={{
        background: 'rgba(239, 68, 68, 0.06)',
        border: '1px solid var(--status-crit-border)',
        borderRadius: '8px',
        padding: 'var(--space-4)',
        marginBottom: 'var(--space-4)',
        boxShadow: '0 0 15px -3px rgba(239, 68, 68, 0.15)',
      }}
    >
      <div className="sre-card-header" style={{ borderColor: 'rgba(239, 68, 68, 0.25)' }}>
        <div>
          <h2 className="sre-card-title" style={{ color: 'var(--status-crit)' }}>
            <span>🚨 Predictive cascade alerts</span>
          </h2>
          <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
            Actionable early warnings from multi-task cascade predictor before service outage
          </span>
        </div>
        <span className="sre-badge sre-badge-crit">
          {alerts.length} active alarm{alerts.length > 1 ? 's' : ''}
        </span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
        {alerts.map((alert) => {
          const isSelected = selectedAlertId === alert.id;
          const isCritical = alert.severity === 'CRITICAL';
          const isHigh = alert.severity === 'HIGH';
          const alertColor = isCritical
            ? 'var(--status-crit)'
            : isHigh
            ? 'var(--status-warn)'
            : 'var(--accent-brand)';

          return (
            <div
              key={alert.id}
              onClick={() => onSelectAlert(alert)}
              style={{
                background: isSelected ? 'rgba(30, 41, 59, 0.95)' : 'var(--bg-surface)',
                border: `1px solid ${isSelected ? alertColor : 'var(--border-card)'}`,
                borderLeft: `4px solid ${alertColor}`,
                borderRadius: '6px',
                padding: '12px 16px',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              {/* Top Headline: Location & Countdown Lead Time */}
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'baseline',
                  marginBottom: '8px',
                  flexWrap: 'wrap',
                  gap: '8px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span className={`sre-badge ${isCritical ? 'sre-badge-crit' : 'sre-badge-warn'}`}>
                    {alert.severity}
                  </span>
                  <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>
                    Predicted failure at <code style={{ color: alertColor, fontSize: '13px' }}>{alert.predictedFailureLocation}</code>
                  </span>
                </div>

                {/* Prominent Lead Time Readout */}
                <div
                  style={{
                    fontFamily: 'JetBrains Mono, monospace',
                    fontSize: '14px',
                    fontWeight: 700,
                    color: alert.estimatedLeadTimeSeconds < 60 ? 'var(--status-crit)' : 'var(--status-warn)',
                    background: 'var(--bg-canvas)',
                    padding: '2px 8px',
                    borderRadius: '4px',
                    border: '1px solid var(--border-quiet)',
                  }}
                >
                  ⏱️ ~{alert.estimatedLeadTimeSeconds.toFixed(0)}s lead time
                </div>
              </div>

              {/* Supporting Details Strip (Quieter Secondary Metrics) */}
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
                  gap: '8px',
                  fontSize: '11.5px',
                  color: 'var(--text-secondary)',
                  background: 'var(--bg-canvas)',
                  padding: '8px 12px',
                  borderRadius: '4px',
                  marginBottom: '8px',
                }}
              >
                <div>
                  Cascade probability:{' '}
                  <strong style={{ color: 'var(--text-primary)', fontFamily: 'monospace' }}>
                    {(alert.cascadeProbability * 100).toFixed(1)}%
                  </strong>
                </div>
                <div>
                  Conformal confidence:{' '}
                  <strong style={{ color: 'var(--text-primary)' }}>{alert.confidence}</strong>
                </div>
                <div>
                  Detected at:{' '}
                  <strong style={{ color: 'var(--text-primary)', fontFamily: 'monospace' }}>
                    {new Date(alert.timestamp).toLocaleTimeString()}
                  </strong>
                </div>
              </div>

              {/* Propagation Path & CTA */}
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  fontSize: '11.5px',
                }}
              >
                <div style={{ color: 'var(--text-secondary)' }}>
                  <span style={{ color: 'var(--text-tertiary)' }}>Propagation path: </span>
                  <code>{alert.affectedDependency}</code>
                </div>

                <div style={{ color: 'var(--accent-brand)', fontWeight: 500, cursor: 'pointer' }}>
                  {isSelected ? '▲ Hide incident brief' : '▼ Click for diagnosis & mitigation brief'}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
