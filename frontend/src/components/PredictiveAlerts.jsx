import React from 'react';

export default function PredictiveAlerts({ alerts, onSelectAlert, selectedAlertId }) {
  if (!alerts || alerts.length === 0) {
    return (
      <div className="sre-card">
        <div className="sre-card-header">
          <span className="sre-card-title">
            <span>🚨 Predictive Cascade Alerts</span>
          </span>
          <span className="sre-badge sre-badge-ok">NOMINAL ENVELOPE</span>
        </div>
        <div style={{ padding: '1.5rem', textAlign: 'center', color: '#9ca3af', fontSize: '0.85rem' }}>
          <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>✅</div>
          <div>No active cascade alerts. Boundary state is operating within nominal statistical thresholds.</div>
          <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.4rem' }}>
            Awaiting model telemetry or chaos fault injection on testbed.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="sre-card">
      <div className="sre-card-header">
        <span className="sre-card-title">
          <span>🚨 Predictive Cascade Alerts</span>
          <span style={{ fontSize: '0.75rem', color: '#9ca3af', fontWeight: 400 }}>(Click an alert to view Incident Detail)</span>
        </span>
        <span className="sre-badge sre-badge-crit">{alerts.length} ACTIVE ALARM</span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
        {alerts.map((alert) => {
          const isSelected = selectedAlertId === alert.id;
          const isCritical = alert.severity === 'CRITICAL';
          const isHigh = alert.severity === 'HIGH';
          const color = isCritical ? '#ef4444' : (isHigh ? '#f59e0b' : '#3b82f6');

          return (
            <div
              key={alert.id}
              onClick={() => onSelectAlert(alert)}
              style={{
                background: isSelected ? 'rgba(30, 41, 59, 0.8)' : '#0f172a',
                border: `1px solid ${isSelected ? color : '#1e293b'}`,
                borderLeft: `4px solid ${color}`,
                borderRadius: '6px',
                padding: '0.9rem 1.1rem',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <span className={`sre-badge ${isCritical ? 'sre-badge-crit' : 'sre-badge-warn'}`}>
                    {alert.severity}
                  </span>
                  <strong style={{ fontSize: '0.88rem' }}>Predicted Failure at: <code>{alert.predictedFailureLocation}</code></strong>
                </div>

                <div style={{ fontSize: '0.82rem', fontWeight: 700, color: alert.estimatedLeadTimeSeconds < 60 ? '#ef4444' : '#f59e0b' }}>
                  ⏱️ Lead Time: ~{alert.estimatedLeadTimeSeconds.toFixed(0)}s
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.8rem', fontSize: '0.78rem', color: '#9ca3af', margin: '0.6rem 0' }}>
                <div>
                  Cascade Probability: <strong style={{ color: '#f3f4f6' }}>{(alert.cascadeProbability * 100).toFixed(1)}%</strong>
                </div>
                <div>
                  Conformal Confidence: <strong style={{ color: '#f3f4f6' }}>{alert.confidence}</strong>
                </div>
                <div>
                  Timestamp: <strong style={{ color: '#f3f4f6' }}>{new Date(alert.timestamp).toLocaleTimeString()}</strong>
                </div>
              </div>

              <div style={{ fontSize: '0.78rem', color: '#cbd5e1' }}>
                <strong>Affected Dependency:</strong> {alert.affectedDependency}
              </div>

              <div style={{ fontSize: '0.72rem', color: '#38bdf8', marginTop: '0.4rem', textAlign: 'right' }}>
                {isSelected ? '▲ Viewing Incident Details' : '▶ Click for full incident brief & recommended mitigation'}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
