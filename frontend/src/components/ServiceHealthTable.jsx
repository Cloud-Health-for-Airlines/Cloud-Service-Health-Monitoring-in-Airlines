import React from 'react';

export default function ServiceHealthTable({ services }) {
  if (!services || services.length === 0) return null;

  return (
    <div className="sre-card">
      <div className="sre-card-header">
        <span className="sre-card-title">
          <span>📊 Service Health Matrix</span>
          <span style={{ fontSize: '0.75rem', color: '#9ca3af', fontWeight: 400 }}>
            (Cross-Generational Telemetry Breakdown)
          </span>
        </span>
        <span className="sre-badge sre-badge-cloud">5 SERVICES MONITORED</span>
      </div>

      <table className="sre-table">
        <thead>
          <tr>
            <th>Service Domain</th>
            <th>Technology Generation</th>
            <th>Status</th>
            <th>Observed Latency</th>
            <th>Request Rate</th>
            <th>Error Rate</th>
            <th>Health Score</th>
          </tr>
        </thead>
        <tbody>
          {services.map((svc) => {
            const isLegacy = svc.type === 'legacy';
            const isGateway = svc.type === 'boundary-gateway';

            let badgeClass = 'sre-badge-cloud';
            if (isLegacy) badgeClass = 'sre-badge-legacy';
            else if (isGateway) badgeClass = 'sre-badge-gateway';

            const isNominal = svc.status === 'Nominal';
            const isStalling = svc.status === 'Stalling' || svc.status === 'Isolated';

            return (
              <tr key={svc.id}>
                <td>
                  <strong>{svc.name}</strong>
                  <div style={{ fontSize: '0.7rem', color: '#9ca3af' }}><code>{svc.id}</code></div>
                </td>
                <td>
                  <span className={`sre-badge ${badgeClass}`}>{svc.type}</span>
                </td>
                <td>
                  <span className={`sre-badge ${isNominal ? 'sre-badge-ok' : (isStalling ? 'sre-badge-crit' : 'sre-badge-warn')}`}>
                    {svc.status}
                  </span>
                </td>
                <td style={{ fontFamily: 'monospace', color: parseFloat(svc.latency) > 500 ? '#ef4444' : '#e2e8f0' }}>
                  {svc.latency}
                </td>
                <td style={{ fontFamily: 'monospace' }}>
                  {svc.requestRate}
                </td>
                <td style={{ fontFamily: 'monospace', color: parseFloat(svc.errorRate) > 0 ? '#ef4444' : '#10b981' }}>
                  {svc.errorRate}
                </td>
                <td>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <div style={{ width: '50px', background: '#1e293b', height: '6px', borderRadius: '3px', overflow: 'hidden' }}>
                      <div
                        style={{
                          height: '100%',
                          width: `${svc.healthScore}%`,
                          backgroundColor: svc.healthScore > 80 ? '#10b981' : (svc.healthScore > 50 ? '#f59e0b' : '#ef4444'),
                        }}
                      />
                    </div>
                    <span style={{ fontSize: '0.75rem', fontWeight: 600 }}>{svc.healthScore}/100</span>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
