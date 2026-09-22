import React from 'react';

export default function ServiceHealthTable({ services }) {
  if (!services || services.length === 0) return null;

  return (
    <div className="sre-card">
      <div className="sre-card-header">
        <div>
          <h2 className="sre-card-title">
            <span>📊 Service health matrix</span>
          </h2>
          <span style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>
            Cross-generational real-time telemetry breakdown across hybrid airline services
          </span>
        </div>
        <span className="sre-badge sre-badge-cloud">{services.length} services monitored</span>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="sre-table">
          <thead>
            <tr>
              <th>Service domain</th>
              <th>Generation</th>
              <th>Health status</th>
              <th>Observed latency</th>
              <th>Request rate</th>
              <th>Error rate</th>
              <th>Health score</th>
            </tr>
          </thead>
          <tbody>
            {services.map((svc) => {
              const isLegacy = svc.type === 'legacy';
              const isGateway = svc.type === 'boundary-gateway';

              let taxBadge = 'badge-tax-cloud';
              let taxLabel = 'Cloud-native';
              if (isLegacy) {
                taxBadge = 'badge-tax-legacy';
                taxLabel = 'Legacy mainframe';
              } else if (isGateway) {
                taxBadge = 'badge-tax-gateway';
                taxLabel = 'Boundary gateway';
              }

              const isNominal = svc.status === 'Nominal';
              const isCrit = svc.status === 'Stalling' || svc.status === 'Isolated';

              let statusBadge = 'sre-badge-ok';
              if (isCrit) statusBadge = 'sre-badge-crit';
              else if (!isNominal) statusBadge = 'sre-badge-warn';

              const latencyNum = parseFloat(svc.latency);
              const errorNum = parseFloat(svc.errorRate);

              let barColor = 'var(--status-ok)';
              if (svc.healthScore <= 50) barColor = 'var(--status-crit)';
              else if (svc.healthScore <= 80) barColor = 'var(--status-warn)';

              return (
                <tr key={svc.id}>
                  <td>
                    <strong style={{ color: 'var(--text-primary)' }}>{svc.name}</strong>
                    <div style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>
                      <code>{svc.id}</code>
                    </div>
                  </td>
                  <td>
                    <span className={`sre-badge ${taxBadge}`}>{taxLabel}</span>
                  </td>
                  <td>
                    <span className={`sre-badge ${statusBadge}`}>{svc.status}</span>
                  </td>
                  <td
                    style={{
                      fontFamily: 'monospace',
                      fontSize: '12.5px',
                      color: latencyNum > 500 ? 'var(--status-crit)' : 'var(--text-primary)',
                    }}
                  >
                    {svc.latency}
                  </td>
                  <td style={{ fontFamily: 'monospace', fontSize: '12.5px', color: 'var(--text-primary)' }}>
                    {svc.requestRate}
                  </td>
                  <td
                    style={{
                      fontFamily: 'monospace',
                      fontSize: '12.5px',
                      color: errorNum > 0 ? 'var(--status-crit)' : 'var(--status-ok)',
                    }}
                  >
                    {svc.errorRate}
                  </td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div
                        style={{
                          width: '64px',
                          background: 'var(--bg-canvas)',
                          height: '6px',
                          borderRadius: '3px',
                          overflow: 'hidden',
                          border: '1px solid var(--border-quiet)',
                        }}
                      >
                        <div
                          style={{
                            height: '100%',
                            width: `${svc.healthScore}%`,
                            backgroundColor: barColor,
                            transition: 'width 0.3s ease, background-color 0.3s ease',
                          }}
                        />
                      </div>
                      <span
                        style={{
                          fontSize: '12px',
                          fontWeight: 600,
                          fontFamily: 'monospace',
                          color: 'var(--text-primary)',
                        }}
                      >
                        {svc.healthScore}/100
                      </span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
