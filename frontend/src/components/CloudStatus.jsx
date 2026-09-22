import React from 'react';

export default function CloudStatus({ cloud }) {
  const mode = cloud?.mode ?? 'mock';
  const region = cloud?.region ?? 'us-east-1';

  return (
    <div className="panel-subtle">
      <div className="sre-card-header">
        <div>
          <h3 className="sre-card-title">
            <span>☁️ AWS cloud observability stack</span>
          </h3>
          <span style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>
            Cloud telemetry and tracing infrastructure
          </span>
        </div>
        <span className="sre-badge sre-badge-cloud">
          {mode} mode ({region})
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
        <div>
          <div style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '12px', marginBottom: '2px' }}>Amazon CloudWatch</div>
          <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Namespace: <code>{cloud?.cloudwatch?.namespace ?? 'BACCP/AirlineHealth'}</code></div>
          <div style={{ color: 'var(--status-ok)', fontSize: '11px', marginTop: '2px' }}>● Metrics stream active</div>
        </div>

        <div>
          <div style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '12px', marginBottom: '2px' }}>AWS X-Ray</div>
          <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Daemon: <code>{cloud?.xray?.daemon_address ?? '127.0.0.1:2000'}</code></div>
          <div style={{ color: 'var(--status-ok)', fontSize: '11px', marginTop: '2px' }}>● Subsegments emitting</div>
        </div>

        <div>
          <div style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '12px', marginBottom: '2px' }}>Amazon SageMaker</div>
          <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Endpoint: <code>{cloud?.sagemaker?.endpoint_name ?? 'baccp-cascade-predictor'}</code></div>
          <div style={{ color: 'var(--status-ok)', fontSize: '11px', marginTop: '2px' }}>● Cascade model online</div>
        </div>

        <div>
          <div style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '12px', marginBottom: '2px' }}>AWS Lambda & SNS</div>
          <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Function: <code>baccp-circuit-breaker-mitigator</code></div>
          <div style={{ color: 'var(--status-ok)', fontSize: '11px', marginTop: '2px' }}>● Mitigation dispatch active</div>
        </div>
      </div>
    </div>
  );
}
