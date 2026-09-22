import React, { useState } from 'react';

export default function AwsObservabilityPanel({ cloudStatus }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const mode = cloudStatus?.mode || 'mock';
  const region = cloudStatus?.region || 'us-east-1';

  return (
    <div
      className="panel-subtle"
      style={{
        marginTop: 'var(--space-3)',
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
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '14px' }}>☁️</span>
          <div>
            <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
              AWS cloud observability infrastructure
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>
              Backend ingestion pipelines, daemon telemetry, and automated mitigation lambdas
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span className="sre-badge sre-badge-cloud">
            {mode} mode ({region})
          </span>
          <button
            className="sre-btn sre-btn-secondary"
            style={{ padding: '2px 8px', fontSize: '11px' }}
            onClick={(e) => {
              e.stopPropagation();
              setIsExpanded(!isExpanded);
            }}
          >
            {isExpanded ? '▲ Hide infrastructure' : '▼ View infrastructure'}
          </button>
        </div>
      </div>

      {isExpanded && (
        <div
          style={{
            marginTop: 'var(--space-3)',
            paddingTop: 'var(--space-3)',
            borderTop: '1px solid var(--border-quiet)',
          }}
        >
          {/* Flattened Grid: No Nested Boxes or Repeated Borders! */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
              gap: '16px',
            }}
          >
            {/* CloudWatch Pipeline */}
            <div style={{ padding: '4px 0' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '3px' }}>
                Amazon CloudWatch
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                Namespace: <code>{cloudStatus?.cloudwatch?.namespace || 'BACCP/AirlineHealth'}</code>
              </div>
              <div style={{ fontSize: '11px', color: 'var(--status-ok)', marginTop: '3px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span>●</span> Ingesting ε(t) & cascade metrics
              </div>
            </div>

            {/* AWS X-Ray */}
            <div style={{ padding: '4px 0' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '3px' }}>
                AWS X-Ray Tracing
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                Daemon: <code>{cloudStatus?.xray?.daemon_address || '127.0.0.1:2000'}</code>
              </div>
              <div style={{ fontSize: '11px', color: 'var(--status-ok)', marginTop: '3px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span>●</span> Cross-generation subsegments active
              </div>
            </div>

            {/* SageMaker Endpoint */}
            <div style={{ padding: '4px 0' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '3px' }}>
                Amazon SageMaker
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                Endpoint: <code>{cloudStatus?.sagemaker?.endpoint_name || 'baccp-cascade-predictor'}</code>
              </div>
              <div style={{ fontSize: '11px', color: 'var(--status-ok)', marginTop: '3px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span>●</span> Multi-task inference head online
              </div>
            </div>

            {/* Lambda & SNS */}
            <div style={{ padding: '4px 0' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '3px' }}>
                AWS Lambda & SNS
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                Function: <code>baccp-circuit-breaker-mitigator</code>
              </div>
              <div style={{ fontSize: '11px', color: 'var(--status-ok)', marginTop: '3px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span>●</span> Automated mitigation dispatch enabled
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
