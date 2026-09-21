import React from 'react';

export default function CloudStatus({ cloud }) {
  const mode = cloud?.mode ?? 'mock';
  const region = cloud?.region ?? 'us-east-1';

  return (
    <div style={{ background: '#111827', border: '1px solid #374151', borderRadius: '10px', padding: '1.25rem', marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', borderBottom: '1px solid #374151', paddingBottom: '0.5rem' }}>
        <span style={{ fontWeight: 600 }}>AWS Cloud Observability Stack</span>
        <span style={{ padding: '0.2rem 0.5rem', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600, background: 'rgba(6, 182, 212, 0.2)', color: '#67e8f9' }}>
          {mode.toUpperCase()} MODE ({region})
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.6rem' }}>
        <div style={{ background: '#1e293b', padding: '0.6rem', borderRadius: '6px', border: '1px solid #334155', fontSize: '0.75rem' }}>
          <div style={{ fontWeight: 600, color: '#06b6d4', marginBottom: '0.2rem' }}>Amazon CloudWatch</div>
          <div>Namespace: {cloud?.cloudwatch?.namespace ?? 'BACCP/AirlineHealth'}</div>
          <div style={{ color: '#10b981', marginTop: '0.2rem' }}>● Metrics Stream Active</div>
        </div>

        <div style={{ background: '#1e293b', padding: '0.6rem', borderRadius: '6px', border: '1px solid #334155', fontSize: '0.75rem' }}>
          <div style={{ fontWeight: 600, color: '#06b6d4', marginBottom: '0.2rem' }}>AWS X-Ray</div>
          <div>Tracing Daemon: {cloud?.xray?.daemon_address ?? '127.0.0.1:2000'}</div>
          <div style={{ color: '#10b981', marginTop: '0.2rem' }}>● Subsegments Emitting</div>
        </div>

        <div style={{ background: '#1e293b', padding: '0.6rem', borderRadius: '6px', border: '1px solid #334155', fontSize: '0.75rem' }}>
          <div style={{ fontWeight: 600, color: '#06b6d4', marginBottom: '0.2rem' }}>Amazon SageMaker</div>
          <div>Endpoint: {cloud?.sagemaker?.endpoint_name ?? 'baccp-cascade-predictor'}</div>
          <div style={{ color: '#10b981', marginTop: '0.2rem' }}>● Cascade Model Online</div>
        </div>

        <div style={{ background: '#1e293b', padding: '0.6rem', borderRadius: '6px', border: '1px solid #334155', fontSize: '0.75rem' }}>
          <div style={{ fontWeight: 600, color: '#06b6d4', marginBottom: '0.2rem' }}>AWS Lambda & SNS</div>
          <div>Breaker Function & Topic</div>
          <div style={{ color: '#10b981', marginTop: '0.2rem' }}>● Automated Alert Dispatch</div>
        </div>
      </div>
    </div>
  );
}
