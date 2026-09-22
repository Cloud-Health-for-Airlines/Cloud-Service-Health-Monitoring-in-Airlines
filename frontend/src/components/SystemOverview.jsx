import React from 'react';

export default function SystemOverview({ overview }) {
  const isHealthy = overview.overallHealth === 'HEALTHY';
  const isCritical = overview.overallHealth === 'CRITICAL';
  const isDegraded = overview.overallHealth === 'DEGRADED';

  const cascadeProb = overview.cascadeProbability;
  const isProbHigh = cascadeProb !== null && cascadeProb > 0.5;
  const isProbCritical = cascadeProb !== null && cascadeProb > 0.8;

  const drift = overview.boundaryHealthScore ?? 0;
  const threshold = overview.boundaryThreshold ?? 45.0;
  const isDriftHigh = drift >= threshold;
  const isDriftCritical = drift >= 70.0;

  return (
    <div style={{ marginBottom: 'var(--space-4)' }}>
      {/* Differentiated KPI Readout Strip (2 Heroes + 2 Secondary Stats) */}
      <div className="overview-grid">
        {/* Hero KPI 1: Cascade Probability */}
        <div
          className="metric-card-hero"
          style={{
            borderLeft: isProbCritical
              ? '3px solid var(--status-crit)'
              : isProbHigh
              ? '3px solid var(--status-warn)'
              : '3px solid var(--status-ok)',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span className="metric-label">Cascade probability</span>
            <span
              className={`sre-badge ${
                isProbCritical
                  ? 'sre-badge-crit'
                  : isProbHigh
                  ? 'sre-badge-warn'
                  : 'sre-badge-ok'
              }`}
            >
              {cascadeProb !== null
                ? cascadeProb > 0.6
                  ? 'Elevated risk'
                  : 'Nominal'
                : 'Calibrating'}
            </span>
          </div>

          <div
            className="metric-hero-val"
            style={{
              color: isProbCritical
                ? 'var(--status-crit)'
                : isProbHigh
                ? 'var(--status-warn)'
                : 'var(--status-ok)',
            }}
          >
            {cascadeProb !== null ? `${(cascadeProb * 100).toFixed(1)}%` : '—'}
          </div>

          <div className="metric-sub">
            {cascadeProb !== null
              ? 'Multi-task RGCN model · 90% conformal coverage'
              : 'Awaiting model telemetry feed'}
          </div>
        </div>

        {/* Hero KPI 2: Boundary Sync Drift ε(t) */}
        <div
          className="metric-card-hero"
          style={{
            borderLeft: isDriftCritical
              ? '3px solid var(--status-crit)'
              : isDriftHigh
              ? '3px solid var(--status-warn)'
              : '3px solid var(--accent-brand)',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span className="metric-label">Boundary sync drift ε(t)</span>
            <span
              className={`sre-badge ${
                isDriftCritical
                  ? 'sre-badge-crit'
                  : isDriftHigh
                  ? 'sre-badge-warn'
                  : 'sre-badge-ok'
              }`}
            >
              {isDriftHigh ? 'Diverged' : 'Synchronized'}
            </span>
          </div>

          <div
            className="metric-hero-val"
            style={{
              color: isDriftCritical
                ? 'var(--status-crit)'
                : isDriftHigh
                ? 'var(--status-warn)'
                : 'var(--text-primary)',
            }}
          >
            {drift.toFixed(1)}%
          </div>

          <div className="metric-sub">
            Warning threshold: {threshold.toFixed(1)}% | Critical: 70.0%
          </div>
        </div>

        {/* Secondary Metric 1: System Severity & Envelope */}
        <div className="metric-card-secondary">
          <div className="metric-label">Assessed impact severity</div>
          <div
            className="metric-secondary-val"
            style={{
              color: isCritical
                ? 'var(--status-crit)'
                : isDegraded
                ? 'var(--status-warn)'
                : 'var(--status-ok)',
            }}
          >
            {overview.severity || 'LOW'}
          </div>
          <div className="metric-sub">
            Status: {overview.systemStatus || 'Nominal'}
          </div>
        </div>

        {/* Secondary Metric 2: Live Telemetry Stream */}
        <div className="metric-card-secondary">
          <div className="metric-label">Telemetry observation</div>
          <div className="metric-secondary-val" style={{ color: 'var(--text-primary)' }}>
            {overview.lastTelemetryUpdate
              ? new Date(overview.lastTelemetryUpdate).toLocaleTimeString()
              : 'Live'}
          </div>
          <div className="metric-sub">
            Zero-instrumentation eBPF stream active
          </div>
        </div>
      </div>
    </div>
  );
}
