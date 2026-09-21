import React from 'react';

export default function SystemOverview({ overview, connectionState, onRefresh }) {
  const isHealthy = overview.overallHealth === 'HEALTHY';
  const isCritical = overview.overallHealth === 'CRITICAL';

  let healthColor = '#10b981';
  if (isCritical) healthColor = '#ef4444';
  else if (!isHealthy) healthColor = '#f59e0b';

  return (
    <div>
      {/* Top Connection & Status Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#9ca3af' }}>SYSTEM OVERVIEW</span>
          <span className={`sre-badge ${connectionState === 'connected' ? 'sre-badge-ok' : 'sre-badge-crit'}`}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: connectionState === 'connected' ? '#10b981' : '#ef4444' }}></span>
            {connectionState === 'connected' ? 'API CONNECTED' : 'BACKEND OFFLINE'}
          </span>
          <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
            Model Engine: <strong>{overview.modelStatus}</strong>
          </span>
        </div>

        <button onClick={onRefresh} className="sre-btn" title="Refresh telemetry from backend">
          <span>↻</span> Refresh Telemetry
        </button>
      </div>

      {/* Overview Metric Pills */}
      <div className="overview-grid">
        <div className="overview-pill" style={{ borderLeft: `3px solid ${healthColor}` }}>
          <div className="overview-pill-label">Overall System Health</div>
          <div className="overview-pill-val" style={{ color: healthColor }}>
            {overview.overallHealth}
          </div>
          <div className="overview-pill-sub">Status: {overview.systemStatus}</div>
        </div>

        <div className="overview-pill">
          <div className="overview-pill-label">Boundary Health Drift ε(t)</div>
          <div className="overview-pill-val">
            {overview.boundaryHealthScore.toFixed(1)}%
          </div>
          <div className="overview-pill-sub">Threshold: {overview.boundaryThreshold.toFixed(1)}%</div>
        </div>

        <div className="overview-pill">
          <div className="overview-pill-label">Cascade Probability</div>
          <div className="overview-pill-val" style={{ color: overview.cascadeProbability !== null ? (overview.cascadeProbability > 0.6 ? '#ef4444' : '#f59e0b') : '#94a3b8' }}>
            {overview.cascadeProbability !== null ? `${(overview.cascadeProbability * 100).toFixed(1)}%` : 'Awaiting'}
          </div>
          <div className="overview-pill-sub">
            {overview.cascadeProbability !== null ? 'RGCN multi-task inference' : 'Awaiting model telemetry'}
          </div>
        </div>

        <div className="overview-pill">
          <div className="overview-pill-label">Current Severity</div>
          <div className="overview-pill-val" style={{ color: healthColor, fontSize: '1.2rem' }}>
            {overview.severity}
          </div>
          <div className="overview-pill-sub">Impact Level Assessment</div>
        </div>

        <div className="overview-pill">
          <div className="overview-pill-label">Active Alerts</div>
          <div className="overview-pill-val" style={{ color: overview.activeAlertsCount > 0 ? '#ef4444' : '#10b981' }}>
            {overview.activeAlertsCount}
          </div>
          <div className="overview-pill-sub">Predictive Boundary Alarms</div>
        </div>

        <div className="overview-pill">
          <div className="overview-pill-label">Last Telemetry Update</div>
          <div className="overview-pill-val" style={{ fontSize: '0.95rem', fontWeight: 600 }}>
            {new Date(overview.lastTelemetryUpdate).toLocaleTimeString()}
          </div>
          <div className="overview-pill-sub">eBPF polling active</div>
        </div>
      </div>
    </div>
  );
}
