import React, { useState, useEffect } from 'react';
import { fetchDashboardData, applyCircuitBreakerAction, triggerChaosSimulation } from './api/adapter';
import { config } from './config';
import './index.css';

import SystemOverview from './components/SystemOverview';
import DependencyGraph from './components/DependencyGraph';
import BoundaryHealth from './components/BoundaryHealth';
import PredictiveAlerts from './components/PredictiveAlerts';
import IncidentDetail from './components/IncidentDetail';
import ServiceHealthTable from './components/ServiceHealthTable';
import CircuitBreakerStatus from './components/CircuitBreakerStatus';
import ChaosControls from './components/ChaosControls';

export default function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [activeFault, setActiveFault] = useState(null);

  const loadData = async () => {
    try {
      const dashboard = await fetchDashboardData();
      if (dashboard.connectionState === 'connected') {
        setData(dashboard);
      } else if (dashboard.lastKnownData) {
        setData({
          ...dashboard.lastKnownData,
          connectionState: 'offline',
          errorMessage: dashboard.errorMessage,
        });
      } else {
        setData(dashboard);
      }
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, config.pollIntervalMs);
    return () => clearInterval(interval);
  }, []);

  const handleApplyMitigation = async (action, throttleRate, reason) => {
    await applyCircuitBreakerAction(action, throttleRate, reason);
    loadData();
  };

  const handleTriggerChaos = async (fault, level) => {
    setActiveFault(fault === 'clear' ? null : fault);
    await triggerChaosSimulation(fault, level);
    loadData();
  };

  const handleSelectAlert = (alert) => {
    setSelectedAlert(selectedAlert?.id === alert.id ? null : alert);
  };

  if (loading && !data) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', backgroundColor: '#090d16', color: '#9ca3af' }}>
        <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>✈️</div>
        <div style={{ fontSize: '1.1rem', fontWeight: 600, color: '#f3f4f6' }}>Connecting to BACCP Airline Telemetry Stream...</div>
        <div style={{ fontSize: '0.8rem', color: '#64748b', marginTop: '0.5rem' }}>Polling backend API at <code>{config.apiUrl}</code></div>
      </div>
    );
  }

  const isOffline = data?.connectionState === 'offline';
  const overview = data?.systemOverview || {
    overallHealth: 'UNKNOWN',
    boundaryHealthScore: 0,
    boundaryThreshold: 45.0,
    cascadeProbability: null,
    severity: 'LOW',
    activeAlertsCount: 0,
    lastTelemetryUpdate: new Date().toISOString(),
    systemStatus: 'INITIALIZING',
    modelStatus: 'Awaiting model telemetry',
  };

  return (
    <div style={{ minHeight: '100vh', backgroundColor: 'var(--bg-dark)', color: 'var(--text-main)', display: 'flex', flexDirection: 'column' }}>
      {/* SRE Navigation Header */}
      <header style={{ background: 'rgba(17, 24, 39, 0.95)', borderBottom: '1px solid var(--border)', padding: '0.8rem 2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', position: 'sticky', top: 0, zIndex: 50 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span style={{ background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)', color: '#fff', fontWeight: 700, padding: '0.3rem 0.6rem', borderRadius: '6px', fontSize: '0.85rem' }}>
            BACCP
          </span>
          <div>
            <h1 style={{ fontSize: '1rem', fontWeight: 600, margin: 0, color: '#f3f4f6' }}>
              Airline IT Boundary-Aware Cascade Predictor
            </h1>
            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', margin: 0 }}>
              Mission-Critical Cloud Service Health Monitoring & Automated Circuit Breaker Console
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1.2rem', fontSize: '0.8rem' }}>
          <div>
            System: <strong style={{ color: overview.overallHealth === 'CRITICAL' ? '#ef4444' : (overview.overallHealth === 'DEGRADED' ? '#f59e0b' : '#10b981') }}>
              {overview.overallHealth}
            </strong>
          </div>
          <div>
            Gateway: <strong style={{ color: '#c4b5fd' }}>
              {data?.circuitBreaker?.state || 'CLOSED'} ({Math.round((data?.circuitBreaker?.throttleRate || 0) * 100)}%)
            </strong>
          </div>
          <div>
            API: <code>{config.apiUrl}</code>
          </div>
        </div>
      </header>

      {/* Offline Alert Banner */}
      {isOffline && (
        <div style={{ background: 'rgba(239, 68, 68, 0.15)', borderBottom: '1px solid rgba(239, 68, 68, 0.4)', padding: '0.6rem 2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.82rem', color: '#fca5a5' }}>
          <div>
            ⚠️ <strong>Backend Connection Unavailable:</strong> Displaying last known cached telemetry from {new Date(overview.lastTelemetryUpdate).toLocaleTimeString()}.
          </div>
          <button onClick={loadData} className="sre-btn" style={{ padding: '0.2rem 0.6rem', fontSize: '0.75rem' }}>
            Retry Connection
          </button>
        </div>
      )}

      {/* Main Container */}
      <main style={{ maxWidth: '1600px', margin: '0 auto', padding: '1.5rem 2rem', width: '100%', flex: 1 }}>
        {/* Section 1: System Overview */}
        <SystemOverview
          overview={overview}
          connectionState={data?.connectionState || 'offline'}
          onRefresh={loadData}
        />

        {/* Selected Incident Detail (Section 6) */}
        {selectedAlert && (
          <IncidentDetail
            incident={selectedAlert}
            onClose={() => setSelectedAlert(null)}
          />
        )}

        {/* Two-Column Layout */}
        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
          {/* Left Column: Dependency Graph & Chaos Controls */}
          <div>
            {/* Section 2: Dependency Graph */}
            <DependencyGraph
              graph={data?.dependencyGraph}
              activeFault={activeFault}
            />

            {/* Section 7: Circuit Breaker Status */}
            <CircuitBreakerStatus
              circuitBreaker={data?.circuitBreaker}
              onApplyMitigation={handleApplyMitigation}
            />

            {/* Testbed Chaos Fault Injection Playground */}
            <ChaosControls onTriggerChaos={handleTriggerChaos} />
          </div>

          {/* Right Column: Boundary Health & Predictive Alerts */}
          <div>
            {/* Section 3: Boundary Health */}
            <BoundaryHealth boundaryHealth={data?.boundaryHealth} />

            {/* Section 4: Predictive Alerts */}
            <PredictiveAlerts
              alerts={data?.alerts}
              onSelectAlert={handleSelectAlert}
              selectedAlertId={selectedAlert?.id}
            />

            {/* AWS Cloud Observability Status */}
            <div className="sre-card">
              <div className="sre-card-header">
                <span className="sre-card-title">
                  <span>☁️ AWS Cloud Observability Infrastructure</span>
                </span>
                <span className="sre-badge sre-badge-cloud">
                  {(data?.cloudStatus?.mode || 'mock').toUpperCase()} MODE
                </span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.6rem', fontSize: '0.75rem' }}>
                <div style={{ background: '#0f172a', padding: '0.6rem', borderRadius: '6px', border: '1px solid #1e293b' }}>
                  <div style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>Amazon CloudWatch</div>
                  <div>Namespace: <code>{data?.cloudStatus?.cloudwatch?.namespace || 'BACCP/AirlineHealth'}</code></div>
                  <div style={{ color: '#10b981', marginTop: '0.2rem' }}>● Ingesting ε(t) & P(cascade)</div>
                </div>
                <div style={{ background: '#0f172a', padding: '0.6rem', borderRadius: '6px', border: '1px solid #1e293b' }}>
                  <div style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>AWS X-Ray Tracing</div>
                  <div>Daemon: <code>{data?.cloudStatus?.xray?.daemon_address || '127.0.0.1:2000'}</code></div>
                  <div style={{ color: '#10b981', marginTop: '0.2rem' }}>● Cross-Generation Subsegments</div>
                </div>
                <div style={{ background: '#0f172a', padding: '0.6rem', borderRadius: '6px', border: '1px solid #1e293b' }}>
                  <div style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>Amazon SageMaker</div>
                  <div>Endpoint: <code>{data?.cloudStatus?.sagemaker?.endpoint_name || 'baccp-cascade-predictor'}</code></div>
                  <div style={{ color: '#10b981', marginTop: '0.2rem' }}>● Multi-Task Head Online</div>
                </div>
                <div style={{ background: '#0f172a', padding: '0.6rem', borderRadius: '6px', border: '1px solid #1e293b' }}>
                  <div style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>AWS Lambda & SNS</div>
                  <div>Function: <code>baccp-circuit-breaker-mitigator</code></div>
                  <div style={{ color: '#10b981', marginTop: '0.2rem' }}>● Automated Mitigation Active</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Section 5: Service Health Matrix (Full Width) */}
        <ServiceHealthTable services={data?.serviceHealth} />
      </main>
    </div>
  );
}
