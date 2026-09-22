import React, { useState, useEffect } from 'react';
import { fetchDashboardData, applyCircuitBreakerAction, triggerChaosSimulation } from './api/adapter';
import { config } from './config';
import './index.css';

import HeroSection from './components/HeroSection';
import SystemOverview from './components/SystemOverview';
import DependencyGraph from './components/DependencyGraph';
import BoundaryHealth from './components/BoundaryHealth';
import PredictiveAlerts from './components/PredictiveAlerts';
import IncidentDetail from './components/IncidentDetail';
import ServiceHealthTable from './components/ServiceHealthTable';
import CircuitBreakerStatus from './components/CircuitBreakerStatus';
import ChaosControls from './components/ChaosControls';
import AwsObservabilityPanel from './components/AwsObservabilityPanel';

export default function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [activeFault, setActiveFault] = useState(null);
  const [isScrolled, setIsScrolled] = useState(false);

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

  // Track scroll position for dynamic Antigravity nav transparency
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 80);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
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

  // Smooth scroll jump helpers
  const scrollToSection = (id) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  if (loading && !data) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', backgroundColor: '#0c0d11', color: '#9aa0a6' }}>
        <div style={{ fontSize: '2.5rem', marginBottom: '1rem', animation: 'pulse 1.5s infinite' }}>✈️</div>
        <div style={{ fontSize: '1.2rem', fontWeight: 500, color: '#ffffff', fontFamily: 'var(--font-sans)' }}>Connecting to BACCP Telemetry Stream...</div>
        <div style={{ fontSize: '0.82rem', color: '#5f6368', marginTop: '0.5rem', fontFamily: 'var(--font-mono)' }}>Polling backend API at <code>{config.apiUrl}</code></div>
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
    <div style={{ minHeight: '100vh', backgroundColor: 'var(--bg-canvas)', color: 'var(--text-primary)', display: 'flex', flexDirection: 'column' }}>
      {/* Antigravity-Style Sticky Navigation Header */}
      <header
        style={{
          background: isScrolled ? 'rgba(18, 19, 23, 0.92)' : 'rgba(12, 13, 17, 0.65)',
          backdropFilter: 'blur(16px)',
          WebkitBackdropFilter: 'blur(16px)',
          borderBottom: isScrolled ? '1px solid var(--border-card)' : '1px solid transparent',
          padding: '0.65rem 2rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          position: 'sticky',
          top: 0,
          zIndex: 50,
          transition: 'all 0.25s var(--ease-snappy)',
        }}
      >
        {/* Left: Brand Mark */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.9rem' }}>
          <div
            style={{
              width: '28px',
              height: '28px',
              borderRadius: '8px',
              background: 'linear-gradient(135deg, #3279f9, #1a73e8)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 2px 10px rgba(50, 121, 249, 0.4)',
            }}
          >
            <span style={{ fontSize: '14px' }}>✈️</span>
          </div>

          <div>
            <div style={{ fontSize: '0.95rem', fontWeight: 600, margin: 0, color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>
              BACCP <span style={{ fontWeight: 400, color: 'var(--text-tertiary)', fontSize: '0.8rem' }}>| Airline IT Cascade Predictor</span>
            </div>
          </div>
        </div>

        {/* Center: Clean Nav Jumps (Hidden on Mobile) */}
        <nav
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid var(--border-quiet)',
            padding: '3px 6px',
            borderRadius: 'var(--shape-corner-rounded)',
          }}
          className="desktop-nav-strip"
        >
          <button
            onClick={() => scrollToSection('overview')}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-secondary)',
              fontSize: '12px',
              fontWeight: 500,
              padding: '4px 12px',
              borderRadius: '999px',
              cursor: 'pointer',
              transition: 'color 0.15s ease',
            }}
            onMouseEnter={(e) => (e.target.style.color = '#ffffff')}
            onMouseLeave={(e) => (e.target.style.color = 'var(--text-secondary)')}
          >
            Overview
          </button>
          <button
            onClick={() => scrollToSection('topology')}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-secondary)',
              fontSize: '12px',
              fontWeight: 500,
              padding: '4px 12px',
              borderRadius: '999px',
              cursor: 'pointer',
              transition: 'color 0.15s ease',
            }}
            onMouseEnter={(e) => (e.target.style.color = '#ffffff')}
            onMouseLeave={(e) => (e.target.style.color = 'var(--text-secondary)')}
          >
            Topology
          </button>
          <button
            onClick={() => scrollToSection('mitigations')}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-secondary)',
              fontSize: '12px',
              fontWeight: 500,
              padding: '4px 12px',
              borderRadius: '999px',
              cursor: 'pointer',
              transition: 'color 0.15s ease',
            }}
            onMouseEnter={(e) => (e.target.style.color = '#ffffff')}
            onMouseLeave={(e) => (e.target.style.color = 'var(--text-secondary)')}
          >
            Mitigations
          </button>
          <button
            onClick={() => scrollToSection('diagnostics')}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-secondary)',
              fontSize: '12px',
              fontWeight: 500,
              padding: '4px 12px',
              borderRadius: '999px',
              cursor: 'pointer',
              transition: 'color 0.15s ease',
            }}
            onMouseEnter={(e) => (e.target.style.color = '#ffffff')}
            onMouseLeave={(e) => (e.target.style.color = 'var(--text-secondary)')}
          >
            Testbed & Cloud
          </button>
        </nav>

        {/* Right: Authoritative Status & Antigravity CTA Button */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
          {/* Authoritative Single Status Summary */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              background: 'rgba(255, 255, 255, 0.03)',
              padding: '4px 10px',
              borderRadius: '999px',
              border: '1px solid var(--border-quiet)',
            }}
          >
            <span
              style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                backgroundColor:
                  overview.overallHealth === 'CRITICAL'
                    ? 'var(--status-crit)'
                    : overview.overallHealth === 'DEGRADED'
                    ? 'var(--status-warn)'
                    : 'var(--status-ok)',
              }}
            />
            <span style={{ fontSize: '11.5px', fontWeight: 600, color: 'var(--text-primary)' }}>
              {overview.overallHealth}
            </span>

            <span style={{ color: 'var(--border-card)', margin: '0 2px' }}>·</span>

            <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
              {overview.activeAlertsCount} alerts
            </span>
          </div>

          {/* Primary Antigravity Pill Action */}
          <button
            onClick={loadData}
            className="btn-antigravity btn-antigravity-primary"
            style={{ padding: '6px 16px', fontSize: '12px' }}
            title="Refresh live telemetry stream"
          >
            <span>↻ Refresh Telemetry</span>
          </button>
        </div>
      </header>

      {/* Offline Alert Banner */}
      {isOffline && (
        <div
          style={{
            background: 'rgba(239, 68, 68, 0.12)',
            borderBottom: '1px solid rgba(239, 68, 68, 0.3)',
            padding: '0.6rem 2rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            fontSize: '0.82rem',
            color: '#fca5a5',
          }}
        >
          <div>
            ⚠️ <strong>Backend connection unavailable:</strong> Displaying last known cached telemetry from {new Date(overview.lastTelemetryUpdate).toLocaleTimeString()}.
          </div>
          <button
            onClick={loadData}
            className="btn-antigravity btn-antigravity-secondary"
            style={{ padding: '4px 12px', fontSize: '11px' }}
          >
            Retry connection
          </button>
        </div>
      )}

      {/* Hero / Entry Moment (Antigravity Style with Magnetic Buttons & Custom Cursor) */}
      <HeroSection
        overview={overview}
        onExploreGraph={() => scrollToSection('topology')}
        onExploreBoundary={() => scrollToSection('mitigations')}
      />

      {/* Main Container — Calmer, Dense Operations SRE Cockpit */}
      <main style={{ maxWidth: '1600px', margin: '0 auto', padding: '2rem 2rem 4rem', width: '100%', flex: 1 }}>
        {/* Section 1: System Overview (Differentiated KPIs) */}
        <div id="overview">
          <SystemOverview overview={overview} />
        </div>

        {/* Section 2: Predictive Alerts & Incident Triage Drawer */}
        <PredictiveAlerts
          alerts={data?.alerts}
          onSelectAlert={handleSelectAlert}
          selectedAlertId={selectedAlert?.id}
        />

        {/* Selected Incident Detail Drawer (If Alert Selected) */}
        {selectedAlert && (
          <IncidentDetail
            incident={selectedAlert}
            onClose={() => setSelectedAlert(null)}
          />
        )}

        {/* Section 3: Hero Stage — Mission-Critical Dependency Topology & Drift Gauge */}
        <div id="topology" className="hero-stage-grid">
          {/* Generation-Typed Dependency Graph */}
          <DependencyGraph
            graph={data?.dependencyGraph}
            activeFault={activeFault}
          />

          {/* Boundary Health & Comparative Latency Meter */}
          <BoundaryHealth boundaryHealth={data?.boundaryHealth} />
        </div>

        {/* Section 4: Boundary Circuit Breaker & Mitigation Controls */}
        <div id="mitigations">
          <CircuitBreakerStatus
            circuitBreaker={data?.circuitBreaker}
            onApplyMitigation={handleApplyMitigation}
          />
        </div>

        {/* Section 5: Service Health Matrix (Full Width) */}
        <ServiceHealthTable services={data?.serviceHealth} />

        {/* Section 6: Diagnostics & Admin Testbed (Collapsible / Progressive Disclosure) */}
        <div id="diagnostics" style={{ marginTop: 'var(--space-6)' }}>
          <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 'var(--space-2)' }}>
            System diagnostics & testing controls:
          </div>

          {/* Testbed Chaos Fault Injection Playground */}
          <ChaosControls onTriggerChaos={handleTriggerChaos} activeFault={activeFault} />

          {/* AWS Cloud Observability Infrastructure (Flattened 4-Column Strip) */}
          <AwsObservabilityPanel cloudStatus={data?.cloudStatus} />
        </div>
      </main>
    </div>
  );
}
