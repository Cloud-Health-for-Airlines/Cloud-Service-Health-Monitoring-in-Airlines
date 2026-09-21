/**
 * BACCP API Client
 *
 * Connects to the backend REST API on port 8000.
 * Automatically falls back to high-fidelity mock data if the backend is unreachable.
 */

const API_BASE = 'http://localhost:8000/api';

const MOCK_GRAPH = {
  nodes: [
    { id: 'reservations', node_type: 'cloud-native', domain: 'reservations', port: 8081, status: 'nominal' },
    { id: 'crew', node_type: 'cloud-native', domain: 'crew-scheduling', port: 8082, status: 'nominal' },
    { id: 'baggage', node_type: 'cloud-native', domain: 'baggage-handling', port: 8083, status: 'nominal' },
    { id: 'boundary-gateway', node_type: 'boundary-gateway', domain: 'integration', port: 8084, status: 'monitoring' },
    { id: 'legacy-core', node_type: 'legacy', domain: 'mainframe-cics', port: 9090, status: 'uninstrumented' },
  ],
  edges: [
    { source: 'reservations', destination: 'boundary-gateway', protocol: 'tcp', destination_port: 8080, observation_count: 1420, avg_latency_ms: 18.5 },
    { source: 'crew', destination: 'boundary-gateway', protocol: 'tcp', destination_port: 8080, observation_count: 890, avg_latency_ms: 22.1 },
    { source: 'baggage', destination: 'boundary-gateway', protocol: 'tcp', destination_port: 8080, observation_count: 1105, avg_latency_ms: 15.4 },
    { source: 'boundary-gateway', destination: 'legacy-core', protocol: 'tcp', destination_port: 9090, observation_count: 3415, avg_latency_ms: 68.2 },
  ],
};

let mockBreaker = {
  state: 'CLOSED',
  throttle_rate: 0.0,
  target_node: 'boundary-gateway',
  recent_actions: [
    { timestamp: Date.now() / 1000 - 3600, action: 'RESET', throttle_rate: 0.0, reason: 'Initial baseline nominal state', invoked_by: 'system:init' },
  ],
};

let mockActiveFault = null;
let mockFaultLevel = null;

export const apiClient = {
  async fetchGraph() {
    try {
      const res = await fetch(`${API_BASE}/graph`);
      if (res.ok) return await res.json();
    } catch (_) {}
    return MOCK_GRAPH;
  },

  async fetchHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) return await res.json();
    } catch (_) {}
    
    let drift = 14.2;
    if (mockActiveFault === 'network-delay') drift = mockFaultLevel === 'high' ? 88.0 : 55.0;
    if (mockActiveFault === 'connection-drop') drift = mockFaultLevel === 'high' ? 96.0 : 68.0;
    if (mockActiveFault === 'batch-job-stall') drift = mockFaultLevel === 'high' ? 92.0 : 62.0;

    return {
      sync_drift_score: drift,
      threshold: 45.0,
      status: drift >= 70.0 ? 'CRITICAL' : (drift >= 45.0 ? 'DEGRADED' : 'HEALTHY'),
      active_fault: mockActiveFault,
      fault_level: mockFaultLevel,
      formula: 'ε(t) = ||Φ(t) - Ψ(t)|| / ||Φ(t)|| * 100',
      timestamp: new Date().toISOString(),
    };
  },

  async fetchAlerts() {
    try {
      const res = await fetch(`${API_BASE}/alerts`);
      if (res.ok) return await res.json();
    } catch (_) {}

    const isFault = Boolean(mockActiveFault);
    const prob = isFault ? (mockFaultLevel === 'high' ? 0.89 : 0.68) : 0.08;
    const leadTime = isFault ? (mockFaultLevel === 'high' ? 24 : 85) : 320;
    const severity = prob > 0.8 ? 'CRITICAL' : (prob > 0.5 ? 'HIGH' : (prob > 0.3 ? 'MEDIUM' : 'LOW'));

    return {
      cascade_probability: prob,
      estimated_lead_time_seconds: leadTime,
      predicted_root_cause_node: 'boundary-gateway',
      predicted_affected_nodes: prob > 0.5 ? ['boundary-gateway', 'reservations', 'crew', 'baggage'] : ['boundary-gateway'],
      severity,
      conformal_bounds: {
        confidence_level: 0.90,
        lower: Math.max(0.0, prob - 0.08),
        upper: Math.min(1.0, prob + 0.08),
      },
      boundary_sync_drift: isFault ? (mockFaultLevel === 'high' ? 88.0 : 65.0) : 14.2,
      explanation: isFault
        ? `High-risk boundary cascade detected at boundary-gateway (P=${(prob * 100).toFixed(1)}%). Estimated lead time: ${leadTime}s. Active chaos fault: ${mockActiveFault} (${mockFaultLevel}).`
        : 'All boundary and microservice channels operating within normal operating envelopes.',
      model_metadata: {
        engine: 'RGCN-Hetero-CascadePredictor-v1',
        calibration: 'Split-Conformal-Sliding-Window',
        mode: 'standalone-mock',
      },
    };
  },

  async fetchCircuitBreaker() {
    try {
      const res = await fetch(`${API_BASE}/circuit-breaker`);
      if (res.ok) return await res.json();
    } catch (_) {}
    return mockBreaker;
  },

  async applyCircuitBreaker(action, throttleRate, reason) {
    try {
      const res = await fetch(`${API_BASE}/circuit-breaker`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, throttle_rate: throttleRate, reason }),
      });
      if (res.ok) return await res.json();
    } catch (_) {}

    mockBreaker.state = action;
    mockBreaker.throttle_rate = throttleRate;
    mockBreaker.recent_actions.push({
      timestamp: Date.now() / 1000,
      action,
      throttle_rate: throttleRate,
      reason,
      invoked_by: 'dashboard-standalone',
    });
    return { status: 'applied', current_state: mockBreaker };
  },

  async simulateChaos(fault, level) {
    try {
      const res = await fetch(`${API_BASE}/simulate/chaos`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fault, level }),
      });
      if (res.ok) return await res.json();
    } catch (_) {}

    if (fault === 'clear') {
      mockActiveFault = null;
      mockFaultLevel = null;
    } else {
      mockActiveFault = fault;
      mockFaultLevel = level;
    }
    return { message: `Simulated ${fault} (${level})` };
  },

  async fetchCloudStatus() {
    try {
      const res = await fetch(`${API_BASE}/cloud/status`);
      if (res.ok) return await res.json();
    } catch (_) {}
    return {
      mode: 'mock',
      region: 'us-east-1',
      cloudwatch: { namespace: 'BACCP/AirlineHealth', is_live: false, buffered_metrics_count: 24 },
      xray: { daemon_address: '127.0.0.1:2000', is_live: false, recorded_traces_count: 12 },
      sagemaker: { endpoint_name: 'baccp-cascade-predictor', is_live: false },
      sns: { topic_arn: 'arn:aws:sns:us-east-1:123456789012:baccp-cascade-alerts', is_live: false, dispatched_alerts_count: 3 },
      lambda: { function_name: 'baccp-circuit-breaker-mitigator' },
    };
  },
};
