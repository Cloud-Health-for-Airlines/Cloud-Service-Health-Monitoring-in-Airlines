import React, { useState } from 'react';

export default function DependencyGraph({ graph, activeFault }) {
  const [selectedNode, setSelectedNode] = useState(null);

  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];

  // Identify nodes
  const legacyNode = nodes.find((n) => n.node_type === 'legacy') || { id: 'legacy-core', node_type: 'legacy', port: 9090 };
  const gatewayNode = nodes.find((n) => n.node_type === 'boundary-gateway') || { id: 'boundary-gateway', node_type: 'boundary-gateway', port: 8084 };
  const cloudNodes = nodes.filter((n) => n.node_type === 'cloud-native');

  const isFaultActive = Boolean(activeFault);

  return (
    <div className="sre-card">
      <div className="sre-card-header">
        <span className="sre-card-title">
          <span>🌐 Generation-Typed Dependency Graph</span>
          <span style={{ fontSize: '0.75rem', color: '#9ca3af', fontWeight: 400 }}>
            (Zero-Instrumentation eBPF Discovered)
          </span>
        </span>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <span className="sre-badge sre-badge-legacy">Legacy (Mainframe)</span>
          <span className="sre-badge sre-badge-gateway">Boundary Gateway</span>
          <span className="sre-badge sre-badge-cloud">Cloud-Native</span>
        </div>
      </div>

      <div style={{ background: '#0a0f1d', borderRadius: '8px', border: '1px solid #1e293b', position: 'relative' }}>
        <svg viewBox="0 0 880 440" style={{ width: '100%', height: '400px' }}>
          <defs>
            <marker id="arrow-blue" viewBox="0 0 10 10" refX="28" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#3b82f6" />
            </marker>
            <marker id="arrow-purple" viewBox="0 0 10 10" refX="28" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill={isFaultActive ? '#ef4444' : '#8b5cf6'} />
            </marker>
          </defs>

          {/* Cloud Nodes to Gateway Edges */}
          <line x1="180" y1="80" x2="440" y2="220" stroke="#3b82f6" strokeWidth="2.5" strokeDasharray="5,4" markerEnd="url(#arrow-blue)" />
          <line x1="180" y1="220" x2="440" y2="220" stroke="#3b82f6" strokeWidth="2.5" strokeDasharray="5,4" markerEnd="url(#arrow-blue)" />
          <line x1="180" y1="360" x2="440" y2="220" stroke="#3b82f6" strokeWidth="2.5" strokeDasharray="5,4" markerEnd="url(#arrow-blue)" />

          {/* Gateway to Legacy Core Edge (The Critical Boundary) */}
          <line
            x1="440"
            y1="220"
            x2="720"
            y2="220"
            stroke={isFaultActive ? '#ef4444' : '#8b5cf6'}
            strokeWidth={isFaultActive ? 5 : 3.5}
            markerEnd="url(#arrow-purple)"
          />

          {/* Edge Metadata Labels */}
          <text x="290" y="135" fill="#94a3b8" fontSize="11" textAnchor="middle">HTTP:8080 (1.4k obs)</text>
          <text x="300" y="210" fill="#94a3b8" fontSize="11" textAnchor="middle">HTTP:8080 (890 obs)</text>
          <text x="290" y="305" fill="#94a3b8" fontSize="11" textAnchor="middle">HTTP:8080 (1.1k obs)</text>
          <text
            x="580"
            y="205"
            fill={isFaultActive ? '#ef4444' : '#c4b5fd'}
            fontSize="12"
            fontWeight="bold"
            textAnchor="middle"
          >
            {isFaultActive ? `⚠️ FAULT: ${activeFault} (TCP:9090)` : 'TCP:9090 (3.4k obs | 68ms)'}
          </text>

          {/* Node 1: Reservations (Cloud-Native) */}
          <g transform="translate(180, 80)" style={{ cursor: 'pointer' }} onClick={() => setSelectedNode(cloudNodes[0] || { id: 'reservations' })}>
            <circle r="36" fill="#1e293b" stroke="#3b82f6" strokeWidth="2.5" />
            <text y="-5" fill="#ffffff" fontSize="12" fontWeight="600" textAnchor="middle">reservations</text>
            <text y="14" fill="#93c5fd" fontSize="10" textAnchor="middle">cloud-native</text>
          </g>

          {/* Node 2: Crew (Cloud-Native) */}
          <g transform="translate(180, 220)" style={{ cursor: 'pointer' }} onClick={() => setSelectedNode(cloudNodes[1] || { id: 'crew' })}>
            <circle r="36" fill="#1e293b" stroke="#3b82f6" strokeWidth="2.5" />
            <text y="-5" fill="#ffffff" fontSize="12" fontWeight="600" textAnchor="middle">crew</text>
            <text y="14" fill="#93c5fd" fontSize="10" textAnchor="middle">cloud-native</text>
          </g>

          {/* Node 3: Baggage (Cloud-Native) */}
          <g transform="translate(180, 360)" style={{ cursor: 'pointer' }} onClick={() => setSelectedNode(cloudNodes[2] || { id: 'baggage' })}>
            <circle r="36" fill="#1e293b" stroke="#3b82f6" strokeWidth="2.5" />
            <text y="-5" fill="#ffffff" fontSize="12" fontWeight="600" textAnchor="middle">baggage</text>
            <text y="14" fill="#93c5fd" fontSize="10" textAnchor="middle">cloud-native</text>
          </g>

          {/* Node 4: Boundary Gateway (Boundary-Gateway) */}
          <g transform="translate(440, 220)" style={{ cursor: 'pointer' }} onClick={() => setSelectedNode(gatewayNode)}>
            <rect
              x="-58"
              y="-58"
              width="116"
              height="116"
              rx="14"
              fill="#1e1b4b"
              stroke={isFaultActive ? '#ef4444' : '#8b5cf6'}
              strokeWidth="3.5"
            />
            <text y="-14" fill="#ffffff" fontSize="13" fontWeight="700" textAnchor="middle">boundary</text>
            <text y="4" fill="#ffffff" fontSize="13" fontWeight="700" textAnchor="middle">gateway</text>
            <text y="24" fill="#c4b5fd" fontSize="10" textAnchor="middle">boundary-gateway</text>
            <text y="40" fill="#a78bfa" fontSize="9" textAnchor="middle">HTTP→TCP Adapter</text>
          </g>

          {/* Node 5: Legacy Core (Legacy Mainframe) */}
          <g transform="translate(720, 220)" style={{ cursor: 'pointer' }} onClick={() => setSelectedNode(legacyNode)}>
            <polygon
              points="0,-50 48,-25 48,25 0,50 -48,25 -48,-25"
              fill="#2d1b0a"
              stroke="#f59e0b"
              strokeWidth="3.5"
            />
            <text y="-8" fill="#ffffff" fontSize="13" fontWeight="700" textAnchor="middle">legacy-core</text>
            <text y="12" fill="#fde68a" fontSize="10" textAnchor="middle">mainframe-cics</text>
            <text y="28" fill="#f59e0b" fontSize="9" textAnchor="middle">TCP:9090</text>
          </g>
        </svg>

        {/* Selected Node Details Drawer */}
        {selectedNode && (
          <div style={{ position: 'absolute', bottom: '10px', left: '10px', background: 'rgba(15, 23, 42, 0.95)', border: '1px solid #374151', borderRadius: '6px', padding: '0.6rem 1rem', fontSize: '0.78rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
            <div>
              <strong>Selected:</strong> <code>{selectedNode.id}</code>
            </div>
            <div>
              <strong>Type:</strong> <span className={`sre-badge ${selectedNode.node_type === 'legacy' ? 'sre-badge-legacy' : (selectedNode.node_type === 'boundary-gateway' ? 'sre-badge-gateway' : 'sre-badge-cloud')}`}>{selectedNode.node_type}</span>
            </div>
            <div>
              <strong>Port:</strong> {selectedNode.port || '8080'}
            </div>
            <button onClick={() => setSelectedNode(null)} style={{ background: 'none', border: 'none', color: '#9ca3af', cursor: 'pointer', fontSize: '1rem' }}>✕</button>
          </div>
        )}
      </div>
    </div>
  );
}
