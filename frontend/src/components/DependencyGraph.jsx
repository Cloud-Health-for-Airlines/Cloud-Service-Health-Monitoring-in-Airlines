import React, { useState } from 'react';

export default function DependencyGraph({ graph, activeFault }) {
  const [selectedNode, setSelectedNode] = useState(null);

  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];

  // Identify nodes
  const legacyNode = nodes.find((n) => n.node_type === 'legacy') || {
    id: 'legacy-core',
    node_type: 'legacy',
    port: 9090,
  };
  const gatewayNode = nodes.find((n) => n.node_type === 'boundary-gateway') || {
    id: 'boundary-gateway',
    node_type: 'boundary-gateway',
    port: 8084,
  };
  const cloudNodes = nodes.filter((n) => n.node_type === 'cloud-native');

  const isFaultActive = Boolean(activeFault);

  return (
    <div className="panel-hero">
      <div className="sre-card-header">
        <div>
          <h2 className="sre-card-title">
            <span>🌐 Generation-typed dependency graph</span>
          </h2>
          <span style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>
            Zero-instrumentation topology discovered via kernel-level eBPF probes
          </span>
        </div>

        <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
          <span className="sre-badge sre-badge-cloud">Cloud-native</span>
          <span className="sre-badge sre-badge-gateway">Boundary gateway</span>
          <span className="sre-badge sre-badge-legacy">Legacy mainframe</span>
        </div>
      </div>

      <div
        style={{
          background: 'var(--bg-canvas)',
          borderRadius: '8px',
          border: '1px solid var(--border-quiet)',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <svg viewBox="0 0 880 420" style={{ width: '100%', height: '390px', display: 'block' }}>
          <defs>
            <marker
              id="arrow-cyan"
              viewBox="0 0 10 10"
              refX="26"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#38bdf8" />
            </marker>
            <marker
              id="arrow-purple"
              viewBox="0 0 10 10"
              refX="28"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path
                d="M 0 0 L 10 5 L 0 10 z"
                fill={isFaultActive ? 'var(--status-crit)' : '#818cf8'}
              />
            </marker>
          </defs>

          {/* Cloud Nodes to Gateway Edges */}
          <line
            x1="180"
            y1="75"
            x2="440"
            y2="210"
            stroke="#38bdf8"
            strokeWidth="2"
            strokeDasharray="5,4"
            opacity="0.75"
            markerEnd="url(#arrow-cyan)"
          />
          <line
            x1="180"
            y1="210"
            x2="440"
            y2="210"
            stroke="#38bdf8"
            strokeWidth="2"
            strokeDasharray="5,4"
            opacity="0.75"
            markerEnd="url(#arrow-cyan)"
          />
          <line
            x1="180"
            y1="345"
            x2="440"
            y2="210"
            stroke="#38bdf8"
            strokeWidth="2"
            strokeDasharray="5,4"
            opacity="0.75"
            markerEnd="url(#arrow-cyan)"
          />

          {/* Gateway to Legacy Core Edge (The Critical Boundary Crossing) */}
          <line
            x1="440"
            y1="210"
            x2="720"
            y2="210"
            stroke={isFaultActive ? 'var(--status-crit)' : '#818cf8'}
            strokeWidth={isFaultActive ? 5 : 3}
            markerEnd="url(#arrow-purple)"
          />

          {/* Edge Metadata Labels (Monospace, readable) */}
          <text x="285" y="130" fill="var(--text-tertiary)" fontSize="10.5" fontFamily="monospace" textAnchor="middle">
            HTTP:8080 · 1.4k obs · 18ms
          </text>
          <text x="300" y="200" fill="var(--text-tertiary)" fontSize="10.5" fontFamily="monospace" textAnchor="middle">
            HTTP:8080 · 890 obs · 22ms
          </text>
          <text x="285" y="295" fill="var(--text-tertiary)" fontSize="10.5" fontFamily="monospace" textAnchor="middle">
            HTTP:8080 · 1.1k obs · 15ms
          </text>

          {/* Boundary Edge Callout */}
          <g transform="translate(580, 195)">
            <rect
              x="-110"
              y="-14"
              width="220"
              height="24"
              rx="4"
              fill="var(--bg-surface)"
              stroke={isFaultActive ? 'var(--status-crit)' : 'var(--border-card)'}
            />
            <text
              x="0"
              y="2"
              fill={isFaultActive ? 'var(--status-crit)' : '#c7c3e8'}
              fontSize="11"
              fontFamily="monospace"
              fontWeight="600"
              textAnchor="middle"
            >
              {isFaultActive
                ? `⚠️ FAULT: ${activeFault} (TCP:9090)`
                : 'TCP:9090 · 3.4k obs · 68ms'}
            </text>
          </g>

          {/* Node 1: Reservations (Cloud-Native) */}
          <g
            transform="translate(180, 75)"
            style={{ cursor: 'pointer' }}
            onClick={() => setSelectedNode(cloudNodes[0] || { id: 'reservations', node_type: 'cloud-native', port: 8081 })}
          >
            <circle r="34" fill="#141c28" stroke="#38bdf8" strokeWidth="2" />
            <text y="-4" fill="var(--text-primary)" fontSize="11.5" fontWeight="600" textAnchor="middle">
              reservations
            </text>
            <text y="14" fill="#94a3b8" fontSize="9.5" textAnchor="middle">
              cloud-native
            </text>
          </g>

          {/* Node 2: Crew (Cloud-Native) */}
          <g
            transform="translate(180, 210)"
            style={{ cursor: 'pointer' }}
            onClick={() => setSelectedNode(cloudNodes[1] || { id: 'crew', node_type: 'cloud-native', port: 8082 })}
          >
            <circle r="34" fill="#141c28" stroke="#38bdf8" strokeWidth="2" />
            <text y="-4" fill="var(--text-primary)" fontSize="11.5" fontWeight="600" textAnchor="middle">
              crew
            </text>
            <text y="14" fill="#94a3b8" fontSize="9.5" textAnchor="middle">
              cloud-native
            </text>
          </g>

          {/* Node 3: Baggage (Cloud-Native) */}
          <g
            transform="translate(180, 345)"
            style={{ cursor: 'pointer' }}
            onClick={() => setSelectedNode(cloudNodes[2] || { id: 'baggage', node_type: 'cloud-native', port: 8083 })}
          >
            <circle r="34" fill="#141c28" stroke="#38bdf8" strokeWidth="2" />
            <text y="-4" fill="var(--text-primary)" fontSize="11.5" fontWeight="600" textAnchor="middle">
              baggage
            </text>
            <text y="14" fill="#94a3b8" fontSize="9.5" textAnchor="middle">
              cloud-native
            </text>
          </g>

          {/* Node 4: Boundary Gateway (Boundary-Gateway) */}
          <g
            transform="translate(440, 210)"
            style={{ cursor: 'pointer' }}
            onClick={() => setSelectedNode(gatewayNode)}
          >
            <rect
              x="-54"
              y="-54"
              width="108"
              height="108"
              rx="12"
              fill="#1e1e2e"
              stroke={isFaultActive ? 'var(--status-crit)' : '#818cf8'}
              strokeWidth="2.5"
            />
            <text y="-12" fill="var(--text-primary)" fontSize="12.5" fontWeight="600" textAnchor="middle">
              boundary
            </text>
            <text y="4" fill="var(--text-primary)" fontSize="12.5" fontWeight="600" textAnchor="middle">
              gateway
            </text>
            <text y="22" fill="#cbd5e1" fontSize="9.5" textAnchor="middle">
              boundary-gateway
            </text>
            <text y="38" fill="#94a3b8" fontSize="8.5" fontFamily="monospace" textAnchor="middle">
              HTTP→TCP Adapter
            </text>
          </g>

          {/* Node 5: Legacy Core (Legacy Mainframe) */}
          <g
            transform="translate(720, 210)"
            style={{ cursor: 'pointer' }}
            onClick={() => setSelectedNode(legacyNode)}
          >
            <polygon
              points="0,-46 45,-23 45,23 0,46 -45,23 -45,-23"
              fill="#221c17"
              stroke="#b45309"
              strokeWidth="2.5"
            />
            <text y="-6" fill="var(--text-primary)" fontSize="12.5" fontWeight="600" textAnchor="middle">
              legacy-core
            </text>
            <text y="12" fill="#d4c5b9" fontSize="9.5" textAnchor="middle">
              mainframe-cics
            </text>
            <text y="27" fill="#a8a29e" fontSize="8.5" fontFamily="monospace" textAnchor="middle">
              TCP:9090
            </text>
          </g>
        </svg>

        {/* Selected Node Details Drawer */}
        {selectedNode && (
          <div
            style={{
              position: 'absolute',
              bottom: '10px',
              left: '12px',
              right: '12px',
              background: 'rgba(15, 23, 42, 0.95)',
              border: '1px solid var(--border-card)',
              borderRadius: '6px',
              padding: '8px 14px',
              fontSize: '12px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              backdropFilter: 'blur(6px)',
            }}
          >
            <div style={{ display: 'flex', gap: '14px', alignItems: 'center' }}>
              <div>
                <span style={{ color: 'var(--text-secondary)' }}>Selected node: </span>
                <code style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{selectedNode.id}</code>
              </div>
              <div>
                <span style={{ color: 'var(--text-secondary)' }}>Type: </span>
                <span
                  className={`sre-badge ${
                    selectedNode.node_type === 'legacy'
                      ? 'sre-badge-legacy'
                      : selectedNode.node_type === 'boundary-gateway'
                      ? 'sre-badge-gateway'
                      : 'sre-badge-cloud'
                  }`}
                >
                  {selectedNode.node_type}
                </span>
              </div>
              <div>
                <span style={{ color: 'var(--text-secondary)' }}>Ingress port: </span>
                <code style={{ color: 'var(--text-primary)' }}>{selectedNode.port || '8080'}</code>
              </div>
            </div>
            <button
              onClick={() => setSelectedNode(null)}
              className="sre-btn"
              style={{ padding: '2px 8px', fontSize: '11px' }}
              title="Close inspection drawer"
            >
              ✕ Close
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
