import React, { useState } from 'react';

export default function CircuitBreakerControl({ breaker, onApplyMitigation }) {
  const [throttleVal, setThrottleVal] = useState(0);

  const state = breaker?.state ?? 'CLOSED';
  const throttleRate = breaker?.throttle_rate ?? 0.0;
  const history = breaker?.recent_actions ?? [];

  const handleApply = () => {
    const rate = throttleVal / 100.0;
    const action = rate > 0.9 ? 'OPEN' : (rate > 0.0 ? 'THROTTLED' : 'CLOSED');
    onApplyMitigation(action, rate, `Operator manual override: ${throttleVal}%`);
  };

  return (
    <div style={{ background: '#111827', border: '1px solid #374151', borderRadius: '10px', padding: '1.25rem', marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', borderBottom: '1px solid #374151', paddingBottom: '0.5rem' }}>
        <span style={{ fontWeight: 600 }}>Boundary Circuit Breaker & Mitigation</span>
        <span style={{ padding: '0.2rem 0.5rem', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600, background: state === 'OPEN' ? '#ef444420' : (state === 'THROTTLED' ? '#f59e0b20' : '#10b98120'), color: state === 'OPEN' ? '#fca5a5' : (state === 'THROTTLED' ? '#fde68a' : '#6ee7b7') }}>
          STATE: {state} ({Math.round(throttleRate * 100)}%)
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.8rem 0', borderBottom: '1px solid #374151' }}>
        <div>
          <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>Manual Throttle Override</div>
          <div style={{ fontSize: '0.75rem', color: '#9ca3af' }}>Dynamic rate limit for boundary-gateway node</div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
          <input
            type="range"
            min="0"
            max="100"
            value={throttleVal}
            onChange={(e) => setThrottleVal(parseInt(e.target.value))}
            style={{ width: '130px' }}
          />
          <span style={{ fontSize: '0.85rem', fontWeight: 600, width: '40px' }}>{throttleVal}%</span>
          <button
            onClick={handleApply}
            style={{ backgroundColor: '#2563eb', color: '#fff', border: '1px solid #3b82f6', padding: '0.45rem 0.8rem', borderRadius: '6px', fontSize: '0.8rem', cursor: 'pointer' }}
          >
            Apply
          </button>
        </div>
      </div>

      <div style={{ marginTop: '1rem' }}>
        <div style={{ fontSize: '0.82rem', fontWeight: 600, color: '#9ca3af', marginBottom: '0.5rem' }}>Mitigation Action History:</div>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.78rem' }}>
          <thead>
            <tr style={{ color: '#9ca3af', borderBottom: '1px solid #374151', textAlign: 'left' }}>
              <th style={{ padding: '0.4rem' }}>Time</th>
              <th style={{ padding: '0.4rem' }}>Action</th>
              <th style={{ padding: '0.4rem' }}>Rate</th>
              <th style={{ padding: '0.4rem' }}>Reason</th>
              <th style={{ padding: '0.4rem' }}>Source</th>
            </tr>
          </thead>
          <tbody>
            {history.slice(-5).reverse().map((item, idx) => (
              <tr key={idx} style={{ borderBottom: '1px solid rgba(55,65,81,0.3)' }}>
                <td style={{ padding: '0.4rem' }}>{new Date(item.timestamp * 1000).toLocaleTimeString()}</td>
                <td style={{ padding: '0.4rem', fontWeight: 600 }}>{item.action}</td>
                <td style={{ padding: '0.4rem' }}>{Math.round(item.throttle_rate * 100)}%</td>
                <td style={{ padding: '0.4rem', color: '#cbd5e1' }}>{item.reason}</td>
                <td style={{ padding: '0.4rem', color: '#9ca3af' }}>{item.invoked_by}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
