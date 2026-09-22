import React, { useState, useEffect, useRef, useCallback } from 'react';

// Technical glyphs inspired by Antigravity's floating icon field
const GLYPHS = [
  { id: 'node', label: 'eBPF Node', icon: '☍', x: 12, y: 22, size: 44, speed: 0.8 },
  { id: 'gateway', label: 'Boundary Gateway', icon: '🛡️', x: 82, y: 18, size: 48, speed: 0.6 },
  { id: 'cloud', label: 'Cloud Native', icon: '☁️', x: 22, y: 68, size: 40, speed: 1.1 },
  { id: 'mainframe', label: 'Mainframe CICS', icon: '⚙️', x: 88, y: 72, size: 46, speed: 0.7 },
  { id: 'pulse', label: 'Sync Drift ε(t)', icon: '⚡', x: 74, y: 44, size: 42, speed: 0.9 },
  { id: 'terminal', label: 'TCP:9090', icon: '›_', x: 10, y: 48, size: 40, speed: 1.2 },
];

export default function HeroSection({ overview, onExploreGraph, onExploreBoundary }) {
  const heroRef = useRef(null);
  const [cursorPos, setCursorPos] = useState({ x: -1000, y: -1000 });
  const [cursorActive, setCursorActive] = useState(false);
  const [cursorHoverText, setCursorHoverText] = useState(null);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  // Magnetic button offsets
  const [primaryMagnetic, setPrimaryMagnetic] = useState({ x: 0, y: 0 });
  const [secondaryMagnetic, setSecondaryMagnetic] = useState({ x: 0, y: 0 });
  const primaryBtnRef = useRef(null);
  const secondaryBtnRef = useRef(null);

  // Smooth cursor follow with lerp
  const followerPos = useRef({ x: -100, y: -100 });
  const followerElemRef = useRef(null);
  const rafId = useRef(null);
  const isVisible = useRef(true);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefersReducedMotion(mediaQuery.matches);
    const handler = (e) => setPrefersReducedMotion(e.matches);
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  // RAF loop for smooth interpolated cursor following
  const updateCursorFollower = useCallback(() => {
    if (!prefersReducedMotion && followerElemRef.current && isVisible.current) {
      const targetX = cursorPos.x;
      const targetY = cursorPos.y;
      followerPos.current.x += (targetX - followerPos.current.x) * 0.18;
      followerPos.current.y += (targetY - followerPos.current.y) * 0.18;

      followerElemRef.current.style.transform = `translate3d(${followerPos.current.x}px, ${followerPos.current.y}px, 0)`;
    }
    rafId.current = requestAnimationFrame(updateCursorFollower);
  }, [cursorPos, prefersReducedMotion]);

  useEffect(() => {
    rafId.current = requestAnimationFrame(updateCursorFollower);
    return () => cancelAnimationFrame(rafId.current);
  }, [updateCursorFollower]);

  // Pause when tab hidden or hero scrolled out
  useEffect(() => {
    const handleVisibility = () => {
      isVisible.current = document.visibilityState === 'visible';
    };
    document.addEventListener('visibilitychange', handleVisibility);

    const observer = new IntersectionObserver(([entry]) => {
      isVisible.current = entry.isIntersecting;
      if (!entry.isIntersecting) {
        setCursorActive(false);
      }
    });

    if (heroRef.current) observer.observe(heroRef.current);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibility);
      observer.disconnect();
    };
  }, []);

  const handleMouseMove = (e) => {
    setCursorPos({ x: e.clientX, y: e.clientY });
    if (!cursorActive) setCursorActive(true);

    // Primary button magnetic attraction
    if (primaryBtnRef.current) {
      const rect = primaryBtnRef.current.getBoundingClientRect();
      const btnCenterX = rect.left + rect.width / 2;
      const btnCenterY = rect.top + rect.height / 2;
      const dist = Math.hypot(e.clientX - btnCenterX, e.clientY - btnCenterY);
      if (dist < 90) {
        setPrimaryMagnetic({
          x: (e.clientX - btnCenterX) * 0.22,
          y: (e.clientY - btnCenterY) * 0.22,
        });
      } else {
        setPrimaryMagnetic({ x: 0, y: 0 });
      }
    }

    // Secondary button magnetic attraction
    if (secondaryBtnRef.current) {
      const rect = secondaryBtnRef.current.getBoundingClientRect();
      const btnCenterX = rect.left + rect.width / 2;
      const btnCenterY = rect.top + rect.height / 2;
      const dist = Math.hypot(e.clientX - btnCenterX, e.clientY - btnCenterY);
      if (dist < 80) {
        setSecondaryMagnetic({
          x: (e.clientX - btnCenterX) * 0.18,
          y: (e.clientY - btnCenterY) * 0.18,
        });
      } else {
        setSecondaryMagnetic({ x: 0, y: 0 });
      }
    }
  };

  const handleMouseLeave = () => {
    setCursorActive(false);
    setCursorHoverText(null);
    setPrimaryMagnetic({ x: 0, y: 0 });
    setSecondaryMagnetic({ x: 0, y: 0 });
  };

  // Determine health tone
  const isCritical = overview?.overallHealth === 'CRITICAL';
  const isDegraded = overview?.overallHealth === 'DEGRADED';
  const statusColor = isCritical ? '#ef4444' : isDegraded ? '#f59e0b' : '#3279f9';
  const statusLabel = isCritical
    ? 'Critical cascade failure risk across hybrid boundary'
    : isDegraded
    ? 'State divergence warning on boundary gateway'
    : 'All hybrid airline boundary channels operating nominally';

  return (
    <section
      ref={heroRef}
      className="hero-entry-zone"
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{
        minHeight: '48vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        textAlign: 'center',
        padding: '60px 24px 48px',
        background: 'radial-gradient(ellipse 70% 60% at 50% 20%, rgba(50, 121, 249, 0.09), transparent 75%), var(--bg-canvas)',
        borderBottom: '1px solid var(--border-quiet)',
      }}
    >
      {/* Antigravity Custom Cursor (Strictly in Hero Zone) */}
      {!prefersReducedMotion && (
        <div
          ref={followerElemRef}
          className="antigravity-cursor-follower"
          style={{ opacity: cursorActive ? 1 : 0 }}
        >
          {cursorHoverText ? (
            <div className="antigravity-cursor-pill">
              <span>{cursorHoverText}</span>
            </div>
          ) : (
            <div className="antigravity-cursor-dot" />
          )}
        </div>
      )}

      {/* Floating Reactive Technical Glyphs (Antigravity Icon Field) */}
      {!prefersReducedMotion && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            pointerEvents: 'none',
            overflow: 'hidden',
          }}
        >
          {GLYPHS.map((glyph) => {
            // Subtle displacement away from cursor
            let offsetX = 0;
            let offsetY = 0;
            if (heroRef.current && cursorPos.x > -500) {
              const rect = heroRef.current.getBoundingClientRect();
              const glyphPixelX = rect.left + (glyph.x / 100) * rect.width;
              const glyphPixelY = rect.top + (glyph.y / 100) * rect.height;
              const dx = cursorPos.x - glyphPixelX;
              const dy = cursorPos.y - glyphPixelY;
              const dist = Math.hypot(dx, dy);
              if (dist < 180 && dist > 0) {
                const force = (180 - dist) / 180;
                offsetX = -(dx / dist) * force * 16;
                offsetY = -(dy / dist) * force * 16;
              }
            }

            return (
              <div
                key={glyph.id}
                style={{
                  position: 'absolute',
                  left: `${glyph.x}%`,
                  top: `${glyph.y}%`,
                  width: `${glyph.size}px`,
                  height: `${glyph.size}px`,
                  borderRadius: '50%',
                  background: 'rgba(183, 191, 217, 0.05)',
                  border: '1px solid rgba(255, 255, 255, 0.09)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '18px',
                  color: 'rgba(255, 255, 255, 0.65)',
                  backdropFilter: 'blur(4px)',
                  transform: `translate3d(${offsetX}px, ${offsetY}px, 0)`,
                  transition: 'transform 0.25s cubic-bezier(0.16, 1, 0.3, 1)',
                  boxShadow: '0 4px 16px rgba(0, 0, 0, 0.25)',
                }}
                title={glyph.label}
              >
                <span>{glyph.icon}</span>
              </div>
            );
          })}
        </div>
      )}

      {/* Hero Content Container */}
      <div style={{ maxWidth: '960px', zIndex: 2, position: 'relative' }}>
        {/* Antigravity Product Eyebrow Pill */}
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            background: 'rgba(50, 121, 249, 0.1)',
            border: '1px solid rgba(50, 121, 249, 0.3)',
            padding: '5px 14px',
            borderRadius: '999px',
            fontSize: '12px',
            fontWeight: 500,
            color: '#93c5fd',
            marginBottom: '20px',
            backdropFilter: 'blur(8px)',
          }}
        >
          <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: statusColor }}></span>
          <span>Airline IT Cascade Intelligence · BACCP SRE Core</span>
        </div>

        {/* Large Confident Two-Line Headline (Antigravity Style: 5rem desktop, line-height 1.1) */}
        <h1
          style={{
            fontSize: 'clamp(2.5rem, 5.5vw, 4.4rem)',
            fontWeight: 450,
            lineHeight: 1.1,
            letterSpacing: '-0.035em',
            margin: '0 0 16px',
            color: 'var(--text-primary)',
          }}
        >
          <span>Predicting cascade liftoff.</span>
          <span
            style={{
              display: 'block',
              color: 'var(--text-secondary)',
              fontSize: 'clamp(1.75rem, 3.8vw, 3.2rem)',
              fontWeight: 400,
              marginTop: '4px',
            }}
          >
            Before failures cross the hybrid boundary.
          </span>
        </h1>

        {/* Real-time Subheadline Status Line */}
        <p
          style={{
            fontSize: '14.5px',
            color: 'var(--text-secondary)',
            maxWidth: '680px',
            margin: '0 auto 28px',
            lineHeight: 1.5,
          }}
        >
          Continuous RGCN multi-task inference, zero-instrumentation eBPF state topology, and automated RL circuit breaker mitigation for mission-critical flight operations.
        </p>

        {/* Live Boundary Health Pill in Hero Copy */}
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            background: 'rgba(18, 19, 23, 0.8)',
            border: '1px solid var(--border-card)',
            padding: '6px 16px',
            borderRadius: '999px',
            fontSize: '12.5px',
            color: 'var(--text-primary)',
            marginBottom: '32px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
          }}
        >
          <span
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: statusColor,
              boxShadow: `0 0 8px ${statusColor}`,
            }}
          />
          <span>{statusLabel}</span>
          <span style={{ color: 'var(--text-tertiary)' }}>·</span>
          <span style={{ fontFamily: 'monospace', color: '#93c5fd' }}>
            ε(t): {overview?.boundaryHealthScore?.toFixed(1) || '12.4'}%
          </span>
        </div>

        {/* Hero Magnetic CTA Buttons */}
        <div
          style={{
            display: 'flex',
            gap: '14px',
            justifyContent: 'center',
            alignItems: 'center',
            flexWrap: 'wrap',
          }}
        >
          {/* Primary CTA with Magnetic Offset & Morphing Cursor Affordance */}
          <button
            ref={primaryBtnRef}
            onClick={onExploreGraph}
            onMouseEnter={() => setCursorHoverText('→ Explore topology')}
            onMouseLeave={() => setCursorHoverText(null)}
            className="btn-antigravity btn-antigravity-primary"
            style={{
              transform: `translate3d(${primaryMagnetic.x}px, ${primaryMagnetic.y}px, 0)`,
            }}
          >
            <span>⚡ Explore Dependency Topology</span>
          </button>

          {/* Secondary CTA with Magnetic Offset */}
          <button
            ref={secondaryBtnRef}
            onClick={onExploreBoundary}
            onMouseEnter={() => setCursorHoverText('🛡️ View drift metrics')}
            onMouseLeave={() => setCursorHoverText(null)}
            className="btn-antigravity btn-antigravity-secondary"
            style={{
              transform: `translate3d(${secondaryMagnetic.x}px, ${secondaryMagnetic.y}px, 0)`,
            }}
          >
            <span>🛡️ Boundary Drift & Mitigations</span>
          </button>
        </div>
      </div>
    </section>
  );
}
