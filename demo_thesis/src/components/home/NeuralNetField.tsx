import { useEffect, useRef } from 'react';

/**
 * NeuralNetField
 * ─────────────────────────────────────────────────────────────────────────
 * Subtle canvas-based neural mesh that sits behind the Home hero. Chosen
 * over Three.js (overkill, would balloon the Home chunk) and over particle
 * libraries (visually generic). Hand-rolled so we control every line:
 *
 *  • Deterministic seeded node layout (no per-mount churn).
 *  • Nearest-neighbor mesh (each node links to its closest few peers, so
 *    the topology reads as a "graph", not a random spaghetti scatter).
 *  • Three depth layers — back nodes are smaller and dimmer, front nodes
 *    larger and brighter — giving the field implicit Z-depth without
 *    real 3D.
 *  • Slow Brownian drift with soft viewport bounce → the field breathes
 *    without ever crossing into "screensaver" territory.
 *  • Traveling pulses along edges → reads as neural activity, not decor.
 *  • DPR-aware, resize-aware, prefers-reduced-motion aware, raf-cancelling
 *    on unmount.
 *
 * Performance footprint: a single canvas, ~60 nodes, ~110 edges, ≤ 6 pulses.
 * That's ~180 primitives drawn per frame and rAF runs only while mounted.
 * No allocations inside the loop beyond pulse spawns.
 */

const PALETTE = {
  edgeBase: 'rgb(123, 156, 255)',
  nodeBase: 'rgb(170, 190, 230)',
  pulseCore: 'rgba(150, 180, 255, 0.65)',
  pulseHalo: 'rgba(150, 180, 255, 0)',
};

// Mulberry32 — deterministic PRNG so the field looks "designed".
function mulberry32(seed: number) {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Node {
  x: number;
  y: number;
  vx: number;
  vy: number;
  layer: 0 | 1 | 2; // 0 = back, 2 = front
  phase: number;    // for breathing
}

interface Edge {
  a: number;
  b: number;
  baseDist: number;
}

interface Pulse {
  edge: number;
  t: number;       // 0..1 progression
  speed: number;
}

function buildField(width: number, height: number) {
  const rand = mulberry32(0x42424242);
  // Density scales with area, capped so we never overrun.
  const area = width * height;
  const target = Math.max(36, Math.min(72, Math.round(area / 22000)));

  const nodes: Node[] = [];
  // Poisson-ish placement: reject too-close candidates so the mesh looks
  // even rather than clumpy.
  const minDist = Math.sqrt(area / (target * 1.6));
  let attempts = 0;
  while (nodes.length < target && attempts < target * 60) {
    attempts++;
    const cand = {
      x: rand() * width,
      y: rand() * height,
      vx: (rand() - 0.5) * 0.08,
      vy: (rand() - 0.5) * 0.08,
      layer: (Math.floor(rand() * 3) as 0 | 1 | 2),
      phase: rand() * Math.PI * 2,
    };
    let ok = true;
    for (const n of nodes) {
      const dx = n.x - cand.x;
      const dy = n.y - cand.y;
      if (dx * dx + dy * dy < minDist * minDist) { ok = false; break; }
    }
    if (ok) nodes.push(cand);
  }

  // Build edges: each node → its 3 nearest neighbors. Deduplicate pairs.
  const edges: Edge[] = [];
  const seen = new Set<string>();
  for (let i = 0; i < nodes.length; i++) {
    const dists: { j: number; d: number }[] = [];
    for (let j = 0; j < nodes.length; j++) {
      if (i === j) continue;
      const dx = nodes[i].x - nodes[j].x;
      const dy = nodes[i].y - nodes[j].y;
      dists.push({ j, d: dx * dx + dy * dy });
    }
    dists.sort((a, b) => a.d - b.d);
    for (let k = 0; k < 3 && k < dists.length; k++) {
      const { j } = dists[k];
      const key = i < j ? `${i}-${j}` : `${j}-${i}`;
      if (seen.has(key)) continue;
      seen.add(key);
      edges.push({ a: Math.min(i, j), b: Math.max(i, j), baseDist: Math.sqrt(dists[k].d) });
    }
  }

  return { nodes, edges };
}

export function NeuralNetField({ className = '' }: { className?: string }) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { alpha: true });
    if (!ctx) return;

    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    let dpr = Math.min(window.devicePixelRatio || 1, 2);
    let cssW = canvas.clientWidth || 1;
    let cssH = canvas.clientHeight || 1;
    let field = buildField(cssW, cssH);
    const pulses: Pulse[] = [];
    const rand = mulberry32(0xABCDEF01); // separate stream for pulses

    function applyDpr() {
      canvas!.width = Math.round(cssW * dpr);
      canvas!.height = Math.round(cssH * dpr);
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function resize() {
      const nw = canvas!.clientWidth || 1;
      const nh = canvas!.clientHeight || 1;
      if (Math.abs(nw - cssW) < 1 && Math.abs(nh - cssH) < 1) return;
      cssW = nw;
      cssH = nh;
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      applyDpr();
      // Rebuild only on substantial size change so we don't churn the layout.
      field = buildField(cssW, cssH);
      pulses.length = 0;
    }

    applyDpr();

    let raf = 0;
    let lastSpawn = 0;

    function frame(t: number) {
      ctx!.clearRect(0, 0, cssW, cssH);

      const { nodes, edges } = field;

      // Drift — very slow, with soft viewport bounce.
      if (!reducedMotion) {
        for (const n of nodes) {
          n.x += n.vx;
          n.y += n.vy;
          if (n.x < 0)   { n.x = 0;    n.vx = Math.abs(n.vx); }
          if (n.x > cssW){ n.x = cssW; n.vx = -Math.abs(n.vx); }
          if (n.y < 0)   { n.y = 0;    n.vy = Math.abs(n.vy); }
          if (n.y > cssH){ n.y = cssH; n.vy = -Math.abs(n.vy); }
        }
      }

      // Edges
      ctx!.lineCap = 'round';
      for (const e of edges) {
        const a = nodes[e.a];
        const b = nodes[e.b];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const d = Math.sqrt(dx * dx + dy * dy);
        // Fade edges that stretch too far past their initial length —
        // keeps the mesh visually local even as nodes drift.
        const stretch = d / e.baseDist;
        const alpha = Math.max(0, 0.085 - (stretch - 1.0) * 0.07);
        if (alpha <= 0.005) continue;
        // Depth weighting — edges between deeper nodes are dimmer.
        const depth = (nodes[e.a].layer + nodes[e.b].layer) / 4; // 0..1
        const finalAlpha = alpha * (0.55 + depth * 0.55);
        ctx!.strokeStyle = `${PALETTE.edgeBase.slice(0, -1)} / ${finalAlpha.toFixed(3)})`;
        // Convert "rgb(...)" to "rgb(... / a)" — replace last paren via regex
        ctx!.strokeStyle = PALETTE.edgeBase.replace('rgb(', 'rgba(').replace(')', `, ${finalAlpha.toFixed(3)})`);
        ctx!.lineWidth = 0.55 + depth * 0.25;
        ctx!.beginPath();
        ctx!.moveTo(a.x, a.y);
        ctx!.lineTo(b.x, b.y);
        ctx!.stroke();
      }

      // Nodes — front layer bigger and brighter
      const breath = reducedMotion ? 0 : Math.sin(t * 0.0008);
      for (const n of nodes) {
        const layerFactor = 0.55 + n.layer * 0.30;            // 0.55, 0.85, 1.15
        const r = (0.9 + n.layer * 0.55) * layerFactor;
        const localPulse = reducedMotion ? 1 : (Math.sin(t * 0.0012 + n.phase) * 0.18 + 1);
        const alpha = (0.18 + n.layer * 0.10 + breath * 0.04) * localPulse;
        ctx!.fillStyle = PALETTE.nodeBase.replace('rgb(', 'rgba(').replace(')', `, ${alpha.toFixed(3)})`);
        ctx!.beginPath();
        ctx!.arc(n.x, n.y, r, 0, Math.PI * 2);
        ctx!.fill();
      }

      // Pulses — spawn occasionally
      if (!reducedMotion) {
        if (t - lastSpawn > 850 && pulses.length < 5 && rand() < 0.45) {
          lastSpawn = t;
          const ei = Math.floor(rand() * edges.length);
          pulses.push({ edge: ei, t: 0, speed: 0.004 + rand() * 0.004 });
        }
        for (let i = pulses.length - 1; i >= 0; i--) {
          const p = pulses[i];
          p.t += p.speed;
          if (p.t >= 1) { pulses.splice(i, 1); continue; }
          const e = edges[p.edge];
          if (!e) { pulses.splice(i, 1); continue; }
          const a = nodes[e.a];
          const b = nodes[e.b];
          const x = a.x + (b.x - a.x) * p.t;
          const y = a.y + (b.y - a.y) * p.t;
          // Soft halo
          const grad = ctx!.createRadialGradient(x, y, 0, x, y, 14);
          grad.addColorStop(0, PALETTE.pulseCore);
          grad.addColorStop(1, PALETTE.pulseHalo);
          ctx!.fillStyle = grad;
          ctx!.beginPath();
          ctx!.arc(x, y, 14, 0, Math.PI * 2);
          ctx!.fill();
          // Bright core dot
          ctx!.fillStyle = PALETTE.pulseCore;
          ctx!.beginPath();
          ctx!.arc(x, y, 1.4, 0, Math.PI * 2);
          ctx!.fill();
        }
      }

      raf = requestAnimationFrame(frame);
    }

    raf = requestAnimationFrame(frame);

    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);

  return (
    <canvas
      ref={ref}
      className={`pointer-events-none absolute inset-0 h-full w-full ${className}`}
      aria-hidden
    />
  );
}
