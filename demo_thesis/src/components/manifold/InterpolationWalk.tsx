import { motion } from 'framer-motion';
import { ProvenanceBadge } from '@/components/pipeline/ProvenanceBadge';
import { PremiumPanel } from '@/components/premium/PremiumPanel';
import { ManifoldSectionHeader } from './ManifoldSectionHeader';
import { KickerRule, MonoNote, LatentOnlyChip } from './atoms';
import {
  interpolationWalk,
  interpolationStats,
  formatMetric,
} from '@/data/neuralManifoldExplorer';
import { DERIVED_PROV, REPLAY_PROV } from '@/lib/provenance';

/**
 * InterpolationWalk
 * ─────────────────────────────────────────────────────────────
 * A semantic trajectory between two decoded embeddings —
 * rendered as a smooth Bézier path on a coordinate plane with
 * five waypoints positioned along it. Each waypoint card shows
 * its t-value, vignette, and top-3 concept stack.
 *
 * Interpolation paths are CLIP-space trajectories, not real
 * neural trajectories.
 */
export function InterpolationWalk() {
  // 5 waypoint screen positions along a gentle Bézier path in the 1000×220 viewBox.
  const waypoints: Array<{ x: number; y: number }> = [
    { x:  80, y: 158 },
    { x: 270, y: 100 },
    { x: 500, y:  80 },
    { x: 730, y: 116 },
    { x: 920, y:  60 },
  ];

  return (
    <motion.section
      id="manifold-walk"
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className="mx-auto max-w-[1280px] scroll-mt-20 px-4 sm:px-6 lg:px-8"
    >
      <ManifoldSectionHeader
        kicker="Manifold walk"
        title="Walking on the decoded perceptual manifold."
        description="Interpolation between two decoded embeddings traverses the CLIP semantic space smoothly. Smoothness and abrupt-transition rate are population aggregates from the V62a semantic reports."
        action={
          <div className="flex items-center gap-2">
            <LatentOnlyChip />
            <ProvenanceBadge provenance={DERIVED_PROV} />
          </div>
        }
      />

      <PremiumPanel className="mt-6 p-5 sm:p-6">
        {/* Trajectory plane */}
        <div className="rounded-2xl border border-white/[0.05] bg-white/[0.014] p-4 sm:p-5">
          <div className="flex items-center justify-between">
            <KickerRule label="Trial A → Trial B" />
            <MonoNote>slerp · t ∈ [0, 1]</MonoNote>
          </div>

          {/* SVG curve with waypoint pins */}
          <TrajectorySvg waypoints={waypoints} />

          {/* Waypoint cards aligned under the SVG */}
          <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-5">
            {interpolationWalk.map((s, i) => (
              <WaypointCard key={s.t} step={s} highlight={i === 0 || i === interpolationWalk.length - 1} />
            ))}
          </div>
        </div>

        {/* Aggregate metrics */}
        <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Stat
            label="Smoothness"
            valVal={interpolationStats.smoothnessValidation.toFixed(7)}
            valS1k={interpolationStats.smoothnessShared1000.toFixed(7)}
            tone="accent"
          />
          <Stat
            label="Abrupt rate"
            valVal={(interpolationStats.abruptValidation * 100).toFixed(2) + '%'}
            valS1k={(interpolationStats.abruptShared1000 * 100).toFixed(2) + '%'}
          />
          <Stat
            label="Mean velocity"
            valVal={formatMetric(interpolationStats.velocityValidation)}
            valS1k={formatMetric(interpolationStats.velocityShared1000)}
          />
          <Stat
            label="Top-concept transitions"
            valVal={String(interpolationStats.transitionsValidation)}
            valS1k={String(interpolationStats.transitionsShared1000)}
          />
        </div>

        <div className="mt-4 flex flex-col gap-2 border-t border-white/[0.05] pt-3 text-[10.5px] text-text-muted sm:flex-row sm:items-center sm:justify-between">
          <p>
            Smoothness is measured in the CLIP text-probe distribution space — adjacent t-steps
            share most of their probe mass. Trajectories are CLIP-space, not real neural paths.
          </p>
          <ProvenanceBadge provenance={REPLAY_PROV} />
        </div>
      </PremiumPanel>
    </motion.section>
  );
}

/* ─── Trajectory ──────────────────────────────────────────── */

function TrajectorySvg({
  waypoints,
}: {
  waypoints: Array<{ x: number; y: number }>;
}) {
  // Smooth cubic-Bézier through the five waypoints.
  const start = waypoints[0];
  const path = waypoints
    .slice(1)
    .map((p, i) => {
      const prev = waypoints[i];
      const c1x = prev.x + (p.x - prev.x) * 0.45;
      const c2x = p.x - (p.x - prev.x) * 0.45;
      return `C ${c1x},${prev.y} ${c2x},${p.y} ${p.x},${p.y}`;
    })
    .join(' ');
  const fullPath = `M ${start.x},${start.y} ${path}`;

  return (
    <svg viewBox="0 0 1000 220" className="mt-3 h-[200px] w-full" role="img" aria-label="Interpolation trajectory">
      <defs>
        <linearGradient id="walk-stroke" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%"   stopColor="rgb(123,156,255)" stopOpacity="0.30" />
          <stop offset="50%"  stopColor="rgb(123,156,255)" stopOpacity="0.95" />
          <stop offset="100%" stopColor="rgb(190,206,238)" stopOpacity="0.55" />
        </linearGradient>
        <linearGradient id="walk-glow" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"  stopColor="rgb(123,156,255)" stopOpacity="0.10" />
          <stop offset="100%" stopColor="rgb(123,156,255)" stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* Hair-thin coordinate grid */}
      <g stroke="rgb(255 255 255 / 0.05)" strokeWidth="0.4">
        {[44, 88, 132, 176].map((y) => (
          <line key={y} x1="0" y1={y} x2="1000" y2={y} />
        ))}
        {[200, 400, 600, 800].map((x) => (
          <line key={x} x1={x} y1="0" x2={x} y2="220" />
        ))}
      </g>

      {/* Soft glow under the curve */}
      <path
        d={`${fullPath} L 920,220 L 80,220 Z`}
        fill="url(#walk-glow)"
        opacity="0.65"
      />

      {/* Phase band annotations — semantic segments between waypoints */}
      {[
        { mid:  175, label: 'animal'     },
        { mid:  385, label: 'outdoor'    },
        { mid:  615, label: 'ambiguous'  },
        { mid:  825, label: 'urban'      },
      ].map((seg) => (
        <text
          key={seg.label}
          x={seg.mid}
          y="205"
          textAnchor="middle"
          fontSize="8.5"
          letterSpacing="0.16em"
          fill="rgb(150,158,172)"
          fillOpacity="0.78"
        >
          {seg.label.toUpperCase()}
        </text>
      ))}

      {/* Main trajectory */}
      <path
        d={fullPath}
        stroke="url(#walk-stroke)"
        strokeWidth="2.4"
        fill="none"
        strokeLinecap="round"
      />

      {/* Waypoint pins — outer ring + bright core + tiny vertical drop to t-label */}
      {waypoints.map((p, i) => {
        const endpoint = i === 0 || i === waypoints.length - 1;
        return (
          <g key={i} transform={`translate(${p.x}, ${p.y})`}>
            {/* tick drop */}
            <line x1="0" y1="0" x2="0" y2="-18" stroke="rgb(123,156,255)" strokeOpacity="0.32" strokeWidth="0.55" />
            <circle r="11" fill="rgb(20,24,32)" stroke="rgb(123,156,255)" strokeOpacity={endpoint ? 0.85 : 0.55} strokeWidth={endpoint ? 1.0 : 0.8} />
            <circle r="4.6" fill="rgb(123,156,255)" fillOpacity={endpoint ? 1.0 : 0.92} />
            {endpoint ? <circle r="1.7" fill="rgb(255,255,255)" fillOpacity="0.9" /> : null}
            <text
              y="-22"
              textAnchor="middle"
              fontSize="9"
              fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
              fill="rgb(180,190,212)"
              letterSpacing="0.06em"
            >
              t = {interpolationWalk[i].t.toFixed(2)}
            </text>
          </g>
        );
      })}

      {/* Endpoint labels */}
      <text x="56"  y="200" fontSize="11" fontWeight="700" letterSpacing="0.12em" fill="rgb(220,228,242)">A</text>
      <text x="936" y="32"  fontSize="11" fontWeight="700" letterSpacing="0.12em" fill="rgb(220,228,242)">B</text>
      <text x="20"  y="212" fontSize="7"  letterSpacing="0.18em" fill="rgb(140,148,168)">TRIAL · START</text>
      <text x="858" y="42"  fontSize="7"  letterSpacing="0.18em" fill="rgb(140,148,168)">TRIAL · END</text>
    </svg>
  );
}

/* ─── Waypoint card ────────────────────────────────────────── */

function WaypointCard({
  step,
  highlight,
}: {
  step: { t: number; topConcepts: [string, string, string]; vignette: string };
  highlight?: boolean;
}) {
  return (
    <div
      className={`rounded-xl border px-3 py-2.5 ${
        highlight
          ? 'border-accent/22 bg-accent/[0.05]'
          : 'border-white/[0.05] bg-white/[0.014]'
      }`}
    >
      <p className="font-mono text-[9.5px] uppercase tracking-[0.14em] text-text-muted">
        t = {step.t.toFixed(2)}
      </p>
      <p className="mt-1 text-[12px] font-semibold tracking-tight text-text-primary">
        {step.vignette}
      </p>
      <ul className="mt-2 space-y-0.5">
        {step.topConcepts.map((c, idx) => (
          <li
            key={c}
            className={`font-mono text-[10.5px] tabular-nums ${
              idx === 0 ? 'text-accent/90' : 'text-text-muted'
            }`}
          >
            {idx === 0 ? '· ' : '  '}
            {c}
          </li>
        ))}
      </ul>
    </div>
  );
}

function Stat({
  label,
  valVal,
  valS1k,
  tone = 'default',
}: {
  label: string;
  valVal: string;
  valS1k: string;
  tone?: 'default' | 'accent';
}) {
  const valueClass = tone === 'accent' ? 'text-accent' : 'text-text-primary';
  return (
    <div className="rounded-xl border border-white/[0.05] bg-white/[0.015] px-4 py-3">
      <p className="premium-kicker">{label}</p>
      <div className="mt-2 flex items-baseline justify-between gap-3">
        <div>
          <p className="font-mono text-[9.5px] uppercase tracking-[0.14em] text-text-muted">val</p>
          <p className={`mt-0.5 font-mono text-[15px] font-semibold tabular-nums leading-none ${valueClass}`}>
            {valVal}
          </p>
        </div>
        <div className="text-right">
          <p className="font-mono text-[9.5px] uppercase tracking-[0.14em] text-text-muted">s1k</p>
          <p className="mt-0.5 font-mono text-[13px] font-semibold tabular-nums leading-none text-text-secondary">
            {valS1k}
          </p>
        </div>
      </div>
    </div>
  );
}
