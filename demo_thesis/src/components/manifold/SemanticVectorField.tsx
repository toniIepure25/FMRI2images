import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ProvenanceBadge } from '@/components/pipeline/ProvenanceBadge';
import { PremiumPanel } from '@/components/premium/PremiumPanel';
import { ManifoldSectionHeader } from './ManifoldSectionHeader';
import { KickerRule, MonoNote, SchematicChip } from './atoms';
import {
  semanticErrorAggregate,
  vectorFieldArrows,
} from '@/data/neuralManifoldExplorer';
import { DERIVED_PROV } from '@/lib/provenance';

const HIGHLIGHT_ID = 'd';

/**
 * Per-arrow scientific note rendered in the selected-callout card.
 * Lives here (UI layer) so the data module stays metric-only.
 */
const EXPLANATIONS: Record<string, string> = {
  a: 'Indoor scenes drift toward shared object identity when room context is sparse.',
  b: 'Animal predictions in cluttered scenes pull toward outdoor environmental hubs.',
  c: 'Vehicles pick up street-context features when the road is the dominant signal.',
  d: 'Face errors track toward broader scene-level interpretation — a stable, repeating direction.',
  e: 'Food errors collapse onto generic object identity when colour cues are weak.',
  f: 'Person-in-scene predictions drift toward scene-only embeddings.',
  g: 'Water-region trials are pulled toward land/terrain composition in the predicted direction.',
  h: 'Natural views skew toward urban CLIP regions when built structures appear at the edges.',
};

/**
 * SemanticVectorField
 * ─────────────────────────────────────────────────────────────
 * Scientific vector-field instrument:
 *   • Large schematic CLIP plane with proper axes, gridlines,
 *     and a tagged legend.
 *   • Predicted points are amber, target points blue-white.
 *   • The selected example is rendered with a soft halo + an
 *     annotated leader line + a short scientific note in the
 *     side panel — same grammar as a publication figure.
 *   • Hovering or focusing any row in the corrections list
 *     lifts the matching arrow on the plane.
 */
export function SemanticVectorField() {
  const [activeId, setActiveId] = useState<string>(HIGHLIGHT_ID);
  const active = vectorFieldArrows.find((a) => a.id === activeId) ?? vectorFieldArrows[0];

  return (
    <motion.section
      id="manifold-error-field"
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className="mx-auto max-w-[1280px] scroll-mt-20 px-4 sm:px-6 lg:px-8"
    >
      <ManifoldSectionHeader
        kicker="Semantic error field"
        title="Errors point somewhere — they are not noise."
        description="Each arrow represents the semantic correction direction z_target − z_pred. Repeated alignment across trials reveals that decoder errors are structured in CLIP space, not random."
        action={
          <div className="flex items-center gap-2">
            <SchematicChip />
            <ProvenanceBadge provenance={DERIVED_PROV} />
          </div>
        }
      />

      <PremiumPanel className="mt-6 p-5 sm:p-6">
        <div className="grid gap-5 lg:grid-cols-[minmax(0,2.2fr)_minmax(240px,1fr)]">

          {/* ── Plane ── */}
          <div className="rounded-2xl border border-white/[0.05] bg-white/[0.014] p-4 sm:p-5">
            <div className="flex items-center justify-between">
              <KickerRule label="CLIP semantic plane" />
              <MonoNote>z_pred → z_target</MonoNote>
            </div>
            <VectorFieldSvg activeId={activeId} />
            <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2">
              <Legend dotColor="rgb(232,174,102)" label="predicted (z_pred)" />
              <Legend dotColor="rgb(180,196,232)" label="target (z_target)" />
              <Legend dotColor="rgb(123,156,255)" label="correction direction" />
              <span className="ml-auto font-mono text-[10px] uppercase tracking-[0.14em] text-text-muted">
                Coordinates illustrative — not a real UMAP
              </span>
            </div>
          </div>

          {/* ── Side panel ── */}
          <div className="flex flex-col gap-3.5">
            {/* Headline + structured chip */}
            <div className="rounded-xl border border-white/[0.05] bg-white/[0.015] p-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="premium-kicker">Mean direction similarity</p>
                  <p className="mt-1.5 font-mono text-[28px] font-semibold leading-none tabular-nums text-accent">
                    {semanticErrorAggregate.meanDirectionSimilarity.toFixed(3)}
                  </p>
                </div>
                <StructuredChip />
              </div>
              <p className="mt-2 text-[10.5px] leading-snug text-text-muted">
                Average cosine between per-trial error vectors and the population mean error
                direction — well above chance for random vectors in 768-D.
              </p>
            </div>

            {/* Selected callout */}
            <AnimatePresence mode="wait">
              <motion.div
                key={active.id}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
                className="rounded-xl border border-accent/22 bg-accent/[0.05] p-4"
              >
                <div className="flex items-center justify-between">
                  <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-accent/90">
                    Selected example
                  </p>
                  <MonoNote>α · structured</MonoNote>
                </div>
                <p className="mt-1.5 font-mono text-[16px] font-semibold leading-none tabular-nums text-text-primary">
                  <span className="text-status-warning/95">{active.fromLabel}</span>
                  <span className="text-text-muted/70"> → </span>
                  <span className="text-accent">{active.toLabel}</span>
                </p>
                <p className="mt-2 text-[11px] leading-snug text-text-secondary">
                  {EXPLANATIONS[active.id]}
                </p>
              </motion.div>
            </AnimatePresence>

            {/* Corrections list */}
            <div className="rounded-xl border border-white/[0.05] bg-white/[0.015] p-4">
              <div className="flex items-center justify-between">
                <p className="premium-kicker">Most-common corrections</p>
                <MonoNote>z_target − z_pred</MonoNote>
              </div>
              <ul className="mt-2 space-y-1">
                {vectorFieldArrows.map((a) => {
                  const isActive = activeId === a.id;
                  return (
                    <li
                      key={a.id}
                      onMouseEnter={() => setActiveId(a.id)}
                      onFocus={() => setActiveId(a.id)}
                      tabIndex={0}
                      className={`flex cursor-default items-center gap-2.5 rounded-md px-2 py-1.5 text-[11.5px] outline-none transition focus-visible:ring-1 focus-visible:ring-accent/40 ${
                        isActive
                          ? 'bg-accent/[0.08] ring-1 ring-accent/20'
                          : 'hover:bg-white/[0.025]'
                      }`}
                    >
                      <span className={`h-1.5 w-1.5 rounded-full ${isActive ? 'bg-accent' : 'bg-text-muted/65'}`} />
                      <span className="font-mono text-text-muted">{a.fromLabel}</span>
                      <span className="text-text-muted/65">→</span>
                      <span className="font-mono text-text-primary">{a.toLabel}</span>
                    </li>
                  );
                })}
              </ul>
            </div>
          </div>
        </div>
      </PremiumPanel>
    </motion.section>
  );
}

/* ─── Plane SVG ─────────────────────────────────────────────── */

function VectorFieldSvg({ activeId }: { activeId: string }) {
  const active = vectorFieldArrows.find((a) => a.id === activeId);
  return (
    <svg
      viewBox="0 0 100 100"
      className="mt-3 h-[340px] w-full"
      role="img"
      aria-label="Schematic semantic error vector field"
    >
      <defs>
        <marker
          id="vfield-arrow"
          viewBox="0 0 6 6"
          refX="5"
          refY="3"
          markerWidth="4"
          markerHeight="4"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M0,0 L6,3 L0,6 Z" fill="rgb(123,156,255)" fillOpacity="0.95" />
        </marker>
        <radialGradient id="vfield-halo" cx="0.5" cy="0.5" r="0.5">
          <stop offset="0%"   stopColor="rgb(123,156,255)" stopOpacity="0.42" />
          <stop offset="100%" stopColor="rgb(123,156,255)" stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* Hair-thin grid */}
      <g stroke="rgb(255 255 255 / 0.045)" strokeWidth="0.35">
        {[20, 40, 60, 80].map((p) => (
          <g key={p}>
            <line x1="0" y1={p} x2="100" y2={p} />
            <line x1={p} y1="0" x2={p} y2="100" />
          </g>
        ))}
      </g>

      {/* Frame axes */}
      <g stroke="rgb(255 255 255 / 0.14)" strokeWidth="0.5">
        <line x1="0" y1="100" x2="100" y2="100" />
        <line x1="0" y1="0"   x2="0"   y2="100" />
      </g>

      {/* Axis labels */}
      <text x="2"  y="6"  fontSize="3" fill="rgb(140,148,168)" letterSpacing="0.15">CLIP DIM B</text>
      <text x="80" y="98" fontSize="3" fill="rgb(140,148,168)" letterSpacing="0.15">CLIP DIM A →</text>

      {/* Background scatter */}
      {Array.from({ length: 92 }).map((_, i) => {
        const x = ((i * 47 + 7)  % 95) + 2;
        const y = ((i * 31 + 23) % 95) + 2;
        return <circle key={`bg-${i}`} cx={x} cy={y} r="0.55" fill="rgb(150,158,172)" fillOpacity="0.18" />;
      })}

      {/* Arrows */}
      {vectorFieldArrows.map((a) => {
        const isActive = activeId === a.id;
        const dimOther = !isActive;
        const opacityScale = dimOther ? 0.32 : 1;
        return (
          <g key={a.id}>
            {isActive ? (
              <circle
                cx={(a.from[0] + a.to[0]) / 2}
                cy={(a.from[1] + a.to[1]) / 2}
                r="12"
                fill="url(#vfield-halo)"
              />
            ) : null}
            <circle cx={a.from[0]} cy={a.from[1]} r={isActive ? 2.1 : 1.55} fill="rgb(232,174,102)" fillOpacity={0.92 * opacityScale} />
            <circle cx={a.to[0]}   cy={a.to[1]}   r={isActive ? 2.1 : 1.55} fill="rgb(180,196,232)" fillOpacity={0.98 * opacityScale} />
            <line
              x1={a.from[0]}
              y1={a.from[1]}
              x2={a.to[0]}
              y2={a.to[1]}
              stroke="rgb(123,156,255)"
              strokeOpacity={(isActive ? 0.98 : 0.55) * (dimOther ? 0.55 : 1)}
              strokeWidth={isActive ? 0.95 : 0.6}
              markerEnd="url(#vfield-arrow)"
            />
            <text
              x={(a.from[0] + a.to[0]) / 2}
              y={(a.from[1] + a.to[1]) / 2 - 2.2}
              fontSize="2.6"
              fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
              fill="rgb(180,196,232)"
              fillOpacity={(isActive ? 0.98 : 0.7) * (dimOther ? 0.55 : 1)}
              textAnchor="middle"
              letterSpacing="0.04"
            >
              {a.fromLabel}→{a.toLabel}
            </text>
          </g>
        );
      })}

      {/* Annotated leader on the active example */}
      {active ? <ActiveLeader from={active.from} to={active.to} /> : null}
    </svg>
  );
}

function ActiveLeader({ from, to }: { from: [number, number]; to: [number, number] }) {
  // Pick a screen quadrant for the leader so it stays inside the plane.
  const midX = (from[0] + to[0]) / 2;
  const midY = (from[1] + to[1]) / 2;
  const leftSide = midX > 50;
  const upper    = midY > 50;
  const lx = leftSide ? Math.max(8, midX - 30) : Math.min(92, midX + 30);
  const ly = upper    ? Math.max(8, midY - 14) : Math.min(94, midY + 14);
  const anchorX = leftSide ? lx + 22 : lx - 22;
  return (
    <g>
      <line
        x1={midX}
        y1={midY}
        x2={anchorX}
        y2={ly}
        stroke="rgb(123,156,255)"
        strokeOpacity="0.55"
        strokeWidth="0.45"
        strokeDasharray="1.4 1.4"
      />
      <circle cx={anchorX} cy={ly} r="0.9" fill="rgb(123,156,255)" fillOpacity="0.85" />
      <text
        x={lx}
        y={ly}
        textAnchor={leftSide ? 'end' : 'start'}
        fontSize="2.6"
        fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
        fill="rgb(180,196,232)"
        fillOpacity="0.95"
        letterSpacing="0.08"
      >
        Active · z_target − z_pred
      </text>
    </g>
  );
}

function Legend({ dotColor, label }: { dotColor: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-2 font-mono text-[10.5px] uppercase tracking-[0.14em] text-text-muted">
      <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: dotColor }} />
      {label}
    </span>
  );
}

function StructuredChip() {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-status-success/24 bg-status-success/[0.07] px-2 py-1 text-[9.5px] font-semibold uppercase tracking-[0.14em] text-status-success/95">
      <span className="h-1.5 w-1.5 rounded-full bg-status-success" />
      Structured · not random
    </span>
  );
}
