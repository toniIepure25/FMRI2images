import { useState } from 'react';
import { motion } from 'framer-motion';
import { ProvenanceBadge } from '@/components/pipeline/ProvenanceBadge';
import { PremiumPanel } from '@/components/premium/PremiumPanel';
import { ManifoldSectionHeader } from './ManifoldSectionHeader';
import { KickerRule, MonoNote, SchematicChip } from './atoms';
import {
  semanticErrorAggregate,
  vectorFieldArrows,
} from '@/data/neuralManifoldExplorer';
import { DERIVED_PROV } from '@/lib/provenance';

/**
 * SemanticVectorField
 * ─────────────────────────────────────────────────────────────
 * A large schematic CLIP plane with proper axes, gridlines, a
 * tagged legend, and selectable arrows. Predicted points are
 * amber, target points are blue-white; hovering or focusing an
 * arrow lifts it above the field and highlights the matching
 * row in the side panel.
 */
export function SemanticVectorField() {
  const [activeId, setActiveId] = useState<string | null>(null);

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className="mx-auto max-w-[1280px] px-4 sm:px-6 lg:px-8"
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
            <VectorFieldSvg
              activeId={activeId}
              onSelect={(id) => setActiveId((cur) => (cur === id ? null : id))}
            />
            <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2">
              <Legend dotColor="rgb(232,174,102)" label="predicted (z_pred)" />
              <Legend dotColor="rgb(180,196,232)" label="target (z_target)" />
              <Legend dotColor="rgb(123,156,255)" label="correction direction" />
              <span className="ml-auto font-mono text-[10px] uppercase tracking-[0.14em] text-text-muted">
                Coordinates illustrative — not a real UMAP
              </span>
            </div>
          </div>

          {/* ── Side panel: aggregate + correction list ── */}
          <div className="flex flex-col gap-3.5">
            <div className="rounded-xl border border-white/[0.05] bg-white/[0.015] p-4">
              <p className="premium-kicker">Mean direction similarity</p>
              <p className="mt-1.5 font-mono text-[28px] font-semibold leading-none tabular-nums text-accent">
                {semanticErrorAggregate.meanDirectionSimilarity.toFixed(3)}
              </p>
              <p className="mt-2 text-[10.5px] leading-snug text-text-muted">
                Average cosine between per-trial error vectors and the population mean error
                direction — well above chance for random vectors in 768-D.
              </p>
            </div>

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
                      onMouseLeave={() => setActiveId(null)}
                      onFocus={() => setActiveId(a.id)}
                      onBlur={() => setActiveId(null)}
                      className={`flex cursor-default items-center gap-2.5 rounded-md px-2 py-1.5 text-[11.5px] transition ${
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

function VectorFieldSvg({
  activeId,
  onSelect,
}: {
  activeId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <svg
      viewBox="0 0 100 100"
      className="mt-3 h-[330px] w-full"
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
          <path d="M0,0 L6,3 L0,6 Z" fill="rgb(123,156,255)" fillOpacity="0.92" />
        </marker>
        <radialGradient id="vfield-halo" cx="0.5" cy="0.5" r="0.5">
          <stop offset="0%"  stopColor="rgb(123,156,255)" stopOpacity="0.32" />
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

      {/* Axes (frame) */}
      <g stroke="rgb(255 255 255 / 0.14)" strokeWidth="0.5">
        <line x1="0"   y1="100" x2="100" y2="100" />
        <line x1="0"   y1="0"   x2="0"   y2="100" />
      </g>

      {/* Axis labels */}
      <text x="2"  y="6"  fontSize="3"   fill="rgb(140,148,168)" letterSpacing="0.15">CLIP DIM B</text>
      <text x="80" y="98" fontSize="3"   fill="rgb(140,148,168)" letterSpacing="0.15">CLIP DIM A →</text>

      {/* Background scatter — quiet population dots so the field doesn't look empty */}
      {Array.from({ length: 90 }).map((_, i) => {
        const x = ((i * 47 + 7)  % 95) + 2;
        const y = ((i * 31 + 23) % 95) + 2;
        return <circle key={`bg-${i}`} cx={x} cy={y} r="0.55" fill="rgb(150,158,172)" fillOpacity="0.18" />;
      })}

      {/* Arrows */}
      {vectorFieldArrows.map((a) => {
        const isActive = activeId === a.id;
        const dimOther = activeId != null && !isActive;
        const opacityBase = dimOther ? 0.35 : 1;
        return (
          <g
            key={a.id}
            onMouseEnter={() => onSelect(a.id)}
            onMouseLeave={() => onSelect(a.id)}
            style={{ cursor: 'default' }}
            tabIndex={0}
          >
            {isActive ? (
              <circle cx={(a.from[0] + a.to[0]) / 2} cy={(a.from[1] + a.to[1]) / 2} r="9" fill="url(#vfield-halo)" />
            ) : null}
            <circle cx={a.from[0]} cy={a.from[1]} r="1.6" fill="rgb(232,174,102)" fillOpacity={0.85 * opacityBase} />
            <circle cx={a.to[0]}   cy={a.to[1]}   r="1.6" fill="rgb(180,196,232)" fillOpacity={0.95 * opacityBase} />
            <line
              x1={a.from[0]}
              y1={a.from[1]}
              x2={a.to[0]}
              y2={a.to[1]}
              stroke="rgb(123,156,255)"
              strokeOpacity={(isActive ? 0.95 : 0.55) * (dimOther ? 0.5 : 1)}
              strokeWidth={isActive ? 0.85 : 0.6}
              markerEnd="url(#vfield-arrow)"
            />
            <text
              x={(a.from[0] + a.to[0]) / 2}
              y={(a.from[1] + a.to[1]) / 2 - 2.1}
              fontSize="2.6"
              fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
              fill="rgb(180,196,232)"
              fillOpacity={(isActive ? 0.95 : 0.7) * (dimOther ? 0.6 : 1)}
              textAnchor="middle"
              letterSpacing="0.04"
            >
              {a.fromLabel}→{a.toLabel}
            </text>
          </g>
        );
      })}
    </svg>
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
