import { motion } from 'framer-motion';
import { ProvenanceBadge } from '@/components/pipeline/ProvenanceBadge';
import { PremiumPanel } from '@/components/premium/PremiumPanel';
import { ManifoldSectionHeader } from './ManifoldSectionHeader';
import { EvidenceImage } from './EvidenceImage';
import { KickerRule, MonoNote } from './atoms';
import { probeTrial, formatMetric } from '@/data/neuralManifoldExplorer';
import { DERIVED_PROV, REPLAY_PROV } from '@/lib/provenance';

/**
 * SemanticProbePanel
 * ─────────────────────────────────────────────────────────────
 * Three-column layout:
 *
 *   left   stimulus card (4:5, NSD thumbnail, with caveat)
 *   mid    decode pipeline — voxels → MLP → vMF unit-sphere → CLIP point → probe
 *   right  ranked text probe readout (#1 emphasised)
 *
 * The middle column is intentionally the visual anchor — every
 * stage is rendered to scale on a hair-thin coordinate plane so
 * the diagram reads like an instrument schematic, not four
 * letters in boxes.
 */
export function SemanticProbePanel() {
  const probes = probeTrial.topProbes;
  const maxScore = Math.max(...probes.map((p) => p.score), 1);

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className="mx-auto max-w-[1280px] px-4 sm:px-6 lg:px-8"
    >
      <ManifoldSectionHeader
        kicker="Semantic probe"
        title="What the brain embedding means in language space."
        description="Language probes convert a 768-D predicted CLIP vector into interpretable concept evidence — the predicted direction is compared against a library of text-concept embeddings."
      />

      <PremiumPanel className="mt-6 p-5 sm:p-6">
        <div className="grid items-stretch gap-5 lg:grid-cols-[minmax(200px,0.8fr)_minmax(0,2.1fr)_minmax(240px,1.3fr)]">

          {/* ── Stimulus ── */}
          <div className="flex flex-col gap-3">
            <KickerRule label="Stimulus" />
            <figure className="overflow-hidden rounded-xl border border-white/[0.05] bg-surface-elevated">
              <EvidenceImage
                src={probeTrial.stimulusImage}
                alt={probeTrial.caption}
                aspect="4/5"
              />
              <figcaption className="px-3 py-2.5">
                <p className="text-[11.5px] font-semibold tracking-tight text-text-primary">
                  {probeTrial.caption}
                </p>
                <p className="mt-0.5 font-mono text-[10px] tabular-nums text-text-muted">
                  NSD #{probeTrial.nsdId} · {probeTrial.subject} · {probeTrial.protocol}
                </p>
              </figcaption>
            </figure>
            <div className="flex items-center justify-between">
              <ProvenanceBadge provenance={REPLAY_PROV} />
              <MonoNote>Trial · {probeTrial.trialId}</MonoNote>
            </div>
          </div>

          {/* ── Decode pipeline ── */}
          <div className="workbench-inset flex flex-col justify-between gap-4">
            <div className="flex items-center justify-between">
              <KickerRule label="Decode pipeline" />
              <MonoNote>voxels → MLP → vMF → CLIP → probe</MonoNote>
            </div>
            <DecodePipelineSvg />
            <p className="text-[10.5px] leading-snug text-text-muted">
              fMRI voxels feed a residual MLP whose output is normalised to a unit-sphere direction
              by the vMF head, projected into CLIP space, then read in language by 29 text-concept
              probes. The probe column on the right shows the resulting concept evidence.
            </p>
          </div>

          {/* ── Top probes ── */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <KickerRule label="Top probes" />
              <ProvenanceBadge provenance={DERIVED_PROV} />
            </div>
            <div className="rounded-xl border border-white/[0.05] bg-white/[0.015] p-3.5">
              <ul className="space-y-1.5">
                {probes.map((p, i) => {
                  const pct = (p.score / maxScore) * 100;
                  const highlight = i === 0;
                  return (
                    <li key={p.concept} className="flex items-center gap-3">
                      <span
                        className={`w-3 text-right font-mono text-[9.5px] tabular-nums ${
                          highlight ? 'text-accent' : 'text-text-muted/85'
                        }`}
                      >
                        {(i + 1).toString().padStart(2, '0')}
                      </span>
                      <span
                        className={`min-w-[88px] text-[11.5px] tracking-tight ${
                          highlight ? 'text-text-primary font-semibold' : 'text-text-secondary'
                        }`}
                      >
                        {p.concept}
                      </span>
                      <div className="relative h-[4px] flex-1 overflow-hidden rounded-full bg-white/[0.04]">
                        <div
                          className={`absolute inset-y-0 left-0 rounded-full ${
                            highlight ? 'bg-accent/85' : 'bg-text-secondary/55'
                          }`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span
                        className={`w-10 text-right font-mono text-[11px] tabular-nums ${
                          highlight ? 'text-text-primary' : 'text-text-muted'
                        }`}
                      >
                        {p.score.toFixed(2)}
                      </span>
                    </li>
                  );
                })}
              </ul>
              <div className="mt-3 flex items-center justify-between border-t border-white/[0.05] pt-2.5">
                <MonoNote>Aggregate agreement @5</MonoNote>
                <p className="font-mono text-[13px] font-semibold tabular-nums text-accent">
                  {formatMetric(probeTrial.agreementAt5, '%')}
                </p>
              </div>
            </div>
            <p className="text-[10.5px] leading-snug text-text-muted">
              Per-trial ranking is DERIVED — illustrative of the probe profile that produces the
              aggregate <span className="text-text-secondary">@5</span> metric above.
            </p>
          </div>
        </div>
      </PremiumPanel>
    </motion.section>
  );
}

/**
 * Decode pipeline diagram — one 360×140 SVG figure with five
 * stages connected by hair-thin flow lines on a coordinate plane.
 * The vMF stage renders a unit-sphere with a single direction vector.
 */
function DecodePipelineSvg() {
  // X-position of each stage centre in the 360-wide viewBox.
  const stages = [
    { x:  44, label: 'voxels',   detail: 'NSD ROI β'      },
    { x: 112, label: 'MLP',      detail: 'residual 4-layer' },
    { x: 180, label: 'vMF',      detail: 'unit sphere'    },
    { x: 248, label: 'CLIP',     detail: 'ViT-L/14 768-D' },
    { x: 316, label: 'probe',    detail: '29 concepts'    },
  ];

  return (
    <svg viewBox="0 0 360 150" className="w-full" role="img" aria-label="Decode pipeline diagram">
      <defs>
        <linearGradient id="probe-rail" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%"   stopColor="rgb(123,156,255)" stopOpacity="0" />
          <stop offset="14%"  stopColor="rgb(123,156,255)" stopOpacity="0.5" />
          <stop offset="86%"  stopColor="rgb(123,156,255)" stopOpacity="0.5" />
          <stop offset="100%" stopColor="rgb(123,156,255)" stopOpacity="0" />
        </linearGradient>
        <radialGradient id="probe-sphere" cx="0.42" cy="0.36" r="0.65">
          <stop offset="0%"  stopColor="rgb(190,206,238)" stopOpacity="0.65" />
          <stop offset="60%" stopColor="rgb(80,108,176)"  stopOpacity="0.20" />
          <stop offset="100%" stopColor="rgb(20,26,40)"   stopOpacity="0.05" />
        </radialGradient>
        <marker id="probe-arrow" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4" markerHeight="4" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L6,3 L0,6 Z" fill="rgb(190,206,238)" fillOpacity="0.85" />
        </marker>
      </defs>

      {/* Hair-thin coordinate plane behind the pipeline */}
      <g stroke="rgb(255 255 255 / 0.05)" strokeWidth="0.4">
        <line x1="0" y1="50" x2="360" y2="50" />
        <line x1="0" y1="100" x2="360" y2="100" />
      </g>

      {/* Connecting rail */}
      <line x1="20" y1="75" x2="340" y2="75" stroke="url(#probe-rail)" strokeWidth="0.7" />

      {/* Stage 1 — voxel grid (4×4 dots) */}
      <Voxels cx={44} cy={75} />

      {/* Stage 2 — MLP funnel (4 layers) */}
      <MlpFunnel cx={112} cy={75} />

      {/* Stage 3 — vMF unit sphere with direction vector */}
      <g transform="translate(180,75)">
        <circle r="22" fill="url(#probe-sphere)" stroke="rgb(180,196,232)" strokeOpacity="0.32" strokeWidth="0.6" />
        {/* Latitudinal arcs to suggest a sphere */}
        <ellipse rx="22" ry="7" fill="none" stroke="rgb(180,196,232)" strokeOpacity="0.16" strokeWidth="0.45" />
        <ellipse rx="22" ry="14" fill="none" stroke="rgb(180,196,232)" strokeOpacity="0.12" strokeWidth="0.4" />
        <line x1="-22" y1="0" x2="22" y2="0" stroke="rgb(180,196,232)" strokeOpacity="0.18" strokeWidth="0.4" />
        {/* Direction vector */}
        <line x1="0" y1="0" x2="14" y2="-12" stroke="rgb(190,206,238)" strokeOpacity="0.95" strokeWidth="1.1" markerEnd="url(#probe-arrow)" />
        <circle r="1.4" fill="rgb(190,206,238)" />
      </g>

      {/* Stage 4 — CLIP point: a 8-strip latent fingerprint */}
      <ClipFingerprint cx={248} cy={75} />

      {/* Stage 5 — Probe column (4 hair bars) */}
      <ProbeColumn cx={316} cy={75} />

      {/* Labels */}
      {stages.map((s) => (
        <g key={s.label}>
          <text x={s.x} y={118} textAnchor="middle" fontSize="9" fontWeight="600" letterSpacing="0.06em" fill="rgb(220,224,236)">
            {s.label.toUpperCase()}
          </text>
          <text x={s.x} y={130} textAnchor="middle" fontSize="8.5" fill="rgb(140,148,168)">
            {s.detail}
          </text>
        </g>
      ))}
    </svg>
  );
}

function Voxels({ cx, cy }: { cx: number; cy: number }) {
  const pts: Array<[number, number]> = [];
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      pts.push([cx - 10.5 + c * 7, cy - 10.5 + r * 7]);
    }
  }
  return (
    <g>
      {pts.map(([x, y], i) => {
        const v = (((i * 47 + 13) % 19) / 19) * 0.7 + 0.2;
        return <rect key={i} x={x - 2.4} y={y - 2.4} width="4.8" height="4.8" rx="0.8" fill="rgb(190,206,238)" fillOpacity={v} />;
      })}
    </g>
  );
}

function MlpFunnel({ cx, cy }: { cx: number; cy: number }) {
  // Four vertical bars whose heights taper (funnel)
  const layers = [22, 18, 14, 10];
  return (
    <g>
      {layers.map((h, i) => {
        const x = cx - 14 + i * 9;
        return (
          <rect
            key={i}
            x={x - 2}
            y={cy - h / 2}
            width="4"
            height={h}
            rx="1.2"
            fill="rgb(180,196,232)"
            fillOpacity={0.18 + i * 0.04}
          />
        );
      })}
      {/* Tiny residual skip arc */}
      <path
        d={`M ${cx - 14} ${cy - 16} Q ${cx} ${cy - 22} ${cx + 13} ${cy - 16}`}
        fill="none"
        stroke="rgb(123,156,255)"
        strokeOpacity="0.32"
        strokeWidth="0.6"
        strokeDasharray="1.5 1.5"
      />
    </g>
  );
}

function ClipFingerprint({ cx, cy }: { cx: number; cy: number }) {
  const N = 8;
  return (
    <g>
      <line x1={cx} y1={cy - 18} x2={cx} y2={cy + 18} stroke="rgb(123,156,255)" strokeOpacity="0.45" strokeWidth="0.5" />
      {Array.from({ length: N }).map((_, i) => {
        const t = i / (N - 1);
        const y = cy - 16 + t * 32;
        const v = Math.sin(t * Math.PI * 2.4 + 0.3) * 0.7 + (((i * 31) % 11) / 11 - 0.5) * 0.3;
        const len = Math.min(12, Math.abs(v) * 14);
        const side = v >= 0 ? 1 : -1;
        return (
          <rect
            key={i}
            x={cx + (side > 0 ? 0 : -len)}
            y={y - 1.4}
            width={len}
            height="2.8"
            rx="0.6"
            fill={side > 0 ? 'rgb(123,156,255)' : 'rgb(190,206,238)'}
            fillOpacity={0.55}
          />
        );
      })}
    </g>
  );
}

function ProbeColumn({ cx, cy }: { cx: number; cy: number }) {
  const heights = [22, 17, 13, 9];
  return (
    <g>
      {heights.map((h, i) => {
        const y = cy - 16 + i * 9;
        const accent = i === 0;
        return (
          <g key={i}>
            <rect x={cx - 12} y={y - 1.4} width="24" height="2.8" rx="1.4" fill="rgb(255,255,255)" fillOpacity="0.04" />
            <rect
              x={cx - 12}
              y={y - 1.4}
              width={h}
              height="2.8"
              rx="1.4"
              fill={accent ? 'rgb(123,156,255)' : 'rgb(190,206,232)'}
              fillOpacity={accent ? 0.92 : 0.45}
            />
          </g>
        );
      })}
    </g>
  );
}
