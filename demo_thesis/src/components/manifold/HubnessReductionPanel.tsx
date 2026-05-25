import { motion } from 'framer-motion';
import { ProvenanceBadge } from '@/components/pipeline/ProvenanceBadge';
import { PremiumPanel } from '@/components/premium/PremiumPanel';
import { ManifoldSectionHeader } from './ManifoldSectionHeader';
import { KickerRule, MonoNote } from './atoms';
import { hubnessShared1000, formatMetric } from '@/data/neuralManifoldExplorer';
import { REPLAY_PROV } from '@/lib/provenance';

/**
 * HubnessReductionPanel — "Why CSLS matters"
 * ─────────────────────────────────────────────────────────────
 * Two-up Lorenz comparison with stacked hub-count rails so the
 * cosine→CSLS shift is the visual centre of the panel. A wide
 * "before → after" delta strip below pins the numerical story.
 */
export function HubnessReductionPanel() {
  return (
    <motion.section
      id="manifold-hubness"
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className="mx-auto max-w-[1280px] scroll-mt-20 px-4 sm:px-6 lg:px-8"
    >
      <ManifoldSectionHeader
        kicker="Hubness correction"
        title="Why CSLS matters."
        description="CSLS reweights candidates by local neighbourhood density, reducing generic high-dimensional hubs. Below, the same SHARED1000 gallery is scored under cosine and under CSLS — the cumulative-share curves and the max-hub counts tell the same story."
        action={<ProvenanceBadge provenance={REPLAY_PROV} />}
      />

      <PremiumPanel className="mt-6 p-5 sm:p-6">
        {/* Leading at-a-glance interpretation */}
        <div className="mb-5 flex flex-col gap-2 rounded-xl border border-white/[0.05] bg-white/[0.014] px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-[12px] leading-snug text-text-secondary">
            <span className="font-semibold text-text-primary">Cosine over-selects generic hub images;</span>{' '}
            CSLS penalises local density and spreads retrieval mass across the gallery.
          </p>
          <span className="inline-flex items-center gap-2 self-start font-mono text-[10.5px] uppercase tracking-[0.14em] sm:self-auto">
            <span className="text-text-muted">Before</span>
            <ArrowRight className="text-text-muted/65" />
            <span className="text-status-success">After</span>
          </span>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <LorenzCard
            title="Cosine retrieval"
            kicker="Before"
            gini={hubnessShared1000.cosineGini}
            r1={hubnessShared1000.cosineR1}
            maxHub={hubnessShared1000.maxHubCosine}
            tone="warning"
          />
          <LorenzCard
            title="CSLS retrieval"
            kicker="After"
            gini={hubnessShared1000.cslsGini}
            r1={hubnessShared1000.cslsR1}
            maxHub={hubnessShared1000.maxHubCsls}
            tone="success"
          />
        </div>

        {/* Before → after pinned summary */}
        <div className="mt-5 grid gap-3 sm:grid-cols-3">
          <BeforeAfter
            label="R@1 (SHARED1000)"
            before={formatMetric(hubnessShared1000.cosineR1, '%')}
            after={formatMetric(hubnessShared1000.cslsR1, '%')}
            delta={`+${((hubnessShared1000.cslsR1 - hubnessShared1000.cosineR1) * 100).toFixed(1)}%`}
            deltaTone="success"
          />
          <BeforeAfter
            label="Hubness Gini"
            before={hubnessShared1000.cosineGini.toFixed(3)}
            after={hubnessShared1000.cslsGini.toFixed(3)}
            delta={`−${hubnessShared1000.giniReduction.toFixed(3)}`}
            deltaTone="success"
          />
          <BeforeAfter
            label="Max-hub count"
            before={String(hubnessShared1000.maxHubCosine)}
            after={String(hubnessShared1000.maxHubCsls)}
            delta={`−${hubnessShared1000.maxHubCosine - hubnessShared1000.maxHubCsls}`}
            deltaTone="success"
          />
        </div>
      </PremiumPanel>
    </motion.section>
  );
}

function LorenzCard({
  title,
  kicker,
  gini,
  r1,
  maxHub,
  tone,
}: {
  title: string;
  kicker: string;
  gini: number;
  r1: number;
  maxHub: number;
  tone: 'success' | 'warning';
}) {
  const stroke = tone === 'success' ? 'rgb(96, 197, 158)' : 'rgb(232, 174, 102)';
  return (
    <div className="rounded-2xl border border-white/[0.05] bg-white/[0.014] p-4 sm:p-5">
      <div className="flex items-center justify-between">
        <KickerRule label={kicker} />
        <MonoNote>SHARED1000 · 1k gallery</MonoNote>
      </div>
      <div className="mt-2 flex items-baseline justify-between">
        <h3 className="text-[16px] font-semibold tracking-tight text-text-primary">{title}</h3>
        <p
          className="font-mono text-[18px] font-semibold tabular-nums leading-none"
          style={{ color: stroke }}
        >
          Gini {gini.toFixed(3)}
        </p>
      </div>

      <div className="mt-3">
        <LorenzCurve gini={gini} stroke={stroke} />
      </div>

      {/* Hub-count rail — visual mass of the worst-offender vs average */}
      <div className="mt-3">
        <p className="premium-kicker">Top-hub count</p>
        <HubRail maxHub={maxHub} stroke={stroke} />
      </div>

      <div className="mt-3 grid grid-cols-3 gap-2 border-t border-white/[0.05] pt-3">
        <Metric label="R@1"     value={formatMetric(r1, '%')} />
        <Metric label="Gini"    value={gini.toFixed(3)} />
        <Metric label="Max hub" value={String(maxHub)} />
      </div>
    </div>
  );
}

/**
 * Lorenz cumulative-share curve.  y = x^(1+k) with k chosen so
 * the analytical Gini matches the reported value: G = k/(k+2).
 */
function LorenzCurve({ gini, stroke }: { gini: number; stroke: string }) {
  const g = Math.min(0.95, Math.max(0.01, gini));
  const k = (2 * g) / (1 - g);
  const samples = 36;
  const pts: string[] = [];
  for (let i = 0; i <= samples; i++) {
    const x = i / samples;
    const y = Math.pow(x, 1 + k);
    pts.push(`${(x * 100).toFixed(2)},${((1 - y) * 100).toFixed(2)}`);
  }
  const fillPath = `M0,100 L${pts.join(' L')} L100,100 Z`;
  const gradientId = `lorenz-fill-${stroke.replace(/[^a-z0-9]/gi, '')}`;
  return (
    <svg viewBox="0 0 100 100" className="h-[170px] w-full" role="img" aria-label="Lorenz cumulative-share curve">
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"  stopColor={stroke} stopOpacity="0.22" />
          <stop offset="100%" stopColor={stroke} stopOpacity="0.02" />
        </linearGradient>
      </defs>
      {/* Hairline grid */}
      <g stroke="rgb(255 255 255 / 0.06)" strokeWidth="0.4">
        {[20, 40, 60, 80].map((p) => (
          <g key={p}>
            <line x1="0" y1={p} x2="100" y2={p} />
            <line x1={p} y1="0" x2={p} y2="100" />
          </g>
        ))}
      </g>
      {/* Axes */}
      <g stroke="rgb(255 255 255 / 0.14)" strokeWidth="0.55">
        <line x1="0" y1="100" x2="100" y2="100" />
        <line x1="0" y1="0"   x2="0"   y2="100" />
      </g>
      {/* Equality line */}
      <line x1="0" y1="100" x2="100" y2="0" stroke="rgb(255 255 255 / 0.14)" strokeWidth="0.5" strokeDasharray="2 2" />
      {/* Curve */}
      <path d={fillPath} fill={`url(#${gradientId})`} />
      <polyline
        points={pts.join(' ')}
        fill="none"
        stroke={stroke}
        strokeOpacity="0.95"
        strokeWidth="1.2"
        strokeLinejoin="round"
      />
      {/* Corner labels */}
      <text x="2"  y="95" fontSize="5.5" fill="rgb(140,148,168)">0%</text>
      <text x="76" y="95" fontSize="5.5" fill="rgb(140,148,168)">items →</text>
      <text x="2"  y="6"  fontSize="5.5" fill="rgb(140,148,168)">cumulative share</text>
    </svg>
  );
}

/**
 * Compact horizontal rail that visualises the max-hub count as a
 * proportion of an arbitrary 60-row reference cap. The visual is
 * deterministic — only the active hub height varies.
 */
function HubRail({ maxHub, stroke }: { maxHub: number; stroke: string }) {
  const cap = 60;
  const cells = 24;
  return (
    <svg viewBox="0 0 240 28" className="mt-1.5 h-[28px] w-full" role="img" aria-label="Top-hub count rail">
      {Array.from({ length: cells }).map((_, i) => {
        // Deterministic decreasing density across the rail
        const base = Math.max(2, maxHub - (i * (maxHub / cells)));
        const h = Math.min(22, (base / cap) * 22);
        const x = 4 + i * 10;
        return (
          <g key={i}>
            <rect x={x} y={4} width="6" height="22" rx="1.4" fill="rgb(255,255,255)" fillOpacity="0.04" />
            <rect
              x={x}
              y={26 - h}
              width="6"
              height={h}
              rx="1.4"
              fill={stroke}
              fillOpacity={0.4 + (i === 0 ? 0.45 : 0)}
            />
          </g>
        );
      })}
      <text x="4"   y="3" fontSize="4.5" fill="rgb(140,148,168)">most-frequent →</text>
      <text x="180" y="3" fontSize="4.5" fill="rgb(140,148,168)">least-frequent</text>
    </svg>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="premium-kicker">{label}</p>
      <p className="mt-1 font-mono text-[14px] font-semibold tabular-nums text-text-primary">
        {value}
      </p>
    </div>
  );
}

function BeforeAfter({
  label,
  before,
  after,
  delta,
  deltaTone = 'default',
}: {
  label: string;
  before: string;
  after: string;
  delta: string;
  deltaTone?: 'success' | 'warning' | 'default';
}) {
  const tone =
    deltaTone === 'success' ? 'text-status-success'
    : deltaTone === 'warning' ? 'text-status-warning'
    : 'text-text-secondary';
  return (
    <div className="rounded-xl border border-white/[0.05] bg-white/[0.015] px-4 py-3">
      <p className="premium-kicker">{label}</p>
      <div className="mt-2 flex items-baseline gap-3">
        <p className="font-mono text-[14px] tabular-nums text-text-muted">{before}</p>
        <ArrowRight className="text-text-muted/65" />
        <p className="font-mono text-[19px] font-semibold tabular-nums text-text-primary leading-none">
          {after}
        </p>
        <span className={`ml-auto font-mono text-[12px] tabular-nums ${tone}`}>{delta}</span>
      </div>
    </div>
  );
}

function ArrowRight({ className }: { className?: string }) {
  return (
    <svg width="14" height="10" viewBox="0 0 14 10" fill="none" className={className} aria-hidden>
      <path d="M1 5h11M9 1l4 4-4 4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
