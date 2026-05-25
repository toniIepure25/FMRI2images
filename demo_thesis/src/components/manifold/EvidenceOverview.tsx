import { motion } from 'framer-motion';
import { ProvenanceBadge } from '@/components/pipeline/ProvenanceBadge';
import { PremiumPanel } from '@/components/premium/PremiumPanel';
import { ManifoldSectionHeader } from './ManifoldSectionHeader';
import {
  evidenceMetrics,
  formatMetric,
  type ProtocolMetric,
  type MetricFamily,
} from '@/data/neuralManifoldExplorer';

/**
 * EvidenceOverview
 * ─────────────────────────────────────────────────────────────
 * "R@1 is not enough." Seven measurement tiles + one synthesis
 * tile arranged in a 4-column matrix. Each tile carries a tiny
 * family tag (Retrieval / Geometry / Language / Robustness /
 * Hubness) so the matrix reads as a scientific table, not a
 * generic KPI grid.
 */

const FAMILY: Record<MetricFamily, { label: string; tone: string }> = {
  retrieval:  { label: 'Retrieval',  tone: 'text-accent'              },
  geometry:   { label: 'Geometry',   tone: 'text-status-success/85'   },
  language:   { label: 'Language',   tone: 'text-text-secondary'      },
  robustness: { label: 'Robustness', tone: 'text-status-warning/85'   },
  hubness:    { label: 'Hubness',    tone: 'text-text-secondary'      },
};

export function EvidenceOverview() {
  return (
    <motion.section
      id="manifold-overview"
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className="mx-auto max-w-[1280px] scroll-mt-20 px-4 sm:px-6 lg:px-8"
    >
      <ManifoldSectionHeader
        kicker="Evidence matrix"
        title="R@1 is not enough."
        description={
          <>
            A decoder can retrieve images; the better scientific question is whether it preserves
            <span className="text-text-secondary"> semantic structure</span>. Each cell reads a value
            from the V62a final semantic reports — validation versus SHARED1000 — grouped by
            scientific family so the matrix reads at a glance.
          </>
        }
        action={
          <div className="flex items-center gap-2">
            <LegendDot tone="accent"  label="Validation" />
            <LegendDot tone="muted"   label="SHARED1000" />
            <ProvenanceBadge provenance={{ kind: 'replay', label: 'Reports' }} />
          </div>
        }
      />

      <div className="mt-7 grid gap-3.5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {evidenceMetrics.map((m) => (
          <EvidenceCard key={m.label} metric={m} />
        ))}
        <SynthesisCard />
      </div>
    </motion.section>
  );
}

function EvidenceCard({ metric }: { metric: ProtocolMetric }) {
  const val = metric.validation;
  const s1k = metric.shared1000;
  const valPct = Math.max(0, Math.min(1, val));
  const s1kPct = Math.max(0, Math.min(1, s1k));
  const gap = Math.abs(val - s1k);
  const gapUnit = metric.unit === '%' ? '%' : '';
  const gapValue = (gap * (metric.unit === '%' ? 100 : 1)).toFixed(2);
  const fam = FAMILY[metric.family];

  return (
    <PremiumPanel className="group relative flex flex-col p-4 transition duration-200 hover:border-white/[0.085]">
      {/* Family tag */}
      <div className="mb-3 flex items-center gap-2.5">
        <span className={`font-mono text-[9.5px] font-semibold uppercase tracking-[0.18em] ${fam.tone}`}>
          {fam.label}
        </span>
        <span className="h-px flex-1 bg-white/[0.05]" aria-hidden />
        {metric.shortLabel ? (
          <span className="font-mono text-[9.5px] uppercase tracking-[0.14em] text-text-muted/85">
            {metric.shortLabel}
          </span>
        ) : null}
      </div>

      {/* Header: metric name + provenance */}
      <div className="flex items-start justify-between gap-3">
        <p className="text-[13px] font-semibold tracking-tight text-text-primary">{metric.label}</p>
        <ProvenanceBadge provenance={metric.provenance} />
      </div>

      {/* Comparison rails */}
      <div className="mt-4 space-y-2.5">
        <ComparisonRow
          label="Val"
          value={formatMetric(val, metric.unit)}
          width={valPct}
          tone="accent"
        />
        <ComparisonRow
          label="S1K"
          value={formatMetric(s1k, metric.unit)}
          width={s1kPct}
          tone="muted"
        />
      </div>

      {/* Footer: interpretation + Δ */}
      {metric.note ? (
        <div className="mt-3 border-t border-white/[0.05] pt-2.5">
          <p className="text-[10.5px] leading-snug text-text-muted">{metric.note}</p>
          {gap > 0.001 ? (
            <p className="mt-1.5 flex items-center gap-1.5">
              <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-text-muted/85">Δ</span>
              <span className="font-mono text-[11px] tabular-nums text-text-secondary">
                {gapValue}{gapUnit}
              </span>
            </p>
          ) : null}
        </div>
      ) : null}
    </PremiumPanel>
  );
}

function ComparisonRow({
  label,
  value,
  width,
  tone,
}: {
  label: string;
  value: string;
  width: number;
  tone: 'accent' | 'muted';
}) {
  const valueClass = tone === 'accent' ? 'text-accent' : 'text-text-primary';
  const barClass = tone === 'accent' ? 'bg-accent/85' : 'bg-text-secondary/55';
  return (
    <div className="flex items-center gap-3">
      <span className="w-7 font-mono text-[9.5px] uppercase tracking-[0.14em] text-text-muted/90">
        {label}
      </span>
      <div className="relative h-[5px] flex-1 overflow-hidden rounded-full bg-white/[0.04]">
        <div
          className={`absolute inset-y-0 left-0 rounded-full ${barClass}`}
          style={{ width: `${(width * 100).toFixed(1)}%` }}
        />
      </div>
      <span className={`w-[58px] text-right font-mono text-[13px] font-semibold tabular-nums leading-none ${valueClass}`}>
        {value}
      </span>
    </div>
  );
}

function LegendDot({
  tone,
  label,
}: {
  tone: 'accent' | 'muted';
  label: string;
}) {
  const dotClass = tone === 'accent' ? 'bg-accent/85' : 'bg-text-secondary/55';
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-white/[0.05] bg-white/[0.015] px-2 py-0.5 font-mono text-[9.5px] uppercase tracking-[0.14em] text-text-muted/90">
      <span className={`h-1.5 w-1.5 rounded-full ${dotClass}`} />
      {label}
    </span>
  );
}

/**
 * Synthesis tile — fills the last cell of a 4-col grid and ties
 * the matrix back to the page's thesis. Visual treatment matches
 * the evidence cards but with a darker, contemplative tone.
 */
function SynthesisCard() {
  return (
    <PremiumPanel className="relative flex flex-col p-4">
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.22]"
        aria-hidden
        style={{
          backgroundImage:
            'repeating-linear-gradient(135deg, rgb(255 255 255 / 0.015) 0 1px, transparent 1px 9px)',
          maskImage:
            'linear-gradient(180deg, transparent 0%, black 35%, black 100%)',
          WebkitMaskImage:
            'linear-gradient(180deg, transparent 0%, black 35%, black 100%)',
        }}
      />
      <div className="relative mb-3 flex items-center gap-2.5">
        <span className="font-mono text-[9.5px] font-semibold uppercase tracking-[0.18em] text-accent/85">
          Synthesis
        </span>
        <span className="h-px flex-1 bg-white/[0.05]" aria-hidden />
        <span className="font-mono text-[9.5px] uppercase tracking-[0.14em] text-text-muted/85">
          R@1 + ρ + Δ
        </span>
      </div>

      <div className="relative">
        <p className="text-[13px] font-semibold tracking-tight text-text-primary">
          One score never proves a manifold.
        </p>
        <p className="mt-2 text-[11.5px] leading-relaxed text-text-secondary">
          A high R@1 with a flat RSA, weak language agreement, or unbounded hubness would be a
          retrieval trick, not understanding. The seven measurements on the left only co-validate
          when read together.
        </p>
      </div>

      <p className="relative mt-3 border-t border-white/[0.05] pt-2.5 font-mono text-[10px] uppercase tracking-[0.14em] text-text-muted">
        Read row-wise · evaluate together
      </p>
    </PremiumPanel>
  );
}
