import { motion } from 'framer-motion';
import { ProvenanceBadge } from '@/components/pipeline/ProvenanceBadge';
import { PremiumPanel } from '@/components/premium/PremiumPanel';
import { ManifoldSectionHeader } from './ManifoldSectionHeader';
import {
  evidenceMetrics,
  formatMetric,
  type ProtocolMetric,
} from '@/data/neuralManifoldExplorer';

/**
 * EvidenceOverview
 * ─────────────────────────────────────────────────────────────
 * "R@1 is not enough." A grid of evidence cards, each pulling a
 * real aggregate from the V62a final semantic reports.
 *
 * Card hierarchy:
 *   kicker (family)  ·  metric name      ·  ProvenanceBadge
 *   ────────────────────────────────────────────────────────
 *   validation row   (primary, accent value + comparison bar)
 *   shared1000 row   (secondary value + comparison bar)
 *   ────────────────────────────────────────────────────────
 *   hair-rule
 *   one-line interpretation + Δ
 *
 * The two rows share the same baseline so the gap reads as a
 * visual story across the dashboard, not just numbers in boxes.
 */
export function EvidenceOverview() {
  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className="mx-auto max-w-[1280px] px-4 sm:px-6 lg:px-8"
    >
      <ManifoldSectionHeader
        kicker="Evidence overview"
        title="R@1 is not enough."
        description={
          <>
            A decoder can retrieve images; the better scientific question is whether it preserves
            <span className="text-text-secondary"> semantic structure</span>. Each card reads a value
            from the V62a final semantic reports — validation versus SHARED1000 — so the
            generalisation gap is visible at a glance.
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

  return (
    <PremiumPanel className="group relative flex flex-col p-4 transition duration-200 hover:border-white/[0.085]">
      {/* Header: family · name · provenance */}
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="premium-kicker">{metric.label}</p>
          {metric.shortLabel ? (
            <p className="mt-1 font-mono text-[10.5px] uppercase tracking-[0.14em] text-text-muted/85">
              {metric.shortLabel}
            </p>
          ) : null}
        </div>
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
