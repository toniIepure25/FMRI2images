import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import type { DemoCase } from '@/types';
import { isDefenseReady } from '@/lib/pipelineNormalize';
import { EmptyState } from '@/components/premium/EmptyState';

export interface PhaseTrialSelectionProps {
  cases: DemoCase[];
  onSelect: (case_: DemoCase) => void;
}

const DIFF_ORDER: Record<DemoCase['difficulty'], number> = {
  best: 0,
  medium: 1,
  hard: 2,
  mystery: 3,
};

type IntegrityTier = 'defense' | 'partial' | 'retrieval';

function integrityTier(c: DemoCase): IntegrityTier {
  if (isDefenseReady(c)) return 'defense';
  if (c.retrievedImages.length > 0 && !!c.targetImage) return 'partial';
  return 'retrieval';
}

const TIER_ORDER: Record<IntegrityTier, number> = { defense: 0, partial: 1, retrieval: 2 };

function sortCases(cases: DemoCase[]): DemoCase[] {
  return [...cases].sort((a, b) => {
    const t = TIER_ORDER[integrityTier(a)] - TIER_ORDER[integrityTier(b)];
    if (t !== 0) return t;
    const d = DIFF_ORDER[a.difficulty] - DIFF_ORDER[b.difficulty];
    return d !== 0 ? d : a.nsdId - b.nsdId;
  });
}

function diffLabel(d: DemoCase['difficulty']): string {
  return d === 'best' ? 'Exact match' : d === 'medium' ? 'Near miss' : d === 'hard' ? 'Hard case' : 'Mystery';
}

function statusLabel(c: DemoCase): string {
  if (c.metrics.r1Correct) return 'Exact match';
  if (c.metrics.r5Correct) return 'Top-5 match';
  if (isDefenseReady(c)) return 'Ready';
  return diffLabel(c.difficulty);
}

export function PhaseTrialSelection({ cases, onSelect }: PhaseTrialSelectionProps) {
  const sorted = useMemo(() => sortCases(cases), [cases]);
  const [filter, setFilter] = useState<string>('all');
  const [defenseOnly, setDefenseOnly] = useState(false);

  const filtered = useMemo(() => {
    let result = sorted;
    if (defenseOnly) result = result.filter((c) => integrityTier(c) === 'defense');
    if (filter !== 'all') result = result.filter((c) => c.difficulty === filter);
    return result;
  }, [sorted, filter, defenseOnly]);

  const defenseCount = useMemo(() => sorted.filter((c) => integrityTier(c) === 'defense').length, [sorted]);
  const counts = useMemo(() => {
    const base = defenseOnly ? sorted.filter((c) => integrityTier(c) === 'defense') : sorted;
    const m: Record<string, number> = { all: base.length };
    for (const c of base) m[c.difficulty] = (m[c.difficulty] || 0) + 1;
    return m;
  }, [sorted, defenseOnly]);

  const filterAction = (
    <div className="flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={() => setDefenseOnly((v) => !v)}
        className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[11.5px] font-semibold transition ${
          defenseOnly
            ? 'border-accent/25 bg-accent/[0.08] text-accent'
            : 'border-white/[0.06] bg-white/[0.025] text-text-secondary hover:border-white/15 hover:text-text-primary'
        }`}
      >
        Defense set
        <span className="font-mono text-[10px] opacity-75">{defenseCount}</span>
      </button>
      <div className="h-4 w-px bg-white/[0.06]" aria-hidden />
      <div className="inline-flex items-center gap-0.5 rounded-full border border-white/[0.05] bg-white/[0.02] p-0.5">
        {['all', 'best', 'medium', 'hard'].map((f) => (
          <button
            key={f}
            type="button"
            onClick={() => setFilter(f)}
            className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px] font-semibold transition ${
              filter === f
                ? 'bg-white/[0.06] text-text-primary'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            {f === 'all' ? 'All' : diffLabel(f as DemoCase['difficulty'])}
            <span className="font-mono text-[10px] tabular-nums opacity-60">{counts[f] || 0}</span>
          </button>
        ))}
      </div>
    </div>
  );

  return (
    <div className="pb-14">
      {/* ── Slim filter row — no card chrome, aligned right; whitespace and
             typography carry the relationship to the header above. ── */}
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 px-1">
        <p className="text-[11.5px] text-text-muted">
          Showing{' '}
          <span className="font-mono tabular-nums text-text-secondary">{filtered.length}</span>{' '}
          of{' '}
          <span className="font-mono tabular-nums text-text-secondary">{sorted.length}</span>{' '}
          trials
        </p>
        {filterAction}
      </div>

      <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
        {filtered.map((case_, idx) => {
          const exact = case_.metrics.rank === 1;
          const top5 = !exact && case_.metrics.r5Correct;
          // Quiet semantic match tag — no full background fill, no thick border
          const matchTag = exact
            ? { ring: 'ring-status-success/30', dot: 'bg-status-success', text: 'text-status-success/90' }
            : top5
            ? { ring: 'ring-status-warning/30', dot: 'bg-status-warning', text: 'text-status-warning/90' }
            : { ring: 'ring-white/15', dot: 'bg-white/55', text: 'text-white/70' };
          return (
            <motion.button
              key={case_.id}
              type="button"
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                delay: Math.min(idx * 0.025, 0.45),
                duration: 0.35,
                ease: [0.22, 1, 0.36, 1],
              }}
              className="group rounded-[14px] text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface-base"
              onClick={() => onSelect(case_)}
            >
              {/* ── Image-first tile (Option A): metadata sits on the image
                     via a graphite gradient. No dark slab footer. ── */}
              <figure
                className={`relative aspect-[4/5] overflow-hidden rounded-[14px] border bg-black/10 transition duration-300 group-hover:-translate-y-[2px] ${
                  exact ? 'border-status-success/22' : 'border-white/[0.05] group-hover:border-white/[0.12]'
                }`}
                style={{
                  boxShadow:
                    '0 1px 0 rgb(255 255 255 / 0.022) inset, 0 18px 44px -32px rgb(0 0 0 / 0.95)',
                }}
              >
                <img
                  src={case_.targetImage}
                  alt={`NSD stimulus ${case_.nsdId}`}
                  className="absolute inset-0 h-full w-full object-cover transition duration-[600ms] ease-out group-hover:scale-[1.025]"
                  loading="lazy"
                />

                {/* Top tint — keeps the top badges legible without darkening
                    the whole frame. */}
                <div
                  className="pointer-events-none absolute inset-x-0 top-0 h-16 bg-gradient-to-b from-black/35 to-transparent"
                  aria-hidden
                />
                {/* Bottom graphite gradient — carries the metadata. Not pure
                    black, hint of cool tone, opacity ramped so the image
                    still breathes into it. */}
                <div
                  className="pointer-events-none absolute inset-x-0 bottom-0 h-[44%]"
                  aria-hidden
                  style={{
                    background:
                      'linear-gradient(to top, rgb(8 10 14 / 0.88) 0%, rgb(8 10 14 / 0.72) 30%, rgb(8 10 14 / 0.30) 70%, transparent 100%)',
                  }}
                />
                {/* Hair-thin inset ring softens the image edge. */}
                <div className="pointer-events-none absolute inset-0 rounded-[13px] ring-1 ring-inset ring-white/[0.045]" aria-hidden />

                {/* ── Top corners: match tag + rank ── */}
                <div className="absolute left-2.5 top-2.5">
                  <span
                    className={`inline-flex items-center gap-1 rounded-full bg-black/45 px-1.5 py-[2px] text-[9px] font-semibold uppercase tracking-[0.08em] backdrop-blur-md ring-1 ${matchTag.ring} ${matchTag.text}`}
                  >
                    <span className={`h-[4px] w-[4px] shrink-0 rounded-full ${matchTag.dot}`} />
                    {statusLabel(case_)}
                  </span>
                </div>
                <div className="absolute right-2.5 top-2.5">
                  <span className="inline-flex items-center rounded-full bg-black/45 px-1.5 py-[2px] font-mono text-[9px] font-semibold tabular-nums text-white/82 ring-1 ring-white/10 backdrop-blur-md">
                    #{case_.metrics.rank}
                  </span>
                </div>

                {/* ── Bottom overlay metadata — image-first caption. ── */}
                <figcaption className="absolute inset-x-0 bottom-0 px-4 pb-3.5 pt-6">
                  <div className="flex items-baseline justify-between gap-3">
                    <p className="truncate text-[14px] font-semibold leading-tight tracking-tight text-white drop-shadow-[0_1px_2px_rgb(0_0_0_/_0.65)]">
                      NSD&nbsp;<span className="font-mono tabular-nums">{case_.nsdId}</span>
                    </p>
                    <span className="shrink-0 font-mono text-[10.5px] tabular-nums text-white/65">
                      κ&nbsp;{case_.uncertainty.kappa.toFixed(0)}
                    </span>
                  </div>
                  <p className="mt-1 truncate text-[10px] uppercase tracking-[0.18em] text-white/55">
                    {case_.subject} · session {case_.session}
                  </p>
                </figcaption>

                {/* ── Exact-match accent: soft inset success ring + bottom
                       hairline rule. Calibration-light, not glow. ── */}
                {exact ? (
                  <>
                    <div
                      className="pointer-events-none absolute inset-0 rounded-[14px] ring-1 ring-inset ring-status-success/18"
                      aria-hidden
                    />
                    <div
                      className="pointer-events-none absolute inset-x-6 bottom-0 h-px bg-gradient-to-r from-transparent via-status-success/60 to-transparent"
                      aria-hidden
                    />
                  </>
                ) : null}
              </figure>
            </motion.button>
          );
        })}
      </div>

      {filtered.length === 0 && (
        <EmptyState
          title="No trials match this filter"
          detail="The dataset is still available; clear filters to return to the full curated trial gallery."
          action={
            <button
              type="button"
              onClick={() => {
                setFilter('all');
                setDefenseOnly(false);
              }}
              className="premium-button-secondary px-4 py-2 text-xs"
            >
              Clear filters
            </button>
          }
          className="mt-8"
        />
      )}
    </div>
  );
}
