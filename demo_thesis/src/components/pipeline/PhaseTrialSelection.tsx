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
          // Refined status tones — dimmer, more elegant, no full bg fill
          const statusTone = exact
            ? 'border-status-success/22 bg-black/35 text-status-success/95'
            : top5
            ? 'border-status-warning/22 bg-black/35 text-status-warning/95'
            : 'border-white/[0.08] bg-black/35 text-white/75';
          const statusDot = exact
            ? 'bg-status-success'
            : top5
            ? 'bg-status-warning'
            : 'bg-white/60';
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
              <div
                className={`relative overflow-hidden rounded-[14px] border bg-surface-elevated/80 transition duration-300 group-hover:-translate-y-0.5 group-hover:border-white/[0.12] ${
                  exact ? 'border-status-success/18' : 'border-white/[0.05]'
                }`}
                style={{ boxShadow: '0 1px 0 rgb(255 255 255 / 0.022) inset' }}
              >
                {/* Image hero — full-bleed, hair-thin inset ring softens edge */}
                <div className="relative aspect-[4/3] overflow-hidden">
                  <img
                    src={case_.targetImage}
                    alt={`NSD stimulus ${case_.nsdId}`}
                    className="h-full w-full object-cover transition duration-500 group-hover:scale-[1.025]"
                    loading="lazy"
                  />
                  <div className="pointer-events-none absolute inset-0 ring-1 ring-inset ring-white/[0.04]" aria-hidden />

                  {/* Status badge — dot + tight label, smaller, dimmer */}
                  <div className="absolute left-2.5 top-2.5">
                    <span
                      className={`inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[9.5px] font-semibold tracking-tight backdrop-blur-md ${statusTone}`}
                    >
                      <span className={`h-1 w-1 shrink-0 rounded-full ${statusDot}`} />
                      {statusLabel(case_)}
                    </span>
                  </div>

                  {/* Rank chip — quiet mono in opposite corner, replaces the
                      "Rank #N" duplicate in the footer */}
                  <div className="absolute right-2.5 top-2.5">
                    <span className="inline-flex items-center rounded-md border border-white/[0.08] bg-black/40 px-1.5 py-0.5 font-mono text-[10px] font-semibold tabular-nums text-white/85 backdrop-blur-md">
                      #{case_.metrics.rank}
                    </span>
                  </div>
                </div>

                {/* Footer — identity-first, secondary metadata quieter */}
                <div className="px-3.5 pb-3 pt-2.5">
                  <p className="text-[13.5px] font-semibold tracking-tight text-text-primary">
                    NSD {case_.nsdId}
                  </p>
                  <div className="mt-1 flex items-baseline justify-between gap-2 text-[11px]">
                    <span className="text-text-muted">
                      {case_.subject} · session {case_.session}
                    </span>
                    <span className="font-mono tabular-nums text-text-muted">
                      κ {case_.uncertainty.kappa.toFixed(0)}
                    </span>
                  </div>
                </div>

                {/* Exact-match accent — a single hair-thin bottom rule */}
                {exact ? (
                  <div
                    className="pointer-events-none absolute inset-x-4 bottom-0 h-px bg-gradient-to-r from-transparent via-status-success/55 to-transparent"
                    aria-hidden
                  />
                ) : null}
              </div>
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
