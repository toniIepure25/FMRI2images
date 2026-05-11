import { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { DemoCase } from '@/types';
import { getDifficultyColor } from '@/lib/data';
import { isDefenseReady } from '@/lib/pipelineNormalize';

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

const DIFF_BADGE: Record<DemoCase['difficulty'], string> = {
  best:    'bg-status-success/10 text-status-success ring-1 ring-status-success/20',
  medium:  'bg-status-warning/8 text-status-warning ring-1 ring-status-warning/15',
  hard:    'bg-status-error/6 text-status-error ring-1 ring-status-error/15',
  mystery: 'bg-accent/8 text-accent ring-1 ring-accent/15',
};

function SafeImg({ src, alt, className }: { src?: string; alt: string; className?: string }) {
  const [ok, setOk] = useState(true);
  if (!src || !ok) {
    return (
      <div className={`flex items-center justify-center bg-surface-raised ${className ?? ''}`}>
        <svg className="h-6 w-6 text-border-emphasis" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M6.75 7.5a.75.75 0 11-1.5 0 .75.75 0 011.5 0z" />
        </svg>
      </div>
    );
  }
  return <img src={src} alt={alt} className={className} onError={() => setOk(false)} loading="lazy" />;
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

  return (
    <div className="relative pb-12">
      {/* Header */}
      <motion.div className="mb-8" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h2 className="text-[24px] font-semibold tracking-tight text-text-primary">
              Select a trial
            </h2>
            <p className="mt-1.5 max-w-lg text-[13px] leading-relaxed text-text-muted">
              Each card is a Natural Scenes Dataset trial &mdash; the subject viewed this stimulus while brain activity was recorded.
            </p>
          </div>

          {/* Filters */}
          <div className="flex shrink-0 items-center gap-2">
            <button
              type="button"
              onClick={() => setDefenseOnly((v) => !v)}
              className={`rounded-full px-3 py-1.5 text-[11px] font-medium transition-colors ${
                defenseOnly
                  ? 'bg-accent/10 text-accent ring-1 ring-accent/20'
                  : 'text-text-muted ring-1 ring-border-subtle hover:text-text-secondary hover:ring-border-emphasis'
              }`}
            >
              Defense set
              <span className="ml-1.5 font-mono text-[10px] opacity-60">{defenseCount}</span>
            </button>

            <div className="flex items-center gap-0.5 rounded-full bg-surface-raised p-0.5 ring-1 ring-border-subtle">
              {['all', 'best', 'medium', 'hard'].map((f) => (
                <button
                  key={f}
                  type="button"
                  onClick={() => setFilter(f)}
                  className={`rounded-full px-3 py-1.5 text-[11px] font-medium transition-colors ${
                    filter === f
                      ? 'bg-surface-elevated text-text-primary shadow-[0_1px_3px_0_rgb(0,0,0,0.2)]'
                      : 'text-text-muted hover:text-text-secondary'
                  }`}
                >
                  {f === 'all' ? 'All' : diffLabel(f as DemoCase['difficulty'])}
                  <span className="ml-1 font-mono text-[9px] opacity-50">{counts[f] || 0}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </motion.div>

      {/* Trial grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {filtered.map((case_, idx) => {
          const tier = integrityTier(case_);

          return (
            <motion.div
              key={case_.id}
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                delay: Math.min(idx * 0.025, 0.5),
                duration: 0.35,
                ease: [0.22, 1, 0.36, 1],
              }}
              className="group relative cursor-pointer overflow-hidden rounded-xl border border-border-subtle bg-surface-elevated transition-all duration-200 hover:border-border-emphasis hover:shadow-surface-lg"
              onClick={() => onSelect(case_)}
            >
              {/* Image */}
              <div className="relative aspect-[4/3] w-full overflow-hidden">
                <SafeImg
                  src={case_.targetImage}
                  alt={`Stimulus nsdId ${case_.nsdId}`}
                  className="h-full w-full object-cover transition-transform duration-500 ease-out group-hover:scale-[1.03]"
                />
                {/* Bottom gradient for text legibility */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/5 to-transparent" />

                {/* Top-left: difficulty badge */}
                <div className="absolute left-2 top-2">
                  <span className={`rounded px-1.5 py-0.5 text-[9px] font-medium ${DIFF_BADGE[case_.difficulty]}`}>
                    {diffLabel(case_.difficulty)}
                  </span>
                </div>

                {/* Bottom overlay */}
                <div className="absolute bottom-0 left-0 right-0 flex items-end justify-between px-2.5 pb-2.5">
                  <span className="font-mono text-[11px] font-medium text-white/85">
                    nsdId {case_.nsdId}
                  </span>
                  <div className="flex items-center gap-2">
                    {case_.metrics.r1Correct && (
                      <span className="flex items-center gap-1 rounded bg-accent/10 px-1.5 py-[2px] text-[8px] font-semibold text-accent ring-1 ring-accent/15">
                        R@1
                      </span>
                    )}
                    <span className="rounded bg-black/30 px-1.5 py-[2px] font-mono text-[9px] text-text-secondary">
                      {case_.subject}
                    </span>
                  </div>
                </div>
              </div>

              {/* Bottom strip: tier indicator */}
              {tier === 'defense' && (
                <div className="flex items-center gap-1.5 px-3 py-1.5 text-[9px] font-medium text-text-muted">
                  <span className="h-1 w-1 rounded-full bg-status-success" />
                  Defense-ready
                </div>
              )}
              {tier !== 'defense' && (
                <div className="h-[3px] w-full bg-border-subtle transition-colors group-hover:bg-border-emphasis" />
              )}
            </motion.div>
          );
        })}
      </div>

      {/* Empty state */}
      {filtered.length === 0 && (
        <motion.div
          className="flex flex-col items-center justify-center gap-3 py-20"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <svg className="h-10 w-10 text-border-emphasis" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
          </svg>
          <p className="text-[13px] text-text-muted">No trials match this filter</p>
          <button
            type="button"
            onClick={() => { setFilter('all'); setDefenseOnly(false); }}
            className="text-[12px] font-medium text-accent transition hover:text-accent-hover"
          >
            Clear filters
          </button>
        </motion.div>
      )}
    </div>
  );
}
