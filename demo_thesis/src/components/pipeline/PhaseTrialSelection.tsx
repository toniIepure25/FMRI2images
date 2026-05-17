import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import type { DemoCase } from '@/types';
import { isDefenseReady } from '@/lib/pipelineNormalize';
import { SectionHeader } from '@/components/premium/SectionHeader';
import { ImageEvidenceCard } from '@/components/premium/ImageEvidenceCard';
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
  if (isDefenseReady(c)) return 'Defense-ready';
  if (c.metrics.r1Correct) return 'Exact match';
  if (c.metrics.r5Correct) return 'Top-5 match';
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
        className={`rounded-xl border px-3 py-2 text-[12px] font-semibold transition ${
          defenseOnly
            ? 'border-accent/25 bg-accent/10 text-accent'
            : 'border-border-subtle bg-surface-elevated text-text-secondary hover:border-border-emphasis hover:text-text-primary'
        }`}
      >
        Defense set <span className="ml-1 font-mono text-[10px] opacity-70">{defenseCount}</span>
      </button>
      <div className="flex flex-wrap items-center gap-1 rounded-xl border border-border-subtle bg-surface-raised p-1">
        {['all', 'best', 'medium', 'hard'].map((f) => (
          <button
            key={f}
            type="button"
            onClick={() => setFilter(f)}
            className={`rounded-lg px-3 py-1.5 text-[11px] font-semibold transition ${
              filter === f ? 'bg-surface-overlay text-text-primary shadow-surface' : 'text-text-muted hover:text-text-primary'
            }`}
          >
            {f === 'all' ? 'All' : diffLabel(f as DemoCase['difficulty'])}
            <span className="ml-1 font-mono text-[9px] opacity-55">{counts[f] || 0}</span>
          </button>
        ))}
      </div>
    </div>
  );

  return (
    <div className="pb-14">
      <SectionHeader
        eyebrow="Stimulus gallery"
        title="Select an NSD visual trial"
        description="Choose a recorded stimulus and replay its subject-specific fMRI → CLIP retrieval path."
        action={filterAction}
        className="mb-8"
      />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
        {filtered.map((case_, idx) => {
          const exact = case_.metrics.rank === 1;
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
              className="group text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              onClick={() => onSelect(case_)}
            >
              <ImageEvidenceCard
                imageSrc={case_.targetImage}
                title={`nsdId ${case_.nsdId}`}
                subtitle={`${case_.subject} · session ${case_.session}`}
                emphasis={exact}
                badge={
                  <span className={`premium-rank-badge ${exact ? 'premium-rank-badge-primary' : ''}`}>
                    {statusLabel(case_)}
                  </span>
                }
                footer={
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-[12px] font-semibold text-text-primary">{case_.subject} · session {case_.session}</p>
                      <p className="mt-0.5 text-[11px] text-text-muted">Rank #{case_.metrics.rank} · {exact ? 'exact match' : diffLabel(case_.difficulty)}</p>
                    </div>
                    <span className="rounded-lg border border-border-subtle bg-surface-raised px-2.5 py-1 font-mono text-[11px] text-text-muted">
                      κ {case_.uncertainty.kappa.toFixed(0)}
                    </span>
                  </div>
                }
              />
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
