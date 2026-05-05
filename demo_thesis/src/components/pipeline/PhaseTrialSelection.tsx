import { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { DemoCase } from '@/types';
import { getDifficultyColor } from '@/lib/data';

export interface PhaseTrialSelectionProps {
  cases: DemoCase[];
  onSelect: (case_: DemoCase) => void;
}

const DIFF_ORDER: Record<DemoCase['difficulty'], number> = { best: 0, medium: 1, hard: 2, mystery: 3 };

function sortCases(cases: DemoCase[]): DemoCase[] {
  return [...cases].sort((a, b) => {
    const d = DIFF_ORDER[a.difficulty] - DIFF_ORDER[b.difficulty];
    return d !== 0 ? d : a.nsdId - b.nsdId;
  });
}

function preview64(c: DemoCase): number[] {
  const p = c.fmriPreview;
  if (p.length >= 64) return p.slice(0, 64);
  return [...p, ...Array.from({ length: 64 - p.length }, () => 0)];
}

function normalize(values: number[]): number[] {
  let min = Infinity, max = -Infinity;
  for (const v of values) { min = Math.min(min, v); max = Math.max(max, v); }
  const span = max - min || 1;
  return values.map((v) => ((v - min) / span) * 100);
}

function diffLabel(d: DemoCase['difficulty']): string {
  return d === 'best' ? 'Perfect' : d === 'medium' ? 'Good' : d === 'hard' ? 'Hard' : 'Mystery';
}

function SafeImg({ src, alt, className }: { src?: string; alt: string; className?: string }) {
  const [ok, setOk] = useState(true);
  if (!src || !ok) {
    return (
      <div className={`flex items-center justify-center bg-gradient-to-br from-slate-900 to-slate-800 ${className ?? ''}`}>
        <span className="text-[10px] text-slate-600">No preview</span>
      </div>
    );
  }
  return <img src={src} alt={alt} className={className} onError={() => setOk(false)} loading="lazy" />;
}

export function PhaseTrialSelection({ cases, onSelect }: PhaseTrialSelectionProps) {
  const sorted = useMemo(() => sortCases(cases), [cases]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>('all');

  const filtered = useMemo(() => {
    if (filter === 'all') return sorted;
    return sorted.filter((c) => c.difficulty === filter);
  }, [sorted, filter]);

  const counts = useMemo(() => {
    const m: Record<string, number> = { all: sorted.length };
    for (const c of sorted) m[c.difficulty] = (m[c.difficulty] || 0) + 1;
    return m;
  }, [sorted]);

  return (
    <div className="relative pb-28">
      {/* Header */}
      <motion.div
        className="mb-6 rounded-2xl border border-white/[0.06] bg-gradient-to-r from-slate-900/80 via-brain-navy/60 to-slate-900/80 p-5 backdrop-blur-xl sm:p-6"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-base font-semibold tracking-tight text-white sm:text-lg">
              Select an fMRI Trial
            </h2>
            <p className="mt-1 text-xs leading-relaxed text-slate-400 sm:text-sm">
              Each card represents a real NSD trial with its cortical response pattern and
              the stimulus image the subject perceived during scanning.
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-1.5 rounded-xl bg-black/30 p-1 ring-1 ring-white/[0.06]">
            {['all', 'best', 'medium', 'hard'].map((f) => (
              <button
                key={f}
                type="button"
                onClick={() => setFilter(f)}
                className={`rounded-lg px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider transition-all ${
                  filter === f
                    ? 'bg-brain-accent/15 text-brain-accent ring-1 ring-brain-accent/30'
                    : 'text-slate-500 hover:text-slate-300'
                }`}
              >
                {f === 'all' ? 'All' : diffLabel(f as DemoCase['difficulty'])}
                <span className="ml-1 text-[9px] opacity-60">{counts[f] || 0}</span>
              </button>
            ))}
          </div>
        </div>
      </motion.div>

      {/* Grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {filtered.map((case_, idx) => {
          const isSel = selectedId === case_.id;
          const heights = normalize(preview64(case_));
          const diffColor = getDifficultyColor(case_.difficulty);

          return (
            <motion.div
              key={case_.id}
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: selectedId && !isSel ? 0.4 : 1, y: 0 }}
              transition={{ delay: Math.min(idx * 0.03, 0.6), duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
              className={`group relative cursor-pointer overflow-hidden rounded-2xl border bg-gradient-to-b from-slate-900/90 to-brain-dark/95 backdrop-blur transition-all duration-300 ${
                isSel
                  ? 'border-brain-accent/50 shadow-[0_0_30px_rgba(0,212,255,0.15)] ring-1 ring-brain-accent/20'
                  : 'border-white/[0.06] hover:border-white/[0.12] hover:shadow-lg'
              }`}
              onClick={() => setSelectedId(case_.id)}
            >
              {/* Stimulus image */}
              <div className="relative aspect-[4/3] w-full overflow-hidden">
                <SafeImg
                  src={case_.targetImage}
                  alt={`Stimulus nsdId ${case_.nsdId}`}
                  className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-slate-900 via-transparent to-transparent" />

                {/* Badges */}
                <div className="absolute left-2.5 top-2.5 flex items-center gap-1.5">
                  <span
                    className="rounded-md px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider text-white shadow-lg"
                    style={{ backgroundColor: `${diffColor}dd` }}
                  >
                    {diffLabel(case_.difficulty)}
                  </span>
                </div>
                <div className="absolute right-2.5 top-2.5">
                  <span className="rounded-md bg-black/60 px-2 py-0.5 font-mono text-[9px] font-semibold text-slate-200 shadow-lg backdrop-blur-sm ring-1 ring-white/10">
                    {case_.subject}
                  </span>
                </div>

                {/* Bottom overlay with nsdId */}
                <div className="absolute bottom-0 left-0 right-0 flex items-end justify-between px-3 pb-2.5">
                  <span className="font-mono text-[10px] text-white/80">
                    nsdId {case_.nsdId}
                  </span>
                  {case_.metrics.r1Correct && (
                    <span className="rounded-full bg-emerald-500/20 px-2 py-0.5 text-[8px] font-bold text-emerald-300 ring-1 ring-emerald-500/30">
                      R@1 ✓
                    </span>
                  )}
                </div>
              </div>

              {/* Card body */}
              <div className="p-3.5">
                {/* fMRI preview */}
                <div className="mb-3">
                  <p className="mb-1.5 text-[8px] font-bold uppercase tracking-[0.2em] text-slate-500">
                    fMRI voxel activity
                  </p>
                  <div className="flex h-8 items-end gap-[1px] rounded-md bg-black/40 px-0.5 py-0.5 ring-1 ring-white/[0.04]">
                    {heights.map((h, i) => (
                      <motion.div
                        key={i}
                        className="min-w-0 flex-1 rounded-[1px] bg-gradient-to-t from-cyan-700/80 to-brain-accent/60"
                        initial={{ height: '0%' }}
                        animate={{ height: `${Math.max(6, h)}%` }}
                        transition={{ delay: i * 0.005, duration: 0.3 }}
                        style={{ minHeight: 2 }}
                      />
                    ))}
                  </div>
                </div>

                {/* Action button */}
                <button
                  type="button"
                  className={`w-full rounded-xl py-2 text-center text-[11px] font-bold uppercase tracking-wider transition-all duration-200 ${
                    isSel
                      ? 'bg-brain-accent/20 text-brain-accent ring-1 ring-brain-accent/40'
                      : 'bg-white/[0.03] text-slate-400 ring-1 ring-white/[0.06] hover:bg-brain-accent/10 hover:text-brain-accent'
                  }`}
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedId(case_.id);
                    onSelect(case_);
                  }}
                >
                  {isSel ? '● Selected' : 'Decode this trial'}
                </button>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Floating CTA */}
      <AnimatePresence>
        {selectedId && (
          <motion.div
            className="pointer-events-none fixed inset-x-0 bottom-0 z-40 flex justify-center pb-6"
            initial={{ opacity: 0, y: 36 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 24 }}
          >
            <div className="pointer-events-auto">
              <button
                type="button"
                className="flex items-center gap-3 rounded-2xl border border-brain-accent/40 bg-brain-navy/95 px-8 py-4 text-sm font-bold text-brain-accent shadow-[0_8px_40px_rgba(0,212,255,0.2)] backdrop-blur-xl transition-all hover:border-brain-accent/60 hover:shadow-[0_8px_50px_rgba(0,212,255,0.3)]"
                onClick={() => {
                  const c = sorted.find((x) => x.id === selectedId);
                  if (c) onSelect(c);
                }}
              >
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-brain-accent/20 text-xs">▶</span>
                Start Decoding Pipeline
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
