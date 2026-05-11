import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { DemoCase } from '@/types';

interface CaseSelectorProps {
  cases: DemoCase[];
  selectedId?: string;
  onSelect: (c: DemoCase) => void;
}

const DIFFICULTIES = ['best', 'medium', 'hard', 'mystery'] as const;

const DIFFICULTY_SELECTED: Record<(typeof DIFFICULTIES)[number], string> = {
  best:    'border-emerald-500/30 bg-emerald-500/12 text-emerald-300',
  medium:  'border-amber-500/30 bg-amber-500/12 text-amber-300',
  hard:    'border-red-500/30 bg-red-500/12 text-red-300',
  mystery: 'border-violet-500/30 bg-violet-500/12 text-violet-300',
};

const DIFFICULTY_DOT: Record<string, string> = {
  best:    'bg-emerald-400 ring-1 ring-emerald-400/30',
  medium:  'bg-amber-400 ring-1 ring-amber-400/30',
  hard:    'bg-red-400 ring-1 ring-red-400/30',
  mystery: 'bg-violet-400 ring-1 ring-violet-400/30',
};

export function CaseSelector({ cases, selectedId, onSelect }: CaseSelectorProps) {
  const [filter, setFilter] = useState<string>('all');
  const [subjectFilter, setSubjectFilter] = useState<string>('all');

  const subjects = useMemo(() => [...new Set(cases.map((c) => c.subject))], [cases]);
  const filtered = useMemo(
    () =>
      cases.filter(
        (c) =>
          (filter === 'all' || c.difficulty === filter) &&
          (subjectFilter === 'all' || c.subject === subjectFilter),
      ),
    [cases, filter, subjectFilter],
  );

  return (
    <div className="space-y-4">
      {/* Difficulty filter */}
      <div>
        <label className="mb-2 block text-2xs font-semibold uppercase tracking-wider text-text-muted">
          Difficulty
        </label>
        <div className="flex flex-wrap gap-1.5">
          <button
            type="button"
            onClick={() => setFilter('all')}
            className={`rounded-md px-2 py-1 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
              filter === 'all'
                ? 'bg-surface-active text-text-primary'
                : 'text-text-muted hover:bg-surface-raised hover:text-text-secondary'
            }`}
          >
            All
          </button>
          {DIFFICULTIES.map((d) => (
            <button
              key={d}
              type="button"
              onClick={() => setFilter(d)}
              className={`rounded-md border px-2 py-1 text-[11px] font-medium capitalize transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                filter === d
                  ? DIFFICULTY_SELECTED[d]
                  : 'border-transparent text-text-muted hover:border-border-subtle hover:text-text-secondary'
              }`}
            >
              {d}
            </button>
          ))}
        </div>
      </div>

      {/* Subject filter */}
      <div>
        <label className="mb-2 block text-2xs font-semibold uppercase tracking-wider text-text-muted">
          Subject
        </label>
        <div className="flex flex-wrap gap-1.5">
          <button
            type="button"
            onClick={() => setSubjectFilter('all')}
            className={`rounded-md px-2 py-1 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
              subjectFilter === 'all'
                ? 'bg-surface-active text-text-primary'
                : 'text-text-muted hover:bg-surface-raised hover:text-text-secondary'
            }`}
          >
            All
          </button>
          {subjects.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setSubjectFilter(s)}
              className={`rounded-md px-2 py-1 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                subjectFilter === s
                  ? 'bg-accent/10 text-accent ring-1 ring-accent/20'
                  : 'text-text-muted hover:bg-surface-raised hover:text-text-secondary'
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* Case list */}
      <div className="max-h-[50vh] space-y-0.5 overflow-y-auto pr-1">
        <AnimatePresence>
          {filtered.map((c) => (
            <motion.button
              key={c.id}
              layout
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              type="button"
              onClick={() => onSelect(c)}
              className={`w-full rounded-lg border px-3 py-2.5 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                selectedId === c.id
                  ? 'border-accent/25 bg-accent/5 text-text-primary'
                  : 'border-transparent text-text-secondary hover:bg-surface-raised hover:text-text-primary'
              }`}
            >
              <div className="mb-0.5 flex items-center gap-2">
                <span
                  className={`h-1.5 w-1.5 flex-shrink-0 rounded-full ${DIFFICULTY_DOT[c.difficulty] ?? 'bg-slate-500'}`}
                  aria-hidden
                />
                <span className="truncate text-xs font-medium">{c.title}</span>
              </div>
              <div className="ml-3.5 flex items-center gap-3 text-[11px] text-text-muted">
                <span>{c.id}</span>
                <span>{c.subject}</span>
                <span className="capitalize">{c.difficulty}</span>
              </div>
            </motion.button>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
