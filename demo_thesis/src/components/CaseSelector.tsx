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
  best: 'border-emerald-400/55 bg-emerald-500/20 text-emerald-100 shadow-[0_0_12px_rgba(16,185,129,0.25)]',
  medium: 'border-amber-400/55 bg-amber-500/20 text-amber-100 shadow-[0_0_12px_rgba(245,158,11,0.22)]',
  hard: 'border-red-400/55 bg-red-500/18 text-red-100 shadow-[0_0_12px_rgba(239,68,68,0.2)]',
  mystery: 'border-violet-400/55 bg-violet-500/20 text-violet-100 shadow-[0_0_12px_rgba(139,92,246,0.22)]',
};

const DIFFICULTY_DOT: Record<string, string> = {
  best: 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.75)]',
  medium: 'bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.7)]',
  hard: 'bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.65)]',
  mystery: 'bg-violet-400 shadow-[0_0_8px_rgba(167,139,250,0.7)]',
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
      <div>
        <label className="mb-2 block text-xs font-medium uppercase tracking-wider text-slate-500">
          Difficulty
        </label>
        <div className="flex flex-wrap gap-1.5">
          <button
            type="button"
            onClick={() => setFilter('all')}
            className={`rounded-md px-2.5 py-1 text-xs font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brain-accent ${
              filter === 'all' ? 'bg-white/10 text-white' : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            All
          </button>
          {DIFFICULTIES.map((d) => (
            <button
              key={d}
              type="button"
              onClick={() => setFilter(d)}
              className={`rounded-md border px-2.5 py-1 text-xs font-medium capitalize transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brain-accent ${
                filter === d
                  ? DIFFICULTY_SELECTED[d]
                  : 'border-transparent text-slate-500 hover:border-slate-600/80 hover:text-slate-300'
              }`}
            >
              {d}
            </button>
          ))}
        </div>
      </div>
      <div>
        <label className="mb-2 block text-xs font-medium uppercase tracking-wider text-slate-500">
          Subject
        </label>
        <div className="flex flex-wrap gap-1.5">
          <button
            type="button"
            onClick={() => setSubjectFilter('all')}
            className={`rounded-md px-2.5 py-1 text-xs font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brain-accent ${
              subjectFilter === 'all' ? 'bg-white/10 text-white' : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            All
          </button>
          {subjects.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setSubjectFilter(s)}
              className={`rounded-md px-2.5 py-1 text-xs font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brain-accent ${
                subjectFilter === s
                  ? 'bg-brain-accent/10 text-brain-accent'
                  : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
      <div className="max-h-[50vh] space-y-1.5 overflow-y-auto pr-1">
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
              className={`w-full rounded-lg border px-3 py-2.5 text-left text-sm transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brain-accent ${
                selectedId === c.id
                  ? 'border-brain-accent/30 bg-brain-accent/10 text-white'
                  : 'border-transparent text-slate-400 hover:bg-white/5 hover:text-slate-200'
              }`}
            >
              <div className="mb-0.5 flex items-center gap-2">
                <span
                  className={`h-2 w-2 flex-shrink-0 rounded-full ring-2 ring-slate-950 ${DIFFICULTY_DOT[c.difficulty] ?? 'bg-slate-400'}`}
                  aria-hidden
                />
                <span className="truncate text-xs font-medium">{c.title}</span>
              </div>
              <div className="ml-4 flex items-center gap-3 text-xs text-slate-500">
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
