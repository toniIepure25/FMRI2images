import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { DemoCase } from '@/types';
import { getDifficultyColor } from '@/lib/data';

interface CaseSelectorProps {
  cases: DemoCase[];
  selectedId?: string;
  onSelect: (c: DemoCase) => void;
}

const DIFFICULTIES = ['best', 'medium', 'hard', 'mystery'] as const;

export function CaseSelector({ cases, selectedId, onSelect }: CaseSelectorProps) {
  const [filter, setFilter] = useState<string>('all');
  const [subjectFilter, setSubjectFilter] = useState<string>('all');

  const subjects = useMemo(() => [...new Set(cases.map(c => c.subject))], [cases]);
  const filtered = useMemo(() =>
    cases.filter(c =>
      (filter === 'all' || c.difficulty === filter) &&
      (subjectFilter === 'all' || c.subject === subjectFilter)
    ), [cases, filter, subjectFilter]);

  return (
    <div className="space-y-4">
      <div>
        <label className="text-xs font-medium uppercase tracking-wider text-gray-500 mb-2 block">Difficulty</label>
        <div className="flex flex-wrap gap-1.5">
          <button
            type="button"
            onClick={() => setFilter('all')}
            className={`px-2.5 py-1 rounded-md text-xs font-medium transition-all ${filter === 'all' ? 'bg-white/10 text-white' : 'text-gray-500 hover:text-gray-300'}`}
          >
            All
          </button>
          {DIFFICULTIES.map(d => (
            <button
              key={d}
              type="button"
              onClick={() => setFilter(d)}
              className={`px-2.5 py-1 rounded-md text-xs font-medium transition-all capitalize ${filter === d ? 'text-white' : 'text-gray-500 hover:text-gray-300'}`}
              style={filter === d ? { backgroundColor: getDifficultyColor(d) + '22', color: getDifficultyColor(d) } : {}}
            >
              {d}
            </button>
          ))}
        </div>
      </div>
      <div>
        <label className="text-xs font-medium uppercase tracking-wider text-gray-500 mb-2 block">Subject</label>
        <div className="flex flex-wrap gap-1.5">
          <button
            type="button"
            onClick={() => setSubjectFilter('all')}
            className={`px-2.5 py-1 rounded-md text-xs font-medium transition-all ${subjectFilter === 'all' ? 'bg-white/10 text-white' : 'text-gray-500 hover:text-gray-300'}`}
          >
            All
          </button>
          {subjects.map(s => (
            <button
              key={s}
              type="button"
              onClick={() => setSubjectFilter(s)}
              className={`px-2.5 py-1 rounded-md text-xs font-medium transition-all ${subjectFilter === s ? 'bg-brain-accent/10 text-brain-accent' : 'text-gray-500 hover:text-gray-300'}`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
      <div className="space-y-1.5 max-h-[50vh] overflow-y-auto pr-1">
        <AnimatePresence>
          {filtered.map(c => (
            <motion.button
              key={c.id}
              layout
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              type="button"
              onClick={() => onSelect(c)}
              className={`w-full text-left px-3 py-2.5 rounded-lg transition-all text-sm ${
                selectedId === c.id
                  ? 'bg-brain-accent/10 border border-brain-accent/30 text-white'
                  : 'hover:bg-white/5 text-gray-400 hover:text-gray-200 border border-transparent'
              }`}
            >
              <div className="flex items-center gap-2 mb-0.5">
                <span
                  className="w-2 h-2 rounded-full flex-shrink-0"
                  style={{ backgroundColor: getDifficultyColor(c.difficulty) }}
                />
                <span className="font-medium text-xs truncate">{c.title}</span>
              </div>
              <div className="flex items-center gap-3 ml-4 text-xs text-gray-600">
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
