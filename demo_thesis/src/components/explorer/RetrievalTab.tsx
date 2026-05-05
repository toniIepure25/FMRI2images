import { useMemo } from 'react';
import type { DemoCase, RetrievedImage } from '@/types';
import { SafeImg } from '@/components/ui/SafeImg';

export interface RetrievalTabProps {
  case_: DemoCase;
}

const LABEL_STYLES: Record<RetrievedImage['label'], { text: string; className: string }> = {
  correct: { text: 'Correct', className: 'border-emerald-500/40 bg-emerald-500/20 text-emerald-300' },
  semantic_neighbor: {
    text: 'Semantic neighbor',
    className: 'border-violet-500/35 bg-violet-500/20 text-violet-200',
  },
  distractor: { text: 'Distractor', className: 'border-slate-500/35 bg-slate-500/20 text-slate-300' },
};

export function RetrievalTab({ case_ }: RetrievalTabProps) {
  const top5 = useMemo(() => {
    return [...case_.retrievedImages].sort((a, b) => a.rank - b.rank).slice(0, 5);
  }, [case_.retrievedImages]);

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div className="glass-panel p-5">
        <h2 className="text-sm font-semibold text-white">Retrieval gallery</h2>
        <p className="mt-2 text-xs leading-relaxed text-slate-400">
          Even when top-1 is wrong, top-<span className="font-mono text-slate-300">k</span> often reveals semantic proximity in
          CLIP space—nearby items share category or visual structure despite pixel differences.
        </p>
      </div>

      <div className="flex flex-col gap-6 lg:flex-row lg:items-stretch">
        <div className="glass-panel flex w-full flex-shrink-0 flex-col p-4 lg:w-[320px]">
          <p className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-slate-500">Target stimulus</p>
          <div className="relative flex-1 overflow-hidden rounded-xl border-2 border-brain-accent/35 bg-black/50 shadow-[0_0_32px_rgba(0,212,255,0.08)]">
            <SafeImg
              src={case_.targetImage}
              alt="Target"
              className="block h-full min-h-[240px] w-full object-cover lg:min-h-[280px]"
            />
          </div>
          <p className="mt-3 text-center text-[11px] text-slate-500">
            Ground-truth image for trial <span className="font-mono text-slate-400">{case_.id}</span>
          </p>
        </div>

        <div className="min-w-0 flex-1 space-y-4">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Top-5 candidates</p>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
            {top5.map((item) => {
              const isCorrect = item.label === 'correct';
              const ls = LABEL_STYLES[item.label];
              return (
                <div
                  key={item.rank}
                  className={`glass-panel-hover flex min-w-0 flex-col overflow-hidden p-3 ${
                    isCorrect ? 'ring-2 ring-emerald-400/80 ring-offset-2 ring-offset-[#050914]' : ''
                  }`}
                >
                  <div className="relative aspect-square overflow-hidden rounded-lg border border-brain-border/40 bg-black/40">
                    <SafeImg src={item.image} alt={`Rank ${item.rank}`} className="block h-full w-full object-cover" />
                    <div className="absolute left-2 top-2 rounded-md border border-white/20 bg-black/70 px-2 py-0.5 font-mono text-[11px] font-bold text-white backdrop-blur-sm">
                      #{item.rank}
                    </div>
                  </div>
                  <div className="mt-3 space-y-2">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="rounded-md border border-white/10 bg-white/5 px-2 py-0.5 font-mono text-[11px] text-cyan-200">
                        cos {item.score.toFixed(3)}
                      </span>
                      <span className="rounded-md border border-white/10 bg-white/5 px-2 py-0.5 font-mono text-[11px] text-slate-300">
                        CSLS {item.csls.toFixed(3)}
                      </span>
                    </div>
                    <span className={`inline-flex rounded-md border px-2 py-0.5 text-[11px] font-medium ${ls.className}`}>
                      {ls.text}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
