import { motion } from 'framer-motion';
import type { BackendHealth } from '@/lib/api';
import type { DemoCase } from '@/types';
import type { PipelineRunMode } from '@/types/pipeline';

interface PipelineStatusHeaderProps {
  runMode: PipelineRunMode;
  backendHealth: BackendHealth | null;
  selectedCase: DemoCase | null;
  phase: string;
}

const MODE_LABELS: Record<PipelineRunMode, { label: string; detail: string }> =
  {
    checking: {
      label: 'Checking',
      detail: 'Verifying backend availability',
    },
    live: { label: 'Live inference', detail: 'Live model inference active' },
    replay: { label: 'Replay', detail: 'Backend online, replay assets active' },
    'offline-replay': {
      label: 'Offline replay',
      detail: 'Backend unavailable',
    },
    hybrid: {
      label: 'Hybrid',
      detail: 'Live retrieval + cached reconstruction',
    },
    error: { label: 'Error', detail: 'Connection failed' },
  };

const MODE_DOT: Record<PipelineRunMode, string> = {
  checking: 'bg-slate-400',
  live: 'bg-emerald-400',
  replay: 'bg-cyan-400',
  'offline-replay': 'bg-amber-400',
  hybrid: 'bg-blue-400',
  error: 'bg-red-400',
};

export function PipelineStatusHeader({
  runMode,
  backendHealth,
  selectedCase,
}: PipelineStatusHeaderProps) {
  const mode = MODE_LABELS[runMode];
  const dotStyle = MODE_DOT[runMode];

  return (
    <motion.header
      className="mb-2"
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h1 className="text-[28px] font-semibold tracking-tight text-white">
            Cortex2Canvas Pipeline
          </h1>
          <p className="mt-1 text-[13px] text-slate-500">
            fMRI activity → live CLIP retrieval → cached qualitative reconstruction
          </p>
        </div>

        {/* Status chip — compact */}
        <span
          className="inline-flex shrink-0 items-center gap-1.5 rounded-full bg-white/[0.04] px-3 py-1.5 text-[11px] font-medium text-slate-400 ring-1 ring-white/[0.06]"
          title={mode.detail}
        >
          <motion.span
            className={`inline-block h-1.5 w-1.5 rounded-full ${dotStyle}`}
            animate={
              runMode === 'live' || runMode === 'checking'
                ? { opacity: [1, 0.4, 1] }
                : {}
            }
            transition={{ duration: 1.5, repeat: Infinity }}
          />
          {mode.label} · {mode.detail}
        </span>
      </div>

      {/* Trial metadata — shown when a case is selected */}
      {selectedCase && (
        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[11px] text-slate-500">
          <span>{selectedCase.subject}</span>
          <span>nsdId {selectedCase.nsdId}</span>
          <span>session {selectedCase.session}</span>
          {selectedCase.metrics.rank != null && (
            <span
              className={
                selectedCase.metrics.rank === 1
                  ? 'text-emerald-400'
                  : selectedCase.metrics.rank <= 5
                    ? 'text-amber-400'
                    : 'text-slate-400'
              }
            >
              rank #{selectedCase.metrics.rank}
            </span>
          )}
        </div>
      )}

      {/* Backend chip — only if online */}
      {backendHealth && (
        <div className="mt-2">
          <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/5 px-2.5 py-1 text-[10px] text-emerald-400/80 ring-1 ring-emerald-500/15">
            <span className="h-1 w-1 rounded-full bg-emerald-400" />
            Backend · {backendHealth.device}
          </span>
        </div>
      )}
    </motion.header>
  );
}
