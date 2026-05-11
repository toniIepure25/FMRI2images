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

const MODE_LABELS: Record<PipelineRunMode, { label: string; detail: string }> = {
  checking:       { label: 'Checking',          detail: 'Verifying backend availability' },
  live:           { label: 'Live inference',    detail: 'Live model inference active' },
  replay:         { label: 'Replay',            detail: 'Backend online, replay assets active' },
  'offline-replay': { label: 'Offline replay',  detail: 'Backend unavailable — cached results' },
  hybrid:         { label: 'Hybrid',            detail: 'Live retrieval + cached reconstruction' },
  error:          { label: 'Error',             detail: 'Connection failed' },
};

const MODE_DOT: Record<PipelineRunMode, string> = {
  checking:       'bg-text-muted',
  live:           'bg-status-success',
  replay:         'bg-accent',
  'offline-replay': 'bg-status-warning',
  hybrid:         'bg-status-info',
  error:          'bg-status-error',
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
      className="mb-1 space-y-4"
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
    >
      {/* Title row */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-[30px] font-semibold tracking-tight text-text-primary sm:text-[34px]">
            Cortex2Canvas Pipeline
          </h1>
          <p className="mt-1.5 text-sm text-text-muted">
            Neural decoding workbench &middot; fMRI &rarr; CLIP &rarr; reconstruction
          </p>
        </div>

        {/* Run mode chip */}
        <span
          className="inline-flex shrink-0 items-center gap-2 rounded-lg bg-surface-raised px-3.5 py-2 text-[12px] font-medium text-text-secondary ring-1 ring-border-subtle"
          title={mode.detail}
        >
          <motion.span
            className={`inline-block h-2 w-2 rounded-full ${dotStyle}`}
            animate={
              runMode === 'live' || runMode === 'checking'
                ? { opacity: [1, 0.35, 1] }
                : {}
            }
            transition={{ duration: 1.5, repeat: Infinity }}
          />
          {mode.label}
        </span>
      </div>

      {/* Trial metadata + backend strip */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg bg-surface-raised px-4 py-2.5">
        {selectedCase ? (
          <>
            <span className="font-mono text-[13px] font-medium text-text-primary">{selectedCase.subject}</span>
            <span className="text-border-emphasis">&middot;</span>
            <span className="font-mono text-[13px] text-text-secondary">nsdId {selectedCase.nsdId}</span>
            <span className="text-border-emphasis">&middot;</span>
            <span className="font-mono text-[13px] text-text-secondary">session {selectedCase.session}</span>
            {selectedCase.metrics.rank != null && (
              <>
                <span className="text-border-emphasis">&middot;</span>
                <span className={`font-mono text-[13px] font-semibold ${
                  selectedCase.metrics.rank === 1 ? 'text-status-success' : selectedCase.metrics.rank <= 5 ? 'text-status-warning' : 'text-text-muted'
                }`}>
                  rank #{selectedCase.metrics.rank}
                </span>
              </>
            )}
          </>
        ) : (
          <span className="text-[12px] text-text-muted">Select a trial to begin</span>
        )}
        {/* Backend chip inline */}
        {backendHealth && (
          <span className="ml-auto inline-flex items-center gap-1.5 rounded-full bg-accent/[0.06] px-2.5 py-1 text-[10px] text-accent">
            <span className="h-1 w-1 rounded-full bg-accent" />
            {backendHealth.device}
          </span>
        )}
      </div>
    </motion.header>
  );
}
