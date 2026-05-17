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

function modeLabel(h: BackendHealth | null): string {
  if (!h) return 'Offline replay';
  if (h.live_retrieval_available || h.status === 'inference_ready') return 'Hybrid live';
  if (h.server_online || h.status === 'data_ready' || h.status === 'server_online') return 'Cached replay';
  return 'Offline replay';
}

function deviceLabel(h: BackendHealth | null): string {
  return h?.device || 'local';
}

function phaseCopy(phase: string, selectedCase: DemoCase | null) {
  if (phase === 'retrieval' && selectedCase) {
    return {
      eyebrow: 'Decode replay',
      title: 'Decode replay',
      subtitle: `${selectedCase.subject} · nsdId ${selectedCase.nsdId} · session ${selectedCase.session} · rank #${selectedCase.metrics.rank}`,
    };
  }

  if (phase === 'reconstruction') {
    return {
      eyebrow: 'Visual evidence',
      title: 'Compare retrieval evidence',
      subtitle: selectedCase
        ? `Target stimulus, top-ranked retrieval, and available reconstruction metrics for nsdId ${selectedCase.nsdId}.`
        : 'Target stimulus, top-ranked retrieval, and available reconstruction metrics.',
    };
  }

  return {
    eyebrow: 'Stimulus gallery',
    title: 'Cortex2Canvas',
    subtitle: 'Select an NSD stimulus and replay the subject-specific fMRI → CLIP evidence path.',
  };
}

export function PipelineStatusHeader({
  runMode,
  backendHealth,
  selectedCase,
  phase,
}: PipelineStatusHeaderProps) {
  const h = backendHealth;
  const isLive = h?.live_retrieval_available === true;
  const copy = phaseCopy(phase, selectedCase);

  return (
    <motion.header
      className="rounded-[22px] border border-border-subtle bg-[#0b0e16]/88 p-3 shadow-[0_24px_90px_-60px_rgba(0,0,0,0.95)]"
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-center">
        <div className="flex min-w-0 items-center gap-4 rounded-2xl border border-white/[0.045] bg-white/[0.018] px-4 py-3">
          <div className="hidden h-10 w-px bg-gradient-to-b from-transparent via-accent/45 to-transparent sm:block" />
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-text-muted">{copy.eyebrow}</p>
              {selectedCase ? (
                <span className="rounded-md bg-surface-raised px-2 py-0.5 font-mono text-[10px] text-text-muted">
                  nsdId {selectedCase.nsdId}
                </span>
              ) : null}
            </div>
            <h1 className="mt-1 text-[24px] font-semibold leading-tight tracking-[-0.025em] text-text-primary sm:text-[28px]">
              {copy.title}
            </h1>
            <p className="mt-1 max-w-3xl truncate text-[13px] text-text-secondary">
              {copy.subtitle}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 lg:justify-end">
          <span
            className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-[11px] font-semibold ${
              isLive
                ? 'border-accent/25 bg-accent/10 text-accent'
                : h?.server_online
                  ? 'border-border-subtle bg-surface-raised text-text-secondary'
                  : 'border-status-warning/20 bg-status-warning/8 text-status-warning'
            }`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${isLive ? 'bg-accent' : h?.server_online ? 'bg-text-muted' : 'bg-status-warning'}`} />
            {modeLabel(h)}
          </span>
          <span className="rounded-full border border-border-subtle bg-surface-raised px-3 py-1.5 text-[11px] font-semibold text-text-secondary">
            V62a · CLIP ViT-L/14
          </span>
          {selectedCase ? (
            <span className="rounded-full border border-border-subtle bg-surface-raised px-3 py-1.5 text-[11px] font-semibold text-text-muted">
              {selectedCase.subject}
            </span>
          ) : null}
          <span className="rounded-full border border-border-subtle bg-surface-raised px-3 py-1.5 text-[11px] font-semibold text-text-muted">
            {deviceLabel(h)}
          </span>
          {runMode === 'checking' ? (
            <span className="rounded-full border border-border-subtle bg-surface-raised px-3 py-1.5 text-[11px] text-text-muted">
              checking backend
            </span>
          ) : null}
        </div>
      </div>
    </motion.header>
  );
}
