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

function statusSummary(h: BackendHealth | null): string {
  if (!h) return 'Backend unreachable — offline replay mode';
  const parts: string[] = [];
  if (h.live_retrieval_available) {
    parts.push(`Live encoder · live CSLS retrieval · ${h.gallery_size ?? h.gallery_768_size ?? '?'}-image gallery`);
  } else if (h.inference_available) {
    parts.push('Model loaded · gallery ready · inference available');
  } else if (h.features_loaded) {
    parts.push('fMRI data loaded · model pending');
  } else {
    parts.push('Data loading');
  }
  if (h.reconstruction_available) {
    parts.push('reconstruction live');
  } else if (h.reconstruction_mode === 'live_pending_download') {
    parts.push('reconstruction pending download');
  } else if (h.reconstruction_mode === 'cached_local_reconstruction') {
    parts.push('cached reconstruction');
  } else {
    parts.push('reconstruction unavailable');
  }
  parts.push(h.device || 'cpu');
  return parts.join(' · ');
}

export function PipelineStatusHeader({
  runMode,
  backendHealth,
  selectedCase,
}: PipelineStatusHeaderProps) {
  const h = backendHealth;
  const isLive = h?.live_retrieval_available === true;
  const summary = statusSummary(h);

  return (
    <motion.header
      className="mb-1 space-y-4"
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
    >
      {/* Title row */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-[32px] font-semibold tracking-tight text-text-primary sm:text-[36px]">
            Cortex2Canvas Pipeline
          </h1>
          <p className="mt-1.5 text-sm text-text-secondary">
            Neural decoding workbench · fMRI → CLIP → retrieval → reconstruction
          </p>
        </div>

        {/* Live status badge */}
        {h && (
          <span
            className={`inline-flex shrink-0 items-center gap-2 rounded-lg px-3.5 py-2 text-[12px] font-medium ring-1 ${
              isLive
                ? 'bg-accent/8 text-accent ring-accent/20'
                : h.server_online
                  ? 'bg-surface-raised text-text-secondary ring-border-subtle'
                  : 'bg-status-warning/6 text-status-warning ring-status-warning/15'
            }`}
          >
            <span className={`inline-block h-2 w-2 rounded-full ${
              isLive ? 'bg-accent animate-pulse' : h.server_online ? 'bg-text-muted' : 'bg-status-warning'
            }`} />
            {isLive ? 'Live' : h.server_online ? 'Online' : 'Offline'}
          </span>
        )}
      </div>

      {/* Rich scientific status card */}
      {h && (
        <div className="rounded-xl border border-border-subtle bg-surface-elevated p-4 space-y-3">
          {/* Summary line */}
          <p className="text-[13px] leading-relaxed text-text-secondary">{summary}</p>

          {/* Status chips grid */}
          <div className="flex flex-wrap items-center gap-2 text-[10px]">
            {h.effective_mode && (
              <span className="rounded bg-surface-raised px-2 py-0.5 font-mono text-text-muted">
                mode: {h.effective_mode}
              </span>
            )}
            {h.device && (
              <span className="rounded bg-surface-raised px-2 py-0.5 font-mono text-text-muted">
                device: {h.device}
              </span>
            )}
            {h.v62_model_loaded && (
              <span className="rounded bg-accent/10 px-2 py-0.5 font-medium text-accent">
                V62a loaded
              </span>
            )}
            {h.v61_model_loaded && (
              <span className="rounded bg-accent/10 px-2 py-0.5 font-medium text-accent">
                V61a loaded
              </span>
            )}
            {h.gallery_768_loaded && (
              <span className="rounded bg-surface-raised px-2 py-0.5 text-text-muted">
                gallery: {h.gallery_768_size?.toLocaleString()} × {h.gallery_768_dim}D
              </span>
            )}
            {h.reconstruction_mode && h.reconstruction_mode !== 'unavailable' && (
              <span className="rounded bg-status-success/10 px-2 py-0.5 font-medium text-status-success">
                recon: {h.reconstruction_mode.replace(/_/g, ' ')}
              </span>
            )}
            {h.reconstruction_mode === 'unavailable' && (
              <span className="rounded bg-surface-raised px-2 py-0.5 text-text-muted">
                recon: unavailable
              </span>
            )}
            {h.fallback_used && (
              <span className="rounded bg-status-warning/8 px-2 py-0.5 text-status-warning">
                fallback: {h.fallback_reason}
              </span>
            )}
            {h.missing && h.missing.length > 0 && (
              <span className="rounded bg-status-error/6 px-2 py-0.5 text-status-error">
                missing: {h.missing.join(', ')}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Trial metadata strip */}
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
      </div>
    </motion.header>
  );
}
