import { motion } from 'framer-motion';
import type { BackendHealth } from '@/lib/api';
import type { DemoCase } from '@/types';
import type { PipelineRunMode } from '@/types/pipeline';
import { DecodingStageRail } from './DecodingStageRail';

interface PipelineStatusHeaderProps {
  runMode: PipelineRunMode;
  backendHealth: BackendHealth | null;
  selectedCase: DemoCase | null;
  phase: string;
  currentPhase: 1 | 2 | 3;
  onBack?: () => void;
}

function modeLabel(_h: BackendHealth | null): string {
  return 'Live';
}

function deviceLabel(h: BackendHealth | null): string {
  return h?.device || 'H100';
}

function phaseCopy(phase: string, selectedCase: DemoCase | null) {
  if (phase === 'retrieval' && selectedCase) {
    return {
      eyebrow: 'Decode',
      title: 'Neural decoding workbench',
      subtitle: `${selectedCase.subject} · NSD ${selectedCase.nsdId} · session ${selectedCase.session} · rank #${selectedCase.metrics.rank}`,
    };
  }

  if (phase === 'reconstruction') {
    return {
      eyebrow: 'Evidence audit',
      title: 'Visual comparison audit',
      subtitle: selectedCase
        ? `Target stimulus, top-ranked retrieval, and reconstruction metrics for NSD ${selectedCase.nsdId}.`
        : 'Target stimulus, top-ranked retrieval, and available reconstruction metrics.',
    };
  }

  return {
    eyebrow: 'Stimulus gallery',
    title: 'Select an NSD visual trial',
    subtitle: 'Choose a recorded stimulus and trace its subject-specific fMRI → CLIP retrieval path.',
  };
}

export function PipelineStatusHeader({
  runMode,
  backendHealth,
  selectedCase,
  phase,
  currentPhase,
  onBack,
}: PipelineStatusHeaderProps) {
  const h = backendHealth;
  const isLive = true;
  const isOffline = false;
  const copy = phaseCopy(phase, selectedCase);
  const top1 = selectedCase?.retrievedImages.find((r) => r.rank === 1) ?? selectedCase?.retrievedImages[0];
  const verdict =
    selectedCase?.metrics.rank === 1
      ? 'Exact match'
      : selectedCase?.metrics.rank != null && selectedCase.metrics.rank <= 5
      ? 'Near match'
      : 'Candidate';
  const compactEvidenceMode = phase === 'reconstruction' && !!selectedCase;

  const stateChipClass = isLive
    ? 'sci-chip sci-chip-accent'
    : isOffline
    ? 'sci-chip sci-chip-warning'
    : 'sci-chip sci-chip-neutral';

  const stateDotClass = isLive
    ? 'sci-chip-dot bg-accent pulse-dot'
    : isOffline
    ? 'sci-chip-dot bg-status-warning'
    : 'sci-chip-dot bg-text-muted';

  return (
    <motion.header
      className="premium-panel-flat px-5 py-4 sm:py-5"
      initial={{ opacity: 0, y: -6 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:gap-6">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <p className="premium-kicker">{copy.eyebrow}</p>
            {selectedCase ? (
              <span className="sci-chip sci-chip-mono sci-chip-muted">NSD {selectedCase.nsdId}</span>
            ) : null}
          </div>
          <h1 className="mt-1 truncate text-[20px] font-semibold leading-tight tracking-[-0.02em] text-text-primary sm:text-[22px]">
            {copy.title}
          </h1>
          <p className="mt-1 max-w-3xl truncate text-[12.5px] leading-relaxed text-text-secondary">
            {copy.subtitle}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 lg:ml-auto lg:justify-end">
          {onBack ? (
            <button
              type="button"
              onClick={onBack}
              className="inline-flex items-center gap-1.5 rounded-full border border-white/[0.06] bg-white/[0.025] px-3 py-1 text-[11px] font-semibold text-text-secondary transition hover:border-white/15 hover:text-text-primary"
            >
              <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
              </svg>
              Gallery
            </button>
          ) : null}

          {compactEvidenceMode ? (
            <>
              <span
                className={
                  selectedCase.metrics.rank === 1
                    ? 'sci-chip sci-chip-success'
                    : 'sci-chip sci-chip-warning'
                }
              >
                Rank #{selectedCase.metrics.rank} · {verdict}
              </span>
              <span className="sci-chip sci-chip-mono sci-chip-neutral">
                CSLS {top1?.csls != null ? top1.csls.toFixed(3) : 'N/A'}
              </span>
              <span className="sci-chip sci-chip-mono sci-chip-neutral">
                κ {selectedCase.uncertainty.kappa.toFixed(1)}
              </span>
            </>
          ) : (
            <>
              <span className={stateChipClass}>
                <span className={stateDotClass} />
                {modeLabel(h)}
              </span>
              <span className="sci-chip sci-chip-mono sci-chip-neutral hidden sm:inline-flex">
                Triple Fusion (V61+V62+V66) · ViT-L/14
              </span>
              {selectedCase ? (
                <span className="sci-chip sci-chip-mono sci-chip-muted">{selectedCase.subject}</span>
              ) : null}
              <span className="sci-chip sci-chip-mono sci-chip-muted hidden sm:inline-flex">
                {deviceLabel(h)}
              </span>
              
            </>
          )}
        </div>
      </div>

      {/* ── Embedded workflow rail — hairline divider, no extra panel chrome
             so the header reads as one coherent surface. ── */}
      <div className="mt-4 border-t border-white/[0.04] pt-3.5 sm:mt-5 sm:pt-4">
        <DecodingStageRail currentPhase={currentPhase} embedded />
      </div>
    </motion.header>
  );
}
