import { useMemo, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import type { DemoCase } from '@/types';
import { isMetricAvailable } from '@/lib/metrics';
import { REPLAY_PROV, LIVE_PROV, UNKNOWN_PROV, type Provenance } from '@/lib/provenance';
import {
  metricProvenance,
  uncertaintyProvenance,
  duaCfgProvenance,
  reconstructionProvenance,
} from '@/lib/pipelineNormalize';
import { ProvenanceBadge } from './ProvenanceBadge';
import { MetricCell } from './MetricCell';
import { ComparisonTriptych } from './ComparisonTriptych';
import {
  isBackendAvailable,
  resolveNsdIdToTrial,
  fetchInferenceWithRecon,
  type ReconstructionResult,
} from '@/lib/api';

export interface PhaseReconstructionProps {
  case_: DemoCase;
  liveMode?: boolean;
  onReset: () => void;
}

function clamp01(x: number) { return Math.max(0, Math.min(1, x)); }

function kappaLabel(kn: number): string {
  if (kn > 0.7) return 'High directional certainty';
  if (kn > 0.3) return 'Moderate certainty';
  return 'Low certainty';
}

function deltaLabel(d: number): string {
  if (d < 0.1) return 'Low ROI disagreement';
  if (d < 0.4) return 'Moderate disagreement';
  return 'High disagreement';
}

export function PhaseReconstruction({
  case_,
  liveMode = false,
  onReset,
}: PhaseReconstructionProps) {
  const navigate = useNavigate();
  const { uncertainty: u, duaCfg, metrics: m } = case_;

  const deltaNorm = u.delta <= 1 ? u.delta : clamp01(u.delta / (u.delta + 1));
  const kappa01 = clamp01(u.kappaNorm);

  const uncProv = uncertaintyProvenance(case_);
  const duaProv = duaCfgProvenance(case_);
  const reconProv = reconstructionProvenance(case_);

  // ── Live reconstruction from backend ──
  const [liveRecon, setLiveRecon] = useState<ReconstructionResult | null>(null);
  const [reconLoading, setReconLoading] = useState(false);

  useEffect(() => {
    if (!liveMode || !isBackendAvailable()) return;
    let cancelled = false;

    async function tryLiveRecon() {
      setReconLoading(true);
      try {
        const trialIdx = await resolveNsdIdToTrial(case_.nsdId);
        if (cancelled || trialIdx == null) return;
        const result = await fetchInferenceWithRecon(trialIdx);
        if (cancelled || !result?.reconstruction) return;
        setLiveRecon(result.reconstruction);
      } finally {
        if (!cancelled) setReconLoading(false);
      }
    }
    tryLiveRecon();
    return () => { cancelled = true; };
  }, [case_.nsdId, liveMode]);

  // Determine effective reconstruction source
  const hasLiveRecon = liveRecon?.ok && liveRecon.mode === 'LIVE_LOCAL_RECONSTRUCTION';
  const hasCachedRecon = !!case_.diffusionFinal || !!case_.reconstructionImage;
  const hasRecon = hasLiveRecon || hasCachedRecon;
  const reconImageSrc = hasLiveRecon ? liveRecon.image_url : (case_.diffusionFinal ?? case_.reconstructionImage);
  const effectiveReconProv: Provenance = hasLiveRecon
    ? LIVE_PROV
    : hasCachedRecon
      ? reconProv
      : { kind: 'unknown', detail: 'No reconstruction asset available' };

  const top1 = useMemo(
    () => case_.retrievedImages.find((r) => r.rank === 1) ?? case_.retrievedImages[0],
    [case_.retrievedImages],
  );

  const verdict = useMemo(() => {
    if (m.rank === 1) return { text: 'Exact match', cls: 'text-accent', bg: 'bg-accent/8' };
    if (m.rank <= 5) return { text: 'Near miss', cls: 'text-status-warning', bg: 'bg-status-warning/8' };
    return { text: 'Hard case', cls: 'text-status-error', bg: 'bg-status-error/6' };
  }, [m.rank]);

  function getMetricProv(field: 'pixcorr' | 'ssim' | 'cosine' | 'rank'): Provenance {
    if (liveMode) {
      if (field === 'rank' || field === 'cosine') return LIVE_PROV;
      if (field === 'pixcorr' || field === 'ssim') {
        if (!isMetricAvailable(m.pixcorr) && !isMetricAvailable(m.ssim)) {
          return { kind: 'unknown', detail: 'Not available — requires reconstruction asset' };
        }
        return { kind: 'replay', detail: 'From cached reconstruction' };
      }
    }
    return metricProvenance(case_, field);
  }

  return (
    <div className="pb-12">
      <motion.div
        className="premium-panel overflow-hidden"
        initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
      >
        {/* ── 1. RESULT SUMMARY HERO ── */}
        <div className="px-5 py-5 sm:px-6 sm:py-6">
          <div className="flex flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
            <div className="min-w-0 space-y-2">
              <p className="premium-kicker">Evidence audit · NSD {case_.nsdId}</p>
              <div className="flex items-baseline gap-3">
                <span
                  className={`font-mono text-[60px] font-semibold tabular-nums leading-none tracking-tight ${
                    m.rank === 1
                      ? 'text-status-success'
                      : m.rank <= 5
                      ? 'text-status-warning'
                      : 'text-text-primary'
                  }`}
                >
                  #{m.rank}
                </span>
                <span
                  className={`inline-flex rounded-full px-3 py-0.5 text-[12px] font-semibold ${verdict.bg} ${verdict.cls}`}
                >
                  {verdict.text}
                </span>
              </div>
              <p className="text-[12px] text-text-muted">
                10,000-gallery CSLS ranking · {case_.subject} · session {case_.session}
              </p>
            </div>

            {/* Stat readouts — no card chrome; just tight, aligned numbers
                separated by hairline rules. Reads as instrument data. */}
            <div className="flex divide-x divide-white/[0.06] self-stretch">
              <div className="flex flex-col items-center justify-center px-5 first:pl-0">
                <p className="premium-kicker">κ</p>
                <p className="mt-1 font-mono text-[22px] font-semibold tabular-nums leading-none text-accent">
                  {u.kappa.toFixed(1)}
                </p>
                <p className="mt-1 text-[10px] text-text-muted">{kappaLabel(kappa01)}</p>
              </div>
              <div className="flex flex-col items-center justify-center px-5">
                <p className="premium-kicker">δ</p>
                <p className="mt-1 font-mono text-[22px] font-semibold tabular-nums leading-none text-text-primary">
                  {liveMode ? 'N/A' : u.delta.toFixed(3)}
                </p>
                <p className="mt-1 text-[10px] text-text-muted">{liveMode ? 'no per-ROI' : deltaLabel(u.delta)}</p>
              </div>
              <div className="flex flex-col items-center justify-center px-5 last:pr-0">
                <p className="premium-kicker">CSLS</p>
                <p className="mt-1 font-mono text-[22px] font-semibold tabular-nums leading-none text-text-primary">
                  {top1?.csls != null ? top1.csls.toFixed(3) : 'N/A'}
                </p>
                <p className="mt-1 text-[10px] text-text-muted">top-1 score</p>
              </div>
            </div>
          </div>

          {/* Provenance row — single hair-rule, no second container */}
          <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-white/[0.05] pt-3.5">
            {hasLiveRecon ? (
              <ProvenanceBadge provenance={LIVE_PROV} />
            ) : (
              <ProvenanceBadge provenance={uncProv} />
            )}
            <ProvenanceBadge provenance={duaProv} />
            <ProvenanceBadge provenance={effectiveReconProv} />
            {reconLoading && (
              <span className="text-[10px] text-text-muted">· loading live reconstruction…</span>
            )}
          </div>
        </div>

        {/* ── 2. COMPARISON TRIPTYCH — inset, hair-rule separators ── */}
        <div className="border-y border-white/[0.05] bg-white/[0.01] px-5 py-5 sm:px-6 sm:py-6">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="premium-kicker">
              {hasRecon ? 'Visual comparison' : 'Retrieval evidence'}
            </h3>
            {!liveMode && <span className="text-[10.5px] text-text-muted">Cached assets</span>}
          </div>
        <ComparisonTriptych
          panels={[
            {
              title: 'Subject perceived',
              subtitle: 'Reference stimulus',
              imageSrc: case_.targetImage,
              provenance: { kind: 'replay', detail: 'Cached NSD stimulus image' },
              accent: 'emerald' as const,
              revealBlur: true,
              isMatch: m.rank === 1,
            },
            {
              title: 'Model retrieved',
              subtitle: liveMode ? 'Live retrieval · CUDA' : `Top-1 gallery · Rank #${top1?.rank ?? 'N/A'}`,
              imageSrc: top1?.image,
              provenance: liveMode ? { kind: 'derived', detail: 'Live CSLS ranking' } : REPLAY_PROV,
              accent: 'violet' as const,
              isMatch: m.rank === 1,
            },
            {
              title: hasLiveRecon
                ? 'Live reconstruction'
                : hasCachedRecon
                  ? 'Cached reconstruction'
                  : 'Reconstruction not cached',
              subtitle: hasLiveRecon
                ? `Karlo UnCLIP · ${liveRecon?.steps ?? '?'} steps, seed ${liveRecon?.seed ?? '?'}`
                : hasCachedRecon
                  ? 'Stable Diffusion 2.1 output (cached)'
                  : 'Asset not available',
              imageSrc: reconImageSrc,
              provenance: effectiveReconProv,
              accent: 'cyan' as const,
              isMatch: false,
              emptyText: hasLiveRecon
                ? (liveRecon?.reason || liveRecon?.last_error || undefined)
                : 'This replay contains retrieval evidence only. Enable local reconstruction export to compare generated assets.',
            },
          ]}
        />
        </div>

        {/* ── 3. EVIDENCE SUMMARY METRICS ── */}
        <div className="px-5 py-5 sm:px-6 sm:py-6">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="premium-kicker">Evidence summary</h3>
            <span className={`inline-flex items-center gap-2 rounded-full px-3 py-0.5 text-[11px] font-semibold ${verdict.bg} ${verdict.cls}`}>
              {verdict.text} · Rank #{m.rank}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricCell label="Rank" value={String(m.rank)} description="Gallery retrieval rank"
              provenance={getMetricProv('rank')}
              valueColor={m.rank === 1 ? 'text-status-success' : m.rank <= 5 ? 'text-status-warning' : 'text-status-error'} />
            <MetricCell
              label={isMetricAvailable(m.cosine) ? 'Cosine' : 'CSLS'}
              value={isMetricAvailable(m.cosine) ? m.cosine : m.csls}
              description={isMetricAvailable(m.cosine) ? 'Top-1 cosine similarity' : 'Top-1 CSLS score'}
              provenance={getMetricProv('cosine')} />
            <MetricCell label="κ" value={u.kappa} description="Directional concentration" provenance={uncProv} />
            <MetricCell label="δ" value={liveMode ? null : u.delta} description="ROI disagreement" provenance={uncProv} />
            {isMetricAvailable(m.pixcorr) ? (
              <MetricCell label="PixCorr" value={m.pixcorr} description="Pixel correlation" provenance={getMetricProv('pixcorr')} />
            ) : null}
            {isMetricAvailable(m.ssim) ? (
              <MetricCell label="SSIM" value={m.ssim} description="Structural similarity" provenance={getMetricProv('ssim')} />
            ) : null}
          </div>
          {(!isMetricAvailable(m.pixcorr) || !isMetricAvailable(m.ssim)) && (
            <p className="mt-3.5 text-[11px] leading-relaxed text-text-muted">
              <span className="font-medium text-text-secondary">No reconstruction asset for this trial.</span> PixCorr and SSIM are hidden — this audit relies on retrieval rank, CSLS, and uncertainty evidence.
            </p>
          )}
        </div>

        {/* ── 4. AUDIT FOOTER: actions integrated into the audit panel itself ── */}
        <div className="border-t border-white/[0.05] bg-white/[0.012] px-5 py-3.5 sm:px-6">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <button
              type="button"
              onClick={onReset}
              className="inline-flex items-center gap-1.5 self-start rounded-lg border border-border-subtle bg-surface-raised px-3 py-2 text-[12px] font-medium text-text-secondary transition hover:border-border-emphasis hover:text-text-primary sm:self-auto"
            >
              <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
              </svg>
              Try another trial
            </button>
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => navigate(`/explorer/${case_.id}`)}
                className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-[12px] font-medium text-text-secondary transition hover:bg-surface-raised hover:text-text-primary"
              >
                Explore in detail
                <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                </svg>
              </button>
              <span className="hidden h-4 w-px bg-border-subtle/60 sm:inline-block" aria-hidden />
              <button
                type="button"
                onClick={() => navigate('/challenge')}
                className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-[12px] font-medium text-text-secondary transition hover:bg-surface-raised hover:text-text-primary"
              >
                Take the challenge
                <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
