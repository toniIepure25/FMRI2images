import { useMemo, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import type { DemoCase } from '@/types';
import { isMetricAvailable } from '@/lib/metrics';
import { REPLAY_PROV, LIVE_PROV, DERIVED_PROV, UNKNOWN_PROV, type Provenance } from '@/lib/provenance';
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
    <div className="space-y-6 pb-16">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
        <div className="premium-panel premium-panel-hero flex flex-col gap-4 p-5 sm:p-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <h2 className="text-3xl font-semibold tracking-tight text-text-primary">
                {hasRecon ? 'Compare retrieval evidence' : 'Retrieval-only evidence review'}
              </h2>
              {hasLiveRecon ? (
                <span className="inline-flex items-center gap-1 rounded bg-accent/15 px-2.5 py-0.5 text-[10px] font-semibold text-accent">
                  Karlo UnCLIP &middot; live
                </span>
              ) : (
                <ProvenanceBadge provenance={liveMode ? { kind: 'derived', detail: 'Live retrieval + cached reconstruction' } : REPLAY_PROV} />
              )}
            </div>
            <p className="max-w-3xl text-[13px] leading-relaxed text-text-secondary">
              {hasLiveRecon
                ? `Live local reconstruction · generated from V62a CLIP embedding · ${liveRecon?.generation_ms?.toFixed(0) ?? '?'}ms`
                : hasCachedRecon
                  ? 'Cached qualitative reconstruction'
                  : 'Target stimulus and top-ranked retrieval are shown side by side. No reconstruction asset is cached for this trial.'}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className={`rounded-xl px-3 py-2 text-[11px] font-semibold ${verdict.bg} ${verdict.cls}`}>
              Rank #{m.rank} · {verdict.text}
            </span>
            <span className="rounded-xl border border-border-subtle bg-surface-raised px-3 py-2 font-mono text-[11px] text-text-secondary">
              CSLS {top1?.csls != null ? top1.csls.toFixed(3) : 'n/a'}
            </span>
            <span className="rounded-xl border border-border-subtle bg-surface-raised px-3 py-2 font-mono text-[11px] text-text-secondary">
              κ {u.kappa.toFixed(1)}
            </span>
          </div>
        </div>
      </motion.div>

      {/* ── RESULT SUMMARY HERO ── */}
      <motion.div
        className="premium-panel p-6 sm:p-7"
        initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
          <div className="space-y-2">
            <p className="premium-kicker">Retrieval result</p>
            <div className="flex items-baseline gap-4">
              <span className={`font-mono text-[76px] font-semibold tabular-nums leading-none ${
                m.rank === 1 ? 'text-accent' : m.rank <= 5 ? 'text-status-warning' : 'text-text-primary'
              }`}>
                #{m.rank}
              </span>
              <span className={`inline-flex rounded-full px-4 py-1.5 text-base font-semibold ${verdict.bg} ${verdict.cls}`}>
                {verdict.text}
              </span>
            </div>
            <p className="text-[13px] text-text-muted">10,000-gallery CSLS ranking &middot; {case_.subject}</p>
          </div>
          <div className="flex flex-wrap gap-5 sm:gap-8">
            <div className="text-center min-w-[70px]">
              <p className="text-[11px] font-semibold text-text-muted">κ</p>
              <p className="font-mono text-3xl font-bold text-accent">{u.kappa.toFixed(1)}</p>
              <p className="text-[10px] text-text-muted">{kappaLabel(kappa01)}</p>
            </div>
            <div className="text-center min-w-[70px]">
              <p className="text-[11px] font-semibold text-text-muted">δ</p>
              <p className="font-mono text-3xl font-bold text-text-primary">{liveMode ? 'N/A' : u.delta.toFixed(3)}</p>
              <p className="text-[10px] text-text-muted">{liveMode ? 'no per-ROI' : deltaLabel(u.delta)}</p>
            </div>
            <div className="text-center min-w-[70px]">
              <p className="text-[11px] font-semibold text-text-muted">CSLS</p>
              <p className="font-mono text-3xl font-bold text-text-primary">
                {top1?.csls != null ? top1.csls.toFixed(3) : '—'}
              </p>
              <p className="text-[10px] text-text-muted">top-1 score</p>
            </div>
          </div>
        </div>
        <div className="mt-5 flex flex-wrap items-center gap-3 pt-4 border-t border-border-subtle">
          {hasLiveRecon ? (
            <ProvenanceBadge provenance={LIVE_PROV} />
          ) : (
            <ProvenanceBadge provenance={uncProv} />
          )}
          <ProvenanceBadge provenance={duaProv} />
          <ProvenanceBadge provenance={effectiveReconProv} />
          {reconLoading && (
            <span className="text-[10px] text-text-muted">&middot; loading live reconstruction...</span>
          )}
        </div>
      </motion.div>

      {/* Comparison triptych */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <p className="text-[13px] font-semibold text-text-primary">{hasRecon ? 'Visual comparison' : 'Retrieval evidence'}</p>
          {!liveMode && <span className="text-[10px] text-text-muted">Cached assets</span>}
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
              subtitle: liveMode ? 'Live retrieval · CUDA' : `Top-1 gallery · Rank #${top1?.rank ?? '—'}`,
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

      {/* ── Reconstruction Status Panel (shown when reconstruction unavailable) ── */}
      {!hasLiveRecon && !hasCachedRecon && (
        <div className="rounded-2xl border border-border-subtle bg-surface-raised/75 p-5">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-text-primary">Reconstruction not cached</h3>
              <p className="mt-1 text-[11px] text-text-muted">This replay contains retrieval evidence only.</p>
            </div>
            <span className="rounded-lg border border-border-subtle bg-surface-base px-2.5 py-1 text-[10px] font-semibold text-text-muted">Unavailable</span>
          </div>

          <div className="flex items-center gap-2 flex-wrap text-[11px] mb-3">
            <span className="rounded bg-surface-raised px-2 py-1 font-mono text-text-secondary">15,724 voxels</span>
            <span className="text-border-emphasis">&rarr;</span>
            <span className="rounded bg-accent/10 px-2 py-1 font-mono text-accent">V62a MLP</span>
            <span className="text-border-emphasis">&rarr;</span>
            <span className="rounded bg-surface-raised px-2 py-1 font-mono text-text-secondary">μ ∈ R⁷⁶⁸</span>
            <span className="text-border-emphasis">&rarr;</span>
            <span className="rounded bg-surface-raised px-2 py-1 font-mono text-text-secondary">CSLS 10k</span>
            <span className="text-border-emphasis">&rarr;</span>
            <span className="rounded bg-surface-raised px-2 py-1 font-mono text-text-secondary">Rank #{m.rank}</span>
            <span className="text-border-emphasis">&rarr;</span>
            <span className="rounded bg-surface-raised px-1.5 py-1 text-[10px] text-text-muted">optional reconstruction</span>
          </div>

          <div className="flex items-center gap-x-4 gap-y-1 flex-wrap text-[11px]">
            <span className="text-status-success text-xs">✓</span><span className="text-text-secondary">μ available</span>
            <span className="text-status-success text-xs ml-3">✓</span><span className="text-text-secondary">pipeline valid</span>
            <span className="text-text-muted ml-3">—</span><span className="text-text-muted">weights not cached</span>
            <span className="text-[10px] text-text-muted sm:ml-auto">
              Set <span className="font-mono text-accent/70">C2C_RECON_ALLOW_DOWNLOAD=true</span> for live local reconstruction.
            </span>
          </div>
        </div>
      )}

      {/* Trial metrics */}
      <div className="premium-panel p-5 sm:p-6">
        <div className="mb-5 flex flex-col items-center justify-between gap-3 sm:flex-row">
          <h3 className="text-base font-semibold text-text-primary">Evidence summary</h3>
          <span className={`inline-flex items-center gap-2 rounded-full px-3.5 py-1.5 text-[11px] font-semibold ${verdict.bg} ${verdict.cls}`}>
            {verdict.text} &middot; Rank #{m.rank}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <MetricCell label="Rank" value={String(m.rank)} description="Gallery retrieval rank"
            provenance={getMetricProv('rank')}
            valueColor={m.rank === 1 ? 'text-accent' : m.rank <= 5 ? 'text-status-warning' : 'text-status-error'} />
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
          <p className="mt-4 rounded-xl border border-border-subtle bg-surface-raised/70 px-4 py-3 text-[11px] leading-relaxed text-text-muted">
            PixCorr and SSIM are hidden from the primary metric row when no reconstruction asset is available. This replay is evaluated through retrieval rank, CSLS score, and uncertainty evidence.
          </p>
        )}
      </div>

      {/* Actions */}
      <div className="flex flex-col gap-3 sm:flex-row sm:justify-center">
        <button type="button" onClick={onReset}
          className="rounded-xl border border-border-subtle bg-surface-raised px-6 py-3 text-[13px] font-medium text-text-secondary transition hover:border-border-emphasis hover:text-text-primary">
          &larr; Try another trial
        </button>
        <button type="button" onClick={() => navigate(`/explorer/${case_.id}`)}
          className="rounded-xl border border-border-subtle bg-surface-raised px-6 py-3 text-[13px] font-medium text-text-secondary transition hover:border-border-emphasis hover:text-text-primary">
          Explore in detail &rarr;
        </button>
        <button type="button" onClick={() => navigate('/challenge')}
          className="rounded-xl border border-border-subtle bg-surface-raised px-6 py-3 text-[13px] font-medium text-text-secondary transition hover:border-border-emphasis hover:text-text-primary">
          Take the challenge &rarr;
        </button>
      </div>
    </div>
  );
}
