import { useMemo, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import type { DemoCase } from '@/types';
import { isMetricAvailable, formatMetric } from '@/lib/metrics';
import { REPLAY_PROV, LIVE_PROV, type Provenance } from '@/lib/provenance';
import {
  metricProvenance,
  uncertaintyProvenance,
  duaCfgProvenance,
  reconstructionProvenance,
} from '@/lib/pipelineNormalize';
import { ProvenanceBadge } from './ProvenanceBadge';
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
    if (m.rank === 1) return { text: 'Exact match', cls: 'text-status-success', bg: 'bg-status-success/10' };
    if (m.rank <= 5) return { text: 'Near miss', cls: 'text-status-warning', bg: 'bg-status-warning/10' };
    return { text: 'Hard case', cls: 'text-status-error', bg: 'bg-status-error/8' };
  }, [m.rank]);

  // Top-1 CSLS — single source of truth used in both the report header and
  // evidence summary so the two readouts cannot drift.
  const top1Csls = top1?.csls;
  const cslsDisplay = top1Csls != null ? top1Csls.toFixed(3) : 'N/A';
  const cosineDisplay = isMetricAvailable(m.cosine) ? formatMetric(m.cosine, 3) : null;
  const showCosine = cosineDisplay != null;
  const summarySimilarityLabel = showCosine ? 'Cosine · top-1' : 'CSLS · top-1';
  const summarySimilarityValue = showCosine ? cosineDisplay : cslsDisplay;
  const summarySimilarityProv: Provenance = showCosine ? metricProvenance(case_, 'cosine') : metricProvenance(case_, 'rank');

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

  // ── Compact evidence-strip cell. Hair-rule columns, no card chrome.
  //    Vertical rhythm: kicker → value → sub → provenance. Provenance is
  //    pinned at the bottom of each cell so all chips share a baseline, no
  //    matter how short or long the sub-caption gets above them. ──
  type SummaryCell = {
    label: string;
    value: string;
    sub?: string;
    color?: string;
    provenance?: Provenance;
  };
  function SummaryStripCell({ cell }: { cell: SummaryCell }) {
    return (
      <div className="flex flex-col px-4 py-3 first:pl-0 sm:px-5">
        <p className="premium-kicker">{cell.label}</p>
        <p
          className={`mt-1.5 font-mono text-[22px] font-semibold tabular-nums leading-none tracking-tight ${cell.color ?? 'text-text-primary'}`}
        >
          {cell.value}
        </p>
        {cell.sub ? (
          <p className="mt-1.5 text-[10.5px] leading-snug text-text-muted">{cell.sub}</p>
        ) : null}
        {cell.provenance ? (
          <div className="mt-2.5">
            <ProvenanceBadge provenance={cell.provenance} />
          </div>
        ) : null}
      </div>
    );
  }

  const summaryCells: SummaryCell[] = [
    {
      label: 'Rank',
      value: `#${m.rank}`,
      sub: 'Gallery retrieval',
      color: m.rank === 1 ? 'text-status-success' : m.rank <= 5 ? 'text-status-warning' : 'text-status-error',
      provenance: getMetricProv('rank'),
    },
    {
      label: summarySimilarityLabel,
      value: summarySimilarityValue ?? 'N/A',
      sub: 'gallery similarity',
      provenance: summarySimilarityProv,
    },
    {
      label: 'κ',
      value: u.kappa.toFixed(1),
      sub: kappaLabel(kappa01),
      color: 'text-accent',
      provenance: uncProv,
    },
    {
      label: 'δ',
      value: liveMode ? 'N/A' : u.delta.toFixed(3),
      sub: liveMode ? 'no per-ROI' : deltaLabel(u.delta),
      provenance: uncProv,
    },
  ];
  if (isMetricAvailable(m.pixcorr)) {
    summaryCells.push({
      label: 'PixCorr',
      value: formatMetric(m.pixcorr, 3),
      sub: 'pixel correlation',
      provenance: getMetricProv('pixcorr'),
    });
  }
  if (isMetricAvailable(m.ssim)) {
    summaryCells.push({
      label: 'SSIM',
      value: formatMetric(m.ssim, 3),
      sub: 'structural similarity',
      provenance: getMetricProv('ssim'),
    });
  }

  return (
    <div className="pb-12">
      <motion.div
        className="premium-panel overflow-hidden"
        initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
      >
        {/* ── 1. REPORT HEADER — kicker rule, balanced rank glyph, refined
                stat strip with proper hair-rule columns. ── */}
        <div className="px-5 pt-5 pb-4 sm:px-6">
          <div className="flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
            <div className="min-w-0">
              <div className="flex items-center gap-2.5">
                <span className="h-px w-7 bg-accent/45" aria-hidden />
                <p className="premium-kicker">Evidence audit</p>
              </div>
              <div className="mt-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
                <span
                  className={`font-mono text-[30px] font-semibold tabular-nums leading-none tracking-tight ${
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
                  className={`inline-flex items-center gap-1 rounded-full px-2 py-[2px] text-[10.5px] font-semibold uppercase tracking-[0.08em] ring-1 ${
                    m.rank === 1
                      ? 'bg-status-success/[0.08] text-status-success/95 ring-status-success/22'
                      : m.rank <= 5
                      ? 'bg-status-warning/[0.08] text-status-warning/95 ring-status-warning/22'
                      : 'bg-white/[0.04] text-text-secondary ring-white/[0.08]'
                  }`}
                >
                  <span
                    className={`h-[4px] w-[4px] rounded-full ${
                      m.rank === 1
                        ? 'bg-status-success'
                        : m.rank <= 5
                        ? 'bg-status-warning'
                        : 'bg-text-secondary'
                    }`}
                  />
                  {verdict.text}
                </span>
                <span className="font-mono text-[11px] tabular-nums text-text-muted/85">
                  NSD&nbsp;{case_.nsdId} · {case_.subject} · session&nbsp;{case_.session}
                </span>
              </div>
            </div>

            {/* Stat strip — κ · δ · CSLS · top-1. Hair-rule columns share
                the exact same label/value rhythm as the evidence summary
                below so the same metric never reads differently. */}
            <div className="flex divide-x divide-white/[0.06] self-stretch">
              <div className="flex flex-col justify-center px-4 first:pl-0 sm:px-5">
                <p className="premium-kicker">κ</p>
                <p className="mt-1.5 font-mono text-[20px] font-semibold tabular-nums leading-none text-accent">
                  {u.kappa.toFixed(1)}
                </p>
                <p className="mt-1.5 text-[10px] leading-tight text-text-muted">{kappaLabel(kappa01)}</p>
              </div>
              <div className="flex flex-col justify-center px-4 sm:px-5">
                <p className="premium-kicker">δ</p>
                <p className="mt-1.5 font-mono text-[20px] font-semibold tabular-nums leading-none text-text-primary">
                  {liveMode ? 'N/A' : u.delta.toFixed(3)}
                </p>
                <p className="mt-1.5 text-[10px] leading-tight text-text-muted">{liveMode ? 'no per-ROI' : deltaLabel(u.delta)}</p>
              </div>
              <div className="flex flex-col justify-center px-4 last:pr-0 sm:px-5">
                <p className="premium-kicker">CSLS · top-1</p>
                <p className="mt-1.5 font-mono text-[20px] font-semibold tabular-nums leading-none text-text-primary">
                  {cslsDisplay}
                </p>
                <p className="mt-1.5 text-[10px] leading-tight text-text-muted">gallery rank-1</p>
              </div>
            </div>
          </div>

          {/* Provenance row — quiet ribbon under the header, badges sized
              consistently with the rest of the workbench. */}
          <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-white/[0.05] pt-3">
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

        {/* ── 2. Slim reconstruction-unavailable annotation (only when no
                recon). Visually quiet single-row notice. ── */}
        {!hasRecon ? (
          <div className="border-t border-white/[0.05] bg-white/[0.010] px-5 py-2 sm:px-6">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[11px]">
              <svg className="h-3.5 w-3.5 shrink-0 text-text-muted/70" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.6}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m0-10.5a8.25 8.25 0 100 16.5 8.25 8.25 0 000-16.5zm0 12.75h.008v.008H12v-.008z" />
              </svg>
              <p className="text-text-secondary">
                <span className="font-medium text-text-primary">Reconstruction unavailable for this replay.</span>{' '}
                <span className="text-text-muted">Audit relies on retrieval rank, CSLS, and uncertainty.</span>
              </p>
              <span className="ml-auto">
                <ProvenanceBadge provenance={effectiveReconProv} />
              </span>
            </div>
          </div>
        ) : null}

        {/* ── 3. RETRIEVAL / VISUAL COMPARISON — image pair, hairline rule
                so the section reads as a labeled report block. ── */}
        <div className="border-t border-white/[0.05] bg-white/[0.01] px-5 py-5 sm:px-6 sm:py-6">
          <div className="mb-3.5 flex items-baseline justify-between gap-3">
            <div className="flex items-center gap-2.5">
              <span className="h-px w-6 bg-accent/40" aria-hidden />
              <h3 className="premium-kicker">
                {hasRecon ? 'Visual comparison' : 'Retrieval evidence'}
              </h3>
            </div>
            {!liveMode && (
              <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-text-muted/70">
                cached · replay
              </span>
            )}
          </div>
          <ComparisonTriptych
            suppressUnavailableNotice
            panels={[
              {
                title: 'Subject perceived',
                subtitle: 'Reference NSD stimulus',
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
                  : 'This replay contains retrieval evidence only.',
              },
            ]}
          />
        </div>

        {/* ── 4. EVIDENCE SUMMARY — hairline strip, section label sits to
                the left so the strip reads as a labeled report readout. ── */}
        <div className="border-t border-white/[0.05] px-5 py-5 sm:px-6 sm:py-6">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-stretch lg:gap-7">
            {/* Left: kicker rule + section label + quiet caption */}
            <div className="lg:w-[180px] lg:shrink-0 lg:border-r lg:border-white/[0.05] lg:pr-6">
              <div className="flex items-center gap-2.5">
                <span className="h-px w-6 bg-accent/40" aria-hidden />
                <p className="premium-kicker">Evidence summary</p>
              </div>
              <p className="mt-2 text-[10.5px] leading-snug text-text-muted/85">
                Final report values, derived from the gallery rank-1 retrieval.
              </p>
            </div>

            {/* Right: metric strip */}
            <div
              className={`flex-1 grid divide-y divide-white/[0.05] sm:divide-y-0 sm:divide-x ${
                summaryCells.length > 4 ? 'sm:grid-cols-3 lg:grid-cols-6' : 'sm:grid-cols-4'
              }`}
            >
              {summaryCells.map((cell) => (
                <SummaryStripCell key={cell.label} cell={cell} />
              ))}
            </div>
          </div>
        </div>

        {/* ── 5. AUDIT FOOTER — ghost buttons, calibration-light alignment. ── */}
        <div className="border-t border-white/[0.05] px-5 py-3 sm:px-6">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <button
              type="button"
              onClick={onReset}
              className="inline-flex items-center gap-1.5 self-start rounded-lg px-2.5 py-1.5 text-[11.5px] font-medium uppercase tracking-[0.08em] text-text-muted transition hover:bg-white/[0.04] hover:text-text-primary sm:self-auto"
            >
              <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
              </svg>
              Try another trial
            </button>
            <div className="flex flex-wrap items-center gap-0.5">
              <button
                type="button"
                onClick={() => navigate(`/explorer/${case_.id}`)}
                className="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[11.5px] font-medium uppercase tracking-[0.08em] text-text-muted transition hover:bg-white/[0.04] hover:text-text-primary"
              >
                Explore in detail
                <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                </svg>
              </button>
              <span className="hidden h-3 w-px bg-white/[0.05] sm:inline-block" aria-hidden />
              <button
                type="button"
                onClick={() => navigate('/challenge')}
                className="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[11.5px] font-medium uppercase tracking-[0.08em] text-text-muted transition hover:bg-white/[0.04] hover:text-text-primary"
              >
                Take the challenge
                <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
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
