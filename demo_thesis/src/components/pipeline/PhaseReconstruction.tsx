import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import type { DemoCase } from '@/types';
import { isMetricAvailable } from '@/lib/metrics';
import {
  REPLAY_PROV,
  LIVE_PROV,
  DERIVED_PROV,
  UNKNOWN_PROV,
  type Provenance,
} from '@/lib/provenance';
import {
  metricProvenance,
  uncertaintyProvenance,
  duaCfgProvenance,
  reconstructionProvenance,
} from '@/lib/pipelineNormalize';
import { ProvenanceBadge } from './ProvenanceBadge';
import { MetricCell } from './MetricCell';
import { ComparisonTriptych } from './ComparisonTriptych';

export interface PhaseReconstructionProps {
  case_: DemoCase;
  liveMode?: boolean;
  onReset: () => void;
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

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

  const top1 = useMemo(
    () => case_.retrievedImages.find((r) => r.rank === 1) ?? case_.retrievedImages[0],
    [case_.retrievedImages],
  );
  const reconFinal = case_.diffusionFinal ?? case_.reconstructionImage;
  const hasReconstruction = !!reconFinal;

  const animationSkipRef = useRef(false);
  const [stage, setStage] = useState<'uncertainty' | 'triptych' | 'metrics'>('uncertainty');

  const skipAnimation = useCallback(() => {
    animationSkipRef.current = true;
    setStage('metrics');
  }, []);

  useEffect(() => {
    animationSkipRef.current = false;
    setStage('uncertainty');
    const t1 = setTimeout(() => { if (!animationSkipRef.current) setStage('triptych'); }, 1600);
    const t2 = setTimeout(() => { if (!animationSkipRef.current) setStage('metrics'); }, 3000);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, [case_.id]);

  const showTriptych = stage === 'triptych' || stage === 'metrics';
  const showMetrics = stage === 'metrics';

  const verdict = useMemo(() => {
    if (m.rank === 1) return { text: 'Exact match', cls: 'bg-emerald-500/10 text-emerald-300 ring-1 ring-emerald-500/20' };
    if (m.rank <= 5) return { text: 'Near miss', cls: 'bg-amber-500/8 text-amber-300 ring-1 ring-amber-500/20' };
    return { text: 'Hard case', cls: 'bg-red-500/8 text-red-300 ring-1 ring-red-500/15' };
  }, [m.rank]);

  function getMetricProv(field: 'pixcorr' | 'ssim' | 'cosine' | 'rank'): Provenance {
    if (liveMode) {
      if (field === 'rank') return LIVE_PROV;
      if (field === 'cosine') return LIVE_PROV;
      if (field === 'pixcorr') {
        if (!isMetricAvailable(m.pixcorr)) return { kind: 'unknown', detail: 'Not available — requires reconstruction asset' };
        return { kind: 'replay', detail: 'From cached reconstruction, not current live retrieval' };
      }
      if (field === 'ssim') {
        if (!isMetricAvailable(m.ssim)) return { kind: 'unknown', detail: 'Not available — requires reconstruction asset' };
        return { kind: 'replay', detail: 'From cached reconstruction, not current live retrieval' };
      }
    }
    const p = metricProvenance(case_, field);
    if (field === 'pixcorr' && !isMetricAvailable(m.pixcorr)) {
      return { kind: 'unknown', detail: 'Not available — requires reconstruction asset' };
    }
    if (field === 'ssim' && !isMetricAvailable(m.ssim)) {
      return { kind: 'unknown', detail: 'Not available — requires reconstruction asset' };
    }
    return p;
  }

  return (
    <div className="space-y-5 pb-16">
      {/* ── Phase header ── */}
      <motion.div
        className="flex items-center justify-between"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="space-y-1">
          <div className="flex items-center gap-3">
            <h2 className="text-[22px] font-semibold tracking-tight text-white">
              Reconstruct & compare
            </h2>
            <ProvenanceBadge provenance={liveMode ? { kind: 'derived', detail: 'Live retrieval + cached reconstruction' } : REPLAY_PROV} />
          </div>
          <p className="text-[13px] text-slate-500">
            {hasReconstruction ? 'Cached qualitative reconstruction · retrieval comparison' : 'No reconstruction asset · retrieval comparison only'}
          </p>
        </div>
        {stage !== 'metrics' && (
          <button type="button" onClick={skipAnimation}
            className="rounded-full px-4 py-1.5 text-[11px] font-medium text-slate-500 ring-1 ring-white/[0.06] transition hover:text-white hover:ring-white/[0.15]">
            Skip animation
          </button>
        )}
      </motion.div>

      {/* ── Uncertainty + Diffusion policy row ── */}
      <div className="grid gap-4 lg:grid-cols-2">
        {/* Uncertainty summary */}
        <motion.div
          className="pip-surface p-5"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="mb-4 flex items-center justify-between">
            <p className="text-[12px] font-semibold text-white">Uncertainty summary</p>
            <ProvenanceBadge provenance={uncProv} />
          </div>
          <div className="space-y-4">
            {/* κ */}
            <div>
              <div className="flex items-baseline justify-between gap-2">
                <div className="flex items-baseline gap-2">
                  <span className="font-mono text-[11px] text-slate-500">κ</span>
                  <span className="font-mono text-xl font-semibold text-white">{u.kappa.toFixed(1)}</span>
                </div>
                <span className="text-[10px] text-slate-500">{kappaLabel(kappa01)}</span>
              </div>
              <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-white/[0.04]">
                <motion.div
                  className="h-full rounded-full bg-cyan-400/40"
                  initial={{ width: '0%' }}
                  animate={{ width: `${kappa01 * 100}%` }}
                  transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
                />
              </div>
              <p className="mt-1 text-[9px] text-slate-600">Directional certainty in CLIP space</p>
            </div>

            {/* δ */}
            <div>
              <div className="flex items-baseline justify-between gap-2">
                <div className="flex items-baseline gap-2">
                  <span className="font-mono text-[11px] text-slate-500">δ</span>
                  {liveMode ? (
                    <span className="font-mono text-xl font-semibold text-slate-600">N/A</span>
                  ) : (
                    <span className="font-mono text-xl font-semibold text-white">{u.delta.toFixed(3)}</span>
                  )}
                </div>
                <span className="text-[10px] text-slate-500">
                  {liveMode ? 'V62a MLP — no per-ROI δ' : deltaLabel(u.delta)}
                </span>
              </div>
              {!liveMode && (
                <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-white/[0.04]">
                  <motion.div
                    className="h-full rounded-full bg-amber-400/35"
                    initial={{ width: '0%' }}
                    animate={{ width: `${deltaNorm * 100}%` }}
                    transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
                  />
                </div>
              )}
              <p className="mt-1 text-[9px] text-slate-600">Cross-ROI directional tension</p>
            </div>
          </div>
        </motion.div>

        {/* Diffusion policy */}
        <motion.div
          className="pip-surface p-5"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
        >
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="text-[12px] font-semibold text-white">Diffusion policy</p>
              <p className="mt-0.5 text-[9px] text-slate-600">Derived from uncertainty estimate</p>
            </div>
            <ProvenanceBadge provenance={duaProv} />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <PolicyCell label="Guidance scale" value={duaCfg.guidanceScale.toFixed(2)} detail="Classifier-free guidance" />
            <PolicyCell label="Diffusion steps" value={String(duaCfg.diffusionSteps)} detail="Sampling budget" />
            <PolicyCell label="Ensemble K" value={String(duaCfg.ensembleK)} detail="Reconstruction samples" />
            <PolicyCell
              label="Abstain"
              value={duaCfg.abstain ? 'Abstaining' : 'Active'}
              detail={duaCfg.abstain ? 'Policy abstained' : 'Full decoding enabled'}
              valueColor={duaCfg.abstain ? 'text-rose-300' : 'text-emerald-300'}
            />
          </div>
        </motion.div>
      </div>

      {/* ── Comparison triptych ── */}
      <AnimatePresence>
        {showTriptych && (
          <motion.section
            className="space-y-3"
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ type: 'spring', stiffness: 220, damping: 24 }}
          >
            <div className="flex items-center justify-between">
              <p className="text-[12px] font-semibold text-white">Comparison</p>
              {!liveMode && <span className="text-[9px] text-slate-600">Cached assets</span>}
            </div>
            <ComparisonTriptych
              panels={[
                {
                  title: 'Subject perceived',
                  subtitle: 'Reference stimulus (cached)',
                  imageSrc: case_.targetImage,
                  provenance: { kind: 'replay', detail: 'Cached NSD stimulus image' },
                  accent: 'emerald',
                  revealBlur: true,
                },
                {
                  title: 'Model retrieved',
                  subtitle: `Top-1 gallery image · Rank #${top1?.rank ?? '—'}`,
                  imageSrc: top1?.image,
                  provenance: liveMode ? { kind: 'derived', detail: 'Live retrieval rank · cached gallery image' } : REPLAY_PROV,
                  accent: 'violet',
                },
                {
                  title: hasReconstruction ? 'Cached reconstruction' : 'Reconstruction',
                  subtitle: hasReconstruction ? 'Qualitative AI reconstruction' : 'No reconstruction asset',
                  imageSrc: reconFinal,
                  provenance: reconProv,
                  accent: 'cyan',
                  emptyText: 'No reconstruction asset available for this trial.',
                },
              ]}
            />
          </motion.section>
        )}
      </AnimatePresence>

      {/* ── Trial metrics ── */}
      <AnimatePresence>
        {showMetrics && (
          <motion.div
            className="space-y-5"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ type: 'spring', stiffness: 220, damping: 24 }}
          >
            <div className="pip-surface p-5 sm:p-6">
              <div className="mb-5 flex flex-col items-center justify-between gap-3 sm:flex-row">
                <h3 className="text-[15px] font-semibold text-white">Trial metrics</h3>
                <motion.span
                  className={`flex items-center gap-2 rounded-full px-3.5 py-1.5 text-[11px] font-semibold ${verdict.cls}`}
                  initial={{ opacity: 0, scale: 0.96 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ type: 'spring', stiffness: 400, damping: 26 }}
                >
                  {verdict.text} · Rank #{m.rank}
                </motion.span>
              </div>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <MetricCell
                  label="Rank"
                  value={String(m.rank)}
                  description="Gallery retrieval rank"
                  provenance={getMetricProv('rank')}
                  valueColor={m.rank === 1 ? 'text-emerald-300' : m.rank <= 5 ? 'text-amber-300' : 'text-red-300'}
                />
                <MetricCell
                  label="PixCorr"
                  value={m.pixcorr}
                  description="Pixel correlation"
                  provenance={getMetricProv('pixcorr')}
                />
                <MetricCell
                  label="SSIM"
                  value={m.ssim}
                  description="Structural similarity"
                  provenance={getMetricProv('ssim')}
                />
                <MetricCell
                  label={isMetricAvailable(m.cosine) ? 'Cosine' : 'CSLS'}
                  value={isMetricAvailable(m.cosine) ? m.cosine : m.csls}
                  description={isMetricAvailable(m.cosine) ? 'Top-1 cosine similarity' : 'Top-1 CSLS retrieval score'}
                  provenance={getMetricProv('cosine')}
                />
              </div>
              {!hasReconstruction && (
                <p className="mt-3 text-center text-[10px] text-slate-600">
                  PixCorr and SSIM require a reconstruction asset.
                  {!isMetricAvailable(m.pixcorr) && ' Values shown as — because no reconstruction was available.'}
                </p>
              )}
            </div>

            {/* ── Actions ── */}
            <div className="flex flex-col gap-3 sm:flex-row sm:justify-center">
              <button type="button" onClick={onReset}
                className="rounded-xl px-6 py-3 text-[13px] font-medium text-slate-400 ring-1 ring-white/[0.06] transition hover:text-white hover:ring-white/[0.15]">
                ← Try another trial
              </button>
              <button type="button" onClick={() => navigate(`/explorer/${case_.id}`)}
                className="rounded-xl px-6 py-3 text-[13px] font-medium text-slate-400 ring-1 ring-white/[0.06] transition hover:text-white hover:ring-white/[0.15]">
                Explore in detail →
              </button>
              <button type="button" onClick={() => navigate('/challenge')}
                className="rounded-xl px-6 py-3 text-[13px] font-medium text-slate-400 ring-1 ring-white/[0.06] transition hover:text-white hover:ring-white/[0.15]">
                Take the challenge →
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function PolicyCell({
  label,
  value,
  detail,
  valueColor = 'text-white',
}: {
  label: string;
  value: string;
  detail: string;
  valueColor?: string;
}) {
  return (
    <div className="rounded-xl bg-white/[0.025] p-3.5">
      <p className="text-[9px] font-medium text-slate-500">{label}</p>
      <p className={`mt-1 font-mono text-lg font-bold tabular-nums ${valueColor}`}>{value}</p>
      <p className="mt-0.5 text-[8px] text-slate-600">{detail}</p>
    </div>
  );
}
