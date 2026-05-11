import { useMemo } from 'react';
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

  const top1 = useMemo(
    () => case_.retrievedImages.find((r) => r.rank === 1) ?? case_.retrievedImages[0],
    [case_.retrievedImages],
  );
  const reconFinal = case_.diffusionFinal ?? case_.reconstructionImage;
  const hasReconstruction = !!reconFinal;

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
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <h2 className="text-[22px] font-semibold tracking-tight text-text-primary">
                Reconstruct &amp; compare
              </h2>
              <ProvenanceBadge provenance={liveMode ? { kind: 'derived', detail: 'Live retrieval + cached reconstruction' } : REPLAY_PROV} />
            </div>
            <p className="text-[13px] text-text-muted">
              {hasReconstruction ? 'Cached qualitative reconstruction' : 'No reconstruction asset'} &middot; retrieval comparison
            </p>
          </div>
        </div>
      </motion.div>

      {/* ── RESULT SUMMARY HERO ── */}
      <motion.div
        className="rounded-xl border border-border-subtle bg-surface-elevated p-6"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          {/* Verdict */}
          <div className="space-y-1">
            <p className="text-[11px] font-semibold uppercase tracking-[0.15em] text-text-muted">Result</p>
            <div className="flex items-baseline gap-3">
              <span className={`font-mono text-5xl font-bold tabular-nums leading-none ${
                m.rank === 1 ? 'text-accent' : m.rank <= 5 ? 'text-status-warning' : 'text-text-primary'
              }`}>
                #{m.rank}
              </span>
              <span className={`rounded-full px-3 py-1 text-[13px] font-semibold ${verdict.bg} ${verdict.cls}`}>
                {verdict.text}
              </span>
            </div>
            <p className="text-[12px] text-text-muted">10,000-gallery CSLS ranking &middot; {case_.subject}</p>
          </div>

          {/* Key metrics inline */}
          <div className="flex flex-wrap gap-4 sm:gap-6">
            <div className="text-center">
              <p className="text-[10px] font-medium text-text-muted">κ</p>
              <p className="font-mono text-2xl font-semibold text-accent">{u.kappa.toFixed(1)}</p>
              <p className="text-[9px] text-text-muted">{kappaLabel(kappa01)}</p>
            </div>
            <div className="text-center">
              <p className="text-[10px] font-medium text-text-muted">δ</p>
              <p className="font-mono text-2xl font-semibold text-text-primary">
                {liveMode ? 'N/A' : u.delta.toFixed(3)}
              </p>
              <p className="text-[9px] text-text-muted">{liveMode ? 'no per-ROI' : deltaLabel(u.delta)}</p>
            </div>
            <div className="text-center">
              <p className="text-[10px] font-medium text-text-muted">CSLS</p>
              <p className="font-mono text-2xl font-semibold text-text-primary">
                {top1?.csls != null ? top1.csls.toFixed(3) : '—'}
              </p>
              <p className="text-[9px] text-text-muted">top-1 score</p>
            </div>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-3 text-[10px]">
          <ProvenanceBadge provenance={uncProv} />
          <ProvenanceBadge provenance={duaProv} />
          <ProvenanceBadge provenance={reconProv} />
        </div>
      </motion.div>

      {/* Comparison triptych */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <p className="text-[13px] font-semibold text-text-primary">Comparison</p>
          {!liveMode && <span className="text-[10px] text-text-muted">Cached assets</span>}
        </div>
        <ComparisonTriptych
          panels={[
            {
              title: 'Subject perceived',
              subtitle: 'Reference stimulus',
              imageSrc: case_.targetImage,
              provenance: { kind: 'replay', detail: 'Cached NSD stimulus image' },
              accent: 'emerald',
              revealBlur: true,
            },
            {
              title: 'Model retrieved',
              subtitle: `Top-1 gallery · Rank #${top1?.rank ?? '—'}`,
              imageSrc: top1?.image,
              provenance: liveMode ? { kind: 'derived', detail: 'Live retrieval rank · cached gallery image' } : REPLAY_PROV,
              accent: 'violet',
            },
            {
              title: hasReconstruction ? 'Reconstruction' : 'Reconstruction',
              subtitle: hasReconstruction ? 'Stable Diffusion 2.1 output' : 'Asset not available',
              imageSrc: reconFinal,
              provenance: reconProv,
              accent: 'cyan',
              emptyText: 'No reconstruction asset was cached for this trial.\nThis replay contains retrieval evidence only.',
            },
          ]}
        />
      </div>

      {/* Trial metrics */}
      <div className="surface-card p-5 sm:p-6">
        <div className="mb-5 flex flex-col items-center justify-between gap-3 sm:flex-row">
          <h3 className="text-base font-semibold text-text-primary">Trial metrics</h3>
          <span className={`inline-flex items-center gap-2 rounded-full px-3.5 py-1.5 text-[11px] font-semibold ${verdict.bg} ${verdict.cls}`}>
            {verdict.text} &middot; Rank #{m.rank}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <MetricCell label="Rank" value={String(m.rank)} description="Gallery retrieval rank"
            provenance={getMetricProv('rank')}
            valueColor={m.rank === 1 ? 'text-accent' : m.rank <= 5 ? 'text-status-warning' : 'text-status-error'} />
          <MetricCell label="PixCorr" value={m.pixcorr} description="Pixel correlation" provenance={getMetricProv('pixcorr')} />
          <MetricCell label="SSIM" value={m.ssim} description="Structural similarity" provenance={getMetricProv('ssim')} />
          <MetricCell
            label={isMetricAvailable(m.cosine) ? 'Cosine' : 'CSLS'}
            value={isMetricAvailable(m.cosine) ? m.cosine : m.csls}
            description={isMetricAvailable(m.cosine) ? 'Top-1 cosine similarity' : 'Top-1 CSLS score'}
            provenance={getMetricProv('cosine')} />
        </div>
        {!hasReconstruction && (
          <p className="mt-3 text-center text-[11px] text-text-muted">
            PixCorr and SSIM require a reconstruction asset.
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

function PolicyCell({
  label, value, detail, valueColor = 'text-text-primary',
}: { label: string; value: string; detail: string; valueColor?: string }) {
  return (
    <div className="rounded-lg bg-surface-raised p-3.5">
      <p className="text-[9px] font-medium text-text-muted">{label}</p>
      <p className={`mt-1 font-mono text-lg font-bold tabular-nums ${valueColor}`}>{value}</p>
      <p className="mt-0.5 text-[8px] text-text-muted">{detail}</p>
    </div>
  );
}
