import { useMemo } from 'react';
import type { DemoCase } from '@/types';
import { getConfidenceColor, getConfidenceLabel } from '@/lib/data';
import { SafeImg } from '@/components/ui/SafeImg';

export interface OverviewTabProps {
  case_: DemoCase;
}

function heatmapColor(t: number): string {
  const x = Math.max(0, Math.min(1, t));
  const r = Math.round(20 + x * 180);
  const g = Math.round(40 + (1 - x) * 120);
  const b = Math.round(80 + x * 140);
  return `rgb(${r},${g},${b})`;
}

export function OverviewTab({ case_ }: OverviewTabProps) {
  const top1 = useMemo(
    () => case_.retrievedImages.find((r) => r.rank === 1) ?? case_.retrievedImages[0],
    [case_.retrievedImages],
  );

  const fmriNorm = useMemo(() => {
    const v = case_.fmriPreview;
    if (!v.length) return [];
    const min = Math.min(...v);
    const max = Math.max(...v);
    const span = max - min || 1;
    return v.map((x) => (x - min) / span);
  }, [case_.fmriPreview]);

  const confColor = getConfidenceColor(case_.uncertainty.confidenceLevel);
  const confLabel = getConfidenceLabel(case_.uncertainty.confidenceLevel);

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div className="glass-panel p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold tracking-tight text-white">{case_.title}</h1>
            <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">{case_.description}</p>
            <p className="mt-3 text-xs text-slate-500">
              <span className="font-mono text-slate-400">{case_.id}</span>
              <span className="mx-2 text-slate-600">·</span>
              {case_.subject}
              <span className="mx-2 text-slate-600">·</span>
              nsdId {case_.nsdId}
            </p>
          </div>
          <div
            className="rounded-lg border px-3 py-1.5 text-xs font-medium"
            style={{
              borderColor: `${confColor}55`,
              backgroundColor: `${confColor}18`,
              color: confColor,
            }}
          >
            {confLabel}
          </div>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <figure className="glass-panel overflow-hidden p-4 lg:col-span-1">
          <figcaption className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            Ground truth
          </figcaption>
          <div className="relative aspect-square overflow-hidden rounded-lg border border-brain-border/40 bg-black/40">
            <SafeImg src={case_.targetImage} alt="Target stimulus" className="block h-full w-full object-cover" />
          </div>
        </figure>
        <figure className="glass-panel overflow-hidden p-4 lg:col-span-1">
          <figcaption className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            Reconstruction
          </figcaption>
          <div className="relative aspect-square overflow-hidden rounded-lg border border-brain-border/40 bg-black/40">
            <SafeImg src={case_.reconstructionImage} alt="Model reconstruction" className="block h-full w-full object-cover" />
          </div>
        </figure>
        <figure className="glass-panel overflow-hidden p-4 lg:col-span-1">
          <figcaption className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            Top-1 retrieval
          </figcaption>
          <div className="relative aspect-square overflow-hidden rounded-lg border border-brain-border/40 bg-black/40">
            {top1 ? (
              <SafeImg src={top1.image} alt={`Retrieval rank ${top1.rank}`} className="block h-full w-full object-cover" />
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-slate-500">No retrieval</div>
            )}
          </div>
        </figure>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { label: 'Rank', value: `#${case_.metrics.rank}`, hint: 'gallery position' },
          { label: 'Cosine', value: case_.metrics.cosine.toFixed(4), hint: 'embedding similarity' },
          {
            label: 'R@1',
            value: case_.metrics.r1Correct ? 'Correct' : 'Incorrect',
            hint: 'top-1 match',
          },
          { label: 'Confidence', value: confLabel, hint: 'decoder certainty' },
        ].map((m) => (
          <div key={m.label} className="glass-panel px-4 py-3">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">{m.label}</p>
            <p className="mt-1 text-lg font-semibold text-white">{m.value}</p>
            <p className="mt-0.5 text-[11px] text-slate-600">{m.hint}</p>
          </div>
        ))}
      </div>

      <div className="glass-panel p-5">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Interpretation</p>
        <p className="mt-2 text-sm leading-relaxed text-slate-300">{case_.interpretation}</p>
      </div>

      <div className="glass-panel p-5">
        <p className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
          fMRI preview (ROI activity strip)
        </p>
        <div className="flex h-10 w-full overflow-hidden rounded-md border border-brain-border/40 bg-[#050914]">
          {fmriNorm.map((t, i) => (
            <div
              key={i}
              className="h-full flex-1 border-r border-black/30 last:border-r-0"
              style={{ backgroundColor: heatmapColor(t) }}
              title={`${case_.fmriPreview[i]?.toFixed?.(3) ?? ''}`}
            />
          ))}
        </div>
        <p className="mt-2 text-[11px] text-slate-600">
          Normalized voxel-response snapshot for this trial (qualitative; not anatomical).
        </p>
      </div>
    </div>
  );
}
