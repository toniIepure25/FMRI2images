import { useMemo, useState } from 'react';
import type { DemoCase } from '@/types';
import { SafeImg } from '@/components/ui/SafeImg';

export interface ReconstructionTabProps {
  case_: DemoCase;
}

type Preset = 'conservative' | 'balanced' | 'creative';

const PRESET_META: Record<Preset, { label: string; caption: string }> = {
  conservative: {
    label: 'Conservative',
    caption: 'Lower effective guidance\u2014closer to median prior, fewer high-frequency details.',
  },
  balanced: {
    label: 'Balanced',
    caption: 'Default DUA-informed schedule matching this trial\u2019s \u03BA and \u03B4.',
  },
  creative: {
    label: 'Creative',
    caption: 'Higher exploration under the same diffusion backbone\u2014richer texture, higher variance.',
  },
};

export function ReconstructionTab({ case_ }: ReconstructionTabProps) {
  const [preset, setPreset] = useState<Preset>('balanced');

  const top1 = useMemo(
    () => case_.retrievedImages.find((r) => r.rank === 1) ?? case_.retrievedImages[0],
    [case_.retrievedImages],
  );

  const mainImage = useMemo(() => {
    if (preset === 'conservative') return case_.conservativeRecon;
    if (preset === 'creative') return case_.creativeRecon;
    return case_.reconstructionImage;
  }, [preset, case_.conservativeRecon, case_.creativeRecon, case_.reconstructionImage]);

  const hasDiffusionPair = !!(case_.diffusionPrior && case_.diffusionFinal);

  const presetRadiogroupId = 'recon-preset-mode';

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div className="glass-panel p-5">
        <h2 className="text-sm font-semibold text-white">Reconstruction comparison</h2>
        <p className="mt-2 text-xs leading-relaxed text-slate-400">
          Diffusion is conditioned on the predicted CLIP direction.{' '}
          <span className="text-slate-300">Uncertainty-Aware CFG (DUA-CFG)</span> maps decoder concentration (\u03BA) and
          inter-ROI disagreement (\u03B4) into guidance scale, sampling steps, and optional ensembling\u2014trading fidelity against
          diversity when brain evidence is ambiguous.
        </p>
      </div>

      <div className="glass-panel p-5" role="radiogroup" aria-labelledby={presetRadiogroupId}>
        <p id={presetRadiogroupId} className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
          Diffusion preset
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          {(Object.keys(PRESET_META) as Preset[]).map((p) => (
            <button
              key={p}
              type="button"
              role="radio"
              aria-checked={preset === p}
              onClick={() => setPreset(p)}
              className={`rounded-lg border px-4 py-2 text-xs font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brain-accent ${
                preset === p
                  ? 'border-brain-accent/50 bg-brain-accent/15 text-brain-accent'
                  : 'border-brain-border/50 bg-slate-950/50 text-slate-400 hover:border-brain-accent/25 hover:text-slate-200'
              }`}
            >
              {PRESET_META[p].label}
            </button>
          ))}
        </div>
        <p className="mt-4 text-[11px] leading-relaxed text-slate-500">{PRESET_META[preset].caption}</p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <CompareCard title="Ground truth" subtitle="Shown during fMRI" src={case_.targetImage} accent="border-cyan-500/30" />
        <CompareCard
          title="Top-1 retrieved"
          subtitle={top1 ? `Rank #${top1.rank}` : '\u2014'}
          src={top1?.image ?? ''}
          accent="border-violet-500/30"
          empty={!top1}
        />
        <CompareCard
          title={`Reconstruction (${PRESET_META[preset].label})`}
          subtitle={`CFG ${case_.duaCfg.guidanceScale.toFixed(1)} \xB7 ${case_.duaCfg.diffusionSteps} steps`}
          src={mainImage}
          accent="border-emerald-500/30"
        />
      </div>

      {hasDiffusionPair && (
        <div className="glass-panel p-5">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Diffusion progression</p>
          <p className="mt-1 text-xs text-slate-400">
            Left: initial latent prior. Right: final denoised reconstruction after {case_.duaCfg.diffusionSteps} steps.
          </p>
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <CompareCard title="Diffusion prior" subtitle="Latent initialization" src={case_.diffusionPrior!} accent="border-amber-500/30" />
            <CompareCard title="Final reconstruction" subtitle="Denoised output" src={case_.diffusionFinal!} accent="border-emerald-500/30" />
          </div>
        </div>
      )}

      <div className="glass-panel p-5">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">DUA-CFG parameters</p>
        <dl className="mt-3 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
          <div className="rounded-lg border border-white/5 bg-slate-950/50 px-3 py-2">
            <dt className="text-[11px] text-slate-500">Guidance scale</dt>
            <dd className="mt-1 font-mono text-white">{case_.duaCfg.guidanceScale.toFixed(1)}</dd>
          </div>
          <div className="rounded-lg border border-white/5 bg-slate-950/50 px-3 py-2">
            <dt className="text-[11px] text-slate-500">Diffusion steps</dt>
            <dd className="mt-1 font-mono text-white">{case_.duaCfg.diffusionSteps}</dd>
          </div>
          <div className="rounded-lg border border-white/5 bg-slate-950/50 px-3 py-2">
            <dt className="text-[11px] text-slate-500">Ensemble K</dt>
            <dd className="mt-1 font-mono text-white">{case_.duaCfg.ensembleK}</dd>
          </div>
          <div className="rounded-lg border border-white/5 bg-slate-950/50 px-3 py-2">
            <dt className="text-[11px] text-slate-500">Abstain</dt>
            <dd className="mt-1 font-mono text-white">{case_.duaCfg.abstain ? 'Yes' : 'No'}</dd>
          </div>
        </dl>
      </div>
    </div>
  );
}

function CompareCard({
  title,
  subtitle,
  src,
  accent,
  empty,
}: {
  title: string;
  subtitle: string;
  src: string;
  accent: string;
  empty?: boolean;
}) {
  return (
    <div className={`glass-panel overflow-hidden ${accent} border`}>
      <div className="border-b border-brain-border/30 px-4 py-3">
        <p className="text-xs font-semibold text-white">{title}</p>
        <p className="mt-0.5 text-[11px] text-slate-500">{subtitle}</p>
      </div>
      <div className="relative aspect-square bg-[#050914]">
        {empty ? (
          <div className="flex h-full items-center justify-center text-sm text-slate-600">\u2014</div>
        ) : (
          <SafeImg src={src} alt={title} className="block h-full w-full object-cover" />
        )}
      </div>
    </div>
  );
}
