import { useMemo, useState } from 'react';
import type { DemoCase, RoiScore, RoiDefinition } from '@/types';
import { BrainViewer3D } from '@/components/BrainViewer3D';
import { useRoiLayout } from '@/lib/hooks';
import { resolveRoiColor } from '@/components/brain/roiColors';

export interface BrainTabProps {
  case_: DemoCase;
}

export function BrainTab({ case_ }: BrainTabProps) {
  const { layout } = useRoiLayout();
  const [selectedRoi, setSelectedRoi] = useState<string | null>(null);
  const [viewerKey, setViewerKey] = useState(0);
  const [consensusMode, setConsensusMode] = useState(false);

  const deltaForViewer = useMemo(() => {
    if (consensusMode) return 0;
    const d = case_.uncertainty.delta;
    if (d <= 1) return Math.max(0, d);
    return Math.min(1, d / (d + 1));
  }, [case_.uncertainty.delta, consensusMode]);

  const roiDefByName = useMemo(() => {
    const m = new Map<string, RoiDefinition>();
    for (const r of layout?.rois ?? []) m.set(r.name, r);
    return m;
  }, [layout]);

  const scoreByName = useMemo(() => {
    const m = new Map<string, RoiScore>();
    for (const s of case_.roiScores) m.set(s.name, s);
    return m;
  }, [case_.roiScores]);

  const selectedDetail = useMemo(() => {
    if (!selectedRoi) return null;
    const def = roiDefByName.get(selectedRoi);
    const score = scoreByName.get(selectedRoi);
    if (!def && !score) return null;
    return { def, score };
  }, [selectedRoi, roiDefByName, scoreByName]);

  return (
    <div className="mx-auto max-w-6xl space-y-4">
      <div className="glass-panel p-4">
        <p className="text-sm text-slate-300">
          Interactive glass brain with ROI markers. Marker dynamics reflect activation, contribution, and regional
          agreement. Toggle <span className="font-medium text-cyan-300/90">consensus mode</span> to visualize a
          synchronized field (δ → 0) versus the trial&apos;s measured disagreement.
        </p>
      </div>

      <BrainViewer3D
        key={viewerKey}
        roiScores={case_.roiScores}
        delta={deltaForViewer}
        selectedRoi={selectedRoi}
        onRoiSelect={setSelectedRoi}
        autoRotate
        className="neon-border"
      />

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => setViewerKey((k) => k + 1)}
          className="rounded-lg border border-brain-border/60 bg-slate-950/70 px-4 py-2 text-xs font-medium text-slate-200 transition hover:border-brain-accent/40 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brain-accent"
        >
          Reset camera
        </button>
        <button
          type="button"
          onClick={() => setConsensusMode((v) => !v)}
          className={`rounded-lg border px-4 py-2 text-xs font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brain-accent ${
            consensusMode
              ? 'border-emerald-500/50 bg-emerald-500/15 text-emerald-200'
              : 'border-brain-border/60 bg-slate-950/70 text-slate-200 hover:border-brain-accent/40'
          }`}
        >
          {consensusMode ? 'Consensus mode on' : 'Consensus mode off'}
        </button>
        <span className="text-[11px] text-slate-500">
          δ (trial) = {case_.uncertainty.delta.toFixed(4)} → viewer: {deltaForViewer.toFixed(3)}
        </span>
      </div>

      {!selectedDetail && (
        <div className="glass-panel px-5 py-10 text-center">
          <p className="text-sm leading-relaxed text-slate-400">
            Select an ROI marker on the brain viewer to open activation, contribution, and agreement details for that region.
          </p>
        </div>
      )}

      {selectedDetail && layout && (
        <div className="glass-panel p-5">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-wide text-cyan-300/90">
                {selectedDetail.def?.fullName ?? selectedDetail.score?.name ?? selectedRoi}
              </p>
              {selectedDetail.def?.category && (
                <span
                  className="mt-2 inline-flex items-center rounded-md border px-2 py-0.5 text-[11px]"
                  style={{
                    borderColor: `${resolveRoiColor(selectedDetail.def.category, layout, selectedDetail.def.color)}55`,
                    color: resolveRoiColor(selectedDetail.def.category, layout, selectedDetail.def.color),
                  }}
                >
                  {selectedDetail.def.category}
                </span>
              )}
            </div>
            <button
              type="button"
              onClick={() => setSelectedRoi(null)}
              className="shrink-0 rounded-md border border-slate-600/80 bg-slate-900/60 px-2 py-1 text-[11px] text-slate-400 transition hover:text-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brain-accent"
            >
              Clear
            </button>
          </div>
          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            {selectedDetail.score && (
              <>
                <MetricChip label="Activation" value={selectedDetail.score.activation.toFixed(3)} />
                <MetricChip label="Contribution" value={selectedDetail.score.contribution.toFixed(3)} />
                <MetricChip label="Agreement" value={selectedDetail.score.agreement.toFixed(3)} />
              </>
            )}
          </div>
          <p className="mt-4 text-xs leading-relaxed text-slate-300">
            {selectedDetail.score?.interpretation ??
              selectedDetail.def?.functionSummary ??
              selectedDetail.def?.description ??
              '—'}
          </p>
        </div>
      )}
    </div>
  );
}

function MetricChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-brain-border/40 bg-black/25 px-3 py-2">
      <p className="text-[11px] uppercase tracking-wider text-slate-500">{label}</p>
      <p className="mt-0.5 font-mono text-sm text-slate-200">{value}</p>
    </div>
  );
}
