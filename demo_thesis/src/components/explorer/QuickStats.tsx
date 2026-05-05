import type { DemoCase } from '@/types';
import { getConfidenceColor, getConfidenceLabel, getDifficultyColor } from '@/lib/data';

export interface QuickStatsProps {
  case_: DemoCase;
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-brain-border/20 py-2.5 last:border-b-0">
      <span className="text-[10px] font-medium uppercase tracking-wider text-slate-500">{label}</span>
      <span className={`text-right text-xs text-slate-200 ${mono ? 'font-mono' : ''}`}>{value}</span>
    </div>
  );
}

export function QuickStats({ case_ }: QuickStatsProps) {
  const confColor = getConfidenceColor(case_.uncertainty.confidenceLevel);
  const diffColor = getDifficultyColor(case_.difficulty);

  return (
    <div className="space-y-4">
      <h2 className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">At a glance</h2>

      <div
        className="glass-panel px-3 py-2 text-center"
        style={{
          borderColor: `${confColor}44`,
          background: `linear-gradient(135deg, ${confColor}12, transparent)`,
        }}
      >
        <p className="text-[9px] uppercase tracking-wider text-slate-500">Confidence</p>
        <p className="mt-1 text-xs font-semibold" style={{ color: confColor }}>
          {getConfidenceLabel(case_.uncertainty.confidenceLevel)}
        </p>
      </div>

      <div className="glass-panel px-3 py-3">
        <Row label="Rank" value={`#${case_.metrics.rank}`} mono />
        <Row label="Cosine" value={case_.metrics.cosine.toFixed(4)} mono />
        <Row label="CSLS" value={case_.metrics.csls.toFixed(4)} mono />
        <Row label="κ" value={case_.uncertainty.kappa.toFixed(3)} mono />
        <Row label="δ" value={case_.uncertainty.delta.toFixed(3)} mono />
        <Row label="CFG scale" value={case_.duaCfg.guidanceScale.toFixed(2)} mono />
        <Row label="Steps" value={String(case_.duaCfg.diffusionSteps)} mono />
        <Row label="Ensemble K" value={String(case_.duaCfg.ensembleK)} mono />
        <Row label="Subject" value={case_.subject} />
        <div className="flex items-center justify-between gap-3 border-b border-brain-border/20 py-2.5 last:border-b-0">
          <span className="text-[10px] font-medium uppercase tracking-wider text-slate-500">Difficulty</span>
          <span
            className="rounded-md border px-2 py-0.5 text-[10px] font-semibold capitalize"
            style={{ borderColor: `${diffColor}55`, color: diffColor }}
          >
            {case_.difficulty}
          </span>
        </div>
      </div>
    </div>
  );
}
