import type { InferenceResponse } from '@/lib/api';

interface ManifoldMetricsPanelProps {
  inference: InferenceResponse | null;
}

export function ManifoldMetricsPanel({ inference }: ManifoldMetricsPanelProps) {
  const m = inference?.manifold_metrics;

  if (!m || !m.available) {
    return (
      <div className="rounded-xl border border-border-subtle bg-surface-elevated p-6">
        <h3 className="text-sm font-semibold text-text-primary mb-2">Semantic manifold</h3>
        <div className="flex flex-col items-center gap-3 py-10 text-center">
          <svg className="h-10 w-10 text-border-emphasis" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
          </svg>
          <p className="text-[13px] font-medium text-text-muted">Manifold metrics</p>
          <p className="text-[11px] text-text-muted max-w-xs">
            {m?.reason || 'Computing prediction and target embeddings.'}
          </p>
        </div>
      </div>
    );
  }

  const overlap5 = m.neighborhood_overlap_at_5;
  const density = m.on_manifold_score ?? 0;
  const densityPct = Math.round(Math.max(0, Math.min(1, density)) * 100);

  return (
    <div className="rounded-xl border border-border-subtle bg-surface-elevated p-5 sm:p-6">
      <h3 className="text-sm font-semibold text-text-primary mb-4">Semantic manifold</h3>

      {/* Metric grid */}
      <div className="grid gap-3 sm:grid-cols-3 mb-4">
        <div className="rounded-lg bg-surface-raised px-4 py-3 text-center">
          <p className="text-[10px] font-medium text-text-muted">Overlap @5</p>
          <p className="font-mono text-xl font-bold text-text-primary mt-1">
            {overlap5 != null ? `${(overlap5 * 100).toFixed(0)}%` : '—'}
          </p>
        </div>
        <div className="rounded-lg bg-surface-raised px-4 py-3 text-center">
          <p className="text-[10px] font-medium text-text-muted">Overlap @10</p>
          <p className="font-mono text-xl font-bold text-text-primary mt-1">
            {m.neighborhood_overlap_at_10 != null ? `${(m.neighborhood_overlap_at_10 * 100).toFixed(0)}%` : '—'}
          </p>
        </div>
        <div className="rounded-lg bg-surface-raised px-4 py-3 text-center">
          <p className="text-[10px] font-medium text-text-muted">Overlap @20</p>
          <p className="font-mono text-xl font-bold text-text-primary mt-1">
            {m.neighborhood_overlap_at_20 != null ? `${(m.neighborhood_overlap_at_20 * 100).toFixed(0)}%` : '—'}
          </p>
        </div>
      </div>

      {/* On-manifold score gauge */}
      <div className="rounded-lg bg-surface-raised p-4 mb-4">
        <div className="flex items-center justify-between mb-2">
          <p className="text-[11px] font-medium text-text-muted">On-manifold score</p>
          <span className="font-mono text-[13px] font-semibold text-text-primary">{densityPct}%</span>
        </div>
        <div className="h-2 rounded-full bg-surface-active overflow-hidden">
          <div
            className="h-full rounded-full bg-accent/50"
            style={{ width: `${densityPct}%` }}
          />
        </div>
        <p className="mt-1.5 text-[10px] text-text-muted">
          {density > 0.6 ? 'Well within gallery manifold.' : density > 0.3 ? 'Moderate manifold alignment.' : 'May be off-manifold.'}
        </p>
      </div>

      {/* Target local rank */}
      {m.target_local_rank != null && (
        <div className="rounded-lg bg-surface-raised px-4 py-3 flex items-center justify-between mb-4">
          <span className="text-[11px] font-medium text-text-muted">Target rank in predicted neighborhood</span>
          <span className="font-mono text-lg font-bold text-text-primary">#{m.target_local_rank}</span>
        </div>
      )}

      {/* Manifold density */}
      {m.manifold_density != null && (
        <div className="rounded-lg bg-surface-raised px-4 py-3 flex items-center justify-between mb-4">
          <span className="text-[11px] font-medium text-text-muted">Manifold density</span>
          <span className="font-mono text-[13px] font-semibold text-text-primary">{m.manifold_density.toFixed(4)}</span>
        </div>
      )}

      {/* Interpretation */}
      {m.interpretation && (
        <div className="rounded-lg bg-surface-raised px-4 py-2.5 text-[11px] leading-relaxed text-text-muted">
          {m.interpretation}
        </div>
      )}
    </div>
  );
}
