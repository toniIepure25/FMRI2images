import { motion } from 'framer-motion';
import type { ReactNode } from 'react';
import {
  retrievalMetrics, semanticProbe, semanticAxes, semanticAxisAggregates,
  counterfactualMetrics, interpolationMetrics, claims, caveats,
} from '@/data/manifoldEvidence';
import { manifoldEvidenceGenerated as D } from '@/data/generated/manifoldEvidence.generated';

/* ─── Reusable helpers ─── */

function SectionTitle({ children }: { children: ReactNode }) {
  return <h3 className="text-sm font-semibold text-text-primary mb-3">{children}</h3>;
}

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, string> = {
    supported: 'bg-status-success/10 text-status-success ring-1 ring-status-success/20',
    partially_supported: 'bg-status-warning/8 text-status-warning ring-1 ring-status-warning/15',
    partial: 'bg-status-warning/8 text-status-warning ring-1 ring-status-warning/15',
    not_supported: 'bg-status-error/6 text-status-error ring-1 ring-status-error/12',
    unavailable: 'bg-surface-raised text-text-muted ring-1 ring-border-subtle',
  };
  return (
    <span className={`rounded px-2 py-0.5 text-[10px] font-semibold ${map[status] ?? map.unavailable}`}>
      {status.replace(/_/g, ' ')}
    </span>
  );
}

function MetricBadge({ label, value, unit, highlight }: { label: string; value: string | number; unit?: string; highlight?: boolean }) {
  return (
    <div className={`rounded-lg px-3 py-2 text-center ${highlight ? 'bg-accent/8 ring-1 ring-accent/15' : 'bg-surface-raised'}`}>
      <p className="text-[10px] font-medium text-text-muted">{label}</p>
      <p className={`font-mono text-lg font-bold mt-0.5 ${highlight ? 'text-accent' : 'text-text-primary'}`}>
        {typeof value === 'number' ? (value < 0.01 ? value.toExponential(2) : value.toFixed(4)) : value}{unit ?? ''}
      </p>
    </div>
  );
}

function CaveatBanner({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-lg bg-surface-raised px-3 py-2 text-[11px] leading-relaxed text-text-muted italic border-l-2 border-border-emphasis">
      {children}
    </div>
  );
}

function ClaimRow({ claim }: { claim: typeof claims[number] }) {
  return (
    <div className="flex items-center gap-3 rounded-lg bg-surface-raised px-4 py-2.5">
      <span className="w-8 text-[10px] font-mono font-semibold text-text-muted">{claim.id}</span>
      <div className="flex-1 min-w-0">
        <p className="text-[11px] font-semibold text-text-primary">{claim.label}</p>
        {claim.note && <p className="text-[10px] text-text-muted mt-0.5">{claim.note}</p>}
      </div>
      <div className="flex items-center gap-1.5 shrink-0">
        <span className="text-[9px] text-text-muted">val</span>
        <StatusBadge status={claim.validation} />
        <span className="text-[9px] text-text-muted ml-1">s1k</span>
        <StatusBadge status={claim.shared1000} />
      </div>
    </div>
  );
}

/* ─── Section component ─── */

export function ManifoldEvidenceSection() {
  return (
    <motion.section
      className="space-y-6 py-8"
      initial={{ opacity: 0 }} animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
    >
      {/* Section header */}
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-text-primary">Neural Manifold Evidence</h2>
        <p className="mt-2 text-sm text-text-secondary max-w-3xl">
          Beyond top-1 retrieval: scientific evidence that the decoded CLIP embeddings preserve
          semantic geometry, interpretable concept dimensions, and coherent latent structure.
        </p>
      </div>

      {/* ── 1. Retrieval + CSLS + Hubness ── */}
      <div className="grid gap-4 lg:grid-cols-2">
        {/* Retrieval */}
        <div className="rounded-xl border border-border-subtle bg-surface-elevated p-5">
          <div className="flex items-center gap-2 mb-4">
            <SectionTitle>Retrieval &amp; beyond R@1</SectionTitle>
            <StatusBadge status="supported" />
          </div>
          <div className="grid grid-cols-3 gap-2 mb-3">
            <MetricBadge label="CSLS R@1 val" value={retrievalMetrics.validation.cslsR1} highlight />
            <MetricBadge label="CSLS R@5 s1k" value={retrievalMetrics.shared1000.cslsR5} />
            <MetricBadge label="RSA Spearman val" value={D.rsa.validation.spearmanRho} />
          </div>
          <div className="grid grid-cols-3 gap-2">
            <MetricBadge label="CSLS R@1 s1k" value={retrievalMetrics.shared1000.cslsR1} />
            <MetricBadge label="Neighbor Overlap@10" value={D.neighborhood.shared1000.overlapAt10} />
            <MetricBadge label="Lift over random" value={`${D.neighborhood.shared1000.overlapAt10Lift.toFixed(1)}×`} />
          </div>
          <p className="mt-3 text-[11px] leading-relaxed text-text-muted">
            CSLS retrieval outperforms cosine and preserves local semantic neighborhoods significantly above random baseline.
          </p>
        </div>

        {/* Hubness */}
        <div className="rounded-xl border border-border-subtle bg-surface-elevated p-5">
          <div className="flex items-center gap-2 mb-4">
            <SectionTitle>Hubness reduction</SectionTitle>
            <StatusBadge status="supported" />
          </div>
          <p className="text-[10px] text-text-muted mb-2">SHARED1000</p>
          <div className="grid grid-cols-3 gap-2 mb-2">
            <MetricBadge label="Cosine Gini" value={retrievalMetrics.hubness.shared1000.cosineGini} />
            <MetricBadge label="CSLS Gini" value={retrievalMetrics.hubness.shared1000.cslsGini} highlight />
            <MetricBadge label="Reduction" value={retrievalMetrics.hubness.shared1000.derivedGiniReduction} />
          </div>
          <p className="text-[10px] text-text-muted mb-2">Validation</p>
          <div className="grid grid-cols-3 gap-2">
            <MetricBadge label="Cosine Gini" value={retrievalMetrics.hubness.validation.cosineGini} />
            <MetricBadge label="CSLS Gini" value={retrievalMetrics.hubness.validation.cslsGini} />
            <MetricBadge label="Reduction" value={retrievalMetrics.hubness.validation.derivedGiniReduction} />
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-text-muted">
            CSLS reduces hubness on both splits. Gini drops more on SHARED1000 (0.386→0.226) than validation (0.600→0.517).
          </p>
        </div>
      </div>

      {/* ── 2. Text Probe ── */}
      <div className="rounded-xl border border-border-subtle bg-surface-elevated p-5">
        <div className="flex items-center gap-2 mb-4">
          <SectionTitle>CLIP Text Probe</SectionTitle>
          <StatusBadge status="supported" />
          <span className="rounded bg-accent/10 px-2 py-0.5 text-[9px] font-semibold text-accent">Not mind reading</span>
        </div>
        <p className="text-[12px] text-text-muted mb-4">
          What does the decoded brain embedding mean in language space? We compare predicted CLIP embeddings against text concept vectors.
        </p>
        <div className="grid grid-cols-3 gap-2 sm:grid-cols-6 mb-3">
          <MetricBadge label="Agreement @1" value={semanticProbe.shared1000.agreementAt1} />
          <MetricBadge label="Agreement @5" value={semanticProbe.shared1000.agreementAt5} highlight />
          <MetricBadge label="Agreement @10" value={semanticProbe.shared1000.agreementAt10} />
          <MetricBadge label="Score ρ" value={semanticProbe.shared1000.scoreVectorSpearman} />
          <MetricBadge label="Score r" value={semanticProbe.shared1000.scoreVectorPearson} />
          <MetricBadge label="JS div" value={semanticProbe.shared1000.jsDivergenceMean} />
        </div>
        <CaveatBanner>CLIP language probe — not reading thoughts, only comparing decoded vectors to text concept embeddings.</CaveatBanner>
      </div>

      {/* ── 3. Semantic Axes ── */}
      <div className="rounded-xl border border-border-subtle bg-surface-elevated p-5">
        <div className="flex items-center gap-2 mb-4">
          <SectionTitle>Semantic Axes</SectionTitle>
          <StatusBadge status="supported" />
        </div>
        <p className="text-[12px] text-text-muted mb-4">
          Interpretable concept dimensions are preserved in the decoded CLIP space.
          All values read directly from semantic_axes_metrics.json — no manual approximations.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px]">
            <thead>
              <tr className="border-b border-border-subtle text-text-muted">
                <th className="text-left py-1.5 pr-3">Axis</th>
                <th className="text-right py-1.5 px-2">val ρ</th>
                <th className="text-right py-1.5 px-2">s1k ρ</th>
                <th className="text-right py-1.5 px-2">val MAE</th>
                <th className="text-right py-1.5 px-2">s1k MAE</th>
              </tr>
            </thead>
            <tbody>
              {semanticAxes.map((axis) => (
                <tr key={axis.name} className="border-b border-border-subtle/50 hover:bg-surface-raised">
                  <td className="py-1.5 pr-3 font-medium text-text-secondary">{axis.name.replace(/_/g, ' ')}</td>
                  <td className="py-1.5 px-2 text-right font-mono text-text-primary">{axis.spearman.validation.toFixed(3)}</td>
                  <td className="py-1.5 px-2 text-right font-mono text-text-primary">{axis.spearman.shared1000.toFixed(3)}</td>
                  <td className="py-1.5 px-2 text-right font-mono text-text-muted">{axis.mae.validation.toFixed(4)}</td>
                  <td className="py-1.5 px-2 text-right font-mono text-text-muted">{axis.mae.shared1000.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <CaveatBanner>All {semanticAxes.length} axes from semantic_axes_metrics.json shown. Mean Spearman: val {semanticAxisAggregates.validation.meanSpearman.toFixed(3)}, s1k {semanticAxisAggregates.shared1000.meanSpearman.toFixed(3)}.</CaveatBanner>
      </div>

      {/* ── 4. Counterfactuals ── */}
      <div className="rounded-xl border border-border-subtle bg-surface-elevated p-5">
        <div className="flex items-center gap-2 mb-4">
          <SectionTitle>Counterfactual latent editing</SectionTitle>
          <StatusBadge status="supported" />
        </div>
        <p className="text-[12px] text-text-muted mb-3">
          z_modified = normalize(z_pred + α · semantic_axis). Editing decoded CLIP vectors along interpretable axes.
        </p>
        <div className="grid grid-cols-4 gap-2 mb-3">
          <MetricBadge label="Margin mean" value={counterfactualMetrics.shared1000.robustnessMarginMean} />
          <MetricBadge label="Margin median" value={counterfactualMetrics.shared1000.robustnessMarginMedian} />
          <MetricBadge label="Transition rate" value={counterfactualMetrics.shared1000.transitionRate} />
          <MetricBadge label="N transitions" value={counterfactualMetrics.shared1000.nTransitionsObserved} highlight />
        </div>
        <CaveatBanner>Latent-space counterfactual only — does not manipulate brain activity.</CaveatBanner>
      </div>

      {/* ── 5. Interpolation ── */}
      <div className="rounded-xl border border-border-subtle bg-surface-elevated p-5">
        <div className="flex items-center gap-2 mb-4">
          <SectionTitle>Walking on the decoded manifold</SectionTitle>
          <StatusBadge status="supported" />
        </div>
        <p className="text-[12px] text-text-muted mb-3">
          Interpolation (slerp) between decoded embeddings reveals smooth semantic trajectories with rare abrupt transitions.
        </p>
        <div className="grid grid-cols-4 gap-2 mb-3">
          <MetricBadge label="Smoothness" value={interpolationMetrics.shared1000.meanSmoothness} highlight />
          <MetricBadge label="Abrupt rate" value={interpolationMetrics.shared1000.abruptTransitionRate} />
          <MetricBadge label="Velocity" value={interpolationMetrics.shared1000.meanSemanticVelocity.toExponential(2)} />
          <MetricBadge label="Transitions" value={interpolationMetrics.shared1000.topConceptTransitionCount} />
        </div>
        <CaveatBanner>Latent CLIP trajectory — not a real neural trajectory.</CaveatBanner>
      </div>

      {/* ── 6. Claims Dashboard ── */}
      <div className="rounded-xl border border-border-subtle bg-surface-elevated p-5">
        <SectionTitle>Scientific claims (C01–C14)</SectionTitle>
        <div className="space-y-1.5 mt-3">
          {claims.map((c) => <ClaimRow key={c.id} claim={c} />)}
        </div>
      </div>

      {/* ── 7. Scientific Honesty ── */}
      <div className="rounded-xl border border-border-subtle bg-surface-elevated p-5">
        <SectionTitle>Scientific honesty</SectionTitle>
        <div className="space-y-2 mt-3">
          {caveats.map((c, i) => (
            <div key={i} className="flex items-start gap-2 text-[11px] text-text-muted">
              <span className="text-border-emphasis mt-0.5">—</span>
              <span>{c}</span>
            </div>
          ))}
        </div>
      </div>
    </motion.section>
  );
}
