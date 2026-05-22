import { ManifoldHero } from '@/components/manifold/ManifoldHero';
import { EvidenceOverview } from '@/components/manifold/EvidenceOverview';
import { SemanticProbePanel } from '@/components/manifold/SemanticProbePanel';
import { RetrievalEvidencePanel } from '@/components/manifold/RetrievalEvidencePanel';
import { HubnessReductionPanel } from '@/components/manifold/HubnessReductionPanel';
import { SemanticVectorField } from '@/components/manifold/SemanticVectorField';
import { CounterfactualSemanticLab } from '@/components/manifold/CounterfactualSemanticLab';
import { InterpolationWalk } from '@/components/manifold/InterpolationWalk';
import { FutureWorkModules } from '@/components/manifold/FutureWorkModules';

/**
 * NeuralManifoldExplorer
 * ─────────────────────────────────────────────────────────────
 * Premium scientific-OS page for the V62a fMRI→CLIP decoder,
 * viewed as a point on a semantic manifold rather than a single
 * top-1 score.
 *
 * Section spacing is intentionally tight (mt-10 / space-y-10)
 * so the page reads as one continuous instrument, not a series
 * of separated cards with dead space between them.
 */
export function NeuralManifoldExplorer() {
  return (
    <div className="premium-page-bg relative min-h-screen overflow-hidden pb-20">
      <ManifoldHero />
      <div className="mt-12 space-y-12">
        <EvidenceOverview />
        <SemanticProbePanel />
        <RetrievalEvidencePanel />
        <HubnessReductionPanel />
        <SemanticVectorField />
        <CounterfactualSemanticLab />
        <InterpolationWalk />
        <FutureWorkModules />
      </div>
    </div>
  );
}
