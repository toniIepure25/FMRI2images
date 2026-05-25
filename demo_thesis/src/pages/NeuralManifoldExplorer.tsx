import { ManifoldHero } from '@/components/manifold/ManifoldHero';
import { ThesisClaim } from '@/components/manifold/ThesisClaim';
import { StoryNav } from '@/components/manifold/StoryNav';
import { EvidenceOverview } from '@/components/manifold/EvidenceOverview';
import { SemanticProbePanel } from '@/components/manifold/SemanticProbePanel';
import { RetrievalEvidencePanel } from '@/components/manifold/RetrievalEvidencePanel';
import { HubnessReductionPanel } from '@/components/manifold/HubnessReductionPanel';
import { SemanticVectorField } from '@/components/manifold/SemanticVectorField';
import { CounterfactualSemanticLab } from '@/components/manifold/CounterfactualSemanticLab';
import { InterpolationWalk } from '@/components/manifold/InterpolationWalk';
import { HonestyStrip } from '@/components/manifold/HonestyStrip';
import { FutureWorkModules } from '@/components/manifold/FutureWorkModules';

/**
 * NeuralManifoldExplorer
 * ─────────────────────────────────────────────────────────────
 * Premium scientific-OS page for the V62a fMRI→CLIP decoder,
 * viewed as a point on a semantic manifold rather than a single
 * top-1 score.
 *
 * Reading order:
 *   1. Hero          — thesis statement + report capsule.
 *   2. ThesisClaim   — evaluation contract plaque (five domains).
 *   3. StoryNav      — eight-step analysis sequence with anchors.
 *   4. Overview      — evidence matrix (R@1 is not enough).
 *   5. Probe         — language readout.
 *   6. Retrieval     — top-K evidence board.
 *   7. Hubness       — Lorenz before/after.
 *   8. Error field   — schematic CLIP plane with selected callout.
 *   9. Counterfactual — latent-space axis lab.
 *   10. Walk         — manifold interpolation.
 *   11. Scope        — honesty contract.
 *   12. Future work  — explicitly locked modules.
 */
export function NeuralManifoldExplorer() {
  return (
    <div className="premium-page-bg relative min-h-screen overflow-hidden pb-20">
      <ManifoldHero />
      <div className="mt-10 space-y-10">
        <ThesisClaim />
        <StoryNav />
        <EvidenceOverview />
        <SemanticProbePanel />
        <RetrievalEvidencePanel />
        <HubnessReductionPanel />
        <SemanticVectorField />
        <CounterfactualSemanticLab />
        <InterpolationWalk />
        <HonestyStrip />
        <FutureWorkModules />
      </div>
    </div>
  );
}
