/**
 * Export manifold evidence data from source JSON reports.
 * Usage: node scripts/export_manifold_evidence_data.mjs
 * 
 * Reads: ../reports/manifold_analysis/20260515_V62a_{val,shared1000}_full_semantic/
 * Writes: src/data/generated/manifoldEvidence.generated.ts + .provenance.json
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DEMO_DIR = resolve(__dirname, '..');
const REPORTS = resolve(DEMO_DIR, '..', 'reports', 'manifold_analysis');
const OUT_DIR = resolve(DEMO_DIR, 'src', 'data', 'generated');

const SPLITS = {
  validation: resolve(REPORTS, '20260515_V62a_val_full_semantic'),
  shared1000: resolve(REPORTS, '20260515_V62a_shared1000_full_semantic'),
};

function readJSON(path) {
  return JSON.parse(readFileSync(path, 'utf-8'));
}

function extractMetric(json, path) {
  // Flatten key paths like ['csls', 'R@1']
  let v = json;
  for (const k of path) v = v[k];
  return v;
}

const provenance = {};
const evidence = { retrieval: {}, rsa: {}, neighborhood: {}, hubness: {}, semanticProbe: {}, semanticAxes: {}, counterfactual: {}, interpolation: {}, claimTests: {}, provenance: {} };

for (const [split, dir] of Object.entries(SPLITS)) {
  // Retrieval
  const ret = readJSON(resolve(dir, 'retrieval_metrics.json'));
  evidence.retrieval[split] = {
    cosineR1: ret.cosine['R@1'],
    cosineR5: ret.cosine['R@5'],
    cosineR10: ret.cosine['R@10'],
    cosineMeanRank: ret.cosine['mean_rank'],
    cslsR1: ret.csls['R@1'],
    cslsR5: ret.csls['R@5'],
    cslsR10: ret.csls['R@10'],
  };
  provenance.retrieval = { sourceFile: 'retrieval_metrics.json', fieldsExtracted: Object.keys(evidence.retrieval[split]) };

  // RSA
  const rsa = readJSON(resolve(dir, 'rsa_metrics.json'));
  evidence.rsa[split] = {
    spearmanRho: rsa.spearman_rho,
    pearsonRho: rsa.pearson_rho,
  };
  provenance.rsa = { sourceFile: 'rsa_metrics.json', fieldsExtracted: Object.keys(evidence.rsa[split]) };

  // Neighborhood
  const nb = readJSON(resolve(dir, 'neighborhood_metrics.json'));
  evidence.neighborhood[split] = {
    overlapAt5: nb['overlap@5_mean'],
    overlapAt5Lift: nb['overlap@5_lift_over_random'],
    overlapAt10: nb['overlap@10_mean'],
    overlapAt10Lift: nb['overlap@10_lift_over_random'],
    overlapAt20: nb['overlap@20_mean'],
    overlapAt20Lift: nb['overlap@20_lift_over_random'],
  };
  provenance.neighborhood = { sourceFile: 'neighborhood_metrics.json', fieldsExtracted: Object.keys(evidence.neighborhood[split]) };

  // Hubness
  const hub = readJSON(resolve(dir, 'hubness_metrics.json'));
  evidence.hubness[split] = {
    cosineGini: hub.cosine_gini,
    cslsGini: hub.csls_gini,
    derivedGiniReduction: hub.cosine_gini - hub.csls_gini,
  };
  provenance.hubness = { sourceFile: 'hubness_metrics.json', fieldsExtracted: Object.keys(evidence.hubness[split]), derivedFields: ['derivedGiniReduction = cosine_gini - csls_gini'] };

  // Semantic probe
  const sp = readJSON(resolve(dir, 'semantic_probe_metrics.json'));
  evidence.semanticProbe[split] = {
    agreementAt1: sp['agreement@1'],
    agreementAt5: sp['agreement@5'],
    agreementAt10: sp['agreement@10'],
    scoreVectorSpearman: sp.score_vector_spearman,
    scoreVectorPearson: sp.score_vector_pearson,
    jsDivergenceMean: sp.js_divergence_mean,
    nConcepts: sp.n_concepts,
  };
  provenance.semanticProbe = { sourceFile: 'semantic_probe_metrics.json', fieldsExtracted: Object.keys(evidence.semanticProbe[split]) };

  // Semantic axes
  const sa = readJSON(resolve(dir, 'semantic_axes_metrics.json'));
  evidence.semanticAxes[split] = {
    nAxesBuilt: sa.n_axes_built,
    meanSpearman: sa.mean_axis_spearman,
    meanPearson: sa.mean_axis_pearson,
    meanPreservation: sa.mean_preservation,
    meanMae: sa.mean_axis_mae,
    bestAxis: sa.best_axis,
    weakestAxis: sa.weakest_axis,
    perAxis: Object.fromEntries(
      sa.axis_names.map(name => {
        const v = sa.per_axis[name];
        return [name, { spearmanR: v.spearman_r, pearsonR: v.pearson_r, mae: v.mae, preservation: sa.mean_preservation ? v.spearman_r : null }];
      })
    ),
  };
  provenance.semanticAxes = { sourceFile: 'semantic_axes_metrics.json', fieldsExtracted: ['nAxesBuilt', 'meanSpearman', 'meanPearson', 'meanMae', 'meanPreservation', 'bestAxis', 'weakestAxis', 'perAxis[]'] };

  // Counterfactual
  const cf = readJSON(resolve(dir, 'counterfactual_metrics.json'));
  evidence.counterfactual[split] = {
    robustnessMarginMean: cf.robustness_margin_mean,
    robustnessMarginMedian: cf.robustness_margin_median,
    transitionRate: cf.transition_rate,
    nTransitionsObserved: cf.n_transitions_observed,
  };
  provenance.counterfactual = { sourceFile: 'counterfactual_metrics.json', fieldsExtracted: Object.keys(evidence.counterfactual[split]) };

  // Interpolation
  const ip = readJSON(resolve(dir, 'interpolation_metrics.json'));
  evidence.interpolation[split] = {
    meanSmoothness: ip.mean_smoothness,
    abruptTransitionRate: ip.abrupt_transition_rate,
    meanSemanticVelocity: ip.mean_semantic_velocity,
    topConceptTransitionCount: ip.top_concept_transition_count,
  };
  provenance.interpolation = { sourceFile: 'interpolation_metrics.json', fieldsExtracted: Object.keys(evidence.interpolation[split]) };

  // Claims
  const ct = readJSON(resolve(dir, 'claim_tests.json'));
  evidence.claimTests[split] = ct.map(c => ({
    claimId: c.claim_id,
    claimText: c.claim_text,
    status: c.status,
    explanation: c.explanation || '',
  }));
  provenance.claimTests = { sourceFile: 'claim_tests.json', fieldsExtracted: ['claimId', 'claimText', 'status', 'explanation'] };
}

// Add metadata
evidence.provenance = {
  generatedAt: new Date().toISOString(),
  sourceDirectories: SPLITS,
  roundingPolicy: 'raw values preserved; UI handles formatting',
  ...provenance,
};

// Write generated TS
mkdirSync(OUT_DIR, { recursive: true });
const ts = `// Auto-generated from ${Object.entries(SPLITS).map(([,d]) => d).join(', ')}
// Generated at: ${evidence.provenance.generatedAt}
// DO NOT EDIT MANUALLY. Run: npm run export:manifold-evidence

export const manifoldEvidenceGenerated = ${JSON.stringify({ retrieval: evidence.retrieval, rsa: evidence.rsa, neighborhood: evidence.neighborhood, hubness: evidence.hubness, semanticProbe: evidence.semanticProbe, semanticAxes: evidence.semanticAxes, counterfactual: evidence.counterfactual, interpolation: evidence.interpolation, claimTests: evidence.claimTests, provenance: evidence.provenance }, null, 2)} as const;

export type ManifoldEvidence = typeof manifoldEvidenceGenerated;
`;
writeFileSync(resolve(OUT_DIR, 'manifoldEvidence.generated.ts'), ts);

// Write provenance JSON
writeFileSync(resolve(OUT_DIR, 'manifoldEvidence.provenance.json'), JSON.stringify({ provenance: evidence.provenance }, null, 2));

console.log(`Generated: src/data/generated/manifoldEvidence.generated.ts`);
console.log(`Provenance: src/data/generated/manifoldEvidence.provenance.json`);
console.log(`Axes exported: ${Object.keys(evidence.semanticAxes.validation.perAxis).length} validation, ${Object.keys(evidence.semanticAxes.shared1000.perAxis).length} shared1000`);
console.log(`Claims exported: ${evidence.claimTests.validation.length} validation, ${evidence.claimTests.shared1000.length} shared1000`);
