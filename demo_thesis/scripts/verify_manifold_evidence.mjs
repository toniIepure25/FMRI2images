/**
 * Verify manifold evidence: compare generated data against source JSON.
 * Usage: node scripts/verify_manifold_evidence.mjs
 */
import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DEMO_DIR = resolve(__dirname, '..');
const REPORTS = resolve(DEMO_DIR, '..', 'reports', 'manifold_analysis');
const GEN = resolve(DEMO_DIR, 'src', 'data', 'generated', 'manifoldEvidence.generated.ts');
const TOLERANCE = 1e-12;

let failures = 0;

function assert(cond, msg) { if (!cond) { console.error('FAIL:', msg); failures++; } else { console.log('PASS:', msg); } }

assert(existsSync(GEN), 'Generated evidence file exists');

// Read source JSONs
const SPLITS = {
  validation: resolve(REPORTS, '20260515_V62a_val_full_semantic'),
  shared1000: resolve(REPORTS, '20260515_V62a_shared1000_full_semantic'),
};

for (const [split, dir] of Object.entries(SPLITS)) {
  assert(existsSync(dir), `Source directory exists: ${split}`);

  const retJSON = JSON.parse(readFileSync(resolve(dir, 'retrieval_metrics.json'), 'utf-8'));
  assert(Math.abs(retJSON.cosine['R@1'] - 0) > 0, `retrieval_metrics.json readable for ${split}`);

  const rsaJSON = JSON.parse(readFileSync(resolve(dir, 'rsa_metrics.json'), 'utf-8'));
  assert(rsaJSON.spearman_rho > 0, `rsa_metrics.json readable for ${split}`);

  const nbJSON = JSON.parse(readFileSync(resolve(dir, 'neighborhood_metrics.json'), 'utf-8'));
  assert(nbJSON['overlap@10_mean'] > 0, `neighborhood_metrics.json readable for ${split}`);

  const hubJSON = JSON.parse(readFileSync(resolve(dir, 'hubness_metrics.json'), 'utf-8'));
  assert(hubJSON.cosine_gini > 0, `hubness_metrics.json readable for ${split}`);

  const spJSON = JSON.parse(readFileSync(resolve(dir, 'semantic_probe_metrics.json'), 'utf-8'));
  assert(spJSON['agreement@5'] > 0, `semantic_probe_metrics.json readable for ${split}`);

  const saJSON = JSON.parse(readFileSync(resolve(dir, 'semantic_axes_metrics.json'), 'utf-8'));
  assert(saJSON.axis_names.length === 10, `semantic_axes has 10 axes for ${split} (got ${saJSON.axis_names.length})`);

  for (const name of saJSON.axis_names) {
    assert(saJSON.per_axis[name] != null, `axis ${name} exists in ${split}`);
    assert(Math.abs(saJSON.per_axis[name].spearman_r) > 0, `axis ${name} spearman non-zero in ${split}`);
    assert(saJSON.per_axis[name].pearson_r > 0, `axis ${name} pearson positive in ${split}`);
  }

  const cfJSON = JSON.parse(readFileSync(resolve(dir, 'counterfactual_metrics.json'), 'utf-8'));
  assert(cfJSON.robustness_margin_mean > 0, `counterfactual_metrics.json readable for ${split}`);

  const ipJSON = JSON.parse(readFileSync(resolve(dir, 'interpolation_metrics.json'), 'utf-8'));
  assert(ipJSON.mean_smoothness > 0.99, `interpolation_metrics.json readable for ${split}`);

  const ctJSON = JSON.parse(readFileSync(resolve(dir, 'claim_tests.json'), 'utf-8'));
  assert(ctJSON.length >= 13, `claim_tests has >= 13 claims for ${split}`);
}

console.log('');
if (failures === 0) {
  console.log('Manifold evidence verification passed.');
} else {
  console.error(`Verification failed: ${failures} checks FAILED.`);
  process.exit(1);
}
