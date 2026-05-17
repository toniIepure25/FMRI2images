/**
 * Sync manifold evidence figures from reports to public/manifold/
 * Usage: node scripts/sync_manifold_figures.mjs
 */
import { readFileSync, writeFileSync, copyFileSync, mkdirSync, readdirSync, existsSync } from 'fs';
import { resolve, dirname, basename } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DEMO_DIR = resolve(__dirname, '..');
const REPORTS = resolve(DEMO_DIR, '..', 'reports', 'manifold_analysis');
const PUBLIC_DIR = resolve(DEMO_DIR, 'public', 'manifold');
const OUT_DIR = resolve(DEMO_DIR, 'src', 'data', 'generated');

const SPLITS = {
  validation: { dir: '20260515_V62a_val_full_semantic', label: 'Validation' },
  shared1000: { dir: '20260515_V62a_shared1000_full_semantic', label: 'Shared1000' },
};

const CATEGORIES = {
  retrieval: ['retrieval_comparison', 'rank_distribution', 'cosine', 'csls'],
  rsa: ['rsa'],
  neighborhood: ['neighborhood'],
  hubness: ['hubness', 'lorenz'],
  semantic_probe: ['semantic_probe', 'semantic_concept'],
  semantic_axes: ['semantic_axes'],
  counterfactual: ['counterfactual'],
  interpolation: ['interpolation'],
  nmas: ['nmas'],
  reliability: ['reliability', 'coverage', 'density'],
  kappa: ['kappa'],
  error: ['error'],
};

function inferCategory(filename) {
  for (const [cat, keywords] of Object.entries(CATEGORIES)) {
    for (const kw of keywords) {
      if (filename.includes(kw)) return cat;
    }
  }
  return 'unknown';
}

const figureRegistry = { validation: [], shared1000: [] };

for (const [split, { dir, label }] of Object.entries(SPLITS)) {
  const figDir = resolve(REPORTS, dir, 'figures');
  const pubDir = resolve(PUBLIC_DIR, split);
  mkdirSync(pubDir, { recursive: true });

  if (!existsSync(figDir)) {
    console.log(`WARNING: Figures directory not found: ${figDir}`);
    continue;
  }

  const files = readdirSync(figDir).filter(f => f.endsWith('.png'));
  for (const file of files) {
    const src = resolve(figDir, file);
    const dst = resolve(pubDir, file);
    copyFileSync(src, dst);
    figureRegistry[split].push({
      filename: file,
      publicPath: `/manifold/${split}/${file}`,
      split,
      category: inferCategory(file),
      title: file.replace(/^figure_/, '').replace(/\.png$/, '').replace(/_/g, ' '),
    });
  }
  console.log(`Synced ${files.length} figures for ${split}`);
}

// Write figure registry
mkdirSync(OUT_DIR, { recursive: true });
const ts = `// Auto-generated figure registry
// DO NOT EDIT MANUALLY. Run: npm run sync:manifold-figures

export const manifoldFigures = ${JSON.stringify(figureRegistry, null, 2)} as const;

export type ManifoldFigureEntry = typeof manifoldFigures['validation'][number];
`;
writeFileSync(resolve(OUT_DIR, 'manifoldFigures.generated.ts'), ts);
console.log(`Generated: src/data/generated/manifoldFigures.generated.ts`);
console.log(`Total figures: validation=${figureRegistry.validation.length}, shared1000=${figureRegistry.shared1000.length}`);
