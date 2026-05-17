import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useMetrics } from '@/lib/hooks';
import type { MetricEntry, MetricsSummary } from '@/types';
import { PremiumPanel } from '@/components/premium/PremiumPanel';
import { MetricTile } from '@/components/premium/MetricTile';
import { SectionHeader } from '@/components/premium/SectionHeader';
import {
  IconAtom,
  IconBrain,
  IconEye,
  IconLayers,
  IconPipeline,
  IconSparkles,
} from '@/components/ui/Icon';

const R1_KEYS = ['r@1', 'r_1', 'r1', 'val_r@1', 'top1', 'retrieval_r1'];

function normalizeMetricKey(k: string): string {
  return k.toLowerCase().replace(/\s/g, '').replace(/_/g, '');
}

function findRetrievalEntry(summary: MetricsSummary | null): MetricEntry | undefined {
  if (!summary?.metrics?.retrieval) return undefined;
  const retrieval = summary.metrics.retrieval;
  const keysNorm = new Map(Object.keys(retrieval).map((k) => [normalizeMetricKey(k), k]));
  for (const key of R1_KEYS) {
    const canonical = keysNorm.get(normalizeMetricKey(key));
    if (canonical) return retrieval[canonical];
  }
  return undefined;
}

function formatPercent(entry: MetricEntry | undefined): string {
  if (entry == null || typeof entry.value !== 'number') return 'Unavailable';
  const v = entry.value;
  if (v >= 0 && v <= 1) return `${(v * 100).toFixed(1)}%`;
  if (v > 1 && v <= 100) return `${v.toFixed(1)}%`;
  return 'Unavailable';
}

const path = [
  { label: 'Brain activity', detail: 'NSD visual-stimulus betas', Icon: IconBrain },
  { label: 'ROI vector', detail: 'nsdgeneral visual cortex', Icon: IconLayers },
  { label: 'vMF embedding', detail: 'directional 768-D latent', Icon: IconAtom },
  { label: 'CLIP / CSLS', detail: 'semantic gallery search', Icon: IconPipeline },
  { label: 'Visual evidence', detail: 'ranked retrieval + optional recon', Icon: IconEye },
] as const;

const trustRows = [
  ['Input provenance', 'Cached or live fMRI/retrieval assets are marked explicitly.'],
  ['No synthetic claims', 'Unavailable reconstructions and metrics stay unavailable.'],
  ['Evidence-first UI', 'Rank, CSLS, κ, margins, target images, and candidates stay inspectable.'],
];

function Cta({ to, children, primary = false }: { to: string; children: ReactNode; primary?: boolean }) {
  return (
    <Link to={to} className={primary ? 'premium-button-primary' : 'premium-button-secondary'}>
      {children}
    </Link>
  );
}

export function Home() {
  const { metrics, loading } = useMetrics();
  const r1 = formatPercent(findRetrievalEntry(metrics));

  return (
    <div className="premium-page-bg min-h-screen overflow-hidden px-4 pb-20 pt-10 sm:px-6 lg:px-8">
      <section className="mx-auto grid min-h-[calc(100vh-4rem)] max-w-[1460px] items-center gap-8 py-10 lg:grid-cols-[1.05fr_0.95fr]">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="max-w-4xl"
        >
          <p className="premium-kicker mb-5">Bachelor thesis research demo</p>
          <h1 className="max-w-4xl text-5xl font-semibold tracking-[-0.03em] text-text-primary sm:text-7xl lg:text-8xl">
            Cortex2Canvas
          </h1>
          <p className="mt-6 max-w-2xl text-xl leading-relaxed text-text-secondary sm:text-2xl">
            A premium workbench for replaying fMRI-to-CLIP decoding from visual cortex activity to ranked visual evidence.
          </p>
          <p className="mt-5 max-w-2xl text-sm leading-7 text-text-muted">
            The demo follows the scientific path from NSD brain responses through ROI features, an encoder, vMF directional embeddings, CLIP semantic space, CSLS retrieval, and optional reconstruction comparison.
          </p>

          <div className="mt-9 flex flex-col gap-3 sm:flex-row">
            <Cta to="/pipeline" primary>
              Launch decoding pipeline
            </Cta>
            <Cta to="/explorer">Open trial explorer</Cta>
            <Cta to="/evidence">Review evidence</Cta>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, delay: 0.08, ease: [0.22, 1, 0.36, 1] }}
        >
          <PremiumPanel variant="hero" className="p-5 sm:p-6">
            <div className="flex items-start justify-between gap-5">
              <div>
                <p className="premium-kicker">Instrument status</p>
                <h2 className="mt-3 text-2xl font-semibold tracking-tight text-text-primary">
                  Visual-stimulus decoding replay
                </h2>
                <p className="mt-2 text-sm leading-relaxed text-text-secondary">
                  Designed for thesis defense: one coherent trace from cortical signal to CLIP-space evidence.
                </p>
              </div>
              <span className="rounded-xl border border-status-success/20 bg-status-success/10 px-3 py-1.5 text-[11px] font-semibold text-status-success">
                Research demo
              </span>
            </div>

            <div className="mt-7 grid grid-cols-2 gap-3">
              <MetricTile label="Top-1 retrieval" value={loading ? 'Loading' : r1} tone="accent" detail="From cached metrics summary" />
              <MetricTile label="Embedding" value="768-D" detail="CLIP ViT-L/14 space" />
              <MetricTile label="Gallery" value="10k" detail="CSLS retrieval context" />
              <MetricTile label="Provenance" value="Tracked" tone="success" detail="Cached / derived / live / unavailable" />
            </div>

            <div className="mt-7 rounded-2xl border border-border-subtle bg-surface-base/70 p-4">
              <p className="premium-kicker mb-4">Decoding path</p>
              <div className="space-y-3">
                {path.map(({ label, detail, Icon }, index) => (
                  <div key={label} className="flex items-center gap-3">
                    <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-border-subtle bg-surface-raised text-accent">
                      <Icon className="h-4 w-4" aria-hidden />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-[10px] text-text-muted">{String(index + 1).padStart(2, '0')}</span>
                        <p className="text-sm font-semibold text-text-primary">{label}</p>
                      </div>
                      <p className="text-xs text-text-muted">{detail}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </PremiumPanel>
        </motion.div>
      </section>

      <section className="mx-auto max-w-[1460px] py-12">
        <SectionHeader
          eyebrow="Workbench logic"
          title="Built for evidence, not spectacle"
          description="The interface keeps the retrieval result visually dominant while preserving the scientific audit trail behind every cached, derived, live, or unavailable artifact."
        />
        <div className="mt-8 grid gap-4 lg:grid-cols-3">
          {trustRows.map(([title, detail], index) => (
            <motion.div
              key={title}
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-80px' }}
              transition={{ delay: index * 0.05, duration: 0.35 }}
            >
              <PremiumPanel className="h-full p-5">
                <div className="mb-5 flex h-10 w-10 items-center justify-center rounded-xl border border-accent/20 bg-accent/10 text-accent">
                  {index === 0 ? <IconLayers className="h-5 w-5" /> : index === 1 ? <IconSparkles className="h-5 w-5" /> : <IconEye className="h-5 w-5" />}
                </div>
                <h3 className="text-lg font-semibold text-text-primary">{title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-text-muted">{detail}</p>
              </PremiumPanel>
            </motion.div>
          ))}
        </div>
      </section>

      <footer className="mx-auto max-w-[1460px] border-t border-border-subtle py-8 text-sm text-text-muted">
        Cortex2Canvas · fMRI-to-CLIP visual-stimulus decoding research prototype · not for clinical use.
      </footer>
    </div>
  );
}
