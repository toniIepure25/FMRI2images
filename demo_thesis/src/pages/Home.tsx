import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useMetrics } from '@/lib/hooks';
import type { MetricEntry, MetricsSummary } from '@/types';
import { PremiumPanel } from '@/components/premium/PremiumPanel';
import { MetricTile } from '@/components/premium/MetricTile';
import {
  IconAtom,
  IconBrain,
  IconEye,
  IconLayers,
  IconPipeline,
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
    <div className="premium-page-bg min-h-screen overflow-hidden px-4 pb-16 pt-8 sm:px-6 lg:px-8">
      <section className="mx-auto grid max-w-[1280px] items-start gap-8 py-10 lg:grid-cols-[1fr_0.92fr] lg:items-center">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="max-w-4xl"
        >
          <p className="premium-kicker mb-4">Bachelor thesis research demo</p>
          <h1 className="max-w-4xl text-5xl font-semibold tracking-[-0.035em] text-text-primary sm:text-7xl lg:text-[80px]">
            Cortex2Canvas
          </h1>
          <p className="mt-5 max-w-2xl text-xl leading-relaxed text-text-secondary sm:text-2xl">
            Neural decoding evidence console for fMRI-to-CLIP retrieval.
          </p>
          <p className="mt-4 max-w-2xl text-sm leading-7 text-text-muted">
            Trace an NSD visual response from cortical activity through a V62a MLP encoder into CLIP-space retrieval evidence, with explicit provenance for cached, replayed, derived, and unavailable artifacts.
          </p>

          <div className="mt-9 flex flex-col gap-3 sm:flex-row">
            <Cta to="/pipeline" primary>
              Launch decoding replay
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
            {/* Console header — feels like an instrument status bar */}
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="inline-flex items-center gap-1.5 rounded-full border border-status-success/22 bg-status-success/10 px-2.5 py-1 text-[10px] font-semibold text-status-success">
                    <span className="h-1.5 w-1.5 rounded-full bg-status-success pulse-dot" />
                    Replay ready
                  </span>
                  <span className="rounded-md border border-border-subtle bg-surface-raised px-2 py-0.5 font-mono text-[10px] text-text-secondary">
                    V62a · CLIP ViT-L/14
                  </span>
                </div>
                <h2 className="mt-3 text-[22px] font-semibold tracking-tight text-text-primary">
                  Visual-stimulus decoding replay
                </h2>
                <p className="mt-1.5 text-[13px] leading-relaxed text-text-secondary">
                  One coherent trace from cortical signal to ranked CLIP-space evidence.
                </p>
              </div>
            </div>

            {/* Instrument readings */}
            <div className="mt-5 grid grid-cols-2 gap-2.5">
              <MetricTile label="Top-1 retrieval" value={loading ? '—' : r1} tone="accent" detail="Cached metrics summary" />
              <MetricTile label="Embedding" value="768-D" detail="CLIP ViT-L/14 space" />
              <MetricTile label="Gallery" value="10k" detail="CSLS retrieval context" />
              <MetricTile label="Provenance" value="Tracked" tone="success" detail="Live / replay / derived / unavailable" />
            </div>

            {/* Decoding path — denser, single line of identity per step,
                surfaces match the rest of the workbench. */}
            <div className="workbench-inset mt-5">
              <div className="mb-3 flex items-center justify-between">
                <p className="premium-kicker">Decoding path</p>
                <Link
                  to="/pipeline"
                  className="text-[10px] font-semibold uppercase tracking-[0.14em] text-accent/85 hover:text-accent"
                >
                  Launch ↗
                </Link>
              </div>
              <ol className="relative space-y-1.5">
                <div className="pointer-events-none absolute left-[18px] top-3 bottom-3 w-px bg-border-subtle/55" aria-hidden />
                {path.map(({ label, detail, Icon }, index) => (
                  <li key={label} className="relative flex items-center gap-3">
                    <div className="relative z-10 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-border-subtle bg-surface-elevated text-accent shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
                      <Icon className="h-4 w-4" aria-hidden />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-baseline gap-2">
                        <span className="font-mono text-[10px] tabular-nums text-text-muted">
                          {String(index + 1).padStart(2, '0')}
                        </span>
                        <p className="truncate text-[13px] font-semibold text-text-primary">
                          {label}
                        </p>
                      </div>
                      <p className="truncate text-[11px] text-text-muted">{detail}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </div>
          </PremiumPanel>
        </motion.div>
      </section>

      {/* Scientific honesty strip — replaces the marketing-style trust trio
          with a compact, scientific reading that fits an instrument page. */}
      <section className="mx-auto max-w-[1280px] pb-12">
        <div className="rounded-2xl border border-border-subtle bg-surface-elevated/55 px-5 py-4 sm:flex sm:items-center sm:justify-between sm:gap-6">
          <div className="flex items-center gap-3">
            <span className="inline-flex h-2 w-2 rounded-full bg-accent" aria-hidden />
            <p className="text-[13px] text-text-secondary">
              Every value carries a provenance badge.
              <span className="ml-1 text-text-muted">Cached, live, derived, and unavailable artifacts are labelled honestly — no synthetic claims.</span>
            </p>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2 sm:mt-0">
            <span className="rounded-md border border-accent/22 bg-accent/10 px-2 py-0.5 text-[10px] font-semibold text-accent">Live</span>
            <span className="rounded-md border border-border-subtle bg-surface-raised px-2 py-0.5 text-[10px] font-semibold text-text-secondary">Replay</span>
            <span className="rounded-md border border-status-info/20 bg-status-info/10 px-2 py-0.5 text-[10px] font-semibold text-status-info">Derived</span>
            <span className="rounded-md border border-border-subtle bg-surface-raised/70 px-2 py-0.5 text-[10px] font-semibold text-text-muted">Unavailable</span>
          </div>
        </div>
      </section>

    </div>
  );
}
