import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useMetrics } from '@/lib/hooks';
import type { MetricEntry, MetricsSummary } from '@/types';
import { PremiumPanel } from '@/components/premium/PremiumPanel';
import { NeuralNetField } from '@/components/home/NeuralNetField';
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
    <div className="premium-page-bg relative min-h-screen overflow-hidden px-4 pb-20 pt-10 sm:px-6 lg:px-8">
      {/* ── Canvas neural field — animated mesh covering the entire page.
             A composite mask: a soft vertical fade at the very edges +
             a gentle radial dim behind the hero text so the content stays
             dominant without hiding the field everywhere else. ── */}
      <div
        className="pointer-events-none absolute inset-0"
        aria-hidden
        style={{
          maskImage: [
            'radial-gradient(ellipse 60% 38% at 26% 32%, rgba(0,0,0,0.55) 0%, rgba(0,0,0,0.85) 45%, black 80%)',
            'linear-gradient(to bottom, transparent 0%, black 5%, black 95%, transparent 100%)',
          ].join(', '),
          WebkitMaskImage: [
            'radial-gradient(ellipse 60% 38% at 26% 32%, rgba(0,0,0,0.55) 0%, rgba(0,0,0,0.85) 45%, black 80%)',
            'linear-gradient(to bottom, transparent 0%, black 5%, black 95%, transparent 100%)',
          ].join(', '),
          maskComposite: 'intersect',
          WebkitMaskComposite: 'source-in',
        }}
      >
        <NeuralNetField />
      </div>

      <section className="relative z-10 mx-auto grid max-w-[1280px] items-start gap-10 py-12 lg:grid-cols-[1fr_0.94fr] lg:items-center lg:gap-16">
        {/* ─────────────────────────────── HERO LEFT ────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="max-w-4xl"
        >
          {/* Kicker — sits under the navbar like a research-lab callsign. */}
          <div className="flex items-center gap-2.5">
            <span className="h-px w-7 bg-accent/55" aria-hidden />
            <p className="text-[10.5px] font-semibold uppercase tracking-[0.22em] text-accent/80">
              Bachelor thesis · neural decoding lab
            </p>
          </div>

          <h1 className="mt-5 max-w-4xl text-[56px] font-semibold leading-[0.94] tracking-[-0.038em] text-text-primary sm:text-[76px] lg:text-[88px]">
            Cortex<span className="text-text-secondary/80">2</span>Canvas
          </h1>

          <p className="mt-6 max-w-2xl text-[22px] leading-[1.35] tracking-[-0.012em] text-text-secondary sm:text-[26px]">
            A neural decoding evidence console for fMRI&nbsp;→&nbsp;CLIP retrieval.
          </p>

          <p className="mt-5 max-w-[34rem] text-[13.5px] leading-[1.7] text-text-muted">
            Trace an NSD visual response from cortical activity through a triple-fusion encoder ensemble (V61a + V62a + V66a) into CLIP-space retrieval evidence — every value carries an explicit
            provenance.
          </p>

          {/* CTA cluster — primary dominates, secondaries quieter. */}
          <div className="mt-9 flex flex-col gap-3 sm:flex-row sm:items-center">
            <Cta to="/pipeline" primary>
              Launch decoding
            </Cta>
            <Cta to="/explorer">Open trial explorer</Cta>
            <Cta to="/evidence">Review evidence</Cta>
          </div>
        </motion.div>

        {/* ────────────────────────── HERO RIGHT: CONSOLE ───────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, delay: 0.08, ease: [0.22, 1, 0.36, 1] }}
        >
          <PremiumPanel variant="hero" className="p-5 sm:p-6">
            {/* Console status bar — kicker rule + status pills */}
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-2.5">
                <span className="h-px w-6 bg-accent/40" aria-hidden />
                <p className="premium-kicker">Decoding console preview</p>
              </div>
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="inline-flex items-center gap-1.5 rounded-full bg-status-success/[0.08] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.06em] text-status-success ring-1 ring-status-success/22">
                  <span className="h-1.5 w-1.5 rounded-full bg-status-success pulse-dot" />
                  Ready
                </span>
                <span className="rounded-full bg-white/[0.025] px-2.5 py-1 font-mono text-[10px] tabular-nums text-text-secondary ring-1 ring-white/[0.06]">
                  Triple Fusion (V61+V62+V66) · ViT-L/14
                </span>
              </div>
              <div>
                <h2 className="text-[20px] font-semibold leading-tight tracking-tight text-text-primary">
                  Visual-stimulus decoding
                </h2>
                <p className="mt-1.5 text-[12.5px] leading-relaxed text-text-secondary">
                  One coherent trace from cortical signal to ranked CLIP-space evidence.
                </p>
              </div>
            </div>

            {/* Instrument readings — flat hairline strip, no surrounding ring */}
            <div className="mt-5 grid grid-cols-2 divide-x divide-y divide-white/[0.05] border-y border-white/[0.05] sm:grid-cols-4 sm:divide-y-0">
              <HomeReading label="Top-1 retrieval" value={loading ? '—' : r1} tone="accent" detail="SHARED1000 · CSLS" />
              <HomeReading label="Embedding" value="768-D" detail="CLIP ViT-L/14" />
              <HomeReading label="Gallery" value="10k" detail="CSLS context" />
              <HomeReading label="Provenance" value="Tracked" tone="success" detail="4-state labels" />
            </div>

            {/* Decoding path — calibrated rail, refined icon column */}
            <div className="mt-5">
              <div className="mb-3.5 flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <span className="h-px w-5 bg-accent/35" aria-hidden />
                  <p className="premium-kicker">Decoding path</p>
                </div>
                <Link
                  to="/pipeline"
                  className="inline-flex items-center gap-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-accent/85 hover:text-accent"
                >
                  Launch
                  <svg className="h-2.5 w-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M7 17L17 7M17 7H9M17 7v8" />
                  </svg>
                </Link>
              </div>
              <ol className="relative space-y-3">
                <div className="pointer-events-none absolute left-[13px] top-3 bottom-3 w-px bg-white/[0.06]" aria-hidden />
                {path.map(({ label, detail, Icon }, index) => (
                  <li key={label} className="relative flex items-center gap-3">
                    <div className="relative z-10 flex h-[26px] w-[26px] shrink-0 items-center justify-center rounded-md bg-white/[0.02] text-accent ring-1 ring-white/[0.06]">
                      <Icon className="h-3 w-3" aria-hidden />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-baseline gap-2">
                        <span className="font-mono text-[10px] tabular-nums text-text-muted/80">
                          {String(index + 1).padStart(2, '0')}
                        </span>
                        <p className="truncate text-[12.5px] font-semibold leading-none tracking-tight text-text-primary">
                          {label}
                        </p>
                      </div>
                      <p className="mt-1 truncate text-[10.5px] leading-tight text-text-muted/85">
                        {detail}
                      </p>
                    </div>
                  </li>
                ))}
              </ol>
            </div>
          </PremiumPanel>
        </motion.div>
      </section>
    </div>
  );
}

function HomeReading({
  label,
  value,
  detail,
  tone = 'default',
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  tone?: 'default' | 'accent' | 'success';
}) {
  const toneClass =
    tone === 'accent' ? 'text-accent'
    : tone === 'success' ? 'text-status-success'
    : 'text-text-primary';
  return (
    <div className="px-3 py-2.5">
      <p className="premium-kicker">{label}</p>
      <p className={`mt-1.5 font-mono text-[19px] font-semibold leading-none tabular-nums ${toneClass}`}>
        {value}
      </p>
      {detail ? (
        <p className="mt-1.5 text-[10.5px] leading-tight text-text-muted">{detail}</p>
      ) : null}
    </div>
  );
}
