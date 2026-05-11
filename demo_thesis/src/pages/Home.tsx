import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { MetricCard } from '@/components/MetricCard';
import { GlassCard } from '@/components/GlassCard';
import { useMetrics } from '@/lib/hooks';
import type { MetricEntry, MetricsSummary } from '@/types';
import {
  IconAtom,
  IconBrain,
  IconChevronDown,
  IconChevronRight,
  IconEye,
  IconLayers,
  IconSparkles,
} from '@/components/ui/Icon';

type SummaryWithExtras = MetricsSummary & {
  architecture?: { roi_tokens?: number; roiTokens?: number };
  generation?: { label?: string; subtitle?: string };
};

const FALLBACK_R1 = '50.8%';
const FALLBACK_ROI = '17';
const FALLBACK_UNCERTAINTY = { value: 'κ + δ', subtitle: 'Dual Uncertainty Signals' };
const FALLBACK_GENERATION = { value: 'DUA-CFG', subtitle: 'Adaptive Diffusion Control' };

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

function formatR1Percent(entry: MetricEntry | undefined): string {
  if (entry == null || typeof entry.value !== 'number') return FALLBACK_R1;
  const v = entry.value;
  if (v >= 0 && v <= 1) return `${(v * 100).toFixed(1)}%`;
  if (v > 1 && v <= 100) return `${v.toFixed(1)}%`;
  return FALLBACK_R1;
}

function pickRoiTokenCount(summary: MetricsSummary | null): string {
  const arch = (summary as SummaryWithExtras | null)?.architecture;
  const n = arch?.roi_tokens ?? arch?.roiTokens;
  if (typeof n === 'number' && Number.isFinite(n)) return String(Math.round(n));
  return FALLBACK_ROI;
}

function pickGenerationFields(summary: MetricsSummary | null): { value: string; subtitle: string } {
  const gen = (summary as SummaryWithExtras | null)?.generation;
  if (gen?.label || gen?.subtitle) {
    return {
      value: gen.label ?? FALLBACK_GENERATION.value,
      subtitle: gen.subtitle ?? FALLBACK_GENERATION.subtitle,
    };
  }
  return FALLBACK_GENERATION;
}

const containerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08, delayChildren: 0.12 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] } },
};

function HeroOrbs() {
  const orbs = [
    {
      className: 'top-[10%] left-[5%] w-[min(42vw,28rem)] h-[min(42vw,28rem)]',
      gradient: 'radial-gradient(circle at 30% 30%, rgba(0,212,255,0.55) 0%, rgba(139,92,246,0.15) 45%, transparent 70%)',
      duration: 22,
      x: [0, 40, -20, 0],
      y: [0, -30, 20, 0],
    },
    {
      className: 'top-[40%] right-[0%] w-[min(48vw,32rem)] h-[min(48vw,32rem)]',
      gradient: 'radial-gradient(circle at 50% 50%, rgba(139,92,246,0.5) 0%, rgba(236,72,153,0.12) 50%, transparent 72%)',
      duration: 26,
      x: [0, -35, 25, 0],
      y: [0, 25, -15, 0],
    },
    {
      className: 'bottom-[-5%] left-[25%] w-[min(55vw,36rem)] h-[min(55vw,36rem)]',
      gradient: 'radial-gradient(circle at 40% 60%, rgba(236,72,153,0.4) 0%, rgba(0,212,255,0.1) 55%, transparent 72%)',
      duration: 30,
      x: [0, 30, -40, 0],
      y: [0, -20, 10, 0],
    },
  ];

  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden>
      <motion.div
        className="absolute inset-0 opacity-[0.35]"
        style={{
          background:
            'radial-gradient(ellipse 100% 80% at 50% -30%, rgba(0,212,255,0.2), transparent 55%), radial-gradient(ellipse 70% 50% at 100% 100%, rgba(139,92,246,0.12), transparent 50%), radial-gradient(ellipse 50% 40% at 0% 90%, rgba(236,72,153,0.08), transparent 45%)',
        }}
        animate={{ opacity: [0.28, 0.42, 0.28] }}
        transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }}
      />
      {orbs.map((orb, i) => (
        <motion.div
          key={i}
          className={`absolute rounded-full blur-3xl will-change-transform ${orb.className}`}
          style={{ background: orb.gradient }}
          animate={{ x: orb.x, y: orb.y, scale: [1, 1.06, 1] }}
          transition={{
            duration: orb.duration,
            repeat: Infinity,
            ease: 'easeInOut',
            scale: { duration: 8 + i * 2, repeat: Infinity, ease: 'easeInOut' },
          }}
        />
      ))}
      <div
        className="absolute inset-0 opacity-[0.04] mix-blend-soft-light"
        style={{
          backgroundImage:
            'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.03) 2px, rgba(255,255,255,0.03) 4px)',
        }}
      />
    </div>
  );
}

function PrimaryCta({ to, children }: { to: string; children: ReactNode }) {
  return (
    <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} className="w-full sm:w-auto">
      <Link
        to={to}
        className="relative block min-w-[min(100%,14rem)] rounded-2xl bg-brain-accent px-10 py-5 text-center text-lg font-semibold tracking-tight text-brain-dark shadow-[0_14px_44px_-14px_rgba(0,212,255,0.55)] transition-shadow duration-300 hover:shadow-[0_18px_52px_-12px_rgba(0,212,255,0.62)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brain-accent focus-visible:ring-offset-2 focus-visible:ring-offset-brain-dark"
      >
        {children}
      </Link>
    </motion.div>
  );
}

function SecondaryCta({ to, children }: { to: string; children: ReactNode }) {
  return (
    <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} className="w-full sm:w-auto">
      <Link
        to={to}
        className="block min-w-[min(100%,11rem)] rounded-xl border border-slate-600/70 bg-brain-panel/45 px-5 py-3 text-center text-sm font-semibold text-slate-100 backdrop-blur-md transition-colors duration-300 hover:border-brain-accent/40 hover:bg-brain-panel/65 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brain-accent focus-visible:ring-offset-2 focus-visible:ring-offset-brain-dark"
      >
        {children}
      </Link>
    </motion.div>
  );
}

function PipelineConnector() {
  return (
    <div className="hidden w-10 shrink-0 flex-col items-center justify-center md:flex lg:w-14" aria-hidden>
      <motion.div
        className="h-0.5 w-full origin-left rounded-full bg-gradient-to-r from-transparent via-brain-accent/80 to-transparent"
        initial={{ scaleX: 0.2, opacity: 0.4 }}
        animate={{ scaleX: [0.5, 1, 0.85], opacity: [0.5, 1, 0.65] }}
        transition={{ duration: 2.8, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="mt-[-2px] text-brain-accent"
        animate={{ x: [0, 3, 0], opacity: [0.7, 1, 0.7] }}
        transition={{ duration: 1.8, repeat: Infinity }}
      >
        <IconChevronRight className="mx-auto h-5 w-5" />
      </motion.div>
    </div>
  );
}

function PipeCard({
  icon,
  title,
  description,
}: {
  icon: ReactNode;
  title: string;
  description: string;
}) {
  return (
    <motion.div variants={itemVariants} className="mx-auto w-full max-w-md min-w-[10rem] flex-1">
      <GlassCard
        hover
        glow
        className="flex h-full flex-col items-center gap-3 border-brain-border/60 p-5 text-center"
        initial={false}
      >
        <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-brain-border/50 bg-brain-navy/80 text-brain-accent">
          {icon}
        </div>
        <h3 className="text-sm font-semibold tracking-tight text-white">{title}</h3>
        <p className="text-xs leading-relaxed text-slate-300">{description}</p>
      </GlassCard>
    </motion.div>
  );
}

function MetricsSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className="glass-panel rounded-xl border border-brain-border/55 p-6" aria-hidden>
          <div className="mb-4 h-3 w-16 rounded-full bg-slate-800/70 shimmer-bg" />
          <div className="mb-3 h-10 w-[42%] rounded-lg bg-slate-800/70 shimmer-bg" />
          <div className="h-3 w-[88%] rounded-full bg-slate-800/60 shimmer-bg" />
        </div>
      ))}
    </div>
  );
}

const pipelineSteps = [
  {
    title: 'fMRI activity',
    description: 'Voxel time-series from NSD trials, z-scored and masked to the visual cortex.',
    icon: <IconBrain className="h-6 w-6" aria-hidden />,
  },
  {
    title: 'Encoder',
    description: 'ROI Transformer maps regional signals into a dense brain representation.',
    icon: <IconLayers className="h-6 w-6" aria-hidden />,
  },
  {
    title: 'CLIP space',
    description: 'Hyperspherical heads produce 768-D embeddings aligned with vision–language geometry.',
    icon: <IconAtom className="h-6 w-6" aria-hidden />,
  },
  {
    title: 'Retrieval & reconstruction',
    description: 'Nearest-neighbor gallery ranking plus uncertainty-guided diffusion from decoded CLIP.',
    icon: (
      <span className="flex items-center justify-center gap-1">
        <IconEye className="h-6 w-6" aria-hidden />
        <IconSparkles className="h-6 w-6" aria-hidden />
      </span>
    ),
  },
];

const contributions = [
  {
    title: 'vMF-NCE Loss',
    text: 'Bessel-free contrastive training on the unit hypersphere for stable spherical embeddings.',
  },
  {
    title: 'ROI Transformer Encoder',
    text: 'Brain-topology tokens fuse early visual areas and higher-level regions before decoding.',
  },
  {
    title: 'ROI-DCF (Directional Consensus Fusion)',
    text: 'Per-ROI vMF experts combine through spherical consensus with calibrated disagreement.',
  },
  {
    title: 'DUA-CFG (Decomposed Uncertainty-Aware Guidance)',
    text: 'Diffusion guidance and sampling adapt to κ–δ uncertainty instead of fixed schedules.',
  },
];

export function Home() {
  const { metrics, loading: metricsLoading } = useMetrics();

  const r1Entry = findRetrievalEntry(metrics);
  const r1Display = formatR1Percent(r1Entry);
  const roiDisplay = pickRoiTokenCount(metrics);
  const generation = pickGenerationFields(metrics);

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-brain-dark text-slate-100">
      <HeroOrbs />

      {/* Hero */}
      <section className="relative z-10 flex min-h-[85vh] flex-col items-center justify-center px-4 py-20 text-center sm:px-6">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="show"
          className="mx-auto flex max-w-4xl flex-col items-center"
        >
          <motion.p
            variants={itemVariants}
            className="mb-4 font-mono text-[10px] uppercase tracking-[0.35em] text-brain-accent/85 sm:text-xs"
          >
            Bachelor thesis demo
          </motion.p>
          <motion.h1
            variants={itemVariants}
            className="mb-6 text-4xl leading-[1.05] font-bold tracking-tight sm:text-6xl md:text-7xl lg:text-8xl"
          >
            <span className="bg-gradient-to-r from-cyan-400 via-brain-accent to-brain-purple bg-clip-text text-transparent drop-shadow-[0_0_40px_rgba(0,212,255,0.25)]">
              Cortex2Canvas
            </span>
          </motion.h1>
          <motion.p variants={itemVariants} className="mb-5 text-lg font-medium text-slate-300 sm:text-xl md:text-2xl">
            An Interactive fMRI-to-Image Decoding Studio
          </motion.p>
          <motion.p variants={itemVariants} className="mb-12 max-w-2xl text-sm leading-relaxed text-slate-300 sm:text-base">
            Decoding visual perception from brain activity using neural networks, von Mises-Fisher distributions, and
            uncertainty-aware generation
          </motion.p>

          <motion.div variants={itemVariants} className="flex w-full max-w-3xl flex-col items-center gap-5">
            <PrimaryCta to="/pipeline">Launch Pipeline</PrimaryCta>
            <div className="flex w-full flex-col items-stretch gap-3 sm:flex-row sm:flex-wrap sm:justify-center">
              <SecondaryCta to="/explorer">Open Explorer</SecondaryCta>
              <SecondaryCta to="/challenge">Blind Committee Challenge</SecondaryCta>
            </div>
          </motion.div>
        </motion.div>

        <div
          className="pointer-events-none absolute bottom-6 left-1/2 flex -translate-x-1/2 flex-col items-center gap-2 opacity-[0.38]"
          aria-hidden
        >
          <span className="h-px w-14 bg-gradient-to-r from-transparent via-brain-accent/60 to-transparent" />
          <IconChevronDown className="h-3.5 w-3.5 text-slate-500" />
        </div>
      </section>

      {/* Key metrics */}
      <section className="relative z-10 border-t border-brain-border/40 bg-brain-navy/30 px-4 py-20 backdrop-blur-sm sm:px-6">
        <div className="mx-auto max-w-6xl">
          <h2 className="mb-3 text-center text-xs font-semibold uppercase tracking-[0.25em] text-slate-500">
            Key metrics
          </h2>
          <p className="mx-auto mb-12 max-w-lg text-center text-sm text-slate-400">
            Performance highlights from the best-performing model configuration across NSD subjects.
          </p>
          {metricsLoading ? (
            <MetricsSkeleton />
          ) : (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <MetricCard
                label="R@1"
                value={r1Display}
                subtitle="Top-1 Retrieval Accuracy"
                color="#00d4ff"
                delay={0}
              />
              <MetricCard
                label="ROI Transformer"
                value={roiDisplay}
                subtitle="Brain Region Tokens"
                color="#8b5cf6"
                delay={0.08}
              />
              <MetricCard
                label="Uncertainty"
                value={FALLBACK_UNCERTAINTY.value}
                subtitle={FALLBACK_UNCERTAINTY.subtitle}
                color="#ec4899"
                delay={0.16}
              />
              <MetricCard
                label="Generation"
                value={generation.value}
                subtitle={generation.subtitle}
                color="#34d399"
                delay={0.24}
              />
            </div>
          )}
        </div>
      </section>

      {/* Pipeline */}
      <section className="relative z-10 px-4 py-20 sm:px-6">
        <div className="mx-auto max-w-6xl">
          <h2 className="mb-3 text-center text-2xl font-bold text-white sm:text-3xl">Pipeline overview</h2>
          <p className="mx-auto mb-12 max-w-xl text-center text-sm text-slate-400">
            From cortical activity to semantic vision embeddings and generative image candidates.
          </p>

          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: '-80px' }}
            className="flex flex-col gap-4 md:flex-row md:items-stretch md:justify-center md:gap-0"
          >
            {pipelineSteps.map((step, i) => (
              <div key={step.title} className="contents md:flex md:flex-row md:items-stretch">
                <PipeCard {...step} />
                {i < pipelineSteps.length - 1 ? (
                  <>
                    <PipelineConnector />
                    <div className="flex justify-center py-1 text-brain-accent/65 md:hidden">
                      <IconChevronDown className="h-5 w-5" aria-hidden />
                    </div>
                  </>
                ) : null}
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Contributions */}
      <section className="relative z-10 bg-gradient-to-b from-transparent to-brain-navy/40 px-4 py-20 sm:px-6">
        <div className="mx-auto max-w-6xl">
          <h2 className="mb-12 text-center text-2xl font-bold text-white sm:text-3xl">Contributions</h2>
          <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
            {contributions.map((c, i) => (
              <motion.div
                key={c.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.06, duration: 0.45 }}
              >
                <GlassCard hover className="h-full border-brain-border/50 p-6">
                  <h3 className="mb-2 text-base font-semibold text-brain-accent">{c.title}</h3>
                  <p className="text-sm leading-relaxed text-slate-300">{c.text}</p>
                </GlassCard>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-brain-border/40 px-4 py-12 text-center text-sm text-slate-500">
        <p className="font-medium text-slate-400">Cortex2Canvas</p>
        <p className="mx-auto mt-2 max-w-lg">
          Interactive demonstration supporting a bachelor thesis on fMRI-to-image neural decoding with the Natural Scenes
          Dataset (Allen et al., 2022).
        </p>
        <p className="mt-4 text-xs text-slate-600">Research prototype — not for clinical use.</p>
      </footer>
    </div>
  );
}
