import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { MetricCard } from '@/components/MetricCard';
import { GlassCard } from '@/components/GlassCard';
import { useMetrics } from '@/lib/hooks';
import type { MetricEntry, MetricsSummary } from '@/types';

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

function CtaButton({
  to,
  children,
  glowClass,
}: {
  to: string;
  children: ReactNode;
  glowClass: string;
}) {
  return (
    <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} className="w-full sm:w-auto min-w-[12rem]">
      <Link
        to={to}
        className={`relative block w-full text-center rounded-2xl px-8 py-4 font-semibold tracking-tight border border-white/10 bg-brain-panel/40 backdrop-blur-md transition-shadow duration-300 ${glowClass}`}
      >
        <span className="relative z-10 text-white">{children}</span>
      </Link>
    </motion.div>
  );
}

function PipelineConnector() {
  return (
    <div className="hidden md:flex flex-col items-center justify-center shrink-0 w-10 lg:w-14" aria-hidden>
      <motion.div
        className="h-0.5 w-full rounded-full bg-gradient-to-r from-transparent via-brain-accent/80 to-transparent origin-left"
        initial={{ scaleX: 0.2, opacity: 0.4 }}
        animate={{ scaleX: [0.5, 1, 0.85], opacity: [0.5, 1, 0.65] }}
        transition={{ duration: 2.8, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="mt-[-2px] text-brain-accent/90 text-lg leading-none"
        animate={{ x: [0, 3, 0], opacity: [0.7, 1, 0.7] }}
        transition={{ duration: 1.8, repeat: Infinity }}
      >
        →
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
    <motion.div variants={itemVariants} className="flex-1 min-w-[10rem] max-w-md mx-auto w-full">
      <GlassCard
        hover
        glow
        className="p-5 h-full flex flex-col items-center text-center gap-3 border-brain-border/60"
        initial={false}
      >
        <div className="w-12 h-12 rounded-xl bg-brain-navy/80 border border-brain-border/50 flex items-center justify-center text-brain-accent">
          {icon}
        </div>
        <h3 className="text-sm font-semibold text-white tracking-tight">{title}</h3>
        <p className="text-xs text-gray-500 leading-relaxed">{description}</p>
      </GlassCard>
    </motion.div>
  );
}

const pipelineSteps = [
  {
    title: 'fMRI activity',
    description: 'Voxel time-series from NSD trials, z-scored and masked to the visual cortex.',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M3.75 12h16.5m-7.5 6.75h7.5M3.75 4.5h7.5m7.5 0v15"
        />
      </svg>
    ),
  },
  {
    title: 'Encoder',
    description: 'ROI Transformer maps regional signals into a dense brain representation.',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 3v18M15.75 3v18M3 8.25h18M3 15.75h18" />
      </svg>
    ),
  },
  {
    title: 'CLIP space',
    description: 'Hyperspherical heads produce 768-D embeddings aligned with vision–language geometry.',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M12 3c4.556 0 8.25 3.694 8.25 8.25S16.556 19.5 12 19.5 3.75 15.806 3.75 11.25 7.444 3 12 3z"
        />
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 8.25v7.5M8.25 12h7.5" />
      </svg>
    ),
  },
  {
    title: 'Retrieval & reconstruction',
    description: 'Nearest-neighbor gallery ranking plus uncertainty-guided diffusion from decoded CLIP.',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3A1.5 1.5 0 001.5 6v12a1.5 1.5 0 001.5 1.5z"
        />
      </svg>
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
  const { metrics } = useMetrics();

  const r1Entry = findRetrievalEntry(metrics);
  const r1Display = formatR1Percent(r1Entry);
  const roiDisplay = pickRoiTokenCount(metrics);
  const generation = pickGenerationFields(metrics);

  return (
    <div className="relative min-h-screen bg-brain-dark text-gray-100 overflow-x-hidden">
      <HeroOrbs />

      {/* Hero */}
      <section className="relative z-10 min-h-[100dvh] flex flex-col items-center justify-center px-4 sm:px-6 py-20 text-center">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="show"
          className="max-w-4xl mx-auto flex flex-col items-center"
        >
          <motion.p
            variants={itemVariants}
            className="mb-4 text-[10px] sm:text-xs font-mono uppercase tracking-[0.35em] text-brain-accent/80"
          >
            Bachelor thesis demo
          </motion.p>
          <motion.h1
            variants={itemVariants}
            className="text-4xl sm:text-6xl md:text-7xl lg:text-8xl font-bold tracking-tight leading-[1.05] mb-6"
          >
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-brain-accent to-brain-purple drop-shadow-[0_0_40px_rgba(0,212,255,0.25)]">
              Cortex2Canvas
            </span>
          </motion.h1>
          <motion.p variants={itemVariants} className="text-lg sm:text-xl md:text-2xl text-gray-300 font-medium mb-5">
            An Interactive fMRI-to-Image Decoding Studio
          </motion.p>
          <motion.p
            variants={itemVariants}
            className="text-sm sm:text-base text-gray-500 max-w-2xl leading-relaxed mb-12"
          >
            Decoding visual perception from brain activity using neural networks, von Mises-Fisher distributions, and
            uncertainty-aware generation
          </motion.p>

          <motion.div
            variants={itemVariants}
            className="flex flex-col sm:flex-row flex-wrap gap-4 justify-center items-stretch w-full max-w-4xl"
          >
            <CtaButton
              to="/pipeline"
              glowClass="shadow-[0_0_32px_rgba(0,212,255,0.35)] hover:shadow-[0_0_48px_rgba(0,212,255,0.45)]"
            >
              Launch Live Pipeline
            </CtaButton>
            <CtaButton
              to="/explorer"
              glowClass="shadow-[0_0_32px_rgba(139,92,246,0.35)] hover:shadow-[0_0_48px_rgba(139,92,246,0.45)]"
            >
              Open Explorer
            </CtaButton>
            <CtaButton
              to="/challenge"
              glowClass="shadow-[0_0_32px_rgba(236,72,153,0.35)] hover:shadow-[0_0_48px_rgba(236,72,153,0.45)]"
            >
              Blind Committee Challenge
            </CtaButton>
          </motion.div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2, duration: 0.8 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2 text-gray-600"
          aria-hidden
        >
          <motion.div animate={{ y: [0, 6, 0] }} transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}>
            <span className="text-xs tracking-widest uppercase">Scroll</span>
            <div className="mx-auto mt-1 h-8 w-px bg-gradient-to-b from-brain-accent/50 to-transparent" />
          </motion.div>
        </motion.div>
      </section>

      {/* Key metrics */}
      <section className="relative z-10 border-t border-brain-border/40 bg-brain-navy/30 backdrop-blur-sm px-4 sm:px-6 py-16">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-center text-xs font-semibold uppercase tracking-[0.25em] text-gray-500 mb-10">
            Key metrics
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
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
        </div>
      </section>

      {/* Pipeline */}
      <section className="relative z-10 px-4 sm:px-6 py-20">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-center text-2xl sm:text-3xl font-bold text-white mb-3">Pipeline overview</h2>
          <p className="text-center text-sm text-gray-500 max-w-xl mx-auto mb-12">
            From cortical activity to semantic vision embeddings and generative image candidates.
          </p>

          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: '-80px' }}
            className="flex flex-col md:flex-row md:items-stretch md:justify-center gap-4 md:gap-0"
          >
            {pipelineSteps.map((step, i) => (
              <div key={step.title} className="contents md:flex md:flex-row md:items-stretch">
                <PipeCard {...step} />
                {i < pipelineSteps.length - 1 && (
                  <>
                    <PipelineConnector />
                    <div className="md:hidden flex justify-center py-1 text-brain-accent/60 text-lg">↓</div>
                  </>
                )}
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Contributions */}
      <section className="relative z-10 px-4 sm:px-6 py-20 bg-gradient-to-b from-transparent to-brain-navy/40">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-center text-2xl sm:text-3xl font-bold text-white mb-12">Contributions</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {contributions.map((c, i) => (
              <motion.div
                key={c.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.06, duration: 0.45 }}
              >
                <GlassCard hover className="p-6 h-full border-brain-border/50">
                  <h3 className="text-base font-semibold text-brain-accent mb-2">{c.title}</h3>
                  <p className="text-sm text-gray-400 leading-relaxed">{c.text}</p>
                </GlassCard>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-brain-border/40 px-4 py-12 text-center text-sm text-gray-500">
        <p className="font-medium text-gray-400">Cortex2Canvas</p>
        <p className="mt-2 max-w-lg mx-auto">
          Interactive demonstration supporting a bachelor thesis on fMRI-to-image neural decoding with the Natural Scenes
          Dataset (Allen et al., 2022).
        </p>
        <p className="mt-4 text-xs text-gray-600">Research prototype — not for clinical use.</p>
      </footer>
    </div>
  );
}
