import { useMemo } from 'react';
import { motion } from 'framer-motion';
import type { DemoCase } from '@/types';
import { GlassCard } from '@/components/GlassCard';

export interface StepEmbeddingProps {
  case_: DemoCase;
}

const GRID = 11;

function resamplePreview(src: number[], n: number): number[] {
  if (src.length === 0) return Array.from({ length: n }, () => 0.35);
  const out: number[] = [];
  for (let i = 0; i < n; i += 1) {
    const t = (i / Math.max(n - 1, 1)) * (src.length - 1);
    const j = Math.floor(t);
    const f = t - j;
    const v = src[j] * (1 - f) + (src[j + 1] ?? src[j]) * f;
    out.push(v);
  }
  const mx = Math.max(...out.map((x) => Math.abs(x)), 1e-6);
  return out.map((x) => Math.abs(x) / mx);
}

function heatColor(t: number): string {
  const hue = 190 + t * 95;
  const sat = 72 + t * 18;
  const light = 28 + t * 42;
  return `hsl(${hue}, ${sat}%, ${light}%)`;
}

function Beam({ delay = 0 }: { delay?: number }) {
  return (
    <div className="relative hidden min-h-[120px] w-full min-w-[40px] flex-1 items-center justify-center md:flex md:min-h-0 md:w-14 md:flex-none lg:w-20">
      <svg className="absolute inset-0 h-full w-full overflow-visible" preserveAspectRatio="none" aria-hidden>
        <defs>
          <linearGradient id="beam-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgba(34, 211, 238, 0.15)" />
            <stop offset="50%" stopColor="rgba(167, 139, 250, 0.85)" />
            <stop offset="100%" stopColor="rgba(34, 211, 238, 0.2)" />
          </linearGradient>
          <filter id="beam-glow">
            <feGaussianBlur stdDeviation="2" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        <motion.line
          x1="0%"
          y1="50%"
          x2="100%"
          y2="50%"
          stroke="url(#beam-grad)"
          strokeWidth="3"
          strokeLinecap="round"
          filter="url(#beam-glow)"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{ delay: 0.35 + delay, duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
        />
      </svg>
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          className="pointer-events-none absolute left-[8%] top-1/2 z-10 h-2.5 w-2.5 -translate-y-1/2 rounded-full bg-cyan-200 shadow-[0_0_16px_6px_rgba(34,211,238,0.55)]"
          initial={{ left: '8%', opacity: 0, scale: 0.6 }}
          animate={{
            left: ['8%', '92%'],
            opacity: [0, 1, 1, 0.85],
            scale: [0.6, 1, 1, 0.9],
          }}
          transition={{
            duration: 2.4,
            repeat: Infinity,
            ease: 'linear',
            delay: 0.5 + delay + i * 0.35,
          }}
        />
      ))}
    </div>
  );
}

function FlowParticles() {
  return (
    <div className="relative mx-auto h-24 w-full max-w-[200px]">
      <svg viewBox="0 0 200 96" className="h-full w-full overflow-visible" aria-hidden>
        <defs>
          <linearGradient id="flow-path-grad" x1="0" y1="0" x2="200" y2="0">
            <stop offset="0%" stopColor="rgba(34, 211, 238, 0.2)" />
            <stop offset="100%" stopColor="rgba(167, 139, 250, 0.9)" />
          </linearGradient>
        </defs>
        <path
          id="encoder-flow"
          d="M 8 48 C 52 8, 92 88, 140 48 S 188 24, 196 48"
          fill="none"
          stroke="url(#flow-path-grad)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeDasharray="6 10"
          className="opacity-80"
        />
      </svg>
      {[0, 1, 2, 3].map((i) => (
        <motion.div
          key={i}
          className="absolute left-0 top-0 h-2 w-2 rounded-full bg-violet-300 shadow-[0_0_12px_rgba(167,139,250,0.9)]"
          style={{ offsetPath: `path('M 8 48 C 52 8, 92 88, 140 48 S 188 24, 196 48')`, offsetRotate: '0deg' }}
          initial={{ offsetDistance: '0%', opacity: 0.3 }}
          animate={{ offsetDistance: '100%', opacity: [0.2, 1, 1, 0.3] }}
          transition={{
            duration: 3.2 + i * 0.2,
            repeat: Infinity,
            ease: 'linear',
            delay: i * 0.45,
          }}
        />
      ))}
    </div>
  );
}

export function StepEmbedding({ case_ }: StepEmbeddingProps) {
  const heat = useMemo(() => resamplePreview(case_.fmriPreview ?? [], GRID * GRID), [case_.fmriPreview]);
  const eqBars = useMemo(() => resamplePreview(case_.fmriPreview ?? [], 42), [case_.fmriPreview]);
  const stripSamples = useMemo(() => resamplePreview(case_.fmriPreview ?? [], 768), [case_.fmriPreview]);

  const stageVariants = {
    hidden: { opacity: 0, y: 28, scale: 0.96 },
    show: (i: number) => ({
      opacity: 1,
      y: 0,
      scale: 1,
      transition: { delay: 0.12 + i * 0.14, duration: 0.65, ease: [0.22, 1, 0.36, 1] as const },
    }),
  };

  return (
    <div className="flex h-full min-h-0 flex-col items-center justify-center gap-6 px-2 py-2 sm:gap-8 sm:px-4">
      <motion.h2
        className="max-w-4xl text-center font-sans text-2xl font-semibold tracking-tight text-white sm:text-3xl md:text-[2rem]"
        style={{
          textShadow:
            '0 0 24px rgba(34, 211, 238, 0.45), 0 0 48px rgba(167, 139, 250, 0.25), 0 0 80px rgba(59, 130, 246, 0.12)',
        }}
        initial={{ opacity: 0, y: -16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      >
        From neural activity to visual-semantic space
      </motion.h2>

      <div className="flex w-full max-w-6xl flex-col gap-3 sm:gap-4 md:flex-row md:items-stretch">
        <motion.div custom={0} initial="hidden" animate="show" variants={stageVariants} className="min-w-0 flex-1">
          <GlassCard
            glow
            className="relative h-full overflow-hidden border-cyan-500/20 bg-gradient-to-br from-slate-950/90 via-slate-900/80 to-violet-950/40 p-4 sm:p-5"
          >
            <div className="pointer-events-none absolute -right-8 -top-8 h-32 w-32 rounded-full bg-cyan-500/15 blur-3xl" />
            <p className="relative text-center text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-200/90">
              fMRI volume
            </p>
            <p className="relative mt-1 text-center text-[11px] text-slate-500">ROI-masked voxel channels</p>
            <div
              className="relative mx-auto mt-4 grid max-w-[200px] gap-0.5 rounded-lg border border-white/10 bg-slate-950/80 p-2 shadow-inner"
              style={{ gridTemplateColumns: `repeat(${GRID}, minmax(0, 1fr))` }}
            >
              {heat.map((v, idx) => (
                <motion.div
                  key={idx}
                  className="aspect-square rounded-[2px]"
                  style={{
                    backgroundColor: heatColor(v),
                    boxShadow: v > 0.65 ? `0 0 8px ${heatColor(v)}66` : undefined,
                  }}
                  initial={{ opacity: 0, scale: 0.5 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.35 + (idx % 24) * 0.012, duration: 0.35 }}
                />
              ))}
            </div>
            <motion.div
              className="relative mt-3 flex justify-center gap-1"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8 }}
              aria-hidden
            >
              {[0, 1, 2].map((i) => (
                <motion.span
                  key={i}
                  className="h-1 w-1 rounded-full bg-cyan-400/60"
                  animate={{ opacity: [0.35, 1, 0.35], scale: [1, 1.2, 1] }}
                  transition={{ duration: 1.6, repeat: Infinity, delay: i * 0.25 }}
                />
              ))}
            </motion.div>
          </GlassCard>
        </motion.div>

        <Beam delay={0} />

        <motion.div custom={1} initial="hidden" animate="show" variants={stageVariants} className="min-w-0 flex-1">
          <GlassCard
            glow
            className="relative h-full overflow-hidden border-violet-500/25 bg-gradient-to-br from-slate-950/90 via-violet-950/35 to-slate-900/80 p-4 sm:p-5"
          >
            <div className="pointer-events-none absolute -left-10 bottom-0 h-36 w-36 rounded-full bg-violet-500/20 blur-3xl" />
            <p className="relative text-center text-[10px] font-semibold uppercase tracking-[0.2em] text-violet-200/90">
              ROI Transformer Encoder
            </p>
            <p className="relative mt-1 text-center text-[11px] text-slate-500">Topology-aware brain tokens → CLS</p>
            <FlowParticles />
            <div className="relative mt-2 flex flex-wrap justify-center gap-2">
              {['V1–V4', 'FFA', 'PPA', '…'].map((tag, i) => (
                <motion.span
                  key={tag}
                  className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[9px] font-medium text-slate-300"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.55 + i * 0.06 }}
                >
                  {tag}
                </motion.span>
              ))}
            </div>
          </GlassCard>
        </motion.div>

        <Beam delay={0.08} />

        <motion.div custom={2} initial="hidden" animate="show" variants={stageVariants} className="min-w-0 flex-1">
          <GlassCard
            glow
            className="relative h-full overflow-hidden border-fuchsia-500/20 bg-gradient-to-br from-slate-950/90 via-slate-900/80 to-fuchsia-950/30 p-4 sm:p-5"
          >
            <div className="pointer-events-none absolute inset-x-0 top-0 h-24 bg-gradient-to-b from-cyan-500/10 to-transparent" />
            <p className="relative text-center text-[10px] font-semibold uppercase tracking-[0.2em] text-fuchsia-200/90">
              768-D CLIP embedding
            </p>
            <p className="relative mt-1 text-center text-[11px] text-slate-500">L2-normalized semantic direction</p>
            <div className="relative mx-auto mt-4 flex h-28 items-end justify-center gap-px overflow-hidden rounded-lg border border-white/10 bg-slate-950/90 px-1 pb-1 pt-3 shadow-inner">
              {eqBars.map((b, i) => {
                const h = Math.max(8, b * 100);
                const hue = 265 + (i / eqBars.length) * 60;
                return (
                  <motion.div
                    key={i}
                    className="min-w-[3px] max-w-[6px] flex-1 rounded-t-sm"
                    style={{
                      background: `linear-gradient(to top, hsl(${hue}, 45%, 22%), hsl(${hue}, 85%, 62%))`,
                      boxShadow: `0 0 10px hsla(${hue}, 90%, 55%, 0.25)`,
                    }}
                    initial={{ height: '0%' }}
                    animate={{ height: `${h}%` }}
                    transition={{
                      delay: 0.45 + (i % 12) * 0.02,
                      duration: 0.65,
                      ease: [0.22, 1, 0.36, 1],
                    }}
                  />
                );
              })}
            </div>
            <p className="relative mt-2 text-center font-mono text-[10px] text-cyan-300/80">d = 768 · ∥z∥₂ = 1</p>
          </GlassCard>
        </motion.div>
      </div>

      <motion.div
        className="w-full max-w-5xl px-1"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.55, duration: 0.7 }}
      >
        <GlassCard className="border-white/10 bg-slate-950/60 p-4 sm:p-5">
          <p className="mb-3 text-center text-[10px] font-semibold uppercase tracking-[0.25em] text-slate-500">
            Embedding spectrum (768 dimensions)
          </p>
          <div className="flex h-7 w-full overflow-hidden rounded-full border border-white/10 shadow-[inset_0_2px_12px_rgba(0,0,0,0.45)]">
            {stripSamples.map((s, i) => (
              <motion.div
                key={i}
                className="min-w-px flex-1"
                style={{
                  background: `hsl(${190 + s * 110}, ${65 + s * 28}%, ${32 + s * 28}%)`,
                }}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.65 + (i % 64) * 0.002 }}
              />
            ))}
          </div>
          <div className="mt-2 flex justify-between font-mono text-[9px] text-slate-600">
            <span>dim 0</span>
            <span>dim 384</span>
            <span>dim 767</span>
          </div>
        </GlassCard>
      </motion.div>

      <motion.p
        className="max-w-3xl text-center text-sm leading-relaxed text-slate-400 sm:text-[15px]"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.85, duration: 0.55 }}
      >
        The encoder compresses high-dimensional fMRI into a single L2-normalized vector aligned with CLIP ViT-L/14 —
        the shared language between brain activity and natural images.
      </motion.p>
    </div>
  );
}
