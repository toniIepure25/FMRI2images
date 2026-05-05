import { useEffect, useMemo, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { DemoCase } from '@/types';
import { GlassCard } from '@/components/GlassCard';

export interface StepReconstructionProps {
  case_: DemoCase;
}

function SafeImage({
  src,
  alt,
  className,
}: {
  src: string | undefined;
  alt: string;
  className?: string;
}) {
  const [ok, setOk] = useState(Boolean(src));
  const onError = useCallback(() => setOk(false), []);
  if (!src || !ok) {
    return (
      <div
        className={`flex items-center justify-center bg-slate-900 text-[11px] text-slate-500 ${className ?? ''}`}
        role="img"
        aria-label={alt}
      >
        —
      </div>
    );
  }
  return <img src={src} alt={alt} className={className} onError={onError} />;
}

export function StepReconstruction({ case_ }: StepReconstructionProps) {
  const dua = case_.duaCfg;
  const prior = case_.diffusionPrior;
  const final = case_.diffusionFinal;
  const recon = case_.reconstructionImage;
  const hasPair = Boolean(prior && final);

  type GenStatus = 'init' | 'denoise' | 'done';
  const [genStatus, setGenStatus] = useState<GenStatus>('init');

  const [phase, setPhase] = useState<'prior' | 'final'>('prior');
  const [stepIdx, setStepIdx] = useState(0);
  const [showBefore, setShowBefore] = useState(true);

  const totalSteps = Math.max(1, dua.diffusionSteps ?? 50);

  useEffect(() => {
    setGenStatus('init');
    const a = window.setTimeout(() => setGenStatus('denoise'), 700);
    const b = window.setTimeout(() => setGenStatus('done'), 5200);
    return () => {
      window.clearTimeout(a);
      window.clearTimeout(b);
    };
  }, [case_.id]);

  useEffect(() => {
    setPhase('prior');
    setStepIdx(0);
    setShowBefore(true);
  }, [case_.id, hasPair]);

  useEffect(() => {
    if (!hasPair) return;
    const t = window.setTimeout(() => setPhase('final'), 2000);
    return () => window.clearTimeout(t);
  }, [hasPair, case_.id]);

  useEffect(() => {
    const durationMs = hasPair ? 2200 : 2800;
    const start = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs);
      setStepIdx(Math.round(t * totalSteps));
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [hasPair, totalSteps, case_.id]);

  const guidanceFrac = useMemo(() => {
    const g = dua.guidanceScale;
    const n = Math.min(1, Math.max(0, (g - 5) / 8));
    return n;
  }, [dua.guidanceScale]);

  const centerImage = hasPair ? (phase === 'prior' ? prior! : final!) : recon;

  return (
    <div className="flex h-full min-h-0 flex-col items-center gap-5 px-2 py-2 sm:gap-6 sm:px-4">
      <motion.div
        className="mx-auto flex w-full max-w-2xl items-center justify-center gap-2 rounded-xl border border-violet-500/25 bg-violet-500/10 px-4 py-3"
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45 }}
      >
        <motion.span
          className="h-2 w-2 shrink-0 rounded-full bg-violet-400"
          animate={genStatus === 'done' ? {} : { opacity: [0.35, 1, 0.35], scale: [1, 1.15, 1] }}
          transition={{ duration: 1.3, repeat: genStatus === 'done' ? 0 : Infinity }}
        />
        <p className="text-center text-sm font-medium text-violet-100">
          {genStatus === 'init'
            ? 'Initializing diffusion model with DUA-CFG parameters…'
            : genStatus === 'denoise'
              ? 'Denoising in progress…'
              : 'Reconstruction complete.'}
        </p>
      </motion.div>

      <motion.h2
        className="text-center font-sans text-2xl font-semibold tracking-tight text-white sm:text-3xl md:text-[2rem]"
        style={{
          textShadow:
            '0 0 24px rgba(167, 139, 250, 0.35), 0 0 48px rgba(34, 211, 238, 0.18), 0 0 72px rgba(59, 130, 246, 0.1)',
        }}
        initial={{ opacity: 0, y: -12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.55 }}
      >
        Generating the visual reconstruction
      </motion.h2>

      <div className="grid w-full max-w-[1400px] gap-4 lg:grid-cols-[minmax(200px,240px)_minmax(0,1fr)_minmax(200px,260px)] lg:items-start">
        {/* Left column: DUA metric cards */}
        <motion.div
          className="order-2 flex flex-col gap-4 lg:order-none lg:col-start-1 lg:row-start-1"
          initial={{ opacity: 0, x: -18 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.08, duration: 0.55 }}
        >
          <GlassCard className="border-cyan-500/20 bg-slate-950/70 p-4">
            <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-300/90">DUA-CFG</p>
            <p className="mt-1 text-xs text-slate-500">Uncertainty-aware guidance at generation time</p>

            <div className="mt-4 space-y-4">
              <div>
                <div className="mb-1.5 flex justify-between text-xs text-slate-400">
                  <span>Guidance scale</span>
                  <span className="font-mono text-cyan-100">{dua.guidanceScale.toFixed(2)}</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                  <motion.div
                    className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-violet-500"
                    initial={{ width: 0 }}
                    animate={{ width: `${guidanceFrac * 100}%` }}
                    transition={{ delay: 0.2, duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
                  />
                </div>
                <div className="mt-1 flex justify-between font-mono text-[9px] text-slate-600">
                  <span>low</span>
                  <span>high</span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                  <p className="text-[9px] uppercase tracking-wider text-slate-500">Diffusion steps</p>
                  <p className="mt-1 font-mono text-xl text-white">{dua.diffusionSteps}</p>
                </div>
                <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                  <p className="text-[9px] uppercase tracking-wider text-slate-500">Ensemble K</p>
                  <p className="mt-1 font-mono text-xl text-white">{dua.ensembleK}</p>
                </div>
              </div>

              <div className="flex items-center justify-between rounded-lg border border-white/10 px-3 py-2 text-xs">
                <span className="text-slate-500">Abstain gate</span>
                <span className={`font-mono ${dua.abstain ? 'text-amber-300' : 'text-emerald-300'}`}>
                  {dua.abstain ? 'true' : 'false'}
                </span>
              </div>
            </div>
          </GlassCard>
        </motion.div>

        {/* Center stage */}
        <motion.div
          className="order-1 min-w-0 lg:order-none lg:col-start-2 lg:row-start-1"
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.05, duration: 0.55 }}
        >
          <GlassCard glow className="overflow-hidden border-violet-500/25 bg-gradient-to-b from-slate-950/95 to-slate-900/90 p-3 sm:p-4">
            <div className="relative mx-auto w-full max-w-lg">
              <div className="relative aspect-square overflow-hidden rounded-2xl border border-white/10 bg-slate-950 shadow-inner">
                <AnimatePresence mode="wait">
                  {hasPair ? (
                    <motion.div
                      key={phase}
                      className="absolute inset-0"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
                    >
                      <SafeImage
                        src={centerImage}
                        alt={phase === 'prior' ? 'Diffusion prior' : 'Denoised reconstruction'}
                        className="h-full w-full object-cover"
                      />
                      {phase === 'prior' ? (
                        <div
                          className="pointer-events-none absolute inset-0 mix-blend-soft-light opacity-40"
                          style={{
                            backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='80' height='80'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4'/%3E%3C/filter%3E%3Crect width='80' height='80' filter='url(%23n)' opacity='0.5'/%3E%3C/svg%3E")`,
                          }}
                        />
                      ) : null}
                    </motion.div>
                  ) : (
                    <motion.div key="sim" className="absolute inset-0">
                      <motion.div
                        className="absolute inset-0"
                        initial={{ filter: 'blur(28px) brightness(0.55) contrast(1.08)' }}
                        animate={{ filter: 'blur(0px) brightness(1) contrast(1)' }}
                        transition={{ duration: 2.4, ease: [0.22, 1, 0.36, 1], delay: 0.15 }}
                      >
                        <SafeImage src={recon} alt="Reconstruction" className="h-full w-full object-cover" />
                      </motion.div>
                      <motion.div
                        className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(34,211,238,0.08),transparent_55%)]"
                        initial={{ opacity: 0.75 }}
                        animate={{ opacity: 0.15 }}
                        transition={{ duration: 2.2 }}
                      />
                    </motion.div>
                  )}
                </AnimatePresence>

                <div className="pointer-events-none absolute inset-0 rounded-2xl ring-1 ring-inset ring-white/10" />

                <div className="absolute bottom-3 left-3 right-3 flex flex-col gap-2">
                  <div className="flex items-center justify-between gap-2 text-[10px] font-medium uppercase tracking-wider text-slate-300">
                    <span>
                      {hasPair
                        ? phase === 'prior'
                          ? 'Prior (noisy)'
                          : 'Denoised output'
                        : 'Simulated denoise (blur→sharp)'}
                    </span>
                    <span className="font-mono text-cyan-200">
                      Step {stepIdx}/{totalSteps}
                    </span>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded-full bg-slate-800/90">
                    <motion.div
                      className="h-full rounded-full bg-gradient-to-r from-violet-500 via-cyan-400 to-emerald-400"
                      style={{ boxShadow: '0 0 14px rgba(34,211,238,0.35)' }}
                      initial={{ width: '0%' }}
                      animate={{ width: `${(stepIdx / totalSteps) * 100}%` }}
                      transition={{ duration: 0.15 }}
                    />
                  </div>
                </div>
              </div>

              <motion.p
                className="mt-3 text-center text-xs leading-relaxed text-slate-400"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.35 }}
              >
                The diffusion model refines an initial noisy prior into a detailed visual reconstruction, guided by the
                decoded CLIP embedding and DUA-CFG parameters.
              </motion.p>
            </div>
          </GlassCard>
        </motion.div>

        {/* Right: before/after */}
        <motion.div
          className="order-3 min-w-0 lg:order-none lg:col-start-3 lg:row-start-1"
          initial={{ opacity: 0, x: 18 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.12, duration: 0.55 }}
        >
          <GlassCard className="h-full border-fuchsia-500/20 bg-slate-950/70 p-4">
            <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-fuchsia-200/90">Before / after</p>
            <p className="mt-1 text-xs text-slate-500">
              Toggle to compare the early latent estimate with the final decode.
            </p>

            <div className="mt-4 flex gap-2 rounded-lg border border-white/10 bg-slate-900/60 p-1">
              <button
                type="button"
                onClick={() => setShowBefore(true)}
                className={`flex-1 rounded-md py-2 text-xs font-semibold transition ${
                  showBefore ? 'bg-violet-600/40 text-white' : 'text-slate-500 hover:text-slate-300'
                }`}
              >
                Before
              </button>
              <button
                type="button"
                onClick={() => setShowBefore(false)}
                className={`flex-1 rounded-md py-2 text-xs font-semibold transition ${
                  !showBefore ? 'bg-cyan-600/35 text-white' : 'text-slate-500 hover:text-slate-300'
                }`}
              >
                After
              </button>
            </div>

            <div className="relative mt-4 aspect-square overflow-hidden rounded-xl border border-white/10">
              {case_.comparisonPanel ? (
                <SafeImage
                  src={case_.comparisonPanel}
                  alt="Comparison panel"
                  className="h-full w-full object-cover"
                />
              ) : (
                <>
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={showBefore ? 'b' : 'a'}
                      className="absolute inset-0"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.35 }}
                    >
                      <SafeImage
                        src={(showBefore ? prior ?? recon : final ?? recon) ?? recon}
                        alt={showBefore ? 'Before reconstruction' : 'After reconstruction'}
                        className="h-full w-full object-cover"
                      />
                    </motion.div>
                  </AnimatePresence>
                  <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-black/40 to-transparent" />
                </>
              )}
            </div>
          </GlassCard>
        </motion.div>
      </div>

      {/* Ground truth vs reconstruction */}
      <motion.div
        className="w-full max-w-6xl"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.55 }}
      >
        <GlassCard className="border-white/10 bg-slate-950/60 p-4 sm:p-5">
          <p className="text-center text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">
            Ground truth vs reconstruction
          </p>
          <div className="mt-4 grid grid-cols-2 gap-3 sm:gap-4">
            <div>
              <p className="mb-2 text-center text-[10px] text-emerald-400/90">Ground truth</p>
              <div className="overflow-hidden rounded-xl border border-emerald-500/30">
                <SafeImage src={case_.targetImage} alt="Ground truth" className="aspect-square w-full object-cover" />
              </div>
            </div>
            <div>
              <p className="mb-2 text-center text-[10px] text-cyan-400/90">Reconstruction</p>
              <div className="overflow-hidden rounded-xl border border-cyan-500/25">
                <SafeImage
                  src={final ?? recon}
                  alt="Model reconstruction"
                  className="aspect-square w-full object-cover"
                />
              </div>
            </div>
          </div>
        </GlassCard>
      </motion.div>
    </div>
  );
}
