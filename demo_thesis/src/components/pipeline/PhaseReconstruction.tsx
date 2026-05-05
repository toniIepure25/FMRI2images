import { useCallback, useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import type { DemoCase, UncertaintyData } from '@/types';
import { SemiCircleUncertaintyGauge } from '@/components/uncertainty/SemiCircleUncertaintyGauge';
import { DuaCfgVisualPanel } from '@/components/uncertainty/DuaMappingPanels';

export interface PhaseReconstructionProps {
  case_: DemoCase;
  onReset: () => void;
}

function clamp01(x: number) { return Math.max(0, Math.min(1, x)); }

function SafeImg({ src, alt, className }: { src?: string; alt: string; className?: string }) {
  const [ok, setOk] = useState(true);
  useEffect(() => { setOk(true); }, [src]);
  if (!src || !ok) return <div className={`flex items-center justify-center bg-slate-900/90 text-slate-600 text-xs ${className ?? ''}`}>—</div>;
  return <img src={src} alt={alt} className={className} onError={() => setOk(false)} />;
}

function useCountUp(target: number, enabled: boolean, ms: number, dec: number): number {
  const [v, setV] = useState(0);
  useEffect(() => {
    if (!enabled) { setV(0); return; }
    let raf = 0;
    const t0 = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - t0) / ms);
      setV(target * (1 - (1 - t) ** 3));
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, enabled, ms]);
  return dec <= 0 ? Math.round(v) : Number(v.toFixed(dec));
}

export function PhaseReconstruction({ case_, onReset }: PhaseReconstructionProps) {
  const navigate = useNavigate();
  const { uncertainty: u, duaCfg, metrics: m } = case_;

  const deltaNorm = u.delta <= 1 ? u.delta : clamp01(u.delta / (u.delta + 1));
  const kappa01 = clamp01(u.kappaNorm);

  const top1 = useMemo(
    () => case_.retrievedImages.find((r) => r.rank === 1) ?? case_.retrievedImages[0],
    [case_.retrievedImages],
  );
  const reconFinal = case_.diffusionFinal ?? case_.reconstructionImage;
  const reconPrior = case_.diffusionPrior;
  const hasPriorFinal = Boolean(reconPrior && reconFinal);

  const [stage, setStage] = useState<'dua' | 'diffusion' | 'reveal' | 'metrics'>('dua');
  const [diffStep, setDiffStep] = useState(0);
  const [diffDone, setDiffDone] = useState(false);
  const [diffBlend, setDiffBlend] = useState(0);

  useEffect(() => {
    setStage('dua'); setDiffStep(0); setDiffDone(false); setDiffBlend(0);
    const t = window.setTimeout(() => setStage('diffusion'), 1200);
    return () => clearTimeout(t);
  }, [case_.id]);

  useEffect(() => {
    if (stage !== 'diffusion') return;
    let killed = false;
    let raf = 0, timer = 0;
    const totalMs = 2200, t0 = performance.now();
    const tick = (now: number) => {
      if (killed) return;
      const t = Math.min(1, (now - t0) / totalMs);
      setDiffBlend(1 - (1 - t) ** 2);
      setDiffStep(Math.min(150, Math.max(1, Math.round(t * 150))));
      if (t < 1) { raf = requestAnimationFrame(tick); }
      else { setDiffDone(true); timer = window.setTimeout(() => { if (!killed) setStage('reveal'); }, 800); }
    };
    raf = requestAnimationFrame(tick);
    return () => { killed = true; cancelAnimationFrame(raf); clearTimeout(timer); };
  }, [stage, case_.id]);

  useEffect(() => {
    if (stage !== 'reveal') return;
    const t = window.setTimeout(() => setStage('metrics'), 1000);
    return () => clearTimeout(t);
  }, [stage]);

  const showMetrics = stage === 'metrics';
  const rankV = useCountUp(m.rank, showMetrics, 600, 0);
  const pixV = useCountUp(m.pixcorr, showMetrics, 900, 3);
  const ssimV = useCountUp(m.ssim, showMetrics, 900, 3);
  const clipV = useCountUp(m.cosine, showMetrics, 900, 3);

  const verdict = useMemo(() => {
    if (m.rank === 1) return { text: 'Perfect Decode', cls: 'border-emerald-400/50 bg-emerald-500/12 text-emerald-200', icon: '◆' };
    if (m.rank <= 5) return { text: 'Near Miss', cls: 'border-amber-400/50 bg-amber-500/12 text-amber-200', icon: '◇' };
    return { text: 'Challenging Trial', cls: 'border-red-400/40 bg-red-500/10 text-red-200', icon: '△' };
  }, [m.rank]);

  const showDiff = stage === 'diffusion' || stage === 'reveal' || stage === 'metrics';
  const showReveal = stage === 'reveal' || stage === 'metrics';

  return (
    <div className="space-y-5 pb-16">
      {/* Header */}
      <motion.div
        className="rounded-2xl border border-white/[0.06] bg-gradient-to-r from-slate-900/80 via-brain-navy/60 to-slate-900/80 p-5 backdrop-blur-xl"
        initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
      >
        <h2 className="text-base font-semibold text-white sm:text-lg">Reconstruction & Reveal</h2>
        <p className="mt-0.5 text-xs text-slate-400">Uncertainty-aware diffusion → side-by-side comparison</p>
      </motion.div>

      {/* DUA-CFG section */}
      <section className="space-y-4">
        <div className="grid gap-4 lg:grid-cols-2">
          <SemiCircleUncertaintyGauge fraction={kappa01} centerValue={u.kappa.toFixed(1)} percentLabel={`${(kappa01 * 100).toFixed(0)}%`}
            title="κ CONCENTRATION" description="Directional certainty on the CLIP hypersphere" accent="cyan" animationDelay={0.02} />
          <SemiCircleUncertaintyGauge fraction={deltaNorm} centerValue={u.delta.toFixed(3)} percentLabel={`${(deltaNorm * 100).toFixed(0)}%`}
            title="δ DISAGREEMENT" description="Cross-ROI directional tension" accent="amber" animationDelay={0.08} />
        </div>

        <motion.div className="rounded-2xl border border-white/[0.06] bg-gradient-to-b from-slate-900/60 to-slate-900/40 p-5"
          initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
          <p className="mb-3 text-[10px] font-bold uppercase tracking-[0.2em] text-brain-accent">DUA-CFG Computation</p>
          <DuaCfgVisualPanel dua={duaCfg} />
        </motion.div>
      </section>

      {/* Diffusion */}
      <AnimatePresence>
        {showDiff && (
          <motion.section className="overflow-hidden rounded-2xl border border-cyan-500/15 bg-gradient-to-b from-cyan-950/12 to-slate-900/40 p-5"
            initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-300">Diffusion Process</h3>
              <span className="font-mono text-[11px] tabular-nums text-slate-500">
                {diffDone ? 'Complete' : `Step ${diffStep} / 150`}
              </span>
            </div>
            {/* Progress bar */}
            <div className="mb-4 h-1 w-full overflow-hidden rounded-full bg-black/30">
              <motion.div className="h-full rounded-full bg-gradient-to-r from-cyan-600 to-brain-accent"
                animate={{ width: `${(diffStep / 150) * 100}%` }} transition={{ duration: 0.05 }} />
            </div>

            <div className="relative mx-auto aspect-square w-full max-w-lg overflow-hidden rounded-xl ring-1 ring-white/10">
              {hasPriorFinal ? (
                <>
                  <SafeImg src={reconPrior} alt="Prior" className="absolute inset-0 h-full w-full object-cover" />
                  <motion.div className="absolute inset-0" animate={{ opacity: diffBlend }}>
                    <SafeImg src={reconFinal} alt="Reconstruction" className="h-full w-full object-cover" />
                  </motion.div>
                </>
              ) : (
                <>
                  <motion.div className="absolute inset-0" style={{
                    background: `linear-gradient(135deg, rgb(15,23,42), rgb(30,41,59) ${30 + diffBlend * 40}%, rgba(0,212,255,${0.1 + diffBlend * 0.2}))`
                  }} />
                  <motion.div className="absolute inset-0" animate={{ opacity: diffBlend }}>
                    <SafeImg src={reconFinal} alt="Reconstruction" className="h-full w-full object-cover" />
                  </motion.div>
                </>
              )}
            </div>
          </motion.section>
        )}
      </AnimatePresence>

      {/* Grand Reveal */}
      <AnimatePresence>
        {showReveal && (
          <motion.section className="space-y-4" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <h3 className="text-center text-[10px] font-bold uppercase tracking-[0.3em] text-slate-400">
              Grand Reveal
            </h3>
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
              {([
                { key: 'gt', title: 'Subject perceived', subtitle: 'Ground truth stimulus', src: case_.targetImage, accent: 'emerald', delay: 0 },
                { key: 'ret', title: 'Model retrieved', subtitle: `Rank #${top1?.rank ?? 1} from gallery`, src: top1?.image, accent: 'violet', delay: 0.1 },
                { key: 'rec', title: 'Model reconstructed', subtitle: 'Diffusion output', src: reconFinal, accent: 'cyan', delay: 0.2 },
              ] as const).map((col) => (
                <motion.div key={col.key}
                  className={`overflow-hidden rounded-2xl border-2 transition-shadow ${
                    col.accent === 'emerald' ? 'border-emerald-500/40 shadow-[0_0_30px_rgba(16,185,129,0.08)]'
                    : col.accent === 'violet' ? 'border-violet-500/40 shadow-[0_0_30px_rgba(139,92,246,0.1)]'
                    : 'border-cyan-500/40 shadow-[0_0_30px_rgba(0,212,255,0.1)]'
                  }`}
                  initial={{ opacity: 0, y: 28, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  transition={{ type: 'spring', stiffness: 240, damping: 22, delay: col.delay }}
                >
                  <div className={`border-b px-4 py-2.5 ${
                    col.accent === 'emerald' ? 'border-emerald-500/20 bg-emerald-950/20'
                    : col.accent === 'violet' ? 'border-violet-500/20 bg-violet-950/20'
                    : 'border-cyan-500/20 bg-cyan-950/20'
                  }`}>
                    <p className={`text-[11px] font-semibold ${
                      col.accent === 'emerald' ? 'text-emerald-200' : col.accent === 'violet' ? 'text-violet-200' : 'text-cyan-200'
                    }`}>{col.title}</p>
                    <p className="text-[9px] text-slate-500">{col.subtitle}</p>
                  </div>
                  <div className="relative aspect-square w-full bg-black/40">
                    {col.key === 'gt' ? (
                      <>
                        <motion.div className="absolute inset-0"
                          initial={{ scale: 1.08, filter: 'blur(16px)', opacity: 0 }}
                          animate={{ scale: 1, filter: 'blur(0px)', opacity: 1 }}
                          transition={{ delay: 0.5, duration: 0.8, ease: [0.22, 1, 0.36, 1] }}>
                          <SafeImg src={col.src} alt={col.title} className="h-full w-full object-cover" />
                        </motion.div>
                        <motion.div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm"
                          initial={{ opacity: 1 }} animate={{ opacity: 0 }} transition={{ delay: 0.4, duration: 0.5 }}>
                          <motion.span className="font-mono text-4xl font-bold text-white"
                            animate={{ scale: [1, 1.1, 1], opacity: [1, 0.5, 1] }} transition={{ duration: 1, repeat: 2 }}>?</motion.span>
                        </motion.div>
                      </>
                    ) : (
                      <SafeImg src={col.src} alt={col.title} className="h-full w-full object-cover" />
                    )}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.section>
        )}
      </AnimatePresence>

      {/* Metrics */}
      <AnimatePresence>
        {showMetrics && (
          <motion.div className="space-y-5" initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}
            transition={{ type: 'spring', stiffness: 200, damping: 24 }}>

            <div className="rounded-2xl border border-white/[0.06] bg-gradient-to-b from-slate-900/60 to-slate-900/40 p-5 sm:p-6">
              <div className="mb-5 flex flex-col items-center justify-between gap-3 sm:flex-row">
                <h3 className="text-sm font-semibold text-white">Trial Metrics</h3>
                <span className={`flex items-center gap-2 rounded-full border px-4 py-1.5 text-xs font-bold ${verdict.cls}`}>
                  <span>{verdict.icon}</span> {verdict.text} — Rank #{m.rank}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                {([
                  { label: 'Rank', value: String(rankV), hint: 'Gallery retrieval rank', color: m.rank === 1 ? 'text-emerald-300' : m.rank <= 5 ? 'text-amber-300' : 'text-red-300' },
                  { label: 'PixCorr', value: pixV.toFixed(3), hint: 'Pixel correlation', color: 'text-white' },
                  { label: 'SSIM', value: ssimV.toFixed(3), hint: 'Structural similarity', color: 'text-white' },
                  { label: 'CLIP cos', value: clipV.toFixed(3), hint: 'Embedding cosine', color: 'text-white' },
                ] as const).map((metric, i) => (
                  <motion.div key={metric.label}
                    className="rounded-xl border border-white/[0.06] bg-black/20 p-3 sm:p-4"
                    initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 * i }}>
                    <p className="text-[9px] font-bold uppercase tracking-wider text-slate-500">{metric.label}</p>
                    <p className={`mt-1 font-mono text-xl font-bold tabular-nums sm:text-2xl ${metric.color}`}>{metric.value}</p>
                    <p className="mt-0.5 text-[8px] text-slate-600">{metric.hint}</p>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Actions */}
            <div className="flex flex-col gap-3 sm:flex-row sm:justify-center">
              <button type="button" onClick={onReset}
                className="rounded-xl border border-white/10 bg-white/[0.03] px-6 py-3 text-sm font-semibold text-slate-300 transition hover:border-brain-accent/30 hover:text-white">
                Try Another Trial
              </button>
              <button type="button" onClick={() => navigate(`/explorer/${case_.id}`)}
                className="rounded-xl border border-violet-500/30 bg-violet-950/20 px-6 py-3 text-sm font-semibold text-violet-200 transition hover:border-violet-400/50">
                Explore in Detail →
              </button>
              <button type="button" onClick={() => navigate('/challenge')}
                className="rounded-xl border border-brain-accent/30 bg-brain-accent/8 px-6 py-3 text-sm font-semibold text-brain-accent transition hover:bg-brain-accent/15">
                Take the Challenge →
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
