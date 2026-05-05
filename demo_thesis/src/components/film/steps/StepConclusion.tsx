import { useMemo, useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import type { DemoCase } from '@/types';
import { GlassCard } from '@/components/GlassCard';
import { getConfidenceColor, getConfidenceLabel } from '@/lib/data';

export interface StepConclusionProps {
  case_: DemoCase;
  onTryAnotherCase?: () => void;
}

function SafeImg({ src, alt, className }: { src: string; alt: string; className?: string }) {
  const [ok, setOk] = useState(true);
  const onErr = useCallback(() => setOk(false), []);
  if (!src || !ok) {
    return (
      <div
        className={`flex items-center justify-center bg-slate-900 text-xs text-slate-500 ${className ?? ''}`}
        role="img"
        aria-label={alt}
      >
        —
      </div>
    );
  }
  return <img src={src} alt={alt} className={className} onError={onErr} />;
}

export function StepConclusion({ case_, onTryAnotherCase }: StepConclusionProps) {
  const navigate = useNavigate();
  const m = case_.metrics;
  const u = case_.uncertainty;

  const topRois = useMemo(
    () => [...(case_.roiScores ?? [])].sort((a, b) => b.contribution - a.contribution).slice(0, 3),
    [case_.roiScores],
  );

  const topRetrieved = useMemo(() => {
    const r = [...(case_.retrievedImages ?? [])].sort((a, b) => a.rank - b.rank);
    return r[0];
  }, [case_.retrievedImages]);

  const reconSrc = case_.diffusionFinal ?? case_.reconstructionImage;
  const [revealed, setRevealed] = useState(false);

  useEffect(() => {
    setRevealed(false);
    const t = window.setTimeout(() => setRevealed(true), 750);
    return () => window.clearTimeout(t);
  }, [case_.id]);

  const confColor = getConfidenceColor(u.confidenceLevel);
  const badgeLabel = getConfidenceLabel(u.confidenceLevel);

  return (
    <div className="flex h-full min-h-0 flex-col gap-6 px-3 pb-6">
      <motion.h2
        className="text-center text-2xl font-semibold text-white sm:text-3xl md:text-4xl"
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        Decoding complete
      </motion.h2>

      <motion.div
        className="mx-auto flex flex-wrap items-center justify-center gap-3"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
      >
        <span
          className="inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold"
          style={{
            borderColor: `${confColor}88`,
            backgroundColor: `${confColor}22`,
            color: confColor,
          }}
        >
          <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: confColor }} aria-hidden />
          {badgeLabel}
        </span>
        <span className="font-mono text-[11px] text-slate-500">
          κ={u.kappa.toFixed(1)} · δ={u.delta.toFixed(2)}
        </span>
      </motion.div>

      <div className="mx-auto grid w-full max-w-6xl gap-4 md:grid-cols-3">
        {[
          {
            label: 'The subject saw',
            src: case_.targetImage,
            tone: 'emerald' as const,
            reveal: true,
          },
          {
            label: 'The model retrieved',
            src: topRetrieved?.image ?? case_.targetImage,
            tone: 'violet' as const,
            reveal: false,
          },
          {
            label: 'The model reconstructed',
            src: reconSrc,
            tone: 'cyan' as const,
            reveal: false,
          },
        ].map((col, i) => (
          <motion.div
            key={col.label}
            initial={{ opacity: 0, y: 22 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.08 + i * 0.12, duration: 0.5 }}
          >
            <GlassCard className="overflow-hidden p-3" glow={col.tone === 'emerald'}>
              <p
                className={`mb-3 text-center text-[11px] font-semibold uppercase tracking-[0.15em] ${
                  col.tone === 'emerald'
                    ? 'text-emerald-300'
                    : col.tone === 'violet'
                      ? 'text-violet-300'
                      : 'text-cyan-300'
                }`}
              >
                {col.label}
              </p>
              <div className="relative overflow-hidden rounded-xl border border-white/10">
                {col.reveal ? (
                  <AnimatePresence mode="wait">
                    {!revealed ? (
                      <motion.div
                        key="q"
                        className="flex aspect-square w-full flex-col items-center justify-center bg-gradient-to-br from-slate-900 to-violet-950/50"
                        initial={{ opacity: 1 }}
                        exit={{ opacity: 0, scale: 0.92, filter: 'blur(8px)' }}
                        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                      >
                        <motion.span
                          className="text-5xl font-black text-white/35"
                          animate={{ scale: [1, 1.08, 1], opacity: [0.35, 0.6, 0.35] }}
                          transition={{ duration: 1.6, repeat: Infinity }}
                        >
                          ?
                        </motion.span>
                        <span className="mt-2 text-[9px] uppercase tracking-widest text-slate-500">
                          Resolving ground truth…
                        </span>
                      </motion.div>
                    ) : (
                      <motion.div
                        key="img"
                        initial={{ opacity: 0, scale: 0.88, filter: 'blur(12px)' }}
                        animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
                        transition={{ type: 'spring', stiffness: 280, damping: 24 }}
                      >
                        <SafeImg src={col.src} alt={col.label} className="aspect-square w-full object-cover" />
                      </motion.div>
                    )}
                  </AnimatePresence>
                ) : (
                  <SafeImg src={col.src} alt={col.label} className="aspect-square w-full object-cover" />
                )}
              </div>
            </GlassCard>
          </motion.div>
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35 }}
      >
        <GlassCard className="mx-auto max-w-6xl p-5">
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-500">Metrics summary</p>
          <dl className="mt-3 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
            <div>
              <dt className="text-slate-500">Gallery rank</dt>
              <dd className="font-mono text-white">{m.rank}</dd>
            </div>
            <div>
              <dt className="text-slate-500">Cosine</dt>
              <dd className="font-mono text-white">{m.cosine.toFixed(3)}</dd>
            </div>
            <div>
              <dt className="text-slate-500">CSLS</dt>
              <dd className="font-mono text-white">{m.csls.toFixed(3)}</dd>
            </div>
            <div>
              <dt className="text-slate-500">R@1 / R@5</dt>
              <dd className="font-mono text-white">
                {m.r1Correct ? '✓' : '✗'} / {m.r5Correct ? '✓' : '✗'}
              </dd>
            </div>
            <div>
              <dt className="text-slate-500">PixCorr</dt>
              <dd className="font-mono text-white">{m.pixcorr.toFixed(3)}</dd>
            </div>
            <div>
              <dt className="text-slate-500">SSIM</dt>
              <dd className="font-mono text-white">{m.ssim.toFixed(3)}</dd>
            </div>
            <div>
              <dt className="text-slate-500">Alex(2)</dt>
              <dd className="font-mono text-white">{m.alex2.toFixed(3)}</dd>
            </div>
            <div>
              <dt className="text-slate-500">Alex(5)</dt>
              <dd className="font-mono text-white">{m.alex5.toFixed(3)}</dd>
            </div>
          </dl>
        </GlassCard>
      </motion.div>

      <div className="mx-auto grid w-full max-w-6xl gap-4 lg:grid-cols-2">
        <motion.div
          initial={{ opacity: 0, x: -14 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
        >
          <GlassCard className="h-full p-5">
            <p className="text-xs font-semibold uppercase tracking-widest text-slate-500">Key ROI contributions</p>
            <p className="mt-1 text-[11px] text-slate-500">Top 3 regions by decoder contribution</p>
            <ul className="mt-4 space-y-3">
              {topRois.length === 0 ? (
                <li className="text-sm text-slate-500">No ROI breakdown available.</li>
              ) : (
                topRois.map((r, idx) => (
                  <li key={r.name} className="flex items-baseline justify-between gap-2 border-b border-white/5 pb-2 text-sm last:border-0">
                    <span className="font-medium text-slate-200">
                      <span className="mr-2 font-mono text-xs text-cyan-500/80">{idx + 1}.</span>
                      {r.name}
                    </span>
                    <span className="font-mono text-xs text-cyan-200/90">
                      {(r.contribution * 100).toFixed(0)}%
                    </span>
                  </li>
                ))
              )}
            </ul>
          </GlassCard>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 14 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.42 }}
        >
          <GlassCard className="h-full p-6">
            <p className="text-sm leading-relaxed text-slate-300">{case_.interpretation}</p>
          </GlassCard>
        </motion.div>
      </div>

      <motion.div
        className="mx-auto flex max-w-6xl flex-wrap justify-center gap-3"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.48 }}
      >
        {onTryAnotherCase ? (
          <motion.button
            type="button"
            onClick={() => onTryAnotherCase?.()}
            className="rounded-xl border border-white/20 bg-white/5 px-5 py-2.5 text-sm font-semibold text-slate-100 transition hover:border-cyan-400/35 hover:bg-white/10"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            Try another case
          </motion.button>
        ) : null}
        <motion.button
          type="button"
          onClick={() => navigate(`/explorer/${case_.id}`)}
          className="rounded-xl border border-cyan-500/40 bg-cyan-500/10 px-5 py-2.5 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-500/20"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          Open in Explorer
        </motion.button>
        <motion.button
          type="button"
          onClick={() => navigate('/challenge')}
          className="rounded-xl border border-violet-500/40 bg-violet-500/10 px-5 py-2.5 text-sm font-semibold text-violet-100 transition hover:bg-violet-500/20"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          Take the Challenge
        </motion.button>
      </motion.div>
    </div>
  );
}
