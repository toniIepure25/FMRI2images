import { useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { DemoCase } from '@/types';
import { GlassCard } from '@/components/GlassCard';
import { getDifficultyColor } from '@/lib/data';

function difficultyColor(difficulty: DemoCase['difficulty']): string {
  return getDifficultyColor(difficulty);
}

export interface StepSelectTrialProps {
  case_: DemoCase;
}

function useTypewriter(full: string, enabled: boolean, msPerChar = 28) {
  const [n, setN] = useState(0);
  useEffect(() => {
    if (!enabled || !full) {
      setN(0);
      return;
    }
    setN(0);
    let i = 0;
    const id = window.setInterval(() => {
      i += 1;
      setN(Math.min(i, full.length));
      if (i >= full.length) window.clearInterval(id);
    }, msPerChar);
    return () => window.clearInterval(id);
  }, [full, enabled, msPerChar]);
  return full.slice(0, n);
}

export function StepSelectTrial({ case_ }: StepSelectTrialProps) {
  const preview = case_.fmriPreview ?? [];
  const maxV = Math.max(...preview.map((x) => Math.abs(x)), 1e-6);

  const [phase, setPhase] = useState<'idle' | 'meta' | 'acquire' | 'done'>('idle');
  const [barsFromZero, setBarsFromZero] = useState(true);

  useEffect(() => {
    setPhase('idle');
    setBarsFromZero(true);
    const a = window.setTimeout(() => setPhase('meta'), 400);
    const b = window.setTimeout(() => setPhase('acquire'), 2200);
    const c = window.setTimeout(() => setBarsFromZero(false), 2800);
    const d = window.setTimeout(() => setPhase('done'), 5200);
    return () => {
      window.clearTimeout(a);
      window.clearTimeout(b);
      window.clearTimeout(c);
      window.clearTimeout(d);
    };
  }, [case_.id]);

  const subjectText = useMemo(() => `${case_.subject}`, [case_.subject]);
  const sessionText = useMemo(() => `Session ${case_.session}`, [case_.session]);
  const trialText = useMemo(
    () => `Trial ${case_.repetition} · nsdId ${case_.nsdId}`,
    [case_.repetition, case_.nsdId],
  );

  const twSubject = useTypewriter(subjectText, phase === 'meta' || phase === 'acquire' || phase === 'done');
  const twSession = useTypewriter(sessionText, phase === 'acquire' || phase === 'done', 24);
  const twTrial = useTypewriter(trialText, phase === 'acquire' || phase === 'done', 22);

  const showProcessing = phase === 'meta' || phase === 'acquire';
  const showCompleteMsg = phase === 'done';

  return (
    <div className="flex h-full min-h-0 flex-col items-center justify-center gap-8 px-4 py-6">
      <motion.h2
        className="text-center text-2xl font-semibold tracking-tight text-white sm:text-3xl md:text-4xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      >
        Incoming fMRI scan…
      </motion.h2>

      <div className="relative flex w-full max-w-lg flex-col items-center">
        <motion.div
          className="pointer-events-none absolute inset-0 flex items-center justify-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: showProcessing ? 1 : 0.35 }}
        >
          {[1, 2, 3].map((ring) => (
            <motion.span
              key={ring}
              className="absolute rounded-full border border-cyan-400/35"
              style={{ width: 120 + ring * 56, height: 120 + ring * 56 }}
              animate={{
                scale: [1, 1.08, 1],
                opacity: [0.15, 0.45 - ring * 0.08, 0.12],
              }}
              transition={{
                duration: 2.4 + ring * 0.35,
                repeat: Infinity,
                ease: 'easeInOut',
                delay: ring * 0.2,
              }}
            />
          ))}
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.96, y: 16 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ delay: 0.08, duration: 0.5 }}
          className="relative z-10 w-full"
        >
          <GlassCard glow className="p-5 sm:p-6">
            <div className="flex flex-wrap items-start justify-between gap-3 border-b border-white/10 pb-4">
              <div className="min-w-0 flex-1">
                <p className="text-xs uppercase tracking-widest text-cyan-300/80">Acquisition stream</p>
                <p className="mt-1 truncate text-lg font-semibold text-white">{case_.title}</p>
                <p className="mt-2 text-sm leading-relaxed text-slate-400">{case_.description}</p>
              </div>
              <span
                className="shrink-0 rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide"
                style={{
                  backgroundColor: `${difficultyColor(case_.difficulty)}22`,
                  color: difficultyColor(case_.difficulty),
                  border: `1px solid ${difficultyColor(case_.difficulty)}55`,
                }}
              >
                {case_.difficulty}
              </span>
            </div>

            <dl className="mt-4 grid grid-cols-1 gap-3 text-xs sm:grid-cols-3">
              <div className="rounded-lg border border-white/5 bg-slate-950/50 px-3 py-2">
                <dt className="text-slate-500">Subject ID</dt>
                <dd className="mt-1 min-h-[1.25rem] font-mono text-cyan-100">{twSubject}</dd>
              </div>
              <div className="rounded-lg border border-white/5 bg-slate-950/50 px-3 py-2">
                <dt className="text-slate-500">Session</dt>
                <dd className="mt-1 min-h-[1.25rem] font-mono text-cyan-100">{twSession}</dd>
              </div>
              <div className="rounded-lg border border-white/5 bg-slate-950/50 px-3 py-2">
                <dt className="text-slate-500">Trial</dt>
                <dd className="mt-1 min-h-[1.25rem] font-mono text-cyan-100">{twTrial}</dd>
              </div>
            </dl>

            <AnimatePresence mode="wait">
              {showProcessing ? (
                <motion.div
                  key="proc"
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  className="mt-4 flex items-center gap-2 rounded-lg border border-cyan-500/25 bg-cyan-500/10 px-3 py-2"
                >
                  <motion.span
                    className="relative flex h-2.5 w-2.5"
                    aria-hidden
                  >
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-cyan-400 opacity-60" />
                    <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-cyan-400" />
                  </motion.span>
                  <span className="text-sm font-medium text-cyan-100">Processing…</span>
                </motion.div>
              ) : null}
              {showCompleteMsg ? (
                <motion.p
                  key="done"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 text-center text-sm font-medium text-emerald-200/95"
                >
                  Scan complete. Entering visual cortex analysis…
                </motion.p>
              ) : null}
            </AnimatePresence>
          </GlassCard>
        </motion.div>
      </div>

      <motion.div
        className="w-full max-w-2xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.5 }}
      >
        <p className="mb-3 text-center text-xs uppercase tracking-wider text-slate-500">
          fMRI signal (ROI channels, streaming)
        </p>
        <div className="flex h-28 items-end justify-center gap-0.5 rounded-xl border border-white/10 bg-slate-950/60 px-2 py-3 sm:h-36 sm:gap-1">
          {preview.length === 0 ? (
            <p className="self-center text-sm text-slate-500">No preview values</p>
          ) : (
            preview.map((v, i) => {
              const h = (Math.abs(v) / maxV) * 100;
              const hue = 175 + (i / Math.max(preview.length - 1, 1)) * 85;
              const active = !barsFromZero && (phase === 'acquire' || phase === 'done');
              return (
                <motion.div
                  key={i}
                  className="min-w-[3px] max-w-[8px] flex-1 rounded-t-sm sm:max-w-[10px]"
                  style={{
                    background: active
                      ? `linear-gradient(to top, hsl(${hue}, 75%, 38%), hsl(${hue}, 85%, 62%))`
                      : 'rgba(51,65,85,0.35)',
                    boxShadow: active ? `0 0 12px hsla(${hue}, 90%, 55%, 0.25)` : undefined,
                  }}
                  initial={{ height: '4%' }}
                  animate={{ height: active ? `${Math.max(h, 6)}%` : '4%' }}
                  transition={{
                    delay: active ? 0.05 + i * 0.018 : 0,
                    duration: active ? 0.55 : 0.35,
                    ease: [0.22, 1, 0.36, 1],
                  }}
                />
              );
            })
          )}
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, scale: 0.92 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.35, duration: 0.55 }}
        className="w-full max-w-xs"
      >
        <p className="mb-2 text-center text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">
          What did the subject see?
        </p>
        <GlassCard className="overflow-hidden p-0">
          <div className="relative aspect-square w-full bg-gradient-to-br from-slate-900 via-slate-800 to-violet-950/60">
            <div className="flex h-full w-full flex-col items-center justify-center gap-2">
              <motion.span
                className="text-5xl font-black text-white/30 sm:text-6xl"
                animate={{ opacity: [0.25, 0.5, 0.25], scale: [1, 1.04, 1] }}
                transition={{ duration: 2.2, repeat: Infinity }}
              >
                ?
              </motion.span>
              <span className="text-[10px] font-medium uppercase tracking-widest text-slate-500">
                Ground truth withheld
              </span>
            </div>
            <div className="pointer-events-none absolute inset-0 ring-1 ring-inset ring-white/10" />
          </div>
        </GlassCard>
      </motion.div>
    </div>
  );
}
