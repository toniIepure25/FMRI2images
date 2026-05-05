import { useMemo, useState, useCallback, useEffect } from 'react';
import { motion } from 'framer-motion';
import type { DemoCase } from '@/types';
import { GlassCard } from '@/components/GlassCard';

export interface StepRetrievalProps {
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
        className={`flex items-center justify-center bg-gradient-to-br from-slate-900 via-slate-800 to-violet-950/50 text-center text-[10px] font-medium uppercase tracking-wider text-slate-500 ${className ?? ''}`}
        role="img"
        aria-label={alt}
      >
        No preview
      </div>
    );
  }
  return <img src={src} alt={alt} className={className} onError={onError} />;
}

function ClipQueryNode() {
  return (
    <motion.div
      className="relative z-20 flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl border border-cyan-400/35 bg-gradient-to-br from-cyan-500/20 via-slate-900/80 to-violet-600/25 shadow-[0_0_24px_rgba(34,211,238,0.25)]"
      initial={{ scale: 0.85, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: 'spring', stiffness: 280, damping: 22 }}
    >
      <svg viewBox="0 0 24 24" className="h-7 w-7 text-cyan-200" fill="none" stroke="currentColor" aria-hidden>
        <path
          strokeWidth="1.5"
          strokeLinecap="round"
          d="M12 4v4M12 16v4M4 12h4M16 12h4M6.34 6.34l2.83 2.83M14.83 14.83l2.83 2.83M6.34 17.66l2.83-2.83M14.83 9.17l2.83-2.83"
        />
        <circle cx="12" cy="12" r="3" strokeWidth="1.5" />
      </svg>
      <motion.span
        className="pointer-events-none absolute inset-0 rounded-2xl ring-2 ring-cyan-400/30"
        animate={{ opacity: [0.35, 0.85, 0.35], scale: [1, 1.04, 1] }}
        transition={{ duration: 2.8, repeat: Infinity }}
      />
      <span className="absolute -bottom-5 whitespace-nowrap text-[9px] font-semibold uppercase tracking-widest text-cyan-200/80">
        CLIP query
      </span>
    </motion.div>
  );
}

export function StepRetrieval({ case_ }: StepRetrievalProps) {
  const top = useMemo(() => {
    const list = [...(case_.retrievedImages ?? [])];
    list.sort((a, b) => a.rank - b.rank);
    return list.slice(0, 5);
  }, [case_.retrievedImages]);

  const [visibleCount, setVisibleCount] = useState(0);
  const [searchDone, setSearchDone] = useState(false);

  useEffect(() => {
    setVisibleCount(0);
    setSearchDone(false);
    if (top.length === 0) return undefined;
    setVisibleCount(1);
    let revealed = 1;
    if (revealed >= top.length) {
      setSearchDone(true);
      return undefined;
    }
    const id = window.setInterval(() => {
      revealed += 1;
      setVisibleCount((c) => Math.min(c + 1, top.length));
      if (revealed >= top.length) {
        window.clearInterval(id);
        setSearchDone(true);
      }
    }, 260);
    return () => window.clearInterval(id);
  }, [case_.id, top.length]);

  const m = case_.metrics;
  const stripSrc = case_.topkStrip;
  const targetSrc = case_.targetImage;

  const maxCosine = useMemo(() => Math.max(0.08, ...top.map((t) => t.score), m?.cosine ?? 0), [top, m]);

  return (
    <div className="flex h-full min-h-0 flex-col items-center gap-6 px-2 py-2 sm:gap-8 sm:px-4">
      <motion.p
        className="text-center text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400/85"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.4 }}
      >
        {searchDone
          ? 'Gallery search complete. Top candidates found.'
          : 'Searching the CLIP image gallery…'}
      </motion.p>

      <motion.h2
        className="text-center font-sans text-2xl font-semibold tracking-tight text-white sm:text-3xl md:text-[2rem]"
        style={{
          textShadow:
            '0 0 22px rgba(52, 211, 153, 0.35), 0 0 40px rgba(34, 211, 238, 0.2), 0 0 64px rgba(167, 139, 250, 0.12)',
        }}
        initial={{ opacity: 0, y: -14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
      >
        Finding visual matches
      </motion.h2>

      <div className="flex w-full max-w-6xl flex-col gap-5">
        {/* Ground truth — top row */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.08, duration: 0.6 }}
        >
          <GlassCard
            glow
            className="relative overflow-hidden border-emerald-500/35 bg-gradient-to-br from-slate-950/95 via-emerald-950/15 to-slate-900/90 p-4 sm:p-6"
          >
            <div className="pointer-events-none absolute -right-16 -top-16 h-48 w-48 rounded-full bg-emerald-400/10 blur-3xl" />
            <p className="relative text-center text-[10px] font-bold uppercase tracking-[0.2em] text-emerald-300/95">
              Ground truth — What the subject actually saw
            </p>
            <div className="relative mx-auto mt-4 max-w-md">
              <motion.div
                className="relative overflow-hidden rounded-2xl border-2 border-emerald-400/50 shadow-[0_0_0_1px_rgba(52,211,153,0.15),0_0_48px_rgba(52,211,153,0.22)]"
                initial={{ scale: 0.98, opacity: 0.85 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.15, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
              >
                <SafeImage src={targetSrc} alt="Target stimulus" className="aspect-[4/3] w-full object-cover" />
                <div className="pointer-events-none absolute inset-0 rounded-2xl ring-2 ring-inset ring-emerald-400/25" />
              </motion.div>
            </div>
          </GlassCard>
        </motion.div>

        {stripSrc ? (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            transition={{ delay: 0.2, duration: 0.45 }}
            className="overflow-hidden rounded-xl border border-white/10 bg-slate-950/70"
          >
            <p className="px-3 pt-2 text-[9px] font-semibold uppercase tracking-widest text-slate-500">
              Top-K gallery strip
            </p>
            <SafeImage src={stripSrc} alt="Top-K candidates strip" className="mt-1 h-16 w-full object-cover sm:h-20" />
          </motion.div>
        ) : null}

        {/* Retrieval row with query hub + connectors */}
        <div className="relative">
          <div className="pointer-events-none absolute left-1/2 top-[72px] z-0 hidden -translate-x-1/2 sm:block lg:top-[84px]">
            <svg width="100%" height="120" className="w-[min(92vw,1100px)] overflow-visible" aria-hidden>
              {top.map((item, i) => {
                const xPct = 12 + (i / Math.max(top.length - 1, 1)) * 76;
                return (
                  <motion.line
                    key={item.rank}
                    x1="50%"
                    y1="0"
                    x2={`${xPct}%`}
                    y2="100%"
                    stroke="url(#retrieval-beam)"
                    strokeWidth="1.5"
                    strokeOpacity="0.45"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ delay: 0.5 + i * 0.1, duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
                  />
                );
              })}
              <defs>
                <linearGradient id="retrieval-beam" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="rgba(34, 211, 238, 0.7)" />
                  <stop offset="100%" stopColor="rgba(167, 139, 250, 0.45)" />
                </linearGradient>
              </defs>
            </svg>
          </div>

          <div className="relative z-10 flex flex-col items-center gap-6 sm:gap-8">
            <ClipQueryNode />

            <div className="grid w-full grid-cols-2 justify-items-center gap-3 sm:flex sm:flex-wrap sm:justify-center sm:gap-4">
              {top.slice(0, visibleCount).map((item, i) => {
                const isRank1 = item.rank === 1;
                const showMatch = isRank1 && m?.r1Correct;
                const barPct = Math.min(100, (item.score / maxCosine) * 100);
                const scale = isRank1 ? 'sm:scale-105' : '';
                return (
                  <motion.div
                    key={`${item.rank}-${item.image}`}
                    className={`relative w-full max-w-[160px] sm:max-w-[176px] ${scale}`}
                    initial={{ opacity: 0, y: 36, rotateX: -8 }}
                    animate={{ opacity: 1, y: 0, rotateX: 0 }}
                    transition={{
                      delay: 0.35 + i * 0.16,
                      type: 'spring',
                      stiffness: 220,
                      damping: 26,
                    }}
                  >
                    <GlassCard
                      glow={showMatch}
                      className={`overflow-hidden border-white/10 p-2.5 ${
                        isRank1 ? 'ring-2 ring-cyan-400/25' : ''
                      }`}
                    >
                      <div className="relative overflow-hidden rounded-xl border border-white/10">
                        <SafeImage
                          src={item.image}
                          alt={`Retrieval rank ${item.rank}`}
                          className={`aspect-square w-full object-cover ${isRank1 ? 'sm:min-h-[168px]' : ''}`}
                        />
                        <div className="absolute left-2 top-2 flex flex-col gap-1">
                          <span className="rounded-lg border border-white/10 bg-black/75 px-2 py-0.5 text-[10px] font-bold text-white backdrop-blur-md">
                            #{item.rank}
                          </span>
                        </div>
                        {showMatch ? (
                          <motion.div
                            className="absolute right-2 top-2 rounded-full border border-emerald-400/60 bg-emerald-500/90 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide text-white shadow-[0_0_20px_rgba(52,211,153,0.55)]"
                            initial={{ scale: 0.8, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            transition={{ delay: 0.85 + i * 0.05, type: 'spring', stiffness: 400, damping: 18 }}
                          >
                            Match!
                          </motion.div>
                        ) : null}
                      </div>
                      <div className="mt-2 space-y-2">
                        <div className="flex justify-between gap-2 text-[10px] text-slate-400">
                          <span className="font-medium text-cyan-200/90">Cosine</span>
                          <span className="font-mono text-slate-200">{(item.score * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-1.5 overflow-hidden rounded-full bg-slate-800/90">
                          <motion.div
                            className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-violet-500"
                            style={{ boxShadow: '0 0 12px rgba(34, 211, 238, 0.35)' }}
                            initial={{ width: 0 }}
                            animate={{ width: `${barPct}%` }}
                            transition={{ delay: 0.5 + i * 0.12, duration: 0.75, ease: [0.22, 1, 0.36, 1] }}
                          />
                        </div>
                        <div className="flex justify-between gap-2 text-[10px] text-slate-400">
                          <span className="font-medium text-violet-200/90">CSLS</span>
                          <span className="font-mono text-slate-200">{item.csls.toFixed(3)}</span>
                        </div>
                      </div>
                    </GlassCard>
                    <div className="pointer-events-none absolute -inset-px rounded-2xl sm:hidden">
                      <svg className="h-full w-full" aria-hidden>
                        <line
                          x1="50%"
                          y1="-24"
                          x2="50%"
                          y2="0"
                          stroke="rgba(34,211,238,0.25)"
                          strokeWidth="1"
                        />
                      </svg>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>
        </div>

        {m && !m.r1Correct ? (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.1, duration: 0.5 }}
            className="rounded-xl border border-amber-500/35 bg-amber-500/10 px-4 py-3 text-center"
          >
            <p className="text-sm font-semibold text-amber-100">
              Rank-1 is not the exact image — ground truth appears at rank{' '}
              <span className="font-mono text-white">{m.rank}</span>.
            </p>
          </motion.div>
        ) : null}

        <motion.p
          className="max-w-3xl px-2 text-center text-sm leading-relaxed text-slate-400"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.05, duration: 0.5 }}
        >
          Even when the exact image isn&apos;t retrieved at rank 1, the top-K candidates often share semantic content
          with the target.
        </motion.p>
      </div>
    </div>
  );
}
