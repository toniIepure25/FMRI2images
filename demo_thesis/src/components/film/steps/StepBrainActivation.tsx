import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import type { DemoCase } from '@/types';
import { BrainViewer3D } from '@/components/BrainViewer3D';
import { GlassCard } from '@/components/GlassCard';

export interface StepBrainActivationProps {
  case_: DemoCase;
}

export function StepBrainActivation({ case_ }: StepBrainActivationProps) {
  const ordered = useMemo(() => {
    const rois = [...(case_.roiScores ?? [])];
    rois.sort((a, b) => (b.activation + b.contribution) - (a.activation + a.contribution));
    return rois.slice(0, 12).map((r) => r.name);
  }, [case_.roiScores]);

  const [idx, setIdx] = useState(0);

  useEffect(() => {
    if (ordered.length === 0) return undefined;
    const t = window.setInterval(() => {
      setIdx((i) => (i + 1) % ordered.length);
    }, 780);
    return () => window.clearInterval(t);
  }, [ordered.length]);

  const highlight = ordered.length ? ordered[idx] : null;

  return (
    <div className="flex h-full min-h-0 flex-col gap-6 px-2 lg:flex-row lg:gap-8">
      <div className="flex min-h-0 flex-1 flex-col">
        <motion.p
          className="mb-2 text-center text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400/85 lg:text-left"
          initial={{ opacity: 0, y: -6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45 }}
        >
          Analyzing visual cortex response…
        </motion.p>
        <motion.h2
          className="mb-4 text-center text-2xl font-semibold text-white lg:text-left lg:text-3xl"
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          Visual cortex responds
        </motion.h2>
        <motion.div
          className="min-h-[420px] flex-1"
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.12, duration: 0.55 }}
        >
          <BrainViewer3D
            roiScores={case_.roiScores}
            delta={case_.uncertainty.delta}
            selectedRoi={highlight}
            autoRotate
            className="h-full min-h-[420px]"
          />
        </motion.div>
      </div>

      <motion.div
        className="lg:w-80 lg:shrink-0"
        initial={{ opacity: 0, x: 22 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.2, duration: 0.5 }}
      >
        <GlassCard className="p-5">
          <p className="text-xs font-semibold uppercase tracking-widest text-cyan-300/80">Regional flow</p>
          <p className="mt-3 text-sm leading-relaxed text-slate-300">
            Early visual areas (V1–V3) and higher-level regions light up as the decoder&apos;s ROI tokens track
            stimulus-driven activation. The 3D atlas highlights where trial evidence concentrates.
          </p>
          <div className="mt-4 max-h-40 overflow-y-auto rounded-lg border border-white/5 bg-slate-950/40">
            {ordered.length === 0 ? (
              <p className="p-3 text-xs text-slate-500">No ROI scores for this case.</p>
            ) : (
              <ul className="divide-y divide-white/5">
                {ordered.map((name, i) => (
                  <li
                    key={name}
                    className={`flex items-center gap-2 px-3 py-2 text-xs transition-colors ${
                      name === highlight ? 'bg-cyan-500/15 text-cyan-100' : 'text-slate-400'
                    }`}
                  >
                    <span className="w-5 font-mono text-slate-500">{i + 1}</span>
                    <span className="font-medium">{name}</span>
                    {name === highlight ? (
                      <motion.span
                        layoutId="roi-pulse"
                        className="ml-auto h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_10px_#22d3ee]"
                      />
                    ) : null}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </GlassCard>
      </motion.div>
    </div>
  );
}
