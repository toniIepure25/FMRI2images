import { motion } from 'framer-motion';
import type { DemoCase } from '@/types';

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function DuaParam({
  label,
  value,
  detail,
  accent,
  bar,
  delay = 0,
}: {
  label: string;
  value: string;
  detail: string;
  accent: string;
  bar?: { fraction: number; colorFrom: string; colorTo: string };
  delay?: number;
}) {
  return (
    <motion.div
      className="flex flex-col rounded-2xl border border-white/[0.06] bg-slate-950/60 p-5"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
    >
      <p className={`text-[10px] font-bold uppercase tracking-[0.2em] ${accent}`}>{label}</p>
      <p className="mt-2 font-mono text-2xl font-bold tabular-nums text-white">{value}</p>
      <p className="mt-1 text-[11px] leading-relaxed text-slate-500">{detail}</p>
      {bar && (
        <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-slate-800/80">
          <motion.div
            className="h-full rounded-full"
            style={{ background: `linear-gradient(90deg, ${bar.colorFrom}, ${bar.colorTo})` }}
            initial={{ width: '0%' }}
            animate={{ width: `${clamp01(bar.fraction) * 100}%` }}
            transition={{ delay: delay + 0.2, duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
          />
        </div>
      )}
    </motion.div>
  );
}

export function DuaCfgVisualPanel({ dua }: { dua: DemoCase['duaCfg'] }) {
  const gFrac = clamp01((dua.guidanceScale - 1.5) / (12 - 1.5));
  const sFrac = clamp01(dua.diffusionSteps / 48);

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <DuaParam
        label="Guidance scale"
        value={dua.guidanceScale.toFixed(1)}
        detail="CFG strength mapped from κ concentration"
        accent="text-cyan-400/90"
        bar={{ fraction: gFrac, colorFrom: '#22d3ee', colorTo: '#06b6d4' }}
        delay={0}
      />
      <DuaParam
        label="Diffusion steps"
        value={String(dua.diffusionSteps)}
        detail="Sampling budget from joint (κ, δ)"
        accent="text-amber-400/90"
        bar={{ fraction: sFrac, colorFrom: '#fbbf24', colorTo: '#f59e0b' }}
        delay={0.06}
      />
      <DuaParam
        label="Ensemble K"
        value={String(dua.ensembleK)}
        detail="Multi-sample breadth from δ disagreement"
        accent="text-violet-400/90"
        delay={0.12}
      />
      <motion.div
        className="flex flex-col rounded-2xl border border-white/[0.06] bg-slate-950/60 p-5"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.18, duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      >
        <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">Abstain</p>
        <div className="mt-3 flex items-center gap-3">
          <div
            className={`h-2.5 w-2.5 rounded-full ${
              dua.abstain
                ? 'bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.5)]'
                : 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.5)]'
            }`}
          />
          <span className={`text-sm font-semibold ${dua.abstain ? 'text-red-300' : 'text-emerald-300'}`}>
            {dua.abstain ? 'Abstaining' : 'Active'}
          </span>
        </div>
        <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
          {dua.abstain ? 'Insufficient confidence for aggressive decoding' : 'Full decoding trajectory enabled'}
        </p>
      </motion.div>
    </div>
  );
}

export function UncertaintyMappingGrid({ case_ }: { case_: DemoCase }) {
  const { duaCfg: d } = case_;
  const rows = [
    { from: 'κ', accent: 'text-cyan-300', to: 'Guidance scale', value: d.guidanceScale.toFixed(1), hint: 'w ≈ fκ(κ̂) · CFG range' },
    { from: 'δ', accent: 'text-amber-300', to: 'Ensemble K', value: String(d.ensembleK), hint: 'K ∝ gδ(δ) · Kmax' },
    { from: '(κ, δ)', accent: 'text-emerald-300', to: 'Diffusion steps', value: String(d.diffusionSteps), hint: 'T = h(κ̂, δ) · Tref' },
  ];

  return (
    <div className="grid gap-3 md:grid-cols-3">
      {rows.map((row, i) => (
        <motion.div
          key={row.to}
          className="flex flex-col rounded-2xl border border-white/[0.06] bg-slate-950/60 p-5"
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 + i * 0.06, duration: 0.4 }}
        >
          <div className="flex items-center gap-2">
            <span className={`font-mono text-lg font-bold ${row.accent}`}>{row.from}</span>
            <span className="text-slate-600" aria-hidden>→</span>
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">{row.to}</span>
          </div>
          <p className="mt-2 font-mono text-2xl font-bold tabular-nums text-white">{row.value}</p>
          <p className="mt-1 font-mono text-[10px] text-slate-500">{row.hint}</p>
        </motion.div>
      ))}
    </div>
  );
}
