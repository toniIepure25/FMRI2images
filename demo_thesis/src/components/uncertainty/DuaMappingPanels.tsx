import { motion } from 'framer-motion';
import type { DemoCase } from '@/types';

const GUIDE_MIN = 1.5;
const GUIDE_MAX = 12;
const STEP_SEGMENTS = 24;
const STEP_REF_MAX = 48;

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function guidanceNorm(v: number): number {
  return clamp01((v - GUIDE_MIN) / (GUIDE_MAX - GUIDE_MIN));
}

export function DuaCfgVisualPanel({ dua }: { dua: DemoCase['duaCfg'] }) {
  const g = guidanceNorm(dua.guidanceScale);
  const filledSteps = Math.min(
    STEP_SEGMENTS,
    Math.max(0, Math.round((dua.diffusionSteps / STEP_REF_MAX) * STEP_SEGMENTS)),
  );
  const ensMax = 8;
  const k = Math.min(ensMax, Math.max(0, dua.ensembleK));

  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <div className="rounded-xl border border-white/10 bg-black/25 p-4">
        <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-cyan-200/85">Guidance scale</p>
        <p className="mt-1 font-mono text-xl font-semibold tabular-nums text-white">{dua.guidanceScale.toFixed(2)}</p>
        <p className="mt-0.5 text-[10px] text-slate-500">
          Range [{GUIDE_MIN}, {GUIDE_MAX}] — uncertainty-aware CFG strength
        </p>
        <div className="relative mt-4 h-4 w-full overflow-hidden rounded-full bg-[#0f172a] ring-1 ring-white/10">
          <div
            className="absolute inset-0 opacity-90"
            style={{
              background: 'linear-gradient(90deg, #22c55e 0%, #eab308 45%, #ef4444 100%)',
            }}
          />
          <motion.div
            className="absolute bottom-0 top-0 w-1.5 rounded-full bg-white shadow-[0_0_12px_rgba(255,255,255,0.9)]"
            style={{ left: `calc(${(g * 100).toFixed(2)}% - 3px)` }}
            initial={{ opacity: 0, scaleY: 0.5 }}
            animate={{ opacity: 1, scaleY: 1 }}
            transition={{ duration: 0.5, delay: 0.15 }}
          />
        </div>
        <div className="mt-1 flex justify-between text-[9px] tabular-nums text-slate-500">
          <span>{GUIDE_MIN}</span>
          <span>{GUIDE_MAX}</span>
        </div>
      </div>

      <div className="rounded-xl border border-white/10 bg-black/25 p-4">
        <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-amber-200/85">Diffusion steps</p>
        <p className="mt-1 font-mono text-xl font-semibold tabular-nums text-white">{dua.diffusionSteps}</p>
        <p className="mt-0.5 text-[10px] text-slate-500">Sampling depth budget modulated by joint (κ, δ)</p>
        <div className="mt-4 flex gap-0.5">
          {Array.from({ length: STEP_SEGMENTS }, (_, i) => (
            <motion.div
              key={i}
              className={`h-3 min-w-0 flex-1 rounded-sm ${
                i < filledSteps
                  ? 'bg-gradient-to-t from-cyan-600/90 via-cyan-400 to-cyan-200 shadow-[0_0_8px_rgba(34,211,238,0.35)]'
                  : 'bg-slate-800/95 ring-1 ring-white/5'
              }`}
              initial={{ scaleY: 0.4, opacity: 0.4 }}
              animate={{ scaleY: 1, opacity: 1 }}
              transition={{ delay: 0.04 + i * 0.02, duration: 0.35 }}
            />
          ))}
        </div>
      </div>

      <div className="rounded-xl border border-white/10 bg-black/25 p-4">
        <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-violet-200/85">Ensemble K</p>
        <p className="mt-1 font-mono text-xl font-semibold tabular-nums text-white">{dua.ensembleK}</p>
        <p className="mt-0.5 text-[10px] text-slate-500">Monte-Carlo / multi-sample breadth (max {ensMax})</p>
        <div className="mt-4 flex flex-wrap gap-2">
          {Array.from({ length: ensMax }, (_, i) => (
            <motion.div
              key={i}
              className={`h-9 w-9 rounded-full border-2 ${
                i < k
                  ? 'border-fuchsia-400/80 bg-gradient-to-br from-fuchsia-500/90 to-violet-600/90 shadow-[0_0_14px_rgba(217,70,239,0.45)]'
                  : 'border-slate-600/60 bg-slate-900/80'
              }`}
              initial={{ scale: 0.6, opacity: 0.5 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.08 + i * 0.045, type: 'spring', stiffness: 320, damping: 22 }}
            />
          ))}
        </div>
      </div>

      <div className="rounded-xl border border-white/10 bg-black/25 p-4">
        <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-300">Abstain</p>
        <div className="mt-3 flex items-center gap-3">
          <motion.div
            className={`h-3 w-3 rounded-full ${dua.abstain ? 'bg-red-500 shadow-[0_0_12px_rgba(239,68,68,0.7)]' : 'bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.55)]'}`}
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 2, repeat: Infinity, repeatDelay: 0.5 }}
          />
          <div>
            <p className={`text-sm font-semibold ${dua.abstain ? 'text-red-300' : 'text-emerald-300'}`}>
              {dua.abstain ? 'Abstaining' : 'Active'}
            </p>
            <p className="text-[10px] text-slate-500">
              {dua.abstain ? 'Pipeline defers aggressive commit on this trial' : 'Full decoding trajectory enabled'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export function UncertaintyMappingGrid({ case_ }: { case_: DemoCase }) {
  const { duaCfg: d } = case_;
  const rows = [
    {
      from: 'κ',
      accent: 'text-cyan-300',
      to: 'Guidance scale',
      value: d.guidanceScale.toFixed(2),
      hint: 'w ≈ fκ(κ̂) · CFG range',
    },
    {
      from: 'δ',
      accent: 'text-amber-300',
      to: 'Ensemble K',
      value: String(d.ensembleK),
      hint: 'K ∝ gδ(δ) · Kmax',
    },
    {
      from: '(κ, δ)',
      accent: 'text-emerald-300',
      to: 'Diffusion steps',
      value: String(d.diffusionSteps),
      hint: 'T = h(κ̂, δ) · Tref',
    },
  ];

  return (
    <div className="grid gap-4 md:grid-cols-3">
      {rows.map((row, i) => (
        <motion.div
          key={row.to}
          className="relative flex flex-col items-stretch rounded-xl border border-white/10 bg-gradient-to-b from-slate-950/80 to-black/40 p-4"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.12 + i * 0.08, duration: 0.45 }}
        >
          <div className="flex items-center justify-center gap-2 sm:justify-start">
            <span className={`font-mono text-lg font-bold ${row.accent}`}>{row.from}</span>
            <motion.span
              className="text-lg text-slate-500"
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.35 + i * 0.06 }}
              aria-hidden
            >
              →
            </motion.span>
            <span className="text-center text-[10px] font-semibold uppercase tracking-wider text-slate-400 sm:text-left">
              {row.to}
            </span>
          </div>
          <p className="mt-3 text-center font-mono text-2xl font-bold tabular-nums text-white sm:text-left">{row.value}</p>
          <p className="mt-2 text-center font-mono text-[10px] leading-relaxed text-slate-500 sm:text-left">{row.hint}</p>
        </motion.div>
      ))}
    </div>
  );
}
