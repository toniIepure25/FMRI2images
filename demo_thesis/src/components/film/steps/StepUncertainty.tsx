import { motion } from 'framer-motion';
import type { DemoCase, UncertaintyData } from '@/types';
import { GlassCard } from '@/components/GlassCard';
import { getConfidenceColor, getConfidenceLabel } from '@/lib/data';
import { SemiCircleUncertaintyGauge } from '@/components/uncertainty/SemiCircleUncertaintyGauge';
import { DuaCfgVisualPanel, UncertaintyMappingGrid } from '@/components/uncertainty/DuaMappingPanels';

export interface StepUncertaintyProps {
  case_: DemoCase;
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function confidenceShort(level: UncertaintyData['confidenceLevel']): string {
  switch (level) {
    case 'high':
      return 'HIGH';
    case 'medium':
      return 'MEDIUM';
    case 'low':
      return 'LOW';
    case 'abstain':
      return 'ABSTAIN';
    default:
      return '—';
  }
}

export function StepUncertainty({ case_ }: StepUncertaintyProps) {
  const u = case_.uncertainty;
  const dua = case_.duaCfg;
  const levelColor = getConfidenceColor(u.confidenceLevel);
  const confLabel = getConfidenceLabel(u.confidenceLevel);
  const badge = confidenceShort(u.confidenceLevel);

  const kappa01 = clamp01(u.kappaNorm);
  const deltaNorm = u.delta <= 1 ? u.delta : clamp01(u.delta / (u.delta + 1));
  const kappaDisplay = u.kappa >= 10 ? u.kappa.toFixed(1) : u.kappa.toFixed(2);
  const deltaDisplay = u.delta.toFixed(2);

  const gallery = (case_.ensembleImages ?? []).slice(0, 8);

  return (
    <div className="flex h-full min-h-0 flex-col gap-5 px-2 py-2 sm:gap-7 sm:px-4 md:gap-8">
      <motion.h2
        className="mx-auto max-w-4xl text-center font-sans text-xl font-semibold leading-snug tracking-tight text-white sm:text-2xl md:text-[1.65rem]"
        style={{
          textShadow:
            '0 0 22px rgba(244, 114, 182, 0.28), 0 0 44px rgba(167, 139, 250, 0.18), 0 0 72px rgba(34, 211, 238, 0.1)',
        }}
        initial={{ opacity: 0, y: -14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      >
        When uncertain, the model produces multiple plausible reconstructions instead of pretending certainty
      </motion.h2>

      <div className="mx-auto grid w-full max-w-6xl gap-5 md:grid-cols-2 md:gap-8">
        <motion.div
          initial={{ opacity: 0, x: -28 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.65, delay: 0.05 }}
        >
          <GlassCard
            glow
            className="h-full border-cyan-500/25 bg-gradient-to-b from-slate-950/95 to-slate-900/85 p-4 sm:p-6"
          >
            <SemiCircleUncertaintyGauge
              fraction={kappa01}
              centerValue={kappaDisplay}
              percentLabel={`${(kappa01 * 100).toFixed(0)}%`}
              title="κ CONCENTRATION"
              description="How tightly the brain evidence points in one direction on the CLIP hypersphere."
              accent="cyan"
              animationDelay={0.02}
            />
            <p className="mt-2 text-center font-mono text-[10px] text-slate-500">κ̂ (norm) = {u.kappaNorm.toFixed(3)}</p>
          </GlassCard>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 28 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.65, delay: 0.08 }}
        >
          <GlassCard
            glow
            className="h-full border-amber-500/25 bg-gradient-to-b from-slate-950/95 to-amber-950/20 p-4 sm:p-6"
          >
            <SemiCircleUncertaintyGauge
              fraction={deltaNorm}
              centerValue={deltaDisplay}
              percentLabel={`${(deltaNorm * 100).toFixed(0)}%`}
              title="δ DISAGREEMENT"
              description="How much cortical ROI experts conflict before directional consensus fusion."
              accent="amber"
              animationDelay={0.08}
            />
          </GlassCard>
        </motion.div>
      </div>

      <motion.div
        className="mx-auto w-full max-w-6xl"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.12, duration: 0.55 }}
      >
        <GlassCard className="border-white/12 bg-slate-950/75 p-4 backdrop-blur-md sm:p-6">
          <p className="text-center text-[10px] font-bold uppercase tracking-[0.22em] text-slate-400">DUA-CFG parameters</p>
          <p className="mx-auto mt-1 max-w-2xl text-center text-xs text-slate-500">
            Generation controls derived from κ, δ — cinematic readout of the same pipeline as the explorer.
          </p>
          <div className="mt-5">
            <DuaCfgVisualPanel dua={dua} />
          </div>
        </GlassCard>
      </motion.div>

      <motion.div
        className="mx-auto w-full max-w-4xl"
        initial={{ opacity: 0, scale: 0.96, y: 12 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ delay: 0.22, type: 'spring', stiffness: 200, damping: 24 }}
      >
        <div
          className="flex flex-col items-center justify-center rounded-2xl border-2 px-4 py-6 text-center shadow-2xl backdrop-blur-xl sm:px-10 sm:py-7"
          style={{
            borderColor: `${levelColor}99`,
            background: `linear-gradient(155deg, ${levelColor}22, rgba(10,14,26,0.92))`,
          }}
        >
          <p className="text-[10px] font-bold uppercase tracking-[0.28em] text-slate-300/90">Confidence level</p>
          <p className="mt-2 text-sm text-slate-300 sm:text-base">{confLabel}</p>
          <motion.div
            className="mt-4 rounded-2xl border-2 px-10 py-3 text-xl font-bold tracking-[0.2em] sm:text-2xl"
            style={{
              borderColor: `${levelColor}aa`,
              background: `linear-gradient(145deg, ${levelColor}38, rgba(15,23,42,0.65))`,
              color: levelColor,
            }}
            animate={{
              scale: [1, 1.06, 1],
              boxShadow: [
                `0 0 32px ${levelColor}44`,
                `0 0 52px ${levelColor}77`,
                `0 0 32px ${levelColor}44`,
              ],
            }}
            transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}
          >
            {badge}
          </motion.div>
        </div>
      </motion.div>

      <motion.div
        className="mx-auto w-full max-w-6xl"
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.28, duration: 0.55 }}
      >
        <GlassCard className="border-violet-500/25 bg-slate-950/70 p-4 sm:p-6">
          <p className="text-center text-[10px] font-bold uppercase tracking-[0.2em] text-violet-200/90">
            κ / δ → DUA-CFG mapping
          </p>
          <p className="mx-auto mt-2 max-w-2xl text-center text-xs text-slate-500">
            Uncertainty drives guidance strength, ensemble breadth, and diffusion depth for this trial.
          </p>
          <div className="mt-6">
            <UncertaintyMappingGrid case_={case_} />
          </div>
        </GlassCard>
      </motion.div>

      <motion.div
        className="mx-auto w-full max-w-6xl"
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.36, duration: 0.55 }}
      >
        <GlassCard className="border-white/10 bg-slate-950/65 p-5">
          <p className="text-center text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-200/85">
            Multiple reconstruction samples under DUA-CFG
          </p>
          <div className="mt-4 flex flex-wrap justify-center gap-3">
            {gallery.length === 0 ? (
              <p className="text-sm text-slate-500">No ensemble images for this case.</p>
            ) : (
              gallery.map((src, i) => (
                <motion.div
                  key={`${src}-${i}`}
                  className="w-28 overflow-hidden rounded-xl border border-white/15 bg-slate-900/80 shadow-[0_0_20px_rgba(0,0,0,0.35)] sm:w-32"
                  initial={{ opacity: 0, y: 20, rotateZ: -2 }}
                  animate={{ opacity: 1, y: 0, rotateZ: 0 }}
                  transition={{ delay: 0.4 + i * 0.06, type: 'spring', stiffness: 280, damping: 24 }}
                >
                  <img src={src} alt={`Ensemble sample ${i + 1}`} className="aspect-square object-cover" />
                </motion.div>
              ))
            )}
          </div>
        </GlassCard>
      </motion.div>

      <motion.ul
        className="mx-auto max-w-3xl space-y-3 pb-2 text-sm leading-relaxed text-slate-300"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.42, duration: 0.5 }}
      >
        {[
          {
            color: 'bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.7)]',
            body: (
              <>
                <span className="font-bold text-emerald-200">High κ and low δ</span> — convergent brain evidence and a tight
                directional prediction: reconstructions tend to be stable and well calibrated.
              </>
            ),
          },
          {
            color: 'bg-amber-400 shadow-[0_0_10px_rgba(251,191,36,0.65)]',
            body: (
              <>
                <span className="font-bold text-amber-200">Low κ or high δ</span> — ambiguous cortical evidence or conflicting
                ROI experts: expect broader CLIP neighborhoods and higher reconstruction diversity.
              </>
            ),
          },
          {
            color: 'bg-red-400 shadow-[0_0_10px_rgba(248,113,113,0.6)]',
            body: (
              <>
                <span className="font-bold text-red-200">Elevated δ</span> — prefer inspecting retrieval shortlists and the
                ensemble grid; multiple semantic modes may fit the data.
              </>
            ),
          },
        ].map((item, i) => (
          <motion.li
            key={i}
            className="flex gap-3 rounded-xl border border-white/8 bg-white/[0.04] px-4 py-3 backdrop-blur-sm"
            initial={{ opacity: 0, x: -14 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.45 + i * 0.08, duration: 0.45 }}
          >
            <span className={`mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full ${item.color}`} />
            <span>{item.body}</span>
          </motion.li>
        ))}
      </motion.ul>
    </div>
  );
}
