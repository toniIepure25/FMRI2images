import { motion } from 'framer-motion';
import type { UncertaintyData } from '@/types';
import { getConfidenceColor, getConfidenceLabel } from '@/lib/data';
import { SemiCircleUncertaintyGauge } from '@/components/uncertainty/SemiCircleUncertaintyGauge';
import { DuaCfgVisualPanel, UncertaintyMappingGrid } from '@/components/uncertainty/DuaMappingPanels';
import type { DemoCase } from '@/types';

export interface UncertaintyTabProps {
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

export function UncertaintyTab({ case_ }: UncertaintyTabProps) {
  const { uncertainty: u, duaCfg } = case_;
  const confColor = getConfidenceColor(u.confidenceLevel);
  const confLabel = getConfidenceLabel(u.confidenceLevel);
  const badge = confidenceShort(u.confidenceLevel);

  const deltaNorm = u.delta <= 1 ? u.delta : clamp01(u.delta / (u.delta + 1));
  const kappa01 = clamp01(u.kappaNorm);
  const kappaDisplay = u.kappa >= 10 ? u.kappa.toFixed(1) : u.kappa.toFixed(2);
  const deltaDisplay = u.delta.toFixed(2);

  return (
    <div className="mx-auto max-w-6xl space-y-8 pb-4">
      <motion.div
        className="glass-panel p-5"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45 }}
      >
        <h2 className="text-sm font-semibold text-white">Decoder uncertainty</h2>
        <p className="mt-2 text-xs leading-relaxed text-slate-400">
          κ summarizes how concentrated the predicted vMF direction is; δ summarizes cross-ROI tension. Together they steer
          diffusion so committee-scale ambiguity does not collapse into an arbitrarily sharp image.
        </p>
      </motion.div>

      <div className="grid items-stretch gap-6 lg:grid-cols-2">
        <SemiCircleUncertaintyGauge
          fraction={kappa01}
          centerValue={kappaDisplay}
          percentLabel={`${(kappa01 * 100).toFixed(0)}%`}
          title="κ CONCENTRATION"
          description="How concentrated / confident is the predicted visual direction on the CLIP hypersphere?"
          accent="cyan"
          animationDelay={0.02}
        />
        <SemiCircleUncertaintyGauge
          fraction={deltaNorm}
          centerValue={deltaDisplay}
          percentLabel={`${(deltaNorm * 100).toFixed(0)}%`}
          title="δ DISAGREEMENT"
          description="How much do ROI-wise expert directions diverge before spherical fusion?"
          accent="amber"
          animationDelay={0.08}
        />
      </div>

      <motion.div
        className="glass-panel p-5"
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1, duration: 0.5 }}
      >
        <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">DUA-CFG parameters</p>
        <p className="mt-1 text-xs text-slate-500">
          Uncertainty-aware classifier-free guidance: each knob is read from κ, δ, and safety flags for this trial.
        </p>
        <div className="mt-5">
          <DuaCfgVisualPanel dua={duaCfg} />
        </div>
      </motion.div>

      <motion.div
        className="glass-panel w-full border-white/15 bg-gradient-to-br from-slate-950/90 via-[#0d1529]/95 to-slate-950/90 p-6 backdrop-blur-xl"
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.15, type: 'spring', stiffness: 120, damping: 20 }}
      >
        <p className="text-center text-[10px] font-bold uppercase tracking-[0.25em] text-slate-400">Confidence level</p>
        <div className="mt-4 flex flex-col items-center gap-3 sm:flex-row sm:justify-between">
          <p className="text-center text-sm text-slate-300 sm:text-left">{confLabel}</p>
          <motion.div
            className="rounded-2xl border-2 px-8 py-3 text-center text-lg font-bold tracking-[0.15em]"
            style={{
              borderColor: `${confColor}aa`,
              background: `linear-gradient(145deg, ${confColor}33, rgba(15,23,42,0.75))`,
              color: confColor,
              boxShadow: `0 0 28px ${confColor}40`,
            }}
            animate={{
              scale: [1, 1.045, 1],
              boxShadow: [
                `0 0 28px ${confColor}40`,
                `0 0 42px ${confColor}66`,
                `0 0 28px ${confColor}40`,
              ],
            }}
            transition={{ duration: 2.4, repeat: Infinity, ease: 'easeInOut' }}
          >
            {badge}
          </motion.div>
        </div>
      </motion.div>

      <motion.div
        className="glass-panel p-5"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.08, duration: 0.5 }}
      >
        <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-200/80">κ / δ → DUA-CFG mapping</p>
        <p className="mt-1 text-xs text-slate-500">
          Each arrow summarizes how decoder statistics set the operating point for this reconstruction (illustrative
          functional form).
        </p>
        <div className="mt-6">
          <UncertaintyMappingGrid case_={case_} />
        </div>
      </motion.div>

      <motion.div
        className="glass-panel border-emerald-500/15 bg-emerald-500/[0.04] p-5"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2, duration: 0.5 }}
      >
        <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-emerald-400/90">Interpretation</p>
        <ul className="mt-4 space-y-3 text-sm leading-relaxed text-slate-300">
          <li className="flex gap-3">
            <span className="mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.7)]" />
            <span>
              <span className="font-bold text-emerald-200">High κ and low δ</span> — convergent brain evidence and a
              tight directional prediction: reconstructions tend to be stable and well calibrated.
            </span>
          </li>
          <li className="flex gap-3">
            <span className="mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full bg-amber-400 shadow-[0_0_10px_rgba(251,191,36,0.65)]" />
            <span>
              <span className="font-bold text-amber-200">Low κ or high δ</span> — ambiguous cortical evidence or conflicting
              ROI experts: expect broader CLIP neighborhoods and higher reconstruction diversity.
            </span>
          </li>
          <li className="flex gap-3">
            <span className="mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full bg-red-400 shadow-[0_0_10px_rgba(248,113,113,0.6)]" />
            <span>
              <span className="font-bold text-red-200">Elevated δ</span> — prefer inspecting retrieval shortlists and the
              ensemble grid; multiple semantic modes may fit the data.
            </span>
          </li>
        </ul>
      </motion.div>
    </div>
  );
}
