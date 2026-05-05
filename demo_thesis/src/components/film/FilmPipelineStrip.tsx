import { motion } from 'framer-motion';

export function filmStepToPipelineStage(step: number): number {
  if (step <= 0) return 0;
  if (step <= 2) return 1;
  if (step === 3) return 2;
  if (step === 4) return 3;
  if (step <= 6) return 4;
  return 5;
}

function IconFmri({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
      <path strokeLinecap="round" d="M4 8h16M4 12h10M4 16h7" />
      <rect x="2" y="5" width="20" height="14" rx="2" />
      <path strokeLinecap="round" d="M17 14h3" />
    </svg>
  );
}

function IconEncoder({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
      <circle cx="8" cy="8" r="3" />
      <circle cx="16" cy="16" r="3" />
      <path strokeLinecap="round" d="M10.5 10.5l5 5M13.5 10.5l-5 5" opacity="0.5" />
    </svg>
  );
}

function IconClip({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
      <path strokeLinecap="round" d="M12 4v16M8 8l8 8M16 8l-8 8" />
      <circle cx="12" cy="12" r="9" opacity="0.35" />
    </svg>
  );
}

function IconSearch({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
      <circle cx="11" cy="11" r="6" />
      <path strokeLinecap="round" d="M20 20l-4-4" />
    </svg>
  );
}

function IconDiffuse({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
      <path
        strokeLinecap="round"
        d="M7 16c2-4 8-4 10 0M6 12c1.5-2 4.5-2 6 0M9 8c.8-1 2.2-1 3 0M12 4v2"
      />
      <path strokeLinecap="round" d="M5 18h14" opacity="0.4" />
    </svg>
  );
}

function IconResult({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
      <rect x="4" y="5" width="16" height="14" rx="2" />
      <path strokeLinecap="round" d="M8 14l3 3 5-6" />
    </svg>
  );
}

const STAGES = [
  { id: 'fmri', label: 'fMRI', Icon: IconFmri },
  { id: 'encoder', label: 'ROI encoder', Icon: IconEncoder },
  { id: 'clip', label: 'CLIP space', Icon: IconClip },
  { id: 'retrieval', label: 'Retrieval', Icon: IconSearch },
  { id: 'diffusion', label: 'Diffusion', Icon: IconDiffuse },
  { id: 'result', label: 'Result', Icon: IconResult },
] as const;

export interface FilmPipelineStripProps {
  currentStep: number;
}

export function FilmPipelineStrip({ currentStep }: FilmPipelineStripProps) {
  const activeStage = filmStepToPipelineStage(currentStep);
  const progressFrac = STAGES.length <= 1 ? 0 : activeStage / (STAGES.length - 1);

  return (
    <div
      className="relative border-b border-white/5 bg-slate-950/80 px-3 py-3 backdrop-blur-md sm:px-5"
      role="navigation"
      aria-label="Decoding pipeline progress"
    >
      <p className="mb-2.5 text-center text-[9px] font-semibold uppercase tracking-[0.22em] text-slate-500">
        Live pipeline
      </p>

      <div className="relative mx-auto max-w-5xl">
        {/* background connector */}
        <div
          className="absolute left-[5%] right-[5%] top-[22px] h-0.5 rounded-full bg-slate-800/90"
          aria-hidden
        />
        <motion.div
          className="absolute left-[5%] top-[22px] h-0.5 rounded-full bg-gradient-to-r from-cyan-500 via-violet-500 to-emerald-400"
          initial={false}
          animate={{
            width: `${progressFrac * 90}%`,
            opacity: [0.85, 1, 0.85],
          }}
          transition={{
            width: { type: 'spring', stiffness: 120, damping: 22 },
            opacity: { duration: 2.2, repeat: Infinity, ease: 'easeInOut' },
          }}
          style={{ marginLeft: '0%', maxWidth: '90%' }}
          aria-hidden
        />
        {/* animated leading dot on the line */}
        <motion.div
          className="pointer-events-none absolute top-[19px] z-10 h-2 w-2 rounded-full bg-cyan-300 shadow-[0_0_12px_3px_rgba(34,211,238,0.55)]"
          style={{ left: `calc(5% + ${progressFrac * 90}% - 4px)` }}
          animate={{ scale: [1, 1.15, 1] }}
          transition={{ duration: 1.2, repeat: Infinity }}
          aria-hidden
        />

        <div className="relative flex justify-between gap-0.5 sm:gap-1">
          {STAGES.map((stage, i) => {
            const done = i < activeStage;
            const active = i === activeStage;
            const Icon = stage.Icon;
            return (
              <div key={stage.id} className="flex flex-1 flex-col items-center gap-1.5">
                <motion.div
                  className={`relative flex h-10 w-10 items-center justify-center rounded-xl border transition-colors sm:h-11 sm:w-11 ${
                    active
                      ? 'border-cyan-400/70 bg-cyan-500/20 text-cyan-100 shadow-[0_0_24px_rgba(34,211,238,0.35)]'
                      : done
                        ? 'border-emerald-500/45 bg-emerald-500/15 text-emerald-200'
                        : 'border-slate-700/80 bg-slate-900/90 text-slate-600'
                  }`}
                  animate={
                    active
                      ? { boxShadow: ['0 0 20px rgba(34,211,238,0.25)', '0 0 28px rgba(34,211,238,0.5)', '0 0 20px rgba(34,211,238,0.25)'] }
                      : {}
                  }
                  transition={{ duration: 2, repeat: active ? Infinity : 0 }}
                >
                  <Icon className="h-5 w-5 sm:h-[22px] sm:w-[22px]" />
                  {active ? (
                    <motion.span
                      layoutId="pipeline-active-ring"
                      className="pointer-events-none absolute inset-0 rounded-xl border-2 border-cyan-400/40"
                      transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                    />
                  ) : null}
                  {done ? (
                    <span className="absolute -right-1 -top-1 flex h-4 w-4 items-center justify-center rounded-full border border-emerald-400/50 bg-emerald-600 text-[9px] font-bold text-white shadow-md">
                      ✓
                    </span>
                  ) : null}
                </motion.div>
                <span
                  className={`max-w-[4.5rem] text-center text-[8px] font-medium leading-tight sm:max-w-none sm:text-[9px] ${
                    active ? 'text-cyan-200' : done ? 'text-slate-400' : 'text-slate-600'
                  }`}
                >
                  {stage.label}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
