import { motion } from 'framer-motion';

export interface PipelineProgressBarProps {
  currentPhase: 1 | 2 | 3;
}

const steps = [
  { phase: 1 as const, label: 'Select Trial', num: '01' },
  { phase: 2 as const, label: 'Encode & Retrieve', num: '02' },
  { phase: 3 as const, label: 'Reconstruct & Reveal', num: '03' },
];

export function PipelineProgressBar({ currentPhase }: PipelineProgressBarProps) {
  return (
    <nav aria-label="Pipeline progress" className="rounded-2xl border border-white/[0.06] bg-gradient-to-r from-slate-900/80 via-brain-navy/40 to-slate-900/80 px-4 py-4 backdrop-blur-xl sm:px-6 sm:py-5">
      <ol className="mx-auto flex max-w-3xl list-none items-start">
        {steps.map((step, i) => {
          const isComplete = currentPhase > step.phase;
          const isActive = currentPhase === step.phase;
          const lineDone = currentPhase > step.phase;
          const isPending = !isComplete && !isActive;

          return (
            <li
              key={step.phase}
              aria-current={isActive ? 'step' : undefined}
              className="flex min-w-0 flex-1 items-start"
            >
              <div className="flex w-[100px] shrink-0 flex-col items-center sm:w-[140px]">
                <motion.div
                  className={`relative z-10 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border-2 text-sm font-bold sm:h-12 sm:w-12 ${
                    isActive
                      ? 'border-brain-accent bg-brain-accent/12 text-brain-accent'
                      : isComplete
                        ? 'border-emerald-500/60 bg-emerald-500/10 text-emerald-400'
                        : 'border-slate-700 bg-slate-900/80 text-slate-600'
                  }`}
                  animate={isActive ? {
                    boxShadow: [
                      '0 0 14px rgba(0,212,255,0.25)',
                      '0 0 28px rgba(0,212,255,0.45)',
                      '0 0 14px rgba(0,212,255,0.25)',
                    ],
                  } : { boxShadow: '0 0 0 transparent' }}
                  transition={isActive ? { duration: 2, repeat: Infinity } : { duration: 0.2 }}
                >
                  {isComplete ? (
                    <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }}
                      className="text-lg font-bold text-emerald-400">✓</motion.span>
                  ) : isPending ? (
                    <span className="max-w-[2.75rem] text-center text-[7px] font-semibold uppercase leading-tight tracking-wide text-slate-500 sm:text-[8px]">
                      Pending
                    </span>
                  ) : (
                    <span className="font-mono text-xs">{step.num}</span>
                  )}
                </motion.div>

                <p className={`mt-2 text-center text-[10px] font-semibold leading-tight tracking-wide sm:text-[11px] ${
                  isActive ? 'text-brain-accent' : isComplete ? 'text-emerald-400/80' : 'text-slate-600'
                }`}>
                  {step.label}
                </p>

                {isActive && (
                  <motion.div className="mt-1.5 h-0.5 w-8 rounded-full bg-brain-accent shadow-[0_0_8px_rgba(0,212,255,0.4)]"
                    layoutId="pip-active" transition={{ type: 'spring', stiffness: 350, damping: 26 }} />
                )}
              </div>

              {i < steps.length - 1 && (
                <div
                  role="presentation"
                  className={`mx-1 mt-[20px] h-[2px] min-h-[2px] flex-1 self-start rounded-full sm:mt-[24px] ${
                    lineDone
                      ? 'bg-gradient-to-r from-emerald-500/85 via-teal-400/75 to-cyan-400/75'
                      : 'bg-gradient-to-r from-slate-800/90 to-slate-700/45'
                  }`}
                />
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
