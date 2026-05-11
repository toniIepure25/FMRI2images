import { motion } from 'framer-motion';

export interface PipelineProgressBarProps {
  currentPhase: 1 | 2 | 3;
}

const steps = [
  { phase: 1 as const, label: 'Select trial' },
  { phase: 2 as const, label: 'Encode & retrieve' },
  { phase: 3 as const, label: 'Reconstruct & compare' },
];

export function PipelineProgressBar({ currentPhase }: PipelineProgressBarProps) {
  return (
    <nav
      aria-label="Pipeline progress"
      className="py-2"
    >
      <ol className="mx-auto flex max-w-xl list-none items-center justify-center">
        {steps.map((step, i) => {
          const isComplete = currentPhase > step.phase;
          const isActive = currentPhase === step.phase;

          return (
            <li
              key={step.phase}
              aria-current={isActive ? 'step' : undefined}
              className="flex min-w-0 flex-1 items-center"
            >
              <div className="flex shrink-0 flex-col items-center gap-1">
                {/* Step indicator */}
                <div
                  className={`flex h-6 w-6 items-center justify-center rounded-lg font-mono text-[10px] font-semibold transition-all duration-300 ${
                    isActive
                      ? 'bg-white/[0.1] text-white ring-1 ring-white/[0.15]'
                      : isComplete
                        ? 'bg-emerald-500/10 text-emerald-400'
                        : 'bg-white/[0.03] text-slate-600'
                  }`}
                >
                  {isComplete ? (
                    <motion.span
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: 'spring', stiffness: 400, damping: 20 }}
                      className="text-[11px] text-emerald-400"
                    >
                      ✓
                    </motion.span>
                  ) : (
                    <span>{String(step.phase).padStart(2, '0')}</span>
                  )}
                </div>

                {/* Label */}
                <p
                  className={`text-center text-[10px] font-medium leading-tight ${
                    isActive
                      ? 'text-white'
                      : isComplete
                        ? 'text-emerald-400/70'
                        : 'text-slate-600'
                  }`}
                >
                  {step.label}
                </p>
              </div>

              {/* Connector line */}
              {i < steps.length - 1 && (
                <div
                  role="presentation"
                  className={`mx-3 mt-[-14px] h-px flex-1 transition-colors duration-500 ${
                    currentPhase > step.phase
                      ? 'bg-emerald-500/40'
                      : 'bg-white/[0.06]'
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
