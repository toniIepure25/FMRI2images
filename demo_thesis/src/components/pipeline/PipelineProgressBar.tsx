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
    <nav aria-label="Pipeline progress" className="py-3">
      <ol className="mx-auto flex max-w-2xl list-none items-center justify-center gap-0">
        {steps.map((step, i) => {
          const isComplete = currentPhase > step.phase;
          const isActive = currentPhase === step.phase;

          return (
            <li
              key={step.phase}
              aria-current={isActive ? 'step' : undefined}
              className="flex min-w-0 flex-1 items-center"
            >
              <div className="flex shrink-0 flex-col items-center gap-2">
                <div
                  className={`flex h-9 w-9 items-center justify-center rounded-full font-mono text-xs font-semibold transition-all duration-300 ${
                    isActive
                      ? 'bg-accent text-white shadow-[0_0_0_4px_rgba(77,124,255,0.15)]'
                      : isComplete
                        ? 'bg-accent/15 text-accent ring-1 ring-accent/25'
                        : 'bg-surface-raised text-text-muted ring-1 ring-border-subtle'
                  }`}
                >
                  {isComplete ? (
                    <motion.span
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: 'spring', stiffness: 400, damping: 20 }}
                    >
                      &#10003;
                    </motion.span>
                  ) : (
                    <span>{String(step.phase).padStart(2, '0')}</span>
                  )}
                </div>
                <p
                  className={`text-center text-xs font-medium ${
                    isActive
                      ? 'text-text-primary'
                      : isComplete
                        ? 'text-text-secondary'
                        : 'text-text-muted'
                  }`}
                >
                  {step.label}
                </p>
              </div>

              {i < steps.length - 1 && (
                <div className="mx-3 mb-8 h-px flex-1">
                  <div
                    className={`h-full rounded-full transition-colors duration-500 ${
                      currentPhase > step.phase
                        ? 'bg-accent/35'
                        : 'bg-border-subtle'
                    }`}
                  />
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
