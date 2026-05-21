import { motion } from 'framer-motion';

export interface DecodingStageRailProps {
  currentPhase: 1 | 2 | 3;
  /** When true, omit the surrounding panel chrome so the rail can be embedded
   *  inside another panel (e.g. the page header). */
  embedded?: boolean;
}

const stages = [
  { phase: 1 as const, number: '01', label: 'Stimulus selection', detail: 'NSD trial' },
  { phase: 2 as const, number: '02', label: 'Neural decoding', detail: 'fMRI → CLIP' },
  { phase: 3 as const, number: '03', label: 'Evidence audit', detail: 'Result review' },
];

export function DecodingStageRail({ currentPhase, embedded = false }: DecodingStageRailProps) {
  return (
    <nav
      aria-label="Decoding stages"
      className={embedded ? '' : 'premium-panel-flat px-5 py-2.5'}
    >
      <ol className="flex items-center justify-between gap-2">
        {stages.map((stage, idx) => {
          const isActive = currentPhase === stage.phase;
          const isComplete = currentPhase > stage.phase;
          const isLast = idx === stages.length - 1;

          return (
            <li
              key={stage.phase}
              className={`relative flex items-center ${isLast ? 'flex-none' : 'flex-1'}`}
            >
              <div className="flex items-center gap-3">
                <div
                  className={`relative flex h-6 w-6 items-center justify-center rounded-full text-[10px] font-bold tabular-nums transition-colors duration-300 ${
                    isActive
                      ? 'bg-accent text-text-inverse'
                      : isComplete
                      ? 'border border-accent/30 bg-accent/[0.08] text-accent'
                      : 'border border-white/[0.07] bg-white/[0.015] text-text-muted'
                  }`}
                  aria-current={isActive ? 'step' : undefined}
                >
                  {isActive ? (
                    <span className="pointer-events-none absolute inset-0 -m-1 rounded-full ring-2 ring-accent/20" aria-hidden />
                  ) : null}
                  {isComplete ? (
                    <svg
                      className="h-3.5 w-3.5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={3}
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    stage.number
                  )}
                </div>
                <div className="flex flex-col leading-tight">
                  <span
                    className={`text-[12px] font-semibold tracking-tight ${
                      isActive
                        ? 'text-text-primary'
                        : isComplete
                        ? 'text-text-secondary'
                        : 'text-text-muted'
                    }`}
                  >
                    {stage.label}
                  </span>
                  <span
                    className={`text-[10px] ${
                      isActive ? 'text-text-secondary' : 'text-text-muted/70'
                    }`}
                  >
                    {stage.detail}
                  </span>
                </div>
              </div>
              {!isLast && (
                <div className="relative mx-4 h-px flex-1 overflow-hidden bg-white/[0.05]">
                  {isComplete ? (
                    <motion.div
                      className="absolute inset-y-0 left-0 bg-accent/55"
                      initial={{ width: 0 }}
                      animate={{ width: '100%' }}
                      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                    />
                  ) : isActive ? (
                    <motion.div
                      className="absolute inset-y-0 left-0 bg-accent/45"
                      initial={{ width: 0 }}
                      animate={{ width: '42%' }}
                      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                    />
                  ) : null}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
