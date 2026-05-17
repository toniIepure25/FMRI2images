import { motion } from 'framer-motion';

export interface DecodingStageRailProps {
  currentPhase: 1 | 2 | 3;
}

const stages = [
  { phase: 1 as const, number: '01', label: 'Stimulus', detail: 'NSD trial' },
  { phase: 2 as const, number: '02', label: 'Decode', detail: 'fMRI → CLIP' },
  { phase: 3 as const, number: '03', label: 'Evidence', detail: 'Retrieval review' },
];

export function DecodingStageRail({ currentPhase }: DecodingStageRailProps) {
  const progressWidth = currentPhase === 1 ? '33.333%' : currentPhase === 2 ? '66.666%' : '100%';

  return (
    <nav
      aria-label="Decoding stages"
      className="relative overflow-hidden rounded-[20px] border border-border-subtle bg-[#080b12]/90 shadow-[0_22px_80px_-54px_rgba(0,0,0,0.95)]"
    >
      <div className="absolute inset-x-0 top-0 h-px bg-white/[0.055]" />
      <motion.div
        className="absolute left-0 top-0 h-px bg-gradient-to-r from-text-secondary/70 via-accent/45 to-transparent"
        initial={false}
        animate={{ width: progressWidth }}
        transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      />

      <ol className="relative grid list-none grid-cols-1 sm:grid-cols-3">
        {stages.map((stage) => {
          const isActive = currentPhase === stage.phase;
          const isComplete = currentPhase > stage.phase;
          return (
            <li key={stage.phase} aria-current={isActive ? 'step' : undefined}>
              <div
                className={`relative min-h-[74px] px-5 py-4 transition duration-300 sm:border-r sm:last:border-r-0 ${
                  isActive
                    ? 'border-border-subtle bg-white/[0.035] text-text-primary'
                    : isComplete
                      ? 'border-border-subtle/70 bg-white/[0.015] text-text-secondary'
                      : 'border-border-subtle/55 bg-transparent text-text-muted'
                }`}
              >
                <div className="flex items-center gap-4">
                  <span
                    className={`font-mono text-[10px] font-semibold tabular-nums ${
                      isActive ? 'text-accent' : isComplete ? 'text-text-secondary' : 'text-text-muted/65'
                    }`}
                  >
                    {stage.number}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-baseline gap-2">
                      <p className="text-[13px] font-semibold leading-tight">{stage.label}</p>
                      <span
                        className={`hidden font-mono text-[9px] uppercase tracking-[0.18em] sm:inline ${
                          isActive ? 'text-accent/80' : isComplete ? 'text-text-muted' : 'text-text-muted/45'
                        }`}
                      >
                        {isActive ? 'active' : isComplete ? 'complete' : 'queued'}
                      </span>
                    </div>
                    <p className="mt-1 truncate text-[11px] text-text-muted">{stage.detail}</p>
                  </div>
                  <div className={`h-7 w-px ${isActive ? 'bg-accent/50' : isComplete ? 'bg-text-muted/30' : 'bg-border-subtle/60'}`} aria-hidden />
                </div>
                {isActive ? (
                  <motion.div
                    layoutId="decoding-stage-underbar"
                    className="absolute inset-x-5 bottom-0 h-[2px] rounded-full bg-accent/70"
                  />
                ) : null}
              </div>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
