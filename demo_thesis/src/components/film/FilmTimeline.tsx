import { motion } from 'framer-motion';

export interface FilmTimelineStepMeta {
  title: string;
  shortLabel: string;
}

export interface FilmTimelineProps {
  steps: FilmTimelineStepMeta[];
  currentStep: number;
  onStepChange: (index: number) => void;
}

export function FilmTimeline({ steps, currentStep, onStepChange }: FilmTimelineProps) {
  return (
    <div className="pointer-events-auto w-full max-w-5xl px-4 pb-1">
      <div className="relative mx-auto">
        {/* connector line */}
        <div
          className="absolute left-[6%] right-[6%] top-[18px] h-px bg-gradient-to-r from-transparent via-slate-600/80 to-transparent"
          aria-hidden
        />
        <div
          className="absolute left-[6%] top-[18px] h-px bg-gradient-to-r from-cyan-400/0 via-cyan-400 to-cyan-400/0 transition-all duration-700 ease-out"
          style={{
            width: `${steps.length <= 1 ? 0 : (currentStep / (steps.length - 1)) * 88}%`,
            marginLeft: '6%',
          }}
          aria-hidden
        />

        <div className="relative flex justify-between gap-1">
          {steps.map((s, i) => {
            const active = i === currentStep;
            const done = i < currentStep;
            return (
              <button
                key={s.shortLabel}
                type="button"
                onClick={() => onStepChange(i)}
                className="group flex flex-1 flex-col items-center gap-1 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/70 rounded-lg"
              >
                <motion.div
                  layout
                  className={`relative flex h-9 w-9 items-center justify-center rounded-full border text-[11px] font-bold transition-colors ${
                    active
                      ? 'border-cyan-300/90 bg-cyan-500/20 text-cyan-100 shadow-[0_0_22px_rgba(34,211,238,0.45)]'
                      : done
                        ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200/90'
                        : 'border-slate-600/80 bg-slate-900/80 text-slate-500 group-hover:border-slate-500 group-hover:text-slate-300'
                  }`}
                >
                  {i + 1}
                  {active ? (
                    <motion.span
                      layoutId="film-step-glow"
                      className="absolute inset-0 rounded-full bg-cyan-400/25 blur-md"
                      transition={{ type: 'spring', stiffness: 380, damping: 28 }}
                    />
                  ) : null}
                </motion.div>
                <span
                  className={`hidden text-center text-[10px] font-medium leading-tight sm:block ${
                    active ? 'text-cyan-100' : done ? 'text-slate-400' : 'text-slate-600'
                  }`}
                >
                  {s.shortLabel}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
