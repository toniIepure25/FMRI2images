import { motion } from 'framer-motion';

/**
 * StoryNav
 * ─────────────────────────────────────────────────────────────
 * "Analysis sequence" rail — a non-sticky strip showing the
 * eight-step scientific story of the page, with each item
 * acting as an anchor link to the matching section id.
 *
 * Visual language: hair-thin spine + numbered dots + tags.
 * Horizontally scrollable on narrow viewports.
 */

export const SECTION_IDS = {
  overview:       'manifold-overview',
  probe:          'manifold-probe',
  retrieval:      'manifold-retrieval',
  hubness:        'manifold-hubness',
  errorField:     'manifold-error-field',
  counterfactual: 'manifold-counterfactual',
  walk:           'manifold-walk',
  scope:          'manifold-scope',
} as const;

const STEPS: Array<{ id: string; label: string; hint: string }> = [
  { id: SECTION_IDS.overview,       label: 'Overview',       hint: 'R@1 is not enough'   },
  { id: SECTION_IDS.probe,          label: 'Probe',          hint: 'CLIP text concepts'  },
  { id: SECTION_IDS.retrieval,      label: 'Retrieval',      hint: 'cosine → CSLS'       },
  { id: SECTION_IDS.hubness,        label: 'Hubness',        hint: 'CSLS reduces Gini'   },
  { id: SECTION_IDS.errorField,     label: 'Error field',    hint: 'z_target − z_pred'   },
  { id: SECTION_IDS.counterfactual, label: 'Counterfactual', hint: 'latent-space axis'   },
  { id: SECTION_IDS.walk,           label: 'Walk',           hint: 'CLIP-space slerp'    },
  { id: SECTION_IDS.scope,          label: 'Scope',          hint: 'honesty contract'    },
];

export function StoryNav() {
  return (
    <motion.nav
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      aria-label="Analysis sequence"
      className="mx-auto max-w-[1280px] px-4 sm:px-6 lg:px-8"
    >
      <div className="overflow-hidden rounded-2xl border border-white/[0.05] bg-white/[0.013]">
        <div className="flex items-center gap-4 px-4 py-3 sm:px-5">
          <div className="hidden shrink-0 items-center gap-2.5 sm:flex">
            <span className="h-px w-5 bg-accent/45" aria-hidden />
            <p className="premium-kicker">Analysis sequence</p>
          </div>

          <ol className="relative flex flex-1 items-center gap-2 overflow-x-auto pb-0.5">
            {/* hair-thin spine */}
            <span
              className="pointer-events-none absolute inset-x-2 top-[15px] -z-0 h-px bg-white/[0.06]"
              aria-hidden
            />
            {STEPS.map((step, i) => (
              <li key={step.id} className="relative z-10 shrink-0">
                <a
                  href={`#${step.id}`}
                  className="group inline-flex items-center gap-2 rounded-md px-2 py-1 text-left transition hover:bg-white/[0.025]"
                >
                  <span className="flex h-[22px] w-[22px] items-center justify-center rounded-full border border-white/[0.075] bg-[#0d1019] font-mono text-[9.5px] font-semibold tabular-nums text-text-secondary group-hover:border-accent/40 group-hover:text-accent">
                    {String(i + 1).padStart(2, '0')}
                  </span>
                  <span className="flex flex-col leading-none">
                    <span className="text-[11.5px] font-semibold tracking-tight text-text-primary">
                      {step.label}
                    </span>
                    <span className="mt-0.5 hidden text-[10px] text-text-muted lg:block">
                      {step.hint}
                    </span>
                  </span>
                </a>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </motion.nav>
  );
}
