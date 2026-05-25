import { motion } from 'framer-motion';
import { KickerRule } from './atoms';

/**
 * ThesisClaim
 * ─────────────────────────────────────────────────────────────
 * A small premium "thesis plaque" — a single calibrated paragraph
 * that frames the rest of the page. Designed to look like the
 * inscription on a scientific instrument, not a marketing tagline.
 *
 * Restrained dark surface, hair-rule border, accent quote glyph,
 * five-domain micro-row underneath.
 */

const DOMAINS = [
  { tag: 'Geometry',     hint: 'RSA + neighborhoods'    },
  { tag: 'Language',     hint: 'CLIP text probes'       },
  { tag: 'Hubness',      hint: 'CSLS correction'        },
  { tag: 'Counterfact.', hint: 'latent-space margins'   },
  { tag: 'Continuity',   hint: 'manifold interpolation' },
] as const;

export function ThesisClaim() {
  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className="mx-auto max-w-[1280px] px-4 sm:px-6 lg:px-8"
    >
      <div className="relative overflow-hidden rounded-2xl border border-white/[0.05] bg-white/[0.013] p-5 sm:p-6">
        <div
          className="pointer-events-none absolute inset-0 opacity-[0.20]"
          aria-hidden
          style={{
            backgroundImage:
              'repeating-linear-gradient(135deg, rgb(255 255 255 / 0.014) 0 1px, transparent 1px 9px)',
            maskImage:
              'linear-gradient(180deg, transparent 0%, black 40%, black 100%)',
            WebkitMaskImage:
              'linear-gradient(180deg, transparent 0%, black 40%, black 100%)',
          }}
        />

        <div className="relative flex items-center justify-between gap-3">
          <KickerRule label="Thesis claim · evaluation contract" />
          <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-text-muted">
            One claim · five measurements
          </span>
        </div>

        <div className="relative mt-3 flex items-start gap-3">
          <QuoteGlyph />
          <p className="max-w-[64rem] text-[14px] leading-[1.55] tracking-tight text-text-secondary sm:text-[15.5px]">
            The decoder is not evaluated only by whether it retrieves the exact image.
            It is evaluated by whether decoded brain embeddings preserve
            <span className="text-text-primary"> CLIP-space geometry</span>,
            <span className="text-text-primary"> language concepts</span>,
            <span className="text-text-primary"> hubness behaviour</span>,
            <span className="text-text-primary"> counterfactual margins</span>, and
            <span className="text-text-primary"> interpolation continuity</span>.
          </p>
        </div>

        <ul className="relative mt-5 grid grid-cols-2 gap-2 sm:grid-cols-5 sm:gap-3">
          {DOMAINS.map((d, i) => (
            <li
              key={d.tag}
              className="flex items-center gap-2.5 rounded-xl border border-white/[0.05] bg-white/[0.015] px-3 py-2"
            >
              <span className="font-mono text-[10px] tabular-nums text-text-muted">
                {String(i + 1).padStart(2, '0')}
              </span>
              <div className="min-w-0">
                <p className="truncate text-[11.5px] font-semibold tracking-tight text-text-primary">
                  {d.tag}
                </p>
                <p className="truncate text-[10px] text-text-muted">{d.hint}</p>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </motion.section>
  );
}

function QuoteGlyph() {
  return (
    <svg
      width="22"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="shrink-0 text-accent/65"
      aria-hidden
    >
      <path d="M7 7h4v4H7zM7 11c0 3 1 5 4 6M13 7h4v4h-4zM13 11c0 3 1 5 4 6" />
    </svg>
  );
}
