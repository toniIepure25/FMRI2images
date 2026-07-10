import { motion } from 'framer-motion';
import { KickerRule } from './atoms';

/**
 * HonestyStrip
 * ─────────────────────────────────────────────────────────────
 * Compact horizontal card summarising the scientific scope of
 * the page. Five short bullets, restrained tone — present so
 * the page anchors its claims rather than reading defensively.
 */

const STATEMENTS: Array<{ label: string; body: string }> = [
  { label: 'Scope',          body: 'Visual-stimulus fMRI → CLIP decoding. Not mind reading.' },
  { label: 'Counterfactuals', body: 'Latent-space explanations only — never brain manipulation.' },
  { label: 'Interpolation',  body: 'CLIP-space paths, not real neural trajectories.' },
  { label: 'Schematics',     body: 'Coordinates and per-trial probes labelled DERIVED.' },
  { label: 'Transparency',   body: 'All modules and data sources documented explicitly.' },
];

export function HonestyStrip() {
  return (
    <motion.section
      id="manifold-scope"
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className="mx-auto max-w-[1280px] scroll-mt-20 px-4 sm:px-6 lg:px-8"
    >
      <div className="rounded-2xl border border-white/[0.05] bg-white/[0.013] p-5 sm:p-6">
        <div className="flex flex-col gap-1.5 sm:flex-row sm:items-center sm:justify-between">
          <KickerRule label="Scientific scope" />
          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-text-muted">
            Claims under measurement · not under hype
          </p>
        </div>
        <ul className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          {STATEMENTS.map((s) => (
            <li key={s.label} className="rounded-xl border border-white/[0.05] bg-white/[0.015] px-3.5 py-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-text-muted/90">
                {s.label}
              </p>
              <p className="mt-1.5 text-[11.5px] leading-snug text-text-secondary">{s.body}</p>
            </li>
          ))}
        </ul>
      </div>
    </motion.section>
  );
}
