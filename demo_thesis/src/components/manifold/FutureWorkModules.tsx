import { motion } from 'framer-motion';
import { ProvenanceBadge } from '@/components/pipeline/ProvenanceBadge';
import { PremiumPanel } from '@/components/premium/PremiumPanel';
import { ManifoldSectionHeader } from './ManifoldSectionHeader';
import { LockChip } from './atoms';
import { lockedModules } from '@/data/neuralManifoldExplorer';

/**
 * FutureWorkModules
 * ─────────────────────────────────────────────────────────────
 * Deliberately-locked surfaces. Each card adds a stippled diagonal
 * mesh + a "Locked" chip + an explicit "requires" line listing the
 * missing artifact. Marking these honestly is what makes the rest
 * of the page credible.
 */
export function FutureWorkModules() {
  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className="mx-auto max-w-[1280px] px-4 sm:px-6 lg:px-8"
    >
      <ManifoldSectionHeader
        kicker="Next extensions"
        title="What this page does not yet show."
        description="Unavailable modules are explicitly marked rather than faked — the artifact each module depends on is listed beside it so it is clear what would be needed to enable it."
      />

      <div className="mt-6 grid gap-3.5 sm:grid-cols-2">
        {lockedModules.map((m) => (
          <PremiumPanel key={m.id} className="relative overflow-hidden p-5">
            {/* Diagonal hairline mesh — visually signals locked surface */}
            <div
              className="pointer-events-none absolute inset-0 opacity-[0.4]"
              aria-hidden
              style={{
                backgroundImage:
                  'repeating-linear-gradient(135deg, rgb(255 255 255 / 0.02) 0 1px, transparent 1px 9px)',
                maskImage:
                  'linear-gradient(180deg, transparent 0%, black 35%, black 100%)',
                WebkitMaskImage:
                  'linear-gradient(180deg, transparent 0%, black 35%, black 100%)',
              }}
            />
            {/* Faint corner lock glyph */}
            <div className="pointer-events-none absolute right-3 bottom-3 text-text-muted/15" aria-hidden>
              <LockBg />
            </div>

            <div className="relative flex items-start justify-between gap-3">
              <div className="min-w-0">
                <LockChip />
                <h3 className="mt-2.5 text-[15.5px] font-semibold tracking-tight text-text-primary">
                  {m.title}
                </h3>
                <p className="mt-1 text-[12px] leading-relaxed text-text-secondary">
                  {m.description}
                </p>
              </div>
              <ProvenanceBadge provenance={m.provenance} />
            </div>

            <div className="relative mt-4 border-t border-white/[0.05] pt-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-text-muted/85">
                Requires
              </p>
              <p className="mt-1 text-[11.5px] leading-snug text-text-muted">{m.reason}</p>
            </div>
          </PremiumPanel>
        ))}
      </div>
    </motion.section>
  );
}

function LockBg() {
  return (
    <svg width="92" height="92" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="0.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="5" y="11" width="14" height="10" rx="2" />
      <path d="M8 11V8a4 4 0 018 0v3" />
    </svg>
  );
}
