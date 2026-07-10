import { useState } from 'react';
import { motion } from 'framer-motion';
import type { Provenance } from '@/lib/provenance';
import { ProvenanceBadge } from './ProvenanceBadge';
import { EmptyState } from '@/components/premium/EmptyState';

interface TriptychPanel {
  title: string;
  subtitle: string;
  imageSrc?: string | null;
  provenance: Provenance;
  accent: 'emerald' | 'violet' | 'cyan';
  emptyText?: string;
  revealBlur?: boolean;
  isMatch?: boolean;
}

interface ComparisonTriptychProps {
  panels: [TriptychPanel, TriptychPanel, TriptychPanel];
  /** When true, suppress the bulky "third panel unavailable" EmptyState that
   *  normally appears below the image grid. The parent is then responsible
   *  for rendering its own slim notice. Defaults to false to preserve the
   *  existing behavior for older callers. */
  suppressUnavailableNotice?: boolean;
}

function SafeImg({ src, alt, className }: { src?: string | null; alt: string; className?: string }) {
  const [ok, setOk] = useState(true);
  if (!src || !ok) return (
    <div className={`flex items-center justify-center bg-surface-raised ${className ?? ''}`}>
      <span className="text-xs text-text-muted">&mdash;</span>
    </div>
  );
  return (
    <img src={src} alt={alt} className={`h-full w-full object-cover ${className ?? ''}`}
      onError={() => setOk(false)} />
  );
}

const ACCENT_BORDER = {
  emerald: 'border-emerald-500/25',
  violet: 'border-accent/25',
  cyan: 'border-cyan-500/25',
};

const ACCENT_DOT = {
  emerald: 'bg-status-success ring-1 ring-status-success/30',
  violet: 'bg-accent ring-1 ring-accent/30',
  cyan: 'bg-status-info ring-1 ring-status-info/30',
};

export function ComparisonTriptych({ panels, suppressUnavailableNotice = false }: ComparisonTriptychProps) {
  const thirdHasImage = !!panels[2]?.imageSrc;
  const visualPanels = panels;

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
      {visualPanels.map((panel, i) => {
        const hasImage = !!panel.imageSrc;

        return (
          <motion.figure
            key={panel.title}
            className={`group overflow-hidden rounded-2xl border bg-surface-elevated/85 transition-all duration-300 hover:-translate-y-0.5 ${
              panel.isMatch
                ? 'border-status-success/22'
                : 'border-white/[0.05] hover:border-white/[0.1]'
            }`}
            style={{ boxShadow: '0 1px 0 rgb(255 255 255 / 0.025) inset' }}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ type: 'spring', stiffness: 240, damping: 24, delay: i * 0.06 }}
          >
            {/* Image area — kept clean, no heavy overlay */}
            <div className="relative aspect-[4/3] max-h-[360px] w-full bg-surface-base/40 lg:aspect-[16/10]">
              {hasImage ? (
                panel.revealBlur ? (
                  <motion.div
                    className="absolute inset-0"
                    initial={{ filter: 'blur(20px)', opacity: 0 }}
                    animate={{ filter: 'blur(0px)', opacity: 1 }}
                    transition={{ delay: 0.3, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
                  >
                    <SafeImg src={panel.imageSrc} alt={panel.title} className="h-full w-full object-cover" />
                  </motion.div>
                ) : (
                  <SafeImg src={panel.imageSrc} alt={panel.title} className="h-full w-full object-cover" />
                )
              ) : (
                <div className="flex h-full w-full flex-col items-center justify-center gap-2 p-6 text-center">
                  <svg className="h-8 w-8 text-border-emphasis" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M6.75 7.5a.75.75 0 11-1.5 0 .75.75 0 011.5 0z" />
                  </svg>
                  <p className="text-[12px] font-medium text-text-muted">{panel.emptyText ?? 'No asset available.'}</p>
                </div>
              )}

              {/* Inner ring to soften the image edge */}
              <div className="pointer-events-none absolute inset-0 ring-1 ring-inset ring-white/[0.04]" aria-hidden />

              {/* Tiny "match" indicator (no glow), top-left, only for the
                  matching panel. Status badges live in the footer below. */}
              {panel.isMatch ? (
                <span className="absolute left-2.5 top-2.5 inline-flex items-center gap-1 rounded-md border border-status-success/25 bg-black/45 px-2 py-0.5 text-[10px] font-semibold text-status-success backdrop-blur-md">
                  <span className="h-1.5 w-1.5 rounded-full bg-status-success" />
                  Match
                </span>
              ) : null}
            </div>

            {/* Clean caption beneath the frame — like a curated scientific
                triptych, not a marketplace image card. */}
            <figcaption className="flex items-center justify-between gap-3 px-4 py-3">
              <div className="min-w-0">
                <p className="truncate text-[12.5px] font-semibold text-text-primary">{panel.title}</p>
                <p className="mt-0.5 truncate text-[11px] text-text-muted">{panel.subtitle}</p>
              </div>
              <div className="shrink-0">
                <ProvenanceBadge provenance={panel.provenance} />
              </div>
            </figcaption>
          </motion.figure>
        );
      })}
      </div>

      {!thirdHasImage && !suppressUnavailableNotice ? (
        <EmptyState
          compact
          title={panels[2].title}
          detail={
            <>
              {panels[2].emptyText ?? 'Reconstruction processing for this trial.'}
              <span className="mt-2 block">
                <ProvenanceBadge provenance={panels[2].provenance} />
              </span>
            </>
          }
        />
      ) : null}
    </div>
  );
}
