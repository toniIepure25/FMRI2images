import { useState, useEffect } from 'react';
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
}

function SafeImg({ src, alt, className }: { src?: string | null; alt: string; className?: string }) {
  const [ok, setOk] = useState(true);
  const [loaded, setLoaded] = useState(false);
  useEffect(() => { setOk(true); setLoaded(false); }, [src]);
  if (!src || !ok) return (
    <div className={`flex items-center justify-center bg-surface-raised ${className ?? ''}`}>
      <span className="text-xs text-text-muted">&mdash;</span>
    </div>
  );
  return (
    <div className={`relative overflow-hidden ${className ?? ''}`}>
      {!loaded && <div className="pointer-events-none absolute inset-0 z-10 shimmer-bg" aria-hidden />}
      <img src={src} alt={alt} className={`relative z-0 h-full w-full object-cover transition-opacity duration-500 ${loaded ? 'opacity-100' : 'opacity-0'}`}
        onLoad={() => setLoaded(true)} onError={() => setOk(false)} loading="lazy" />
    </div>
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

export function ComparisonTriptych({ panels }: ComparisonTriptychProps) {
  const thirdHasImage = !!panels[2]?.imageSrc;
  const visualPanels = thirdHasImage ? panels : ([panels[0], panels[1]] as const);

  return (
    <div className="space-y-3">
      <div className={`grid grid-cols-1 gap-3 ${thirdHasImage ? 'lg:grid-cols-3' : 'lg:grid-cols-2'}`}>
      {visualPanels.map((panel, i) => {
        const hasImage = !!panel.imageSrc;

        return (
          <motion.div
            key={panel.title}
            className={`group overflow-hidden rounded-2xl border bg-surface-elevated transition-all duration-300 hover:-translate-y-0.5 ${
              panel.isMatch ? 'border-status-success/20 shadow-[0_18px_52px_-42px_rgba(74,222,128,0.34)]' : 'border-border-subtle hover:border-border-emphasis'
            }`}
            initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
            transition={{ type: 'spring', stiffness: 240, damping: 24, delay: i * 0.06 }}
          >
            {/* Image area */}
            <div className="relative aspect-[4/3] max-h-[520px] w-full bg-surface-base/40 lg:aspect-[16/10]">
              {hasImage ? (
                panel.revealBlur ? (
                  <motion.div className="absolute inset-0"
                    initial={{ filter: 'blur(20px)', opacity: 0 }}
                    animate={{ filter: 'blur(0px)', opacity: 1 }}
                    transition={{ delay: 0.3, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}>
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

              {/* Bottom overlay — deep black gradient for max readability */}
              <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/95 via-black/60 to-transparent pt-14 pb-3 px-3">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-[12px] font-semibold text-white/95">{panel.title}</p>
                    <p className="text-[10px] text-white/70">{panel.subtitle}</p>
                  </div>
                  <ProvenanceBadge provenance={panel.provenance} />
                </div>
              </div>
            </div>
          </motion.div>
        );
      })}
      </div>

      {!thirdHasImage ? (
        <EmptyState
          compact
          title={panels[2].title}
          detail={
            <>
              {panels[2].emptyText ?? 'No reconstruction asset was cached for this trial. This replay contains retrieval evidence only.'}
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
