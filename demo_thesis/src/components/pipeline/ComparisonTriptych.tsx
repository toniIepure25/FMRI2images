import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import type { Provenance } from '@/lib/provenance';
import { ProvenanceBadge } from './ProvenanceBadge';

interface TriptychPanel {
  title: string;
  subtitle: string;
  imageSrc?: string | null;
  provenance: Provenance;
  accent: 'emerald' | 'violet' | 'cyan';
  emptyText?: string;
  revealBlur?: boolean;
}

interface ComparisonTriptychProps {
  panels: [TriptychPanel, TriptychPanel, TriptychPanel];
}

function SafeImg({
  src, alt, className,
}: { src?: string | null; alt: string; className?: string }) {
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
      {!loaded && (
        <div className="pointer-events-none absolute inset-0 z-10 shimmer-bg" aria-hidden />
      )}
      <img src={src} alt={alt}
        className={`relative z-0 h-full w-full object-cover transition-opacity duration-500 ${loaded ? 'opacity-100' : 'opacity-0'}`}
        onLoad={() => setLoaded(true)}
        onError={() => setOk(false)}
        loading="lazy" />
    </div>
  );
}

const ACCENT_DOT = {
  emerald: 'bg-status-success ring-1 ring-status-success/30',
  violet:  'bg-accent ring-1 ring-accent/30',
  cyan:    'bg-status-info ring-1 ring-status-info/30',
};

export function ComparisonTriptych({ panels }: ComparisonTriptychProps) {
  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
      {panels.map((panel, i) => {
        const hasImage = !!panel.imageSrc;

        return (
          <motion.div
            key={panel.title}
            className="overflow-hidden rounded-xl border border-border-subtle bg-surface-elevated"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ type: 'spring', stiffness: 240, damping: 24, delay: i * 0.06 }}
          >
            {/* Panel header */}
            <div className="flex items-center justify-between px-4 py-3">
              <div className="flex items-center gap-2.5">
                <div className={`h-2 w-2 rounded-full ${ACCENT_DOT[panel.accent]}`} />
                <div>
                  <p className="text-[12px] font-semibold text-text-primary">{panel.title}</p>
                  <p className="text-[10px] text-text-muted">{panel.subtitle}</p>
                </div>
              </div>
              <ProvenanceBadge provenance={panel.provenance} />
            </div>

            {/* Image area */}
            <div className="relative aspect-square w-full bg-surface-base/50">
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
                /* ── PREMIUM EMPTY STATE ── */
                <div className="flex h-full w-full flex-col items-center justify-center gap-3 p-8">
                  {/* Schematic reconstruction icon */}
                  <svg className="h-12 w-12 text-border-emphasis" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.8}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9.53 16.122a3 3 0 00-5.78 1.128 2.25 2.25 0 01-2.4 2.245 4.5 4.5 0 008.4-2.245c0-.399-.078-.78-.22-1.128zm0 0a15.998 15.998 0 003.388-1.62m-5.043-.025a15.994 15.994 0 011.622-3.395m3.42 3.42a15.995 15.995 0 004.764-4.648l3.876-5.814a1.151 1.151 0 00-1.597-1.597L14.146 6.32a15.996 15.996 0 00-4.649 4.763m3.42 3.42a6.776 6.776 0 00-3.42-3.42" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159" strokeWidth={1.5} />
                  </svg>
                  <div className="text-center space-y-1.5">
                    <p className="text-[13px] font-semibold text-text-secondary">
                      No reconstruction available
                    </p>
                    <p className="text-[11px] leading-relaxed text-text-muted max-w-[200px]">
                      {panel.emptyText ?? 'This replay contains retrieval evidence only. Diffusion policy was derived but no reconstruction image was cached for this trial.'}
                    </p>
                  </div>
                  <ProvenanceBadge provenance={panel.provenance} />
                </div>
              )}
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
