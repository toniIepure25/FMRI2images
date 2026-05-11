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
  src,
  alt,
  className,
}: {
  src?: string | null;
  alt: string;
  className?: string;
}) {
  const [ok, setOk] = useState(true);
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    setOk(true);
    setLoaded(false);
  }, [src]);
  if (!src || !ok) {
    return (
      <div
        className={`flex items-center justify-center bg-slate-900/80 ${className ?? ''}`}
      >
        <span className="text-xs text-slate-600">—</span>
      </div>
    );
  }
  return (
    <div className={`relative overflow-hidden ${className ?? ''}`}>
      {!loaded && (
        <div
          className="pointer-events-none absolute inset-0 z-10 shimmer-bg"
          aria-hidden
        />
      )}
      <img
        src={src}
        alt={alt}
        className={`relative z-0 h-full w-full object-cover transition-opacity duration-500 ${loaded ? 'opacity-100' : 'opacity-0'}`}
        onLoad={() => setLoaded(true)}
        onError={() => setOk(false)}
        loading="lazy"
      />
    </div>
  );
}

const ACCENT_CLASSES = {
  emerald: {
    dot: 'bg-emerald-400',
    title: 'text-emerald-300',
    ring: 'ring-emerald-500/10',
  },
  violet: {
    dot: 'bg-violet-400',
    title: 'text-violet-300',
    ring: 'ring-violet-500/10',
  },
  cyan: {
    dot: 'bg-cyan-400',
    title: 'text-cyan-300',
    ring: 'ring-cyan-500/10',
  },
};

export function ComparisonTriptych({ panels }: ComparisonTriptychProps) {
  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
      {panels.map((panel, i) => {
        const a = ACCENT_CLASSES[panel.accent];
        const hasImage = !!panel.imageSrc;

        return (
          <motion.div
            key={panel.title}
            className={`overflow-hidden rounded-2xl bg-white/[0.02] ring-1 ${a.ring}`}
            initial={{ opacity: 0, y: 16, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{
              type: 'spring',
              stiffness: 240,
              damping: 24,
              delay: i * 0.08,
            }}
          >
            {/* Panel header */}
            <div className="flex items-center justify-between px-4 py-3">
              <div className="flex items-center gap-2.5">
                <div className={`h-1.5 w-1.5 rounded-full ${a.dot}`} />
                <div>
                  <p className={`text-[12px] font-semibold ${a.title}`}>
                    {panel.title}
                  </p>
                  <p className="text-[9px] text-slate-600">{panel.subtitle}</p>
                </div>
              </div>
              <ProvenanceBadge provenance={panel.provenance} />
            </div>

            {/* Image area */}
            <div className="relative aspect-square w-full bg-black/30">
              {hasImage ? (
                panel.revealBlur ? (
                  <motion.div
                    className="absolute inset-0"
                    initial={{ filter: 'blur(20px)', opacity: 0 }}
                    animate={{ filter: 'blur(0px)', opacity: 1 }}
                    transition={{
                      delay: 0.3,
                      duration: 0.7,
                      ease: [0.22, 1, 0.36, 1],
                    }}
                  >
                    <SafeImg
                      src={panel.imageSrc}
                      alt={panel.title}
                      className="h-full w-full object-cover"
                    />
                  </motion.div>
                ) : (
                  <SafeImg
                    src={panel.imageSrc}
                    alt={panel.title}
                    className="h-full w-full object-cover"
                  />
                )
              ) : (
                <div className="flex h-full w-full flex-col items-center justify-center gap-3 p-8">
                  <svg className="h-8 w-8 text-slate-700" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M6.75 7.5a.75.75 0 11-1.5 0 .75.75 0 011.5 0z" />
                  </svg>
                  <p className="text-center text-[11px] leading-relaxed text-slate-600">
                    {panel.emptyText ?? 'No asset available for this trial.'}
                  </p>
                </div>
              )}
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
