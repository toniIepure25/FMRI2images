import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { DemoCase } from '@/types';
import { ClipSpaceViewer } from '@/components/ClipSpaceViewer';
import { useClipProjection } from '@/lib/hooks';
import { GlassCard } from '@/components/GlassCard';

export interface StepClipSpaceProps {
  case_: DemoCase;
}

export function StepClipSpace({ case_ }: StepClipSpaceProps) {
  const { projection, loading, error } = useClipProjection();
  const [flyToQuery, setFlyToQuery] = useState(false);
  const [showWorkspace, setShowWorkspace] = useState(false);

  useEffect(() => {
    setFlyToQuery(false);
    setShowWorkspace(false);
    const show = window.setTimeout(() => setShowWorkspace(true), 1200);
    const fly = window.setTimeout(() => setFlyToQuery(true), 2100);
    return () => {
      window.clearTimeout(show);
      window.clearTimeout(fly);
    };
  }, [case_.id]);

  return (
    <div className="flex h-full min-h-0 flex-col gap-5 px-2">
      <motion.h2
        className="text-center text-xl font-semibold leading-snug text-white sm:text-2xl lg:text-3xl"
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        The predicted CLIP embedding enters the retrieval neighborhood
      </motion.h2>

      <AnimatePresence mode="wait">
        {!showWorkspace ? (
          <motion.div
            key="status"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.35 }}
            className="mx-auto flex w-full max-w-xl items-center justify-center gap-3 rounded-xl border border-cyan-500/30 bg-cyan-500/10 px-4 py-4"
          >
            <motion.span
              className="h-2 w-2 rounded-full bg-cyan-400"
              animate={{ opacity: [0.4, 1, 0.4], scale: [1, 1.2, 1] }}
              transition={{ duration: 1.2, repeat: Infinity }}
            />
            <p className="text-center text-sm font-medium text-cyan-100">
              Embedding computed. Searching CLIP gallery…
            </p>
          </motion.div>
        ) : null}
      </AnimatePresence>

      <motion.div
        initial={{ opacity: 0, y: 22 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: showWorkspace ? 0.08 : 0, duration: 0.55 }}
      >
        <GlassCard className={`p-4 sm:p-5 transition-opacity ${showWorkspace ? 'opacity-100' : 'pointer-events-none opacity-25'}`}>
          {error ? (
            <div className="flex h-[420px] flex-col items-center justify-center gap-2 text-sm text-red-400/80">
              <span className="text-xs text-slate-600">Failed to load CLIP projection</span>
            </div>
          ) : loading || !projection ? (
            <div className="flex h-[420px] items-center justify-center gap-3 text-sm text-slate-500">
              <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-brain-accent/60" />
              Loading CLIP projection…
            </div>
          ) : (
            <ClipSpaceViewer
              projection={projection}
              caseClipSpace={case_.clipSpace}
              flyToQuery={flyToQuery}
              showLines
              className="[&_.neon-border]:shadow-[0_0_24px_rgba(34,211,238,0.12)]"
            />
          )}
        </GlassCard>
      </motion.div>

      <motion.p
        className="mx-auto max-w-3xl text-center text-sm leading-relaxed text-slate-400"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.35, duration: 0.45 }}
      >
        UMAP lifts the 768-D hypersphere into a navigable map. Watch the camera tighten on the predicted query;
        retrieved neighbors orbit nearby while the ground-truth target stays a semantic anchor in the gallery.
      </motion.p>
    </div>
  );
}
