import { motion } from 'framer-motion';
import { ChallengeGame } from '@/components/challenge/ChallengeGame';

export function Challenge() {
  return (
    <div className="min-h-[calc(100vh-3rem)] px-4 pb-16 pt-10 md:px-8">
      <motion.header
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
        className="mb-12 max-w-4xl"
      >
        <p className="text-[11px] font-semibold uppercase tracking-[0.25em] text-accent/80">Challenge mode</p>
        <h1 className="mt-3 text-3xl font-bold tracking-tight text-text-primary md:text-4xl">Blind Committee Challenge</h1>
        <p className="mt-3 text-lg font-medium text-text-secondary md:text-xl">Can you outperform the neural decoder?</p>
        <div className="mt-5 flex max-w-2xl flex-col gap-2">
          <p className="text-sm leading-relaxed text-text-secondary md:text-base">
            Identify which natural image matches the brain-decoded reconstruction.
          </p>
          <p className="text-sm leading-relaxed text-text-muted md:text-base">
            Judge the reconstruction in isolation, then compare human intuition with quantitative retrieval—designed for live
            discussion with a thesis committee.
          </p>
        </div>
      </motion.header>
      <ChallengeGame />
    </div>
  );
}
