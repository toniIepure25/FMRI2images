import { motion } from 'framer-motion';
import { ImagePlaceholder } from '@/components/ImagePlaceholder';

export interface ChallengeCardProps {
  src: string;
  alt: string;
  index: number;
  disabled: boolean;
  isSelected: boolean;
  revealed: boolean;
  isCorrect: boolean;
  onSelect: (index: number) => void;
}

export function ChallengeCard({
  src,
  alt,
  index,
  disabled,
  isSelected,
  revealed,
  isCorrect,
  onSelect,
}: ChallengeCardProps) {
  const showWrong = revealed && isSelected && !isCorrect;
  const showCorrect = revealed && isCorrect;
  const idleHover = !disabled && !revealed;

  return (
    <motion.button
      type="button"
      layout
      initial={{ opacity: 0, y: 12, rotateY: -6 }}
      animate={{
        opacity: 1,
        y: 0,
        rotateY: 0,
        scale: revealed && isSelected ? [1, 1.02, 1] : 1,
      }}
      transition={{
        opacity: { duration: 0.35 },
        layout: { duration: 0.4 },
        scale: { duration: 0.45 },
      }}
      whileHover={
        idleHover
          ? {
              boxShadow: '0 0 28px rgba(139, 92, 246, 0.22), 0 0 48px rgba(236, 72, 153, 0.08)',
              borderColor: 'rgba(139, 92, 246, 0.45)',
            }
          : undefined
      }
      whileTap={idleHover ? { scale: 0.98 } : undefined}
      disabled={disabled}
      onClick={() => !disabled && onSelect(index)}
      className={[
        'group relative w-full overflow-hidden rounded-xl text-left transition-colors duration-300',
        'border bg-brain-navy/60 focus:outline-none focus-visible:ring-2 focus-visible:ring-brain-purple/60',
        showCorrect
          ? 'border-emerald-400/70 shadow-[0_0_32px_rgba(16,185,129,0.35)]'
          : showWrong
            ? 'border-red-400/70 shadow-[0_0_28px_rgba(239,68,68,0.35)]'
            : isSelected && !revealed
              ? 'border-fuchsia-400/80 shadow-[0_0_24px_rgba(217,70,239,0.35)]'
              : 'border-brain-border/60 shadow-none',
        disabled ? 'cursor-not-allowed opacity-90' : 'cursor-pointer',
      ].join(' ')}
      aria-pressed={isSelected}
      aria-label={`Candidate ${index + 1}${revealed && isCorrect ? ', correct answer' : ''}${showWrong ? ', incorrect selection' : ''}`}
    >
      <div className="relative aspect-square w-full [perspective:1200px]">
        <motion.div
          className="h-full w-full origin-center"
          initial={false}
          animate={{
            rotateY: revealed ? [0, -6, 0] : 0,
            rotateX: revealed ? [0, 4, 0] : 0,
          }}
          transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
        >
          {src ? (
            <ImagePlaceholder src={src} alt={alt} className="h-full min-h-[140px] rounded-none border-0" type="target" />
          ) : (
            <div className="flex h-full min-h-[140px] w-full items-center justify-center bg-gradient-to-br from-brain-panel to-brain-navy text-xs text-slate-500">
              No image
            </div>
          )}
        </motion.div>
      </div>
      <div className="flex items-center justify-between border-t border-brain-border/40 px-3 py-2">
        <span className="font-mono text-[10px] uppercase tracking-wider text-slate-500">Option {index + 1}</span>
        {revealed && isCorrect && (
          <span className="text-[10px] font-semibold uppercase tracking-wide text-emerald-400">Ground truth</span>
        )}
        {showWrong && (
          <span className="text-[10px] font-semibold uppercase tracking-wide text-red-400">Your pick</span>
        )}
      </div>
    </motion.button>
  );
}
