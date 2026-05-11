import type { ReactNode } from 'react';
import { motion, type HTMLMotionProps } from 'framer-motion';

interface GlassCardProps extends HTMLMotionProps<'div'> {
  children: ReactNode;
  className?: string;
  hover?: boolean;
  glow?: boolean;
}

export function GlassCard({ children, className = '', hover = false, glow = false, ...props }: GlassCardProps) {
  return (
    <motion.div
      className={`${glow ? 'glow-accent' : ''} ${hover ? 'surface-card-hover' : 'surface-card'} ${className}`}
      whileHover={hover ? { y: -1 } : undefined}
      transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
      {...props}
    >
      {children}
    </motion.div>
  );
}
