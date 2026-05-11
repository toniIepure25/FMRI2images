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
      className={`${hover ? 'glass-panel-hover' : 'glass-panel'} ${glow ? 'glow-accent' : ''} ${className}`}
      whileHover={hover ? { y: -2 } : undefined}
      transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
      {...props}
    >
      {children}
    </motion.div>
  );
}
