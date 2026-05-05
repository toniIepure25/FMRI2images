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
      {...props}
    >
      {children}
    </motion.div>
  );
}
