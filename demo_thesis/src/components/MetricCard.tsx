import type { ReactNode } from 'react';
import { motion } from 'framer-motion';

interface MetricCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  color?: string;
  icon?: ReactNode;
  delay?: number;
}

export function MetricCard({ label, value, subtitle, color, icon, delay = 0 }: MetricCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      className="surface-card relative overflow-hidden p-5"
    >
      <div className="mb-4 flex items-start justify-between">
        <span
          className="inline-block text-2xs font-semibold uppercase tracking-wider text-text-muted"
        >
          {label}
        </span>
        {icon && (
          <div className="text-lg opacity-70" style={{ color: color ?? undefined }}>
            {icon}
          </div>
        )}
      </div>
      <div className="mb-1 font-mono text-[1.75rem] font-semibold leading-none tracking-tight text-text-primary">
        {value}
      </div>
      {subtitle && (
        <div className="mt-2 text-xs leading-relaxed text-text-muted">{subtitle}</div>
      )}
      {/* Subtle accent strip at top */}
      {color && (
        <div
          className="absolute inset-x-0 top-0 h-[3px] rounded-t-xl opacity-60"
          style={{ background: color }}
        />
      )}
    </motion.div>
  );
}
