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

export function MetricCard({ label, value, subtitle, color = '#00d4ff', icon, delay = 0 }: MetricCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="glass-panel-hover group p-5"
    >
      <div className="mb-3 flex items-start justify-between">
        <span className="text-xs font-medium uppercase tracking-wider text-slate-500">{label}</span>
        {icon && <div className="text-lg" style={{ color }}>{icon}</div>}
      </div>
      <div
        className="mb-1 text-3xl font-bold tracking-tight transition-transform duration-300 group-hover:translate-x-0.5"
        style={{ color }}
      >
        {value}
      </div>
      {subtitle && <div className="text-xs text-slate-500">{subtitle}</div>}
    </motion.div>
  );
}
