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
      transition={{ delay, duration: 0.5 }}
      className="glass-panel-hover p-5"
    >
      <div className="flex items-start justify-between mb-3">
        <span className="text-xs font-medium uppercase tracking-wider text-gray-500">{label}</span>
        {icon && <div className="text-lg" style={{ color }}>{icon}</div>}
      </div>
      <div className="text-3xl font-bold mb-1" style={{ color }}>{value}</div>
      {subtitle && <div className="text-xs text-gray-500">{subtitle}</div>}
    </motion.div>
  );
}
