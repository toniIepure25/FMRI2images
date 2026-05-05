import { motion } from 'framer-motion';

interface Tab {
  id: string;
  label: string;
}

interface TabBarProps {
  tabs: Tab[];
  activeTab: string;
  onChange: (id: string) => void;
  /** Namespaces layout animation when multiple tab bars exist on a page */
  layoutIdPrefix?: string;
}

export function TabBar({ tabs, activeTab, onChange, layoutIdPrefix = 'tabs' }: TabBarProps) {
  return (
    <div className="flex w-max min-w-full items-center gap-1 p-1 glass-panel">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          onClick={() => onChange(tab.id)}
          className={`relative rounded-lg px-3 py-2 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brain-accent ${
            activeTab === tab.id ? 'text-white' : 'text-slate-500 hover:text-slate-300'
          }`}
        >
          {activeTab === tab.id && (
            <motion.div
              layoutId={`${layoutIdPrefix}-active-indicator`}
              className="absolute inset-0 rounded-lg border border-brain-accent/20 bg-brain-accent/10"
              transition={{ type: 'spring', bounce: 0.15, duration: 0.4 }}
            />
          )}
          <span className="relative z-10">{tab.label}</span>
        </button>
      ))}
    </div>
  );
}
