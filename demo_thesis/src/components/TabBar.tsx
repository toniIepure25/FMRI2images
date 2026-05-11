import { motion } from 'framer-motion';

interface Tab {
  id: string;
  label: string;
}

interface TabBarProps {
  tabs: Tab[];
  activeTab: string;
  onChange: (id: string) => void;
  layoutIdPrefix?: string;
}

export function TabBar({ tabs, activeTab, onChange, layoutIdPrefix = 'tabs' }: TabBarProps) {
  return (
    <div
      role="tablist"
      aria-orientation="horizontal"
      className="flex w-max min-w-full items-center gap-1 p-1 glass-panel"
    >
      {tabs.map((tab) => {
        const isActive = activeTab === tab.id;
        return (
          <button
            key={tab.id}
            type="button"
            role="tab"
            id={`${layoutIdPrefix}-tab-${tab.id}`}
            aria-selected={isActive}
            aria-controls={`${layoutIdPrefix}-panel-${tab.id}`}
            tabIndex={isActive ? 0 : -1}
            onClick={() => onChange(tab.id)}
            className={`relative rounded-lg px-3 py-2 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brain-accent ${
              isActive ? 'text-white' : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            {isActive && (
              <motion.div
                layoutId={`${layoutIdPrefix}-active-indicator`}
                className="absolute inset-0 rounded-lg border border-brain-accent/20 bg-brain-accent/10"
                transition={{ type: 'spring', bounce: 0.15, duration: 0.4 }}
              />
            )}
            <span className="relative z-10">{tab.label}</span>
          </button>
        );
      })}
    </div>
  );
}
