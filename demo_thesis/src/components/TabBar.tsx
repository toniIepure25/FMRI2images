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
      className="flex w-max min-w-full items-center gap-0.5 rounded-lg bg-surface-raised p-0.5"
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
            className={`relative rounded-md px-3 py-1.5 text-[12px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
              isActive ? 'text-text-primary' : 'text-text-muted hover:text-text-secondary'
            }`}
          >
            {isActive && (
              <motion.div
                layoutId={`${layoutIdPrefix}-active-indicator`}
                className="absolute inset-0 rounded-md bg-surface-elevated shadow-[0_1px_3px_0_rgb(0,0,0,0.3)] ring-1 ring-border-subtle"
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
