import { motion } from 'framer-motion';

interface Tab {
  id: string;
  label: string;
  icon?: string;
}

interface TabBarProps {
  tabs: Tab[];
  activeTab: string;
  onChange: (id: string) => void;
}

export function TabBar({ tabs, activeTab, onChange }: TabBarProps) {
  return (
    <div className="flex items-center gap-1 p-1 glass-panel">
      {tabs.map(tab => (
        <button
          key={tab.id}
          type="button"
          onClick={() => onChange(tab.id)}
          className={`relative px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
            activeTab === tab.id ? 'text-white' : 'text-gray-500 hover:text-gray-300'
          }`}
        >
          {activeTab === tab.id && (
            <motion.div
              layoutId="activeTab"
              className="absolute inset-0 bg-brain-accent/10 border border-brain-accent/20 rounded-lg"
              transition={{ type: 'spring', bounce: 0.15, duration: 0.4 }}
            />
          )}
          <span className="relative z-10">
            {tab.icon && <span className="mr-1">{tab.icon}</span>}
            {tab.label}
          </span>
        </button>
      ))}
    </div>
  );
}
