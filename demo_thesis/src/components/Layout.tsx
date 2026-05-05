import { Outlet, Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';

const navItems = [
  { path: '/', label: 'Home', icon: '◈' },
  { path: '/film', label: 'Film Mode', icon: '▶' },
  { path: '/pipeline', label: 'Pipeline', icon: '⚡' },
  { path: '/explorer', label: 'Explorer', icon: '◉' },
  { path: '/challenge', label: 'Challenge', icon: '?' },
];

export function Layout() {
  const location = useLocation();
  const isHome = location.pathname === '/';

  return (
    <div className="min-h-screen flex flex-col">
      {!isHome && (
        <nav className="fixed top-0 left-0 right-0 z-50 glass-panel border-t-0 rounded-t-none border-x-0">
          <div className="max-w-7xl mx-auto px-4 h-14 flex items-center justify-between">
            <Link to="/" className="flex items-center gap-2 group">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brain-accent to-brain-purple flex items-center justify-center text-sm font-bold">
                C2C
              </div>
              <span className="font-semibold text-sm text-gray-300 group-hover:text-white transition-colors">
                Cortex2Canvas
              </span>
            </Link>
            <div className="flex items-center gap-1">
              {navItems.map(item => (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                    location.pathname === item.path || (item.path !== '/' && location.pathname.startsWith(item.path))
                      ? 'bg-brain-accent/10 text-brain-accent border border-brain-accent/20'
                      : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
                  }`}
                >
                  <span className="mr-1.5">{item.icon}</span>
                  {item.label}
                </Link>
              ))}
            </div>
          </div>
        </nav>
      )}
      <main className={!isHome ? 'pt-14' : ''}>
        <motion.div
          key={location.pathname}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.3 }}
        >
          <Outlet />
        </motion.div>
      </main>
    </div>
  );
}
