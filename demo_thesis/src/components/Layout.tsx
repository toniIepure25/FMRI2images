import { useState } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import {
  IconChallenge,
  IconChevronDown,
  IconExplore,
  IconFilm,
  IconHome,
  IconPipeline,
} from '@/components/ui/Icon';

const navItems = [
  { path: '/', label: 'Home', Icon: IconHome },
  { path: '/film', label: 'Film Mode', Icon: IconFilm },
  { path: '/pipeline', label: 'Pipeline', Icon: IconPipeline },
  { path: '/explorer', label: 'Explorer', Icon: IconExplore },
  { path: '/challenge', label: 'Challenge', Icon: IconChallenge },
] as const;

const navLinkClass =
  'inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brain-accent focus-visible:ring-offset-2 focus-visible:ring-offset-brain-dark';

export function Layout() {
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);

  const pathActive = (path: string) =>
    path === '/' ? location.pathname === '/' : location.pathname === path || location.pathname.startsWith(`${path}/`);

  const closeMobile = () => setMobileOpen(false);

  return (
    <div className="flex min-h-screen flex-col bg-brain-dark">
      <nav className="fixed inset-x-0 top-0 z-50 glass-panel rounded-none border-x-0 border-t-0">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4">
          <Link
            to="/"
            onClick={closeMobile}
            className="group flex items-center gap-2 rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brain-accent focus-visible:ring-offset-2 focus-visible:ring-offset-brain-dark"
          >
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-brain-accent to-brain-purple text-xs font-bold text-white shadow-md shadow-brain-accent/15">
              C2C
            </div>
            <span className="text-sm font-semibold text-slate-300 transition-colors group-hover:text-white">
              Cortex2Canvas
            </span>
          </Link>

          <div className="hidden items-center gap-1 md:flex">
            {navItems.map(({ path, label, Icon }) => (
              <Link
                key={path}
                to={path}
                className={`${navLinkClass} ${
                  pathActive(path)
                    ? 'border border-brain-accent/25 bg-brain-accent/10 text-brain-accent'
                    : 'border border-transparent text-slate-400 hover:bg-white/5 hover:text-slate-100'
                }`}
              >
                <Icon className="h-4 w-4 shrink-0 opacity-90" aria-hidden />
                {label}
              </Link>
            ))}
          </div>

          <button
            type="button"
            className={`inline-flex items-center gap-2 rounded-lg border px-3 py-2 text-sm font-medium text-slate-200 transition-colors md:hidden focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brain-accent focus-visible:ring-offset-2 focus-visible:ring-offset-brain-dark ${
              mobileOpen ? 'border-brain-accent/35 bg-brain-accent/10 text-white' : 'border-brain-border/55 bg-white/[0.04]'
            }`}
            aria-expanded={mobileOpen}
            aria-controls="mobile-primary-nav"
            onClick={() => setMobileOpen((o) => !o)}
          >
            Menu
            <motion.span animate={{ rotate: mobileOpen ? 180 : 0 }} transition={{ duration: 0.22 }}>
              <IconChevronDown className="h-4 w-4 text-brain-accent" aria-hidden />
            </motion.span>
          </button>
        </div>

        <AnimatePresence initial={false}>
          {mobileOpen ? (
            <motion.div
              id="mobile-primary-nav"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
              className="overflow-hidden border-t border-brain-border/35 md:hidden"
            >
              <div className="flex flex-col gap-1 px-4 py-3">
                {navItems.map(({ path, label, Icon }) => (
                  <Link
                    key={path}
                    to={path}
                    onClick={closeMobile}
                    className={`${navLinkClass} ${
                      pathActive(path)
                        ? 'border border-brain-accent/25 bg-brain-accent/10 text-brain-accent'
                        : 'border border-transparent text-slate-300 hover:bg-white/5 hover:text-white'
                    }`}
                  >
                    <Icon className="h-5 w-5 shrink-0 opacity-90" aria-hidden />
                    {label}
                  </Link>
                ))}
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </nav>

      <main className="flex-1 pt-14">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.26, ease: [0.22, 1, 0.36, 1] }}
          >
            <Outlet />
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
