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

const navLinkBase =
  'inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[13px] font-medium transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface-base';

export function Layout() {
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);

  const pathActive = (path: string) =>
    path === '/'
      ? location.pathname === '/'
      : location.pathname === path || location.pathname.startsWith(`${path}/`);

  const closeMobile = () => setMobileOpen(false);

  return (
    <div className="flex min-h-screen flex-col bg-surface-base">
      {/* ── Top navigation ── */}
      <nav className="fixed inset-x-0 top-0 z-50 border-b border-border-subtle bg-surface-base/95 backdrop-blur-sm">
        <div className="mx-auto flex h-12 max-w-[1440px] items-center justify-between px-4 sm:px-6">
          {/* Logo */}
          <Link
            to="/"
            onClick={closeMobile}
            className="flex items-center gap-2.5 rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface-base"
          >
            <div className="flex h-7 w-7 items-center justify-center rounded-md bg-accent text-[11px] font-bold text-text-inverse">
              C2
            </div>
            <span className="hidden text-sm font-semibold tracking-tight text-text-primary sm:inline">
              Cortex2Canvas
            </span>
          </Link>

          {/* Desktop nav links */}
          <div className="hidden items-center gap-0.5 md:flex">
            {navItems.map(({ path, label, Icon }) => (
              <Link
                key={path}
                to={path}
                className={`${navLinkBase} ${
                  pathActive(path)
                    ? 'bg-surface-active text-text-primary'
                    : 'text-text-secondary hover:bg-surface-raised hover:text-text-primary'
                }`}
              >
                <Icon className="h-4 w-4 shrink-0 opacity-80" aria-hidden />
                {label}
              </Link>
            ))}
          </div>

          {/* Mobile menu toggle */}
          <button
            type="button"
            className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-[13px] font-medium transition-colors md:hidden focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface-base ${
              mobileOpen
                ? 'border-accent/30 bg-accent/10 text-text-primary'
                : 'border-border-subtle bg-surface-raised text-text-secondary'
            }`}
            aria-expanded={mobileOpen}
            aria-controls="mobile-primary-nav"
            onClick={() => setMobileOpen((o) => !o)}
          >
            Menu
            <motion.span
              animate={{ rotate: mobileOpen ? 180 : 0 }}
              transition={{ duration: 0.2 }}
            >
              <IconChevronDown className="h-4 w-4 text-accent" aria-hidden />
            </motion.span>
          </button>
        </div>

        {/* Mobile dropdown */}
        <AnimatePresence initial={false}>
          {mobileOpen ? (
            <motion.div
              id="mobile-primary-nav"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
              className="overflow-hidden border-t border-border-subtle md:hidden"
            >
              <div className="flex flex-col gap-0.5 px-3 py-2">
                {navItems.map(({ path, label, Icon }) => (
                  <Link
                    key={path}
                    to={path}
                    onClick={closeMobile}
                    className={`${navLinkBase} ${
                      pathActive(path)
                        ? 'bg-surface-active text-text-primary'
                        : 'text-text-secondary hover:bg-surface-raised hover:text-text-primary'
                    }`}
                  >
                    <Icon className="h-4 w-4 shrink-0 opacity-80" aria-hidden />
                    {label}
                  </Link>
                ))}
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </nav>

      {/* ── Page content ── */}
      <main className="flex-1 pt-12">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
          >
            <Outlet />
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
