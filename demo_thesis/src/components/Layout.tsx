import { useState } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import {
  IconAtom,
  IconChevronDown,
  IconExplore,
  IconHome,
  IconPipeline,
  IconChart,
} from '@/components/ui/Icon';

const navItems = [
  { path: '/', label: 'Home', detail: 'Overview', Icon: IconHome },
  { path: '/pipeline', label: 'Pipeline', detail: 'Decode', Icon: IconPipeline },
  { path: '/explorer', label: 'Explorer', detail: 'Trial inspector', Icon: IconExplore },
  { path: '/manifold', label: 'Manifold', detail: 'Semantic lab', Icon: IconAtom },
  { path: '/evidence', label: 'Evidence', detail: 'Audit trail', Icon: IconChart },
] as const;

export function Layout() {
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);

  const pathActive = (path: string) =>
    path === '/'
      ? location.pathname === '/'
      : location.pathname === path || location.pathname.startsWith(`${path}/`);

  const closeMobile = () => setMobileOpen(false);

  return (
    <div className="premium-page-bg relative flex min-h-screen flex-col bg-surface-base">
      <div className="premium-shell-grid pointer-events-none fixed inset-x-0 top-0 h-[38rem]" aria-hidden />
      {/* ── Top navigation ── */}
      <nav className="sticky top-0 z-50 border-b border-white/[0.07] bg-[#07080c]/88 backdrop-blur-xl">
        <div className="mx-auto flex h-[68px] max-w-[1540px] items-center justify-between px-4 sm:px-6 lg:px-8">
          {/* Logo */}
          <Link
            to="/"
            onClick={closeMobile}
            className="group flex items-center gap-3 rounded-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface-base"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-accent/22 bg-accent/10 text-[12px] font-bold text-accent shadow-[inset_0_1px_0_rgba(255,255,255,0.08)] transition group-hover:border-accent/40">
              C2
            </div>
            <div className="hidden sm:block">
              <span className="block text-[15px] font-semibold tracking-tight text-text-primary">Cortex2Canvas</span>
              <span className="block text-[10px] uppercase tracking-[0.16em] text-text-muted">Neural decoding workbench</span>
            </div>
          </Link>

          {/* Desktop nav links */}
          <div className="hidden items-center gap-1 md:flex">
            {navItems.map(({ path, label, detail, Icon }) => (
              <Link
                key={path}
                to={path}
                className={`premium-nav-link ${pathActive(path) ? 'premium-nav-link-active' : ''}`}
              >
                <Icon className="h-4 w-4 shrink-0 opacity-80" aria-hidden />
                <span className="leading-none">
                  <span className="block">{label}</span>
                  <span className="mt-0.5 hidden text-[10px] font-medium text-text-muted lg:block">{detail}</span>
                </span>
              </Link>
            ))}
          </div>

          <div className="hidden items-center gap-2 lg:flex">
            <span className="rounded-full border border-border-subtle bg-surface-raised/75 px-3.5 py-1.5 text-[11px] font-semibold text-text-secondary">
              Triple Fusion · ViT-L/14
            </span>
          </div>

          {/* Mobile menu toggle */}
          <button
            type="button"
            className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-2 text-[13px] font-medium transition-colors md:hidden focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface-base ${
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
              className="overflow-hidden border-t border-border-subtle bg-[#07080c]/95 md:hidden"
            >
              <div className="flex flex-col gap-0.5 px-3 py-2">
                {navItems.map(({ path, label, Icon }) => (
                  <Link
                    key={path}
                    to={path}
                    onClick={closeMobile}
                    className={`premium-nav-link ${pathActive(path) ? 'premium-nav-link-active' : ''}`}
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
      <main className="relative z-10 flex flex-1 flex-col">
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
