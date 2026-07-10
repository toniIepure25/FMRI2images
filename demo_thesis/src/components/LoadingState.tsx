import { motion } from 'framer-motion';

export function LoadingState({ message = 'Loading neural data...' }: { message?: string }) {
  return (
    <div className="premium-page-bg flex min-h-[60vh] flex-col items-center justify-center px-6 py-20">
      <div className="relative mb-6 h-16 w-16">
        <motion.div
          className="absolute inset-0 rounded-full border border-accent/25"
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
        />
        <motion.div
          className="absolute inset-2 rounded-full border border-t-accent border-r-transparent border-b-transparent border-l-transparent"
          animate={{ rotate: -360 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
        />
        <div className="absolute inset-5 rounded-full bg-accent/10" />
      </div>
      <p className="text-sm font-medium text-text-secondary">{message}</p>
      <p className="mt-2 max-w-sm text-center text-xs text-text-muted">Preparing experiment artifacts.</p>
    </div>
  );
}
