import { motion } from 'framer-motion';

export function LoadingState({ message = 'Loading neural data...' }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <div className="relative w-16 h-16 mb-6">
        <motion.div
          className="absolute inset-0 rounded-full border-2 border-brain-accent/30"
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
        />
        <motion.div
          className="absolute inset-2 rounded-full border-2 border-t-brain-purple border-r-transparent border-b-transparent border-l-transparent"
          animate={{ rotate: -360 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
        />
        <div className="absolute inset-4 rounded-full bg-brain-accent/10" />
      </div>
      <p className="text-sm text-gray-400 animate-pulse">{message}</p>
    </div>
  );
}
