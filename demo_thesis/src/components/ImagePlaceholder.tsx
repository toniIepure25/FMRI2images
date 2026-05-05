import { useState } from 'react';

interface ImagePlaceholderProps {
  src: string;
  alt: string;
  className?: string;
  type?: 'target' | 'reconstruction' | 'retrieved' | 'ensemble';
  label?: string;
}

const gradients: Record<string, string> = {
  target: 'from-emerald-900/40 to-cyan-900/40',
  reconstruction: 'from-purple-900/40 to-pink-900/40',
  retrieved: 'from-blue-900/40 to-indigo-900/40',
  ensemble: 'from-amber-900/40 to-orange-900/40',
};

const borderColors: Record<string, string> = {
  target: 'border-emerald-500/20',
  reconstruction: 'border-purple-500/20',
  retrieved: 'border-blue-500/20',
  ensemble: 'border-amber-500/20',
};

export function ImagePlaceholder({ src, alt, className = '', type = 'target', label }: ImagePlaceholderProps) {
  const [failed, setFailed] = useState(false);

  return (
    <div className={`relative overflow-hidden rounded-lg ${borderColors[type]} border ${className}`}>
      {!failed ? (
        <img
          src={src}
          alt={alt}
          className="w-full h-full object-cover"
          onError={() => setFailed(true)}
        />
      ) : (
        <div
          className={`placeholder-inner w-full min-h-[120px] h-full bg-gradient-to-br ${gradients[type]} flex items-center justify-center`}
        >
          <div className="text-center p-4">
            <div className="text-2xl mb-2 opacity-50" aria-hidden>
              🧠
            </div>
            <div className="text-xs text-gray-500">{alt}</div>
          </div>
        </div>
      )}
      {label && (
        <div className="absolute bottom-0 left-0 right-0 bg-black/60 backdrop-blur-sm px-2 py-1">
          <span className="text-[10px] font-medium text-gray-300">{label}</span>
        </div>
      )}
    </div>
  );
}
