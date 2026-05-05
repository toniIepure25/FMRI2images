import { useState } from 'react';
import type { DemoCase } from '@/types';
import { ClipSpaceViewer } from '@/components/ClipSpaceViewer';
import { useClipProjection } from '@/lib/hooks';
import { LoadingState } from '@/components/LoadingState';
import { ErrorState } from '@/components/ErrorState';

export interface ClipTabProps {
  case_: DemoCase;
}

export function ClipTab({ case_ }: ClipTabProps) {
  const { projection, loading, error } = useClipProjection();
  const [mode, setMode] = useState<'2d' | '3d'>('3d');

  if (loading) return <LoadingState />;
  if (error) return <ErrorState message={error} />;

  return (
    <div className="mx-auto max-w-6xl space-y-4">
      <div className="glass-panel p-4">
        <h2 className="text-sm font-semibold text-white">CLIP latent space</h2>
        <p className="mt-2 text-xs leading-relaxed text-slate-400">
          Predicted embedding (cyan), ground-truth target (star), and retrieved neighbors (diamonds) projected with
          the same UMAP used for the full demo gallery. Use 2D for readable local structure; 3D for manifold context.
        </p>
      </div>

      <ClipSpaceViewer
        projection={projection}
        caseClipSpace={case_.clipSpace}
        mode={mode}
        onModeChange={setMode}
        showLines
        className="glass-panel p-4"
      />
    </div>
  );
}
