import { Suspense, useEffect, useMemo, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { Environment, OrbitControls } from '@react-three/drei';
import type { RoiScore, RoiLayout } from '@/types';
import { useRoiLayout } from '@/lib/hooks';
import { BrainMesh, preloadBrainModelIfExists } from '@/components/brain/BrainMesh';
import { RoiMarker } from '@/components/brain/RoiMarker';
import { computeLayoutNormalization, layoutPositionToScene } from '@/components/brain/layoutCoords';
import { resolveRoiColor } from '@/components/brain/roiColors';

export interface BrainViewer3DProps {
  roiScores?: RoiScore[];
  roiLayout?: RoiLayout;
  /** Disagreement level in [0, 1]; drives synchronized pulse vs. independent flicker. */
  delta?: number;
  selectedRoi?: string | null;
  onRoiSelect?: (name: string | null) => void;
  onRoiHover?: (name: string | null) => void;
  autoRotate?: boolean;
  className?: string;
}

function SceneContent({
  roiScores,
  layout,
  delta,
  selectedRoi,
  onRoiSelect,
  onRoiHover,
  autoRotate = true,
}: {
  roiScores: RoiScore[] | undefined;
  layout: RoiLayout;
  delta: number;
  selectedRoi: string | null | undefined;
  onRoiSelect?: (name: string | null) => void;
  onRoiHover?: (name: string | null) => void;
  autoRotate: boolean;
}) {
  const [userInteracting, setUserInteracting] = useState(false);
  const norm = useMemo(() => computeLayoutNormalization(layout.rois), [layout.rois]);
  const roiScenePositions = useMemo(
    () => layout.rois.map((def) => layoutPositionToScene(def.position, norm)),
    [layout.rois, norm],
  );
  const scoreByName = useMemo(() => {
    const m = new Map<string, RoiScore>();
    for (const s of roiScores ?? []) m.set(s.name, s);
    return m;
  }, [roiScores]);

  return (
    <>
      <color attach="background" args={['#020617']} />
      <fog attach="fog" args={['#020617', 1.8, 8.5]} />
      <Environment preset="city" environmentIntensity={0.18} />
      <ambientLight intensity={0.18} color="#1e3a5f" />
      <hemisphereLight args={['#0c1929', '#020617', 0.42]} />
      <directionalLight position={[2.4, 3.2, 2]} intensity={0.72} color="#c9d6f0" castShadow />
      <spotLight
        position={[2.2, 2.6, 1.9]}
        angle={0.42}
        penumbra={0.85}
        intensity={0.55}
        color="#a5c4ff"
        distance={14}
        decay={2}
      />
      <pointLight position={[-1.8, 0.6, 1.2]} intensity={0.38} color="#5ad0ff" distance={8} decay={2} />
      <pointLight position={[0.2, -0.4, -1.8]} intensity={0.2} color="#6c4dff" distance={6} decay={2} />

      <BrainMesh roiScenePositions={roiScenePositions} />

      {layout.rois.map((def, markerIndex) => {
        const s = scoreByName.get(def.name);
        const position = layoutPositionToScene(def.position, norm);
        const colorHex = resolveRoiColor(def.category, layout, def.color);
        return (
          <RoiMarker
            key={def.name}
            name={def.name}
            fullName={def.fullName}
            category={def.category}
            position={position}
            activation={s?.activation ?? 0.45}
            contribution={s?.contribution ?? 0.45}
            confidence={s?.confidence ?? 0.5}
            agreement={s?.agreement ?? 0.5}
            interpretation={s?.interpretation ?? def.functionSummary ?? def.description}
            colorHex={colorHex}
            delta={delta}
            selected={selectedRoi === def.name}
            markerIndex={markerIndex}
            onSelect={(name) => {
              onRoiSelect?.(selectedRoi === name ? null : name);
            }}
            onHover={onRoiHover}
          />
        );
      })}

      <OrbitControls
        enablePan
        enableDamping
        dampingFactor={0.06}
        minDistance={1.15}
        maxDistance={4.2}
        maxPolarAngle={Math.PI * 0.92}
        autoRotate={autoRotate && !userInteracting}
        autoRotateSpeed={0.35}
        onStart={() => setUserInteracting(true)}
        onEnd={() => {
          window.setTimeout(() => setUserInteracting(false), 2200);
        }}
      />
    </>
  );
}

export function BrainViewer3D({
  roiScores,
  roiLayout: layoutProp,
  delta = 0,
  selectedRoi = null,
  onRoiSelect,
  onRoiHover,
  autoRotate = true,
  className = '',
}: BrainViewer3DProps) {
  const { layout: layoutFromHook, loading: hookLoading } = useRoiLayout();
  const effectiveLayout = layoutProp ?? layoutFromHook;
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    preloadBrainModelIfExists();
  }, []);

  const waitingForLayout = !layoutProp && hookLoading && !layoutFromHook;

  const pinned = useMemo(() => {
    if (!selectedRoi || !effectiveLayout) return null;
    const def = effectiveLayout.rois.find((r) => r.name === selectedRoi);
    const score = roiScores?.find((r) => r.name === selectedRoi);
    if (!def && !score) return null;
    return { def, score };
  }, [selectedRoi, effectiveLayout, roiScores]);

  return (
    <div
      className={`relative flex flex-col rounded-xl border border-slate-700/60 bg-gradient-to-b from-[#050b14] to-[#020617] overflow-hidden ${className}`}
      style={{ minHeight: 420 }}
    >
      <div className="relative w-full flex-1" style={{ height: 440 }}>
        {!mounted || waitingForLayout ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-slate-500">
            <div className="h-10 w-10 animate-pulse rounded-full border-2 border-cyan-500/40 border-t-cyan-400" />
            <p className="text-xs tracking-wide">
              {waitingForLayout ? 'Loading brain layout…' : 'Initializing viewer…'}
            </p>
          </div>
        ) : !effectiveLayout ? (
          <div className="absolute inset-0 flex items-center justify-center px-6 text-center text-sm text-slate-500">
            ROI layout is not available. Check that <code className="text-slate-400">/data/roi_layout.json</code>{' '}
            is deployed.
          </div>
        ) : (
          <Canvas
            camera={{ position: [0, 0.12, 1.82], fov: 42, near: 0.08, far: 40 }}
            gl={{
              alpha: false,
              antialias: true,
              powerPreference: 'high-performance',
            }}
            dpr={[1, 2]}
            className="!absolute inset-0 h-full w-full touch-none"
          >
            <Suspense fallback={null}>
              <SceneContent
                roiScores={roiScores}
                layout={effectiveLayout}
                delta={Math.min(1, Math.max(0, delta))}
                selectedRoi={selectedRoi}
                onRoiSelect={onRoiSelect}
                onRoiHover={onRoiHover}
                autoRotate={autoRotate}
              />
            </Suspense>
          </Canvas>
        )}
      </div>

      {pinned ? (
        <div className="relative z-10 border-t border-slate-700/70 bg-[#0a1628]/85 px-4 py-3 backdrop-blur-md">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-wide text-cyan-300/90">
                {pinned.def?.fullName ?? pinned.score?.name ?? selectedRoi}
              </p>
              <p className="mt-1 text-[11px] text-slate-500">
                {pinned.def?.category && <span className="mr-2">Category: {pinned.def.category}</span>}
                {pinned.score?.hemisphere && <span>Hemisphere: {pinned.score.hemisphere}</span>}
              </p>
              <p className="mt-2 text-xs leading-relaxed text-slate-300">
                {pinned.score?.interpretation ??
                  pinned.def?.functionSummary ??
                  pinned.def?.description ??
                  '—'}
              </p>
            </div>
            <button
              type="button"
              onClick={() => onRoiSelect?.(null)}
              className="shrink-0 rounded-md border border-slate-600/80 bg-slate-900/60 px-2 py-1 text-[10px] text-slate-400 transition hover:border-slate-500 hover:text-slate-200"
            >
              Unpin
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
