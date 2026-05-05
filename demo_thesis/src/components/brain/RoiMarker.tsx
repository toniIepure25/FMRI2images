import { useMemo, useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import { Color, Group, Mesh, AdditiveBlending } from 'three';
import { Html } from '@react-three/drei';

export interface RoiMarkerProps {
  name: string;
  fullName: string;
  category: string;
  position: [number, number, number];
  activation: number;
  contribution: number;
  confidence: number;
  agreement: number;
  interpretation: string;
  colorHex: string;
  /** Disagreement in [0, 1]; high values enable independent flicker. */
  delta: number;
  selected: boolean;
  markerIndex: number;
  onSelect?: (name: string) => void;
  onHover?: (name: string | null) => void;
}

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

function hexToColor(hex: string): Color {
  return new Color(hex.startsWith('#') ? hex : `#${hex}`);
}

export function RoiMarker({
  name,
  fullName,
  category,
  position,
  activation,
  contribution,
  confidence,
  agreement,
  interpretation,
  colorHex,
  delta,
  selected,
  markerIndex,
  onSelect,
  onHover,
}: RoiMarkerProps) {
  const groupRef = useRef<Group>(null);
  const coreRef = useRef<Mesh>(null);
  const haloRef = useRef<Mesh>(null);
  const ringRef = useRef<Mesh>(null);
  const outerGlowRef = useRef<Mesh>(null);
  const [hovered, setHovered] = useState(false);

  const baseRadius = 0.035 + clamp01(activation) * 0.038;
  const DISAGREE_THRESHOLD = 0.38;

  const palette = useMemo(() => {
    const c = hexToColor(colorHex);
    const coreBright = c.clone().multiplyScalar(1.35);
    const haloTint = c.clone().lerp(new Color('#ffffff'), 0.25);
    return { c, coreBright, haloTint };
  }, [colorHex]);

  useFrame(({ clock }) => {
    const t = clock.elapsedTime;
    const g = groupRef.current;
    if (!g) return;

    const highDisagreement = delta >= DISAGREE_THRESHOLD;
    const seed = markerIndex * 2.17;

    let pulse = 1;
    let emissiveMul = 1;
    let haloPulse = 0.55 + 0.45 * (0.5 + 0.5 * Math.sin(t * 2.1 + activation * 3));

    if (!highDisagreement) {
      pulse = 1 + 0.09 * Math.sin(t * 1.85 + activation * 2);
      emissiveMul = 0.82 + 0.18 * Math.sin(t * 1.85);
      haloPulse = 0.5 + 0.28 * Math.sin(t * 1.4 + activation * 4);
    } else {
      const f1 = Math.sin(t * 11 + seed * 4);
      const f2 = Math.sin(t * 6.3 + seed * 1.3);
      pulse = 1 + 0.12 * f1 * f2;
      emissiveMul = 0.5 + 0.5 * (0.5 + 0.5 * f1 * f2);
      haloPulse = 0.35 + 0.55 * (0.5 + 0.5 * f1 * f2);
    }

    const hoverBoost = hovered ? 1.14 : 1;
    const selectBoost = selected ? 1.22 : 1;
    const agreementBoost = 0.82 + 0.18 * clamp01(agreement);
    const activationBoost = 0.88 + 0.22 * clamp01(activation);

    g.scale.setScalar(pulse * hoverBoost * selectBoost);

    const core = coreRef.current;
    if (core) {
      const mat = core.material as import('three').MeshStandardMaterial;
      const baseInt = 1.4 + confidence * 1.6;
      mat.emissiveIntensity = baseInt * emissiveMul * agreementBoost * activationBoost;
      mat.emissive.copy(palette.coreBright);
    }

    const halo = haloRef.current;
    if (halo) {
      const mat = halo.material as import('three').MeshBasicMaterial;
      mat.opacity = (0.14 + 0.2 * haloPulse) * activationBoost;
    }

    const outer = outerGlowRef.current;
    if (outer) {
      const mat = outer.material as import('three').MeshBasicMaterial;
      mat.opacity = (0.06 + 0.12 * haloPulse) * agreementBoost;
    }

    const ring = ringRef.current;
    if (ring) {
      ring.rotation.z = t * 0.7 + seed;
      const mat = ring.material as import('three').MeshBasicMaterial;
      mat.opacity = (0.22 + 0.2 * emissiveMul) * (selected ? 1.15 : 1);
    }
  });

  const showTip = hovered || selected;

  return (
    <group ref={groupRef} position={position}>
      <mesh
        ref={outerGlowRef}
        scale={[baseRadius * 3.2, baseRadius * 3.2, baseRadius * 3.2]}
      >
        <sphereGeometry args={[1, 20, 20]} />
        <meshBasicMaterial
          color={palette.c}
          transparent
          opacity={0.1}
          depthWrite={false}
          blending={AdditiveBlending}
          toneMapped={false}
        />
      </mesh>

      <mesh ref={haloRef} scale={[baseRadius * 1.85, baseRadius * 1.85, baseRadius * 1.85]}>
        <sphereGeometry args={[1, 24, 24]} />
        <meshBasicMaterial
          color={palette.haloTint}
          transparent
          opacity={0.2}
          depthWrite={false}
          blending={AdditiveBlending}
          toneMapped={false}
        />
      </mesh>

      <mesh ref={ringRef} rotation={[Math.PI / 2.2, 0.4, 0]}>
        <torusGeometry args={[baseRadius * 1.45, baseRadius * 0.055, 12, 40]} />
        <meshBasicMaterial
          color={palette.coreBright}
          transparent
          opacity={0.32}
          depthWrite={false}
          blending={AdditiveBlending}
          toneMapped={false}
        />
      </mesh>

      <mesh
        ref={coreRef}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
          onHover?.(name);
          document.body.style.cursor = 'pointer';
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          setHovered(false);
          onHover?.(null);
          document.body.style.cursor = 'auto';
        }}
        onClick={(e) => {
          e.stopPropagation();
          onSelect?.(name);
        }}
      >
        <sphereGeometry args={[baseRadius, 20, 20]} />
        <meshStandardMaterial
          color={palette.coreBright}
          emissive={palette.coreBright}
          emissiveIntensity={1.4 + confidence * 1.6}
          roughness={0.22}
          metalness={0.35}
          toneMapped={false}
        />
      </mesh>

      {showTip ? (
        <Html
          center
          position={[0, baseRadius * 3.4, 0]}
          distanceFactor={4}
          style={{ pointerEvents: 'none' }}
          zIndexRange={[100, 0]}
        >
          <div
            className="rounded-lg border border-cyan-500/30 bg-[#0a1628]/95 px-3 py-2 shadow-xl backdrop-blur-md min-w-[200px] max-w-[280px]"
            style={{ boxShadow: `0 0 24px ${colorHex}33` }}
          >
            <p className="text-[11px] font-semibold uppercase tracking-wide text-cyan-200/90">
              {fullName}
            </p>
            <p className="mt-1 text-[10px] text-slate-400">{category}</p>
            <dl className="mt-2 grid grid-cols-2 gap-x-2 gap-y-1 text-[11px] text-slate-200">
              <dt className="text-slate-500">Activation</dt>
              <dd className="font-mono text-cyan-100">{activation.toFixed(3)}</dd>
              <dt className="text-slate-500">Contribution</dt>
              <dd className="font-mono text-cyan-100">{contribution.toFixed(3)}</dd>
              <dt className="text-slate-500">Confidence</dt>
              <dd className="font-mono text-cyan-100">{confidence.toFixed(3)}</dd>
            </dl>
            <p className="mt-2 border-t border-slate-700/80 pt-2 text-[10px] leading-snug text-slate-300">
              {interpretation || '—'}
            </p>
          </div>
        </Html>
      ) : null}
    </group>
  );
}
