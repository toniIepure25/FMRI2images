import {
  Component,
  type ErrorInfo,
  type ReactNode,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { useFrame } from '@react-three/fiber';
import { useGLTF } from '@react-three/drei';
import {
  AdditiveBlending,
  BackSide,
  Box3,
  BufferAttribute,
  BufferGeometry,
  CatmullRomCurve3,
  DoubleSide,
  Float32BufferAttribute,
  FrontSide,
  Group,
  IcosahedronGeometry,
  Material,
  Mesh,
  MeshPhysicalMaterial,
  Object3D,
  Points,
  TubeGeometry,
  Vector3,
} from 'three';

/** Public URL for optional whole-brain GLB (place file under `public/assets/brain/brain.glb`). */
export const BRAIN_GLB_URL = '/assets/brain/brain.glb';

/**
 * Cached probe so we don't double-fetch and optional preload lines up with the same result.
 */
let glbAvailabilityPromise: Promise<boolean> | null = null;

function getGlbAvailability(): Promise<boolean> {
  if (!glbAvailabilityPromise) {
    glbAvailabilityPromise = fetch(BRAIN_GLB_URL, { method: 'HEAD' })
      .then((r) => r.ok)
      .catch(() => false);
  }
  return glbAvailabilityPromise;
}

/** Fire-and-forget preload when the asset is present (called from the viewer on mount). */
export function preloadBrainModelIfExists(): void {
  void getGlbAvailability().then((ok) => {
    if (ok) {
      useGLTF.preload(BRAIN_GLB_URL);
    }
  });
}

/** Lateral (X), vertical (Y), anterior–posterior (Z): wider than tall, longer than wide. */
const BRAIN_SCALE: [number, number, number] = [0.48, 0.32, 0.54];
const HEMI_X_OFFSET = 0.065;
/** Detail 6 ≳ 40k verts / full sphere — good sulcal detail for hero shots. */
const ICOS_DETAIL = 6;

/** Three.js Y-up fix if a given GLB arrives Z-up (toggle if a new mesh lies on its side). */
const GLTF_EXTRA_ROTATION: [number, number, number] = [0, 0, 0];

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/** Layered oscillators — smooth multi-scale “cortical” undulation without external noise. */
function cortexNoise(px: number, py: number, pz: number): number {
  let n = 0;
  let amp = 1;
  let freq = 1;
  for (let o = 0; o < 6; o++) {
    const sx = freq * px;
    const sy = freq * py;
    const sz = freq * pz;
    n +=
      amp *
      Math.sin(sx + 2.0 * Math.sin(0.37 * sy + o)) *
      Math.cos(sy + 1.12 * Math.sin(0.28 * sz + o * 0.7)) *
      Math.sin(sz + 1.9 * Math.cos(0.21 * sx + o * 0.4));
    amp *= 0.52;
    freq *= 2.03;
  }
  return n;
}

function extractHemisphere(source: BufferGeometry, side: 'left' | 'right'): BufferGeometry {
  const pos = source.attributes.position;
  const idx = source.getIndex();
  if (!idx) {
    return source.clone();
  }
  const newPositions: number[] = [];
  const indexRemap = new Map<number, number>();
  const newIndices: number[] = [];
  const eps = 0.018;

  const vertexOk = (x: number) => (side === 'left' ? x <= eps : x >= -eps);

  const mapVert = (vi: number): number => {
    const existing = indexRemap.get(vi);
    if (existing !== undefined) return existing;
    const x = pos.getX(vi);
    const y = pos.getY(vi);
    const z = pos.getZ(vi);
    const ni = (newPositions.length / 3) | 0;
    newPositions.push(x, y, z);
    indexRemap.set(vi, ni);
    return ni;
  };

  for (let t = 0; t < idx.count; t += 3) {
    const i0 = idx.getX(t);
    const i1 = idx.getX(t + 1);
    const i2 = idx.getX(t + 2);
    const x0 = pos.getX(i0);
    const x1 = pos.getX(i1);
    const x2 = pos.getX(i2);
    if (vertexOk(x0) && vertexOk(x1) && vertexOk(x2)) {
      newIndices.push(mapVert(i0), mapVert(i1), mapVert(i2));
    }
  }

  const geom = new BufferGeometry();
  geom.setAttribute('position', new Float32BufferAttribute(newPositions, 3));
  geom.setIndex(newIndices);
  geom.computeVertexNormals();
  return geom;
}

function displaceHemisphere(geo: BufferGeometry, side: 'left' | 'right'): void {
  const attr = geo.attributes.position as BufferAttribute;
  const tmp = new Vector3();
  for (let i = 0; i < attr.count; i++) {
    tmp.set(attr.getX(i), attr.getY(i), attr.getZ(i));
    const len = tmp.length();
    if (len < 1e-7) continue;
    const dir = tmp.clone().divideScalar(len);
    const { x, y, z } = tmp;
    const azimuth = Math.atan2(z, x);
    const elev = Math.asin(clamp(y / len, -1, 1));

    let disp = 0.048 * cortexNoise(dir.x * 3.05, dir.y * 3.05, dir.z * 3.05);
    disp += 0.026 * Math.sin(6.7 * azimuth + 3.4 * elev);
    disp += 0.021 * Math.cos(5.9 * elev + 2.35 * azimuth);
    disp += 0.016 * Math.sin(3.05 * elev) * Math.cos(4.6 * azimuth);

    const distMid = side === 'left' ? -clamp(x, -1, 0) : clamp(x, 0, 1);
    disp -= 0.095 * Math.exp(-((distMid / 0.22) ** 2));

    const sylLat = Math.abs(azimuth);
    const sylvianEnvelope =
      Math.exp(-((elev + 0.32) ** 2) / 0.045) *
      Math.exp(-((sylLat - 1.05) ** 2) / 0.55);
    disp -= 0.062 * sylvianEnvelope;

    const temporoInf = Math.exp(-((elev + 0.55) ** 2) / 0.09) * (0.55 + 0.45 * Math.abs(Math.sin(2.1 * azimuth)));
    disp += 0.028 * temporoInf;

    const frontalX = side === 'left' ? -0.62 : 0.62;
    const frontalBump = Math.exp(-(((x - frontalX) / 0.42) ** 2 + ((z - 0.56) / 0.5) ** 2 + ((y - 0.08) / 0.55) ** 2));
    disp += 0.036 * frontalBump;

    const parietalBump = Math.exp(-(((z + 0.18) / 0.44) ** 2 + ((Math.abs(y) - 0.22) ** 2) / 0.1));
    disp += 0.022 * parietalBump;

    const occipitalCue = Math.exp(-(((z + 0.72) / 0.38) ** 2 + (y ** 2) / 0.28));
    disp += 0.03 * occipitalCue;

    tmp.copy(dir).multiplyScalar(len + disp);
    attr.setXYZ(i, tmp.x, tmp.y, tmp.z);
  }
  attr.needsUpdate = true;
  geo.computeVertexNormals();
}

function mirrorGeometryX(geo: BufferGeometry): BufferGeometry {
  const g = geo.clone();
  const attr = g.attributes.position as BufferAttribute;
  for (let i = 0; i < attr.count; i++) {
    attr.setX(i, -attr.getX(i));
  }
  attr.needsUpdate = true;
  const ix = g.getIndex();
  if (ix) {
    for (let t = 0; t < ix.count; t += 3) {
      const a = ix.getX(t);
      const b = ix.getX(t + 1);
      const c = ix.getX(t + 2);
      ix.setX(t + 1, c);
      ix.setX(t + 2, b);
    }
    ix.needsUpdate = true;
  }
  g.computeVertexNormals();
  return g;
}

function scalePositions(geo: BufferGeometry, sx: number, sy: number, sz: number): void {
  const attr = geo.attributes.position as BufferAttribute;
  for (let i = 0; i < attr.count; i++) {
    attr.setXYZ(i, attr.getX(i) * sx, attr.getY(i) * sy, attr.getZ(i) * sz);
  }
  attr.needsUpdate = true;
  geo.computeVertexNormals();
}

function buildInnerShell(outer: BufferGeometry, factor: number): BufferGeometry {
  const g = outer.clone();
  const attr = g.attributes.position as BufferAttribute;
  const c = new Vector3();
  for (let i = 0; i < attr.count; i++) {
    c.set(attr.getX(i), attr.getY(i), attr.getZ(i));
    c.multiplyScalar(factor);
    attr.setXYZ(i, c.x, c.y, c.z);
  }
  attr.needsUpdate = true;
  g.computeVertexNormals();
  return g;
}

export interface BrainMeshProps {
  roiScenePositions?: Array<[number, number, number]>;
}

class BrainLoadErrorBoundary extends Component<
  { children: ReactNode; fallback: ReactNode },
  { hasError: boolean }
> {
  constructor(props: { children: ReactNode; fallback: ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): { hasError: boolean } {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.warn('[BrainMesh] GLB could not be rendered; using procedural brain.', error.message, info.componentStack);
  }

  render(): ReactNode {
    if (this.state.hasError) return this.props.fallback;
    return this.props.children;
  }
}

function NeuralFibers({ points }: { points: Array<[number, number, number]> }) {
  const groupRef = useRef<Group>(null);

  const segments = useMemo(() => {
    const list: { geometry: TubeGeometry; key: string }[] = [];
    const n = points.length;
    if (n < 2) return list;

    const maxEdges = 8;
    const stride = Math.max(1, Math.floor(n / 3));
    const edges: [number, number][] = [];
    const seen = new Set<string>();
    for (let i = 0; i < n && edges.length < maxEdges; i++) {
      const j = (i + stride) % n;
      if (i === j) continue;
      const lo = Math.min(i, j);
      const hi = Math.max(i, j);
      const key = `${lo}-${hi}`;
      if (seen.has(key)) continue;
      seen.add(key);
      edges.push([lo, hi]);
    }

    for (const [i, j] of edges) {
      if (list.length >= maxEdges) break;
      const a = new Vector3(...points[i]);
      const b = new Vector3(...points[j]);
      const mid = new Vector3().addVectors(a, b).multiplyScalar(0.5);
      const outward = mid.clone().normalize().multiplyScalar(0.14);
      outward.y += 0.06;
      mid.add(outward);
      const curve = new CatmullRomCurve3([a, mid, b]);
      const geo = new TubeGeometry(curve, 36, 0.0045, 6, false);
      list.push({ geometry: geo, key: `${i}-${j}` });
    }
    return list;
  }, [points]);

  useFrame(({ clock }) => {
    const pulse = 0.42 + 0.28 * (0.5 + 0.5 * Math.sin(clock.elapsedTime * 1.25));
    const g = groupRef.current;
    if (!g) return;
    for (const ch of g.children) {
      const mesh = ch as Mesh;
      const mat = mesh.material as { opacity?: number } | undefined;
      if (mat && typeof mat.opacity === 'number') mat.opacity = pulse;
    }
  });

  if (segments.length === 0) return null;

  return (
    <group ref={groupRef}>
      {segments.map(({ geometry, key }) => (
        <mesh key={key} geometry={geometry}>
          <meshBasicMaterial
            color="#5ad0ff"
            transparent
            opacity={0.55}
            depthWrite={false}
            blending={AdditiveBlending}
            toneMapped={false}
          />
        </mesh>
      ))}
    </group>
  );
}

function CortexParticles({ count = 420 }: { count?: number }) {
  const pointsRef = useRef<Points>(null);

  const geometry = useMemo(() => {
    const geo = new BufferGeometry();
    const positions = new Float32Array(count * 3);
    const s = BRAIN_SCALE;
    for (let i = 0; i < count; i++) {
      const u = Math.random() * 2 - 1;
      const v = Math.random() * 2 - 1;
      const w = Math.random() * 2 - 1;
      const len = Math.cbrt(Math.abs(u * v * w)) || 0.3;
      positions[i * 3] = u * s[0] * 0.72 * len;
      positions[i * 3 + 1] = v * s[1] * 0.68 * len;
      positions[i * 3 + 2] = w * s[2] * 0.72 * len;
    }
    geo.setAttribute('position', new BufferAttribute(positions, 3));
    return geo;
  }, [count]);

  useFrame(({ clock }) => {
    const pts = pointsRef.current;
    if (!pts) return;
    const t = clock.elapsedTime;
    pts.rotation.y = t * 0.018;
    const mat = pts.material as import('three').PointsMaterial;
    if (mat?.opacity !== undefined) {
      mat.opacity = 0.12 + 0.06 * Math.sin(t * 0.7);
    }
  });

  return (
    <points ref={pointsRef} geometry={geometry}>
      <pointsMaterial
        color="#7dd3fc"
        size={0.012}
        transparent
        opacity={0.15}
        depthWrite={false}
        blending={AdditiveBlending}
        sizeAttenuation
        toneMapped={false}
      />
    </points>
  );
}

function HemisphereBlock({
  outerGeo,
  innerGeo,
  position,
  rotationY = 0,
}: {
  outerGeo: BufferGeometry;
  innerGeo: BufferGeometry;
  position: [number, number, number];
  rotationY?: number;
}) {
  return (
    <group position={position} rotation={[0, rotationY, 0]}>
      <pointLight position={[0, 0.06, 0]} intensity={0.65} color="#38bdf8" distance={0.95} decay={2} />
      <mesh geometry={outerGeo} castShadow receiveShadow>
        <meshPhysicalMaterial
          color="#1a4a6a"
          emissive="#0c3a55"
          emissiveIntensity={0.55}
          transparent
          opacity={0.5}
          roughness={0.32}
          metalness={0.03}
          transmission={0.38}
          thickness={1.1}
          ior={1.14}
          clearcoat={0.8}
          clearcoatRoughness={0.3}
          attenuationColor="#22d3ee"
          attenuationDistance={0.75}
          side={DoubleSide}
          depthWrite
        />
      </mesh>
      <mesh geometry={outerGeo}>
        <meshBasicMaterial
          color="#67d4ff"
          wireframe
          transparent
          opacity={0.05}
          depthWrite={false}
        />
      </mesh>
      <mesh geometry={innerGeo}>
        <meshStandardMaterial
          color="#061a2a"
          emissive="#0c4a6e"
          emissiveIntensity={0.55}
          transparent
          opacity={0.14}
          roughness={0.9}
          metalness={0}
          side={FrontSide}
          depthWrite={false}
        />
      </mesh>
      <mesh geometry={innerGeo} scale={[1.012, 1.012, 1.012]}>
        <meshBasicMaterial
          color="#22d3ee"
          transparent
          opacity={0.06}
          side={BackSide}
          depthWrite={false}
          blending={AdditiveBlending}
        />
      </mesh>
    </group>
  );
}

function ProceduralBrain({ roiScenePositions }: BrainMeshProps) {
  const { leftOuter, leftInner, rightOuter, rightInner } = useMemo(() => {
    const base = new IcosahedronGeometry(1, ICOS_DETAIL);
    const leftRaw = extractHemisphere(base, 'left');
    displaceHemisphere(leftRaw, 'left');
    const leftScaled = leftRaw.clone();
    scalePositions(leftScaled, BRAIN_SCALE[0], BRAIN_SCALE[1], BRAIN_SCALE[2]);
    const leftIn = buildInnerShell(leftScaled, 0.9);

    const rightRaw = mirrorGeometryX(leftRaw);
    const rightScaled = rightRaw.clone();
    scalePositions(rightScaled, BRAIN_SCALE[0], BRAIN_SCALE[1], BRAIN_SCALE[2]);
    const rightIn = buildInnerShell(rightScaled, 0.9);

    base.dispose();
    leftRaw.dispose();
    rightRaw.dispose();

    return {
      leftOuter: leftScaled,
      leftInner: leftIn,
      rightOuter: rightScaled,
      rightInner: rightIn,
    };
  }, []);

  const fiberPoints = useMemo(() => {
    const pts = roiScenePositions?.length ? [...roiScenePositions] : [];
    if (pts.length >= 2) return pts;
    const fallback: Array<[number, number, number]> = [
      [-0.28, 0.08, 0.1],
      [-0.22, -0.05, -0.14],
      [0.26, 0.06, 0.08],
      [0.2, -0.07, -0.12],
      [-0.12, 0.18, -0.04],
      [0.1, 0.15, 0.06],
    ];
    return fallback;
  }, [roiScenePositions]);

  return (
    <group>
      <CortexParticles count={480} />
      <NeuralFibers points={fiberPoints} />
      <HemisphereBlock
        outerGeo={leftOuter}
        innerGeo={leftInner}
        position={[-HEMI_X_OFFSET, 0, 0]}
      />
      <HemisphereBlock
        outerGeo={rightOuter}
        innerGeo={rightInner}
        position={[HEMI_X_OFFSET, 0, 0]}
      />
    </group>
  );
}

function applyBrainMaterial(root: Object3D): void {
  root.traverse((obj) => {
    const mesh = obj as Mesh;
    if (!mesh.isMesh) return;
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    const prev = mesh.material;
    if (Array.isArray(prev)) {
      for (const m of prev) (m as Material).dispose?.();
    } else if (prev) {
      (prev as Material).dispose?.();
    }

    mesh.material = new MeshPhysicalMaterial({
      color: '#1a4a6a',
      emissive: '#0c3a55',
      emissiveIntensity: 0.6,
      transparent: true,
      opacity: 0.55,
      roughness: 0.35,
      metalness: 0.02,
      transmission: 0.35,
      thickness: 1.2,
      ior: 1.12,
      clearcoat: 0.8,
      clearcoatRoughness: 0.35,
      attenuationColor: '#22d3ee',
      attenuationDistance: 0.8,
      side: DoubleSide,
      depthWrite: true,
    });
  });
}

/**
 * Normalize a rigged GLTF scene to ~unit size, center it, and tint meshes with the glass material.
 */
function prepareGltfScene(gltf: { scene: Object3D }): Object3D {
  const root = gltf.scene.clone(true);
  root.updateMatrixWorld(true);
  const box = new Box3().setFromObject(root);
  const size = new Vector3();
  box.getSize(size);
  const center = new Vector3();
  box.getCenter(center);
  root.position.sub(center);
  const maxDim = Math.max(size.x, size.y, size.z, 1e-6);
  const targetDiameter = 1.05;
  root.scale.setScalar(targetDiameter / maxDim);
  applyBrainMaterial(root);
  return root;
}

function BrainLoadingShell({ roiScenePositions }: BrainMeshProps) {
  const fiberPoints = useMemo(() => {
    const pts = roiScenePositions?.length ? [...roiScenePositions] : [];
    if (pts.length >= 2) return pts;
    const fallback: Array<[number, number, number]> = [
      [-0.28, 0.08, 0.1],
      [-0.22, -0.05, -0.14],
      [0.26, 0.06, 0.08],
      [0.2, -0.07, -0.12],
      [-0.12, 0.18, -0.04],
      [0.1, 0.15, 0.06],
    ];
    return fallback;
  }, [roiScenePositions]);

  return (
    <group>
      <CortexParticles count={360} />
      <NeuralFibers points={fiberPoints} />
    </group>
  );
}

function GltfBrain({ roiScenePositions }: BrainMeshProps) {
  const gltf = useGLTF(BRAIN_GLB_URL);
  const prepared = useMemo(() => prepareGltfScene(gltf), [gltf]);

  const fiberPoints = useMemo(() => {
    const pts = roiScenePositions?.length ? [...roiScenePositions] : [];
    if (pts.length >= 2) return pts;
    const fallback: Array<[number, number, number]> = [
      [-0.28, 0.08, 0.1],
      [-0.22, -0.05, -0.14],
      [0.26, 0.06, 0.08],
      [0.2, -0.07, -0.12],
      [-0.12, 0.18, -0.04],
      [0.1, 0.15, 0.06],
    ];
    return fallback;
  }, [roiScenePositions]);

  return (
    <group rotation={GLTF_EXTRA_ROTATION}>
      <primitive object={prepared} />
      <CortexParticles count={420} />
      <NeuralFibers points={fiberPoints} />
    </group>
  );
}

export function BrainMesh({ roiScenePositions }: BrainMeshProps) {
  const [gltfAvailable, setGltfAvailable] = useState<boolean | null>(null);

  useEffect(() => {
    let live = true;
    void getGlbAvailability().then((ok) => {
      if (live) setGltfAvailable(ok);
    });
    return () => {
      live = false;
    };
  }, []);

  const proceduralFallback = <ProceduralBrain roiScenePositions={roiScenePositions} />;

  if (gltfAvailable === null) {
    return <BrainLoadingShell roiScenePositions={roiScenePositions} />;
  }

  if (!gltfAvailable) {
    return proceduralFallback;
  }

  return (
    <BrainLoadErrorBoundary fallback={proceduralFallback}>
      <Suspense fallback={<BrainLoadingShell roiScenePositions={roiScenePositions} />}>
        <GltfBrain roiScenePositions={roiScenePositions} />
      </Suspense>
    </BrainLoadErrorBoundary>
  );
}
