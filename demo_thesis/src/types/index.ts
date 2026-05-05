export interface DemoCase {
  id: string;
  subject: string;
  difficulty: 'best' | 'medium' | 'hard' | 'mystery';
  title: string;
  description: string;
  nsdId: number;
  session: number;
  repetition: number;
  fmriPreview: number[];
  targetImage: string;
  retrievedImages: RetrievedImage[];
  reconstructionImage: string;
  conservativeRecon: string;
  creativeRecon: string;
  ensembleImages: string[];
  metrics: CaseMetrics;
  uncertainty: UncertaintyData;
  duaCfg: DuaCfgParams;
  roiScores: RoiScore[];
  clipSpace: ClipSpaceRef;
  interpretation: string;
  challengeDistractors: string[];
  semanticCategory: string;
  /** Optional thesis demo assets */
  comparisonPanel?: string;
  topkStrip?: string;
  diffusionPrior?: string;
  diffusionFinal?: string;
}

export interface RetrievedImage {
  rank: number;
  image: string;
  score: number;
  csls: number;
  label: 'correct' | 'semantic_neighbor' | 'distractor';
}

export interface CaseMetrics {
  rank: number;
  cosine: number;
  csls: number;
  r1Correct: boolean;
  r5Correct: boolean;
  pixcorr: number;
  ssim: number;
  alex2: number;
  alex5: number;
}

export interface UncertaintyData {
  kappa: number;
  kappaNorm: number;
  delta: number;
  confidenceLevel: 'high' | 'medium' | 'low' | 'abstain';
}

export interface DuaCfgParams {
  guidanceScale: number;
  diffusionSteps: number;
  ensembleK: number;
  abstain: boolean;
}

export interface RoiScore {
  name: string;
  hemisphere: string;
  activation: number;
  contribution: number;
  confidence: number;
  agreement: number;
  interpretation: string;
}

export interface ClipSpaceRef {
  queryPointId: string;
  targetPointId: string;
  topKPointIds: string[];
}

export interface RoiLayout {
  rois: RoiDefinition[];
  categories: Record<string, RoiCategory>;
}

export interface RoiDefinition {
  name: string;
  fullName: string;
  hemisphere: string;
  position: { x: number; y: number; z: number };
  color: string;
  category: string;
  description: string;
  functionSummary: string;
}

export interface RoiCategory {
  label: string;
  color: string;
  rois: string[];
}

export interface ClipProjection {
  method: string;
  dimensions: number;
  points: ClipPoint[];
  clusters: ClipCluster[];
}

export interface ClipPoint {
  id: string;
  type: 'gallery' | 'query_predicted' | 'query_target' | 'retrieved';
  x: number;
  y: number;
  z: number;
  x2d: number;
  y2d: number;
  category: string;
  label: string;
  nsdId?: number;
  caseId?: string;
  rank?: number;
  score?: number;
}

export interface ClipCluster {
  name: string;
  centroid: { x: number; y: number; z: number };
  color: string;
  count: number;
}

export interface MetricEntry {
  value: number;
  std?: number;
  label: string;
  description?: string;
}

export interface MetricsSummary {
  projectName: string;
  modelVersion: string;
  clipModel: string;
  dataset: string;
  subjects: string[];
  metrics: {
    retrieval: Record<string, MetricEntry>;
    reconstruction: Record<string, MetricEntry>;
    uncertainty: Record<string, MetricEntry>;
  };
  perSubject: Record<string, Record<string, number>>;
  baselines: BaselineEntry[];
  contributions: ContributionEntry[];
}

export interface BaselineEntry {
  name: string;
  r1: number | null;
  pixcorr: number;
  ssim: number;
  alex2: number;
  alex5: number;
}

export interface ContributionEntry {
  name: string;
  description: string;
}

export type FilmStep = {
  id: number;
  title: string;
  subtitle: string;
  description: string;
  icon: string;
};

export type ExplorerTab =
  | 'overview'
  | 'brain'
  | 'clip'
  | 'retrieval'
  | 'reconstruction'
  | 'uncertainty'
  | 'roi'
  | 'report';

export interface ChallengeState {
  caseId: string;
  options: string[];
  correctIndex: number;
  selectedIndex: number | null;
  revealed: boolean;
  score: number;
  total: number;
}
