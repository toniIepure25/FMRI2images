import type { InferenceResponse } from '@/lib/api';

interface WhyTrustPanelProps {
  inference: InferenceResponse | null;
  liveMode: boolean;
  reconstructionAvailable: boolean;
}

interface TrustCheck {
  label: string;
  value: string;
  status: 'pass' | 'warn' | 'fail' | 'unknown';
  explanation: string;
}

function computeChecks(inference: InferenceResponse | null, liveMode: boolean, reconAvail: boolean): TrustCheck[] {
  const checks: TrustCheck[] = [];
  const d = inference?.diagnostics;
  const r = inference?.reliability_features;
  const m = inference?.manifold_metrics;
  const rank = r?.rank;

  // Rank check
  if (rank === 1) {
    checks.push({ label: 'Retrieval rank', value: '#1', status: 'pass', explanation: 'Exact match — top-1 among entire gallery.' });
  } else if (rank != null && rank <= 5) {
    checks.push({ label: 'Retrieval rank', value: `#${rank}`, status: 'warn', explanation: 'Near match — correct image within top-5.' });
  } else if (rank != null) {
    checks.push({ label: 'Retrieval rank', value: `#${rank}`, status: 'fail', explanation: 'Candidate far from top-1.' });
  } else {
    checks.push({ label: 'Retrieval rank', value: '—', status: 'unknown', explanation: 'Rank data not available.' });
  }

  // κ check
  const kappa = r?.kappa;
  if (kappa != null && kappa > 100) {
    checks.push({ label: 'Directional confidence κ', value: kappa.toFixed(1), status: 'pass', explanation: 'High κ: predicted direction is sharply concentrated on the CLIP hypersphere.' });
  } else if (kappa != null && kappa > 30) {
    checks.push({ label: 'Directional confidence κ', value: kappa.toFixed(1), status: 'warn', explanation: 'Moderate κ: some directional ambiguity.' });
  } else if (kappa != null) {
    checks.push({ label: 'Directional confidence κ', value: kappa.toFixed(1), status: 'fail', explanation: 'Low κ: broad directional uncertainty.' });
  } else {
    checks.push({ label: 'Directional confidence κ', value: '—', status: 'unknown', explanation: 'Kappa not available.' });
  }

  // CSLS margin
  const margin = d?.csls_margin;
  if (margin != null && margin > 1.0) {
    checks.push({ label: 'CSLS margin', value: margin.toFixed(4), status: 'pass', explanation: 'Large margin: top-1 is well separated from alternatives.' });
  } else if (margin != null && margin > 0.1) {
    checks.push({ label: 'CSLS margin', value: margin.toFixed(4), status: 'warn', explanation: 'Moderate margin.' });
  } else if (margin != null) {
    checks.push({ label: 'CSLS margin', value: margin.toFixed(4), status: 'fail', explanation: 'Small margin: near-tie with runner-up.' });
  } else {
    checks.push({ label: 'CSLS margin', value: '—', status: 'unknown', explanation: 'Margin data not available.' });
  }

  // Entropy
  const entropy = d?.topk_entropy;
  if (entropy != null && entropy < 1.0) {
    checks.push({ label: 'Top-K entropy', value: entropy.toFixed(2), status: 'pass', explanation: 'Low entropy: scores are concentrated on a few candidates.' });
  } else if (entropy != null && entropy < 2.0) {
    checks.push({ label: 'Top-K entropy', value: entropy.toFixed(2), status: 'warn', explanation: 'Moderate entropy.' });
  } else if (entropy != null) {
    checks.push({ label: 'Top-K entropy', value: entropy.toFixed(2), status: 'fail', explanation: 'High entropy: scores are spread across many candidates.' });
  } else {
    checks.push({ label: 'Top-K entropy', value: '—', status: 'unknown', explanation: 'Entropy data not available.' });
  }

  // Manifold density
  const density = m?.on_manifold_score;
  if (density != null && density > 0.6) {
    checks.push({ label: 'On-manifold score', value: density.toFixed(3), status: 'pass', explanation: 'Prediction is well within the gallery manifold.' });
  } else if (density != null && density > 0.3) {
    checks.push({ label: 'On-manifold score', value: density.toFixed(3), status: 'warn', explanation: 'Moderate manifold alignment.' });
  } else if (density != null) {
    checks.push({ label: 'On-manifold score', value: density.toFixed(3), status: 'fail', explanation: 'Prediction may be off-manifold.' });
  } else {
    checks.push({ label: 'On-manifold score', value: '—', status: 'unknown', explanation: 'Manifold metrics not available.' });
  }

  // Reconstruction
  if (reconAvail) {
    checks.push({ label: 'Reconstruction', value: 'Available', status: 'pass', explanation: 'Live reconstruction from predicted brain embedding is available.' });
  } else {
    checks.push({ label: 'Reconstruction', value: 'Unavailable', status: 'unknown', explanation: 'Live reconstruction is not available. Karlo UnCLIP model not cached.' });
  }

  // Live mode
  if (liveMode) {
    checks.push({ label: 'Computation mode', value: 'Live', status: 'pass', explanation: 'Model forward and CSLS search execute locally on this machine.' });
  } else {
    checks.push({ label: 'Computation mode', value: 'Replay', status: 'warn', explanation: 'Using cached replay data. Backend may be unavailable.' });
  }

  return checks;
}

export function WhyTrustPanel({ inference, liveMode, reconstructionAvailable }: WhyTrustPanelProps) {
  const checks = computeChecks(inference, liveMode, reconstructionAvailable);
  const passCount = checks.filter(c => c.status === 'pass').length;
  const total = checks.length;

  return (
    <div className="rounded-xl border border-border-subtle bg-surface-elevated p-5 sm:p-6">
      <h3 className="text-sm font-semibold text-text-primary mb-4">Why should we trust this prediction?</h3>
      <div className="space-y-2.5">
        {checks.map((check) => {
          const icon = check.status === 'pass' ? (
            <svg className="h-4 w-4 text-status-success shrink-0" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" /></svg>
          ) : check.status === 'warn' ? (
            <svg className="h-4 w-4 text-status-warning shrink-0" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" /></svg>
          ) : check.status === 'fail' ? (
            <svg className="h-4 w-4 text-status-error shrink-0" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" /></svg>
          ) : (
            <svg className="h-4 w-4 text-text-muted shrink-0" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" /></svg>
          );

          return (
            <div key={check.label} className="flex items-start gap-3 rounded-lg bg-surface-raised px-4 py-3">
              {icon}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-[12px] font-semibold text-text-primary">{check.label}</span>
                  <span className="font-mono text-[11px] text-text-secondary">{check.value}</span>
                </div>
                <p className="text-[11px] leading-relaxed text-text-muted mt-0.5">{check.explanation}</p>
              </div>
            </div>
          );
        })}
      </div>
      <div className="mt-4 flex items-center gap-2 text-[11px] text-text-muted">
        <span>{passCount}/{total} checks passed</span>
        <div className="flex-1 h-0.5 rounded-full bg-surface-active">
          <div className="h-full rounded-full bg-accent/50" style={{ width: `${(passCount / total) * 100}%` }} />
        </div>
      </div>
    </div>
  );
}
