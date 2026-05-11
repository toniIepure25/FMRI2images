# Cortex2Canvas

**An Interactive fMRI-to-Image Decoding Studio**

Presentation-grade web application for demonstrating brain-to-image neural decoding
during a bachelor thesis defense. Built to showcase the complete pipeline: fMRI
activity → ROI Transformer encoding → CLIP embedding space → retrieval and
uncertainty-aware diffusion reconstruction.

## Quick Start

```bash
cd demo_thesis

# Install dependencies
npm install

# Prepare demo assets (copies real results from the repo if available,
# generates placeholder images otherwise; requires Python 3 + Pillow)
python3 scripts/prepare_demo_data.py

# Start development server
npm run dev
```

Open **http://localhost:3000** in a browser.

## Features

### 1. Landing Page (`/`)
Cinematic hero with animated gradient orbs, key metric cards, pipeline overview,
and contribution summaries. Three entry points: Film Mode, Explorer, Challenge.

### 2. Film Mode (`/film`)
Step-by-step cinematic walkthrough of the decoding pipeline:

| Step | Content |
|------|---------|
| 1 | Select fMRI trial — subject/session/trial info, animated signal bars |
| 2 | Visual cortex activation — interactive 3D brain with ROI markers |
| 3 | Neural embedding — pipeline diagram with flowing particles |
| 4 | CLIP vector space — 3D scatter with "fly-to-query" animation |
| 5 | Retrieval candidates — Top-K images with scores |
| 6 | Diffusion reconstruction — simulated progressive denoising |
| 7 | Uncertainty — kappa/delta gauges, DUA-CFG parameters, ensemble |
| 8 | Interpretation — side-by-side comparison, ROI explanation |

**Keyboard controls:** Space = play/pause, Arrow keys = navigate, R = reset

### 3. Explorer Dashboard (`/explorer`)
Full interactive dashboard with 8 tabs: Overview, 3D Brain, CLIP Space,
Retrieval, Reconstruction, Uncertainty, ROI Explainability, Report.

Left sidebar: case/subject/difficulty selector.
Right sidebar: quick stats (confidence, rank, kappa, delta, DUA-CFG).

### 4. Blind Committee Challenge (`/challenge`)
Interactive game where committee members guess which image the subject saw,
then compare their answer against the model's top-1 prediction.

### 5. 3D Brain Viewer
Procedurally generated stylized brain with semi-transparent hemispheres
and 17 ROI markers (V1–V4, FFA, PPA, EBA, OFA, OPA, RSC, etc.).

- Color-coded by functional category
- Activation intensity reflects model scores
- Hover tooltips, click-to-pin details
- Consensus/disagreement visualization (delta-driven pulse vs. flicker)

> **Note:** The 3D brain geometry is a presentation visualization.
> ROI names, coordinates, and scores are loaded from `data/roi_layout.json`
> and the demo case data. The positions approximate standard neuroanatomy
> but are not MNI-registered surfaces.

### 6. CLIP Vector Space Viewer
Interactive 2D/3D scatter plot of CLIP embedding projections:
- Gallery points colored by semantic cluster
- Query, target, and top-K points highlighted
- Connecting lines and similarity labels

## Project Structure

```
demo_thesis/
├── public/
│   ├── data/                  # JSON data files served statically
│   │   ├── demo_cases.json    # 14 demo cases (4 best, 4 medium, 4 hard, 2 mystery)
│   │   ├── roi_layout.json    # 17 ROI definitions with 3D coordinates
│   │   ├── clip_projection.json # UMAP projection with ~150 gallery points
│   │   └── metrics_summary.json # Aggregate metrics and baselines
│   └── assets/
│       └── cases/             # Per-case images (target, retrieved, reconstruction)
├── src/
│   ├── components/
│   │   ├── brain/             # 3D brain mesh, ROI markers
│   │   ├── challenge/         # Challenge game components
│   │   ├── explorer/          # Explorer tab components
│   │   ├── film/              # Film timeline and step components
│   │   └── *.tsx              # Shared components
│   ├── lib/
│   │   ├── data.ts            # Data loading with caching
│   │   ├── dua-cfg.ts         # DUA-CFG calculation (TypeScript port)
│   │   └── hooks.ts           # React data hooks
│   ├── pages/                 # Route pages
│   └── types/                 # TypeScript interfaces
├── data/                      # Source data (copied to public/ by prep script)
├── scripts/
│   └── prepare_demo_data.py   # Asset preparation script
├── backend/                   # Optional FastAPI server
│   ├── main.py
│   └── requirements.txt
└── package.json
```

## Data Files

All demo data is precomputed JSON — no GPU, no live inference, no Stable
Diffusion required. The app loads everything from static files.

| File | Contents |
|------|----------|
| `demo_cases.json` | 14 cases with fMRI previews, retrieval results, ROI scores, uncertainty, DUA-CFG params |
| `roi_layout.json` | 17 ROIs with 3D coordinates, category colors, functional descriptions |
| `clip_projection.json` | UMAP-projected CLIP embeddings with semantic clusters |
| `metrics_summary.json` | Aggregate R@1, SSIM, PixCorr, baselines, per-subject stats |

### Loading Real Results

If the repository contains actual reconstruction results:

```
reconstruction_results/best_cases/query_*.png
reconstruction_results/medium_cases/query_*.png
reconstruction_results/hard_cases/query_*.png
```

The `prepare_demo_data.py` script will copy them into the demo assets.
Otherwise, it generates professional placeholder images (requires Pillow).

## Optional Backend

A FastAPI backend is included but **not required** for the demo.
The frontend loads all data from static JSON files.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API endpoints:
- `GET /api/health` — server readiness and device info
- `GET /api/trials` — list available trial indices
- `GET /api/infer/{trial_idx}` — synchronous inference (returns full result JSON)
- `GET /api/infer-stream/{trial_idx}` — streaming inference via SSE (real-time pipeline progress)
- `GET /api/nsd-id-to-trial/{nsd_id}` — resolve an NSD image ID to trial indices

Static data (cases, CLIP projections, ROI layout, metrics) is served from `public/data/*.json`.

## Build Commands

| Command | Description |
|---------|-------------|
| `npm run dev` | Start Vite dev server on port 3000 |
| `npm run build` | TypeScript check + production build |
| `npm run preview` | Preview production build |
| `npm run typecheck` | TypeScript type checking only |
| `npm run prepare-demo-data` | Run data preparation script |

## Technology Stack

- **React 18** + TypeScript
- **Vite 6** — fast bundler
- **Tailwind CSS 3** — utility-first styling, dark scientific theme
- **Framer Motion** — animations and transitions
- **Three.js** via @react-three/fiber + @react-three/drei — 3D brain
- **Plotly.js** via react-plotly.js — CLIP space scatter plots
- **Recharts** — bar charts for ROI contributions
- **React Router 6** — client-side routing

### 6. Pipeline Mode (`/pipeline`)
Interactive step-by-step decoding pipeline with scientific honesty:

1. **Select trial** — pick a real NSD trial from curated examples
2. **Encode & retrieve** — watch the encoding pipeline execute (live or replayed)
3. **Reconstruct & reveal** — compare perceived stimulus, retrieval, and reconstruction

**Evidence integrity summary:**

| UI element | Data source | Provenance |
|---|---|---|
| fMRI preview | Real ROI feature vector (128-bin mean \|activation\| from 15,724 voxels) | Replay |
| CLIP preview | Real ground-truth CLIP ViT-L/14 embedding (96-bin compression from 768-D) | Replay |
| ROI stats | Not exported for demo — shown as unavailable | Unknown |
| Top-K retrieval | Real cached ranking from experiment output (fused CSLS scores) | Replay |
| Reconstruction | Cached qualitative reconstruction from diffusion experiment, if available | Replay |
| PixCorr/SSIM | Real measured values from experiment, if reconstruction exists | Replay |
| κ, δ | Derived from retrieval rank in replay mode; real model output in live mode | Derived |
| DUA-CFG | Derived replay policy from uncertainty estimate | Derived |

**Provenance system:** Every value, image, and metric is labeled with its source:
- **Live** (emerald) — returned by backend during this session
- **Replay** (cyan) — loaded from cached experiment JSON
- **Derived** (amber) — computed from retrieval/uncertainty policy
- **Unknown** (slate) — provenance cannot be determined; shown as unavailable

**Mode detection:** The header shows the current operating mode:
- `Live · backend inference active` — real model inference in progress
- `Replay · cached experiment assets` — all data from precomputed JSON
- `Offline replay · backend unavailable` — backend unreachable
- `Hybrid · backend reachable, replay assets active` — backend online but using cached data

**No fake placeholders:** The UI never shows seeded/synthetic scientific visualizations
as if they were real data. When an artifact is unavailable, it shows an honest
unavailable state or is hidden entirely.

**Why PixCorr/SSIM may show `—`:** These reconstruction quality metrics require
a diffusion reconstruction asset. When no reconstruction was generated for a trial,
the metrics are unavailable and display as `—` (not 0.000).

**Refreshing real artifacts from the K8s pod:**
```bash
export KUBECONFIG=~/Downloads/antoniu_iepure.yaml
kubectl get pods -n runai-romania-dev -o name | grep jupyter
# Run extraction (produces pipeline_real_artifacts_manifest.json)
# Then integrate:
cd demo_thesis
python3 scripts/integrate_real_artifacts.py
```

**Defense presentation tips:**
- Use the **Defense set** filter to show only fully-evidenced cases
- Start with an "Exact match" case to show the system at its best
- Follow with a "Near miss" to demonstrate the uncertainty story
- The provenance badges prove you understand what is live vs cached
- Click "Computation details" in Phase 2 to explain the data pipeline
- Open the "Evidence integrity" panel to show the committee what is real

**If asked "Is this live?":**

> For presentation reliability, this screen runs as a replay of real experiment
> outputs: the fMRI features, CLIP embeddings, retrieval rankings, and
> reconstruction metrics you see were produced by our trained model on the H100
> cluster and cached as JSON. When the backend is connected via port-forward,
> the encoding and retrieval steps execute through the live model path and
> provenance badges change from Replay to Live. Reconstruction remains a cached
> qualitative asset because diffusion sampling takes minutes per image.
> Every UI element carries a provenance badge so you can distinguish live
> inference from cached results at a glance.

## Key Concepts Visualized

| Concept | Location |
|---------|----------|
| **κ (kappa)** — concentration/confidence | Uncertainty tab, Film step 7 |
| **δ (delta)** — ROI disagreement | Uncertainty tab, Brain viewer consensus mode |
| **DUA-CFG** — adaptive diffusion guidance | Uncertainty tab, Reconstruction tab |
| **ROI contributions** — which brain regions matter | ROI tab, Brain viewer, Film step 2 |
| **CLIP embedding space** — semantic retrieval | CLIP tab, Film step 4 |
| **Top-K retrieval** — nearest-neighbor search | Retrieval tab, Film step 5 |

## Backend Live Path Status

Live backend path is supported by the code but was not verified in this final
audit. The Kubernetes pod (`orchestraiq-jupyter-54644cff87-t7786`) was reachable
and data extraction succeeded, but no FastAPI backend server was running at
`localhost:8000` inside the pod at the time of testing. To verify:

```bash
export KUBECONFIG=~/Downloads/antoniu_iepure.yaml
kubectl port-forward -n runai-romania-dev pod/orchestraiq-jupyter-54644cff87-t7786 8000:8000
curl http://localhost:8000/api/health
```

## Requirements

- Node.js ≥ 18
- Python 3 + Pillow (for asset preparation only)
- A modern browser (Chrome, Firefox, Edge, Safari)
- No GPU required
- No live model inference required
