# Cortex2Canvas Local Backend Runbook

## Prerequisites

- Python 3.10+ with venv
- Node.js 18+ (for the frontend)
- ~6 GB disk for backend artifacts

## 1. Install Python dependencies

```bash
cd ~/Desktop/Bachelor\ V2
source .venv/bin/activate
cd demo_thesis/backend
pip install -r requirements.txt
```

### PyTorch installation (separate)

The backend requires PyTorch. Install one of:

**CPU only** (smaller, works everywhere):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**CUDA 12.1** (if you have an NVIDIA GPU):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 2. Fetch artifacts from the K8s pod

```bash
cd ~/Desktop/Bachelor\ V2/demo_thesis
bash scripts/fetch_live_backend_artifacts_from_pod.sh
```

Or manually copy individual files (see the script for paths).

### Expected local layout

```
demo_thesis/local_backend_data/
  subj01/
    fmri_features.npy   # ~1.8 GB, ROI-masked fMRI features (30000 x 15724)
    index.parquet        # ~724 KB, trial-to-nsdId mapping (30000 rows)
  clip/
    clip.parquet         # ~14 MB, CLIP ViT-L/14 gallery (9999 x 768)
  checkpoints/
    V62a_model_only.pt   # ~2.1 GB, model weights only (no optimizer)
```

### Currently available checkpoint

| Checkpoint         | Experiment              | Output dim | CSLS R@1 (shared1000) | Size   |
| ------------------ | ----------------------- | ---------- | --------------------- | ------ |
| V62a_model_only.pt | V62a_cls_retrieval_768d | 768        | 48.3%                 | 2.1 GB |

The original N1v28a + V55b fusion model (76% R@1) operated in 197k-D space
and those checkpoints are no longer available on the pod. V62a is the best
768-D model compatible with the demo's CLIP gallery.

## 3. Verify artifacts

```bash
cd ~/Desktop/Bachelor\ V2/demo_thesis/backend
python verify_artifacts.py
```

Expected output when everything is ready:

```
  PyTorch:  OK v2.x.x  cuda=yes  device=NVIDIA ...
  fMRI features: OK ../local_backend_data/subj01/fmri_features.npy (1803.4 MB)
  Trial index: OK ../local_backend_data/subj01/index.parquet (0.7 MB)
  CLIP gallery: OK ../local_backend_data/clip/clip.parquet (13.6 MB)
  Checkpoint: OK ../local_backend_data/checkpoints/V62a_model_only.pt (2114.0 MB)
Status: INFERENCE READY
```

## 4. Configure environment

```bash
cd ~/Desktop/Bachelor\ V2/demo_thesis/backend
cp .env.example .env
# Edit .env if paths differ from defaults
```

## 5. Start backend

```bash
cd ~/Desktop/Bachelor\ V2/demo_thesis/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Loading takes 30-60 seconds depending on disk speed and available RAM.

## 6. Test backend

In another terminal:

```bash
# Quick health check
curl -s http://127.0.0.1:8000/api/health | python3 -m json.tool

# Full smoke test
cd ~/Desktop/Bachelor\ V2/demo_thesis/backend
python smoke_test_backend.py --base-url http://127.0.0.1:8000
```

## 7. Start frontend

```bash
cd ~/Desktop/Bachelor\ V2/demo_thesis
npm run dev
```

Open http://localhost:3000/pipeline

## 8. Expected mode labels

| Backend state                         | Frontend shows                                  |
| ------------------------------------- | ----------------------------------------------- |
| Backend unreachable                   | Offline replay · Backend unavailable            |
| Backend online, no data/model         | Replay · Backend online, replay assets active   |
| Backend online, data loaded, no model | Replay · Backend online, replay assets active   |
| Backend online, data + model loaded   | Hybrid · Live retrieval + cached reconstruction |

The "Hybrid" label means the encoding and CSLS retrieval run live through
the model, but reconstruction images remain cached because real-time diffusion
sampling takes minutes per image.

## 9. Troubleshooting

### "torch_available: false"

PyTorch is not installed in your Python environment. See step 1.

### "features_loaded: false"

`fmri_features.npy` not found. Check that `local_backend_data/subj01/fmri_features.npy` exists.

### "model: Checkpoint load failed"

The checkpoint format may not match the installed fmri2img version.
Ensure the repo's `fmri2img` package is installed:

```bash
cd ~/Desktop/Bachelor\ V2
pip install -e ".[train]"
```

### Backend starts but model import fails

The model architecture (`create_model`) comes from `fmri2img.models.unified_model`.
If this import fails, install the main package as above.

### OOM on model loading

The V62a model has 554M parameters (~2.1 GB). With features (1.8 GB) and gallery,
total memory usage is ~5 GB. Ensure at least 8 GB RAM available, or use GPU.

## 10. Security notes

- `local_backend_data/` is gitignored — never commit large binary artifacts
- `backend/.env` is gitignored — never commit credentials
- No NGC tokens or API keys are used by the demo backend

(.venv) tonystark@pop-os:~/Desktop/Bachelor V2$ export KUBECONFIG="$HOME/Downloads/antoniu_iepure.yaml"

kubectl exec -n runai-romania-dev orchestraiq-jupyter-54644cff87-nxd9x -- bash -lc \
'rm -rf /home/jovyan/work/FMRI2images/src/analysis'

kubectl cp \
 ./src/analysis \
 runai-romania-dev/orchestraiq-jupyter-54644cff87-nxd9x:/home/jovyan/work/FMRI2images/src/analysis
(.venv) tonystark@pop-os:~/Desktop/Bachelor V2$ cd "/home/tonystark/Desktop/Bachelor V2/demo_thesis"

../.venv/bin/python -c "
import diffusers, transformers, torch
from transformers import CLIPImageProcessor
from diffusers import UnCLIPImageVariationPipeline
print('diffusers:', diffusers.**version**, diffusers.**file**)
print('transformers:', transformers.**version**, transformers.**file**)
print('torch:', torch.**version**, 'cuda=', torch.cuda.is_available())
print('imports ok')
"
diffusers: 0.35.2 /home/tonystark/Desktop/Bachelor V2/.venv/lib/python3.10/site-packages/diffusers/**init**.py
transformers: 4.57.1 /home/tonystark/Desktop/Bachelor V2/.venv/lib/python3.10/site-packages/transformers/**init**.py
torch: 2.8.0+cu128 cuda= True
imports ok
(.venv) tonystark@pop-os:~/Desktop/Bachelor V2/demo_thesis$ cd "/home/tonystark/Desktop/Bachelor V2/demo_thesis"

PYTHONPATH="../src:backend" \
C2C_BACKEND_MODE=v62_single \
C2C_RECON_MODE=live \
C2C_RECON_ALLOW_DOWNLOAD=true \
C2C_RECON_STEPS=25 \
C2C_RECON_GUIDANCE=8.0 \
C2C_RECON_SEED=42 \
../.venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
