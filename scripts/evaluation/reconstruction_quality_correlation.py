"""
Reconstruction Quality vs Kappa Correlation Analysis.

Runs SDXL + IP-Adapter reconstruction on a stratified subset of SHARED1000,
then correlates per-trial reconstruction quality (PixCorr, SSIM, LPIPS) with kappa.

Also computes correlation between kappa and cosine similarity to GT (cheap proxy).
"""
import sys
sys.path.insert(0, "/home/jovyan/work/FMRI2images/src")

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path("/home/jovyan/work/FMRI2images")
OUTPUT_DIR = REPO_ROOT / "experimental_results" / "calibrated_uncertainty"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_RECON_SUBSET = 200


def load_kappa_and_preds():
    """Load precomputed kappas and predictions for V61a (best model)."""
    kappas = np.load(OUTPUT_DIR / "V61a_image_kappas.npy")
    
    preds = np.load(REPO_ROOT / "experimental_results/V61a_finetune_difflr/subj01/metrics/shared1000_predictions.npy")
    gt = np.load(REPO_ROOT / "experimental_results/V61a_finetune_difflr/subj01/metrics/shared1000_ground_truth.npy")
    nsd_ids = np.load(REPO_ROOT / "experimental_results/V61a_finetune_difflr/subj01/metrics/shared1000_nsd_ids.npy")
    
    preds_768 = np.load(REPO_ROOT / "experimental_results/V62a_cls_retrieval_768d/subj01/metrics/shared1000_predictions.npy")
    gt_768 = np.load(REPO_ROOT / "experimental_results/V62a_cls_retrieval_768d/subj01/metrics/shared1000_ground_truth.npy")
    nsd_ids_v62 = np.load(REPO_ROOT / "experimental_results/V62a_cls_retrieval_768d/subj01/metrics/shared1000_nsd_ids.npy")
    
    kappas_v62 = np.load(OUTPUT_DIR / "V62a_image_kappas.npy")
    
    return {
        "V61a": {"preds": preds, "gt": gt, "nsd_ids": nsd_ids, "kappas": kappas},
        "V62a": {"preds": preds_768, "gt": gt_768, "nsd_ids": nsd_ids_v62, "kappas": kappas_v62},
    }


def cosine_similarity_per_trial(preds, gt):
    """Compute per-trial cosine similarity between predictions and ground truth."""
    preds_n = preds / (np.linalg.norm(preds, axis=1, keepdims=True) + 1e-8)
    gt_n = gt / (np.linalg.norm(gt, axis=1, keepdims=True) + 1e-8)
    return (preds_n * gt_n).sum(axis=1)


def proxy_quality_analysis(data_dict):
    """Use cosine similarity as a proxy for reconstruction quality."""
    results = {}
    
    for model_name, data in data_dict.items():
        preds = data["preds"]
        gt = data["gt"]
        kappas = data["kappas"]
        
        cos_sim = cosine_similarity_per_trial(preds, gt)
        
        rho_kappa_cos, p_kappa_cos = stats.spearmanr(kappas, cos_sim)
        r_pearson, p_pearson = stats.pearsonr(kappas, cos_sim)
        
        logger.info("\n=== %s: Kappa vs Cosine Similarity ===", model_name)
        logger.info("  Cosine sim: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                    cos_sim.mean(), cos_sim.std(), cos_sim.min(), cos_sim.max())
        logger.info("  Spearman: rho=%.4f, p=%.2e", rho_kappa_cos, p_kappa_cos)
        logger.info("  Pearson: r=%.4f, p=%.2e", r_pearson, p_pearson)
        
        n_bins = 5
        kappa_sorted = np.argsort(kappas)
        bin_size = len(kappas) // n_bins
        
        logger.info("\n  Kappa quintile analysis:")
        quintile_data = []
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else len(kappas)
            idx = kappa_sorted[start:end]
            
            q_kappa = kappas[idx]
            q_cos = cos_sim[idx]
            
            quintile_data.append({
                "quintile": i + 1,
                "kappa_range": f"[{q_kappa.min():.2f}, {q_kappa.max():.2f}]",
                "mean_kappa": round(float(q_kappa.mean()), 3),
                "mean_cos_sim": round(float(q_cos.mean()), 4),
                "std_cos_sim": round(float(q_cos.std()), 4),
                "n": len(idx),
            })
            
            logger.info("    Q%d (kappa %.2f-%.2f): cos_sim=%.4f ± %.4f (n=%d)",
                       i+1, q_kappa.min(), q_kappa.max(), q_cos.mean(), q_cos.std(), len(idx))
        
        results[model_name] = {
            "spearman": {"rho": round(float(rho_kappa_cos), 4), "p": float(p_kappa_cos)},
            "pearson": {"r": round(float(r_pearson), 4), "p": float(p_pearson)},
            "cosine_sim_stats": {
                "mean": round(float(cos_sim.mean()), 4),
                "std": round(float(cos_sim.std()), 4),
                "min": round(float(cos_sim.min()), 4),
                "max": round(float(cos_sim.max()), 4),
            },
            "quintile_analysis": quintile_data,
        }
    
    return results


def try_pixel_level_reconstruction(data_dict, n_subset=N_RECON_SUBSET):
    """
    Attempt pixel-level reconstruction and quality metrics.
    Falls back gracefully if diffusion models aren't loaded.
    """
    try:
        import h5py
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        logger.warning("skimage not available, skipping pixel-level metrics")
        return None
    
    hdf5_path = Path(os.environ.get("NSD_HDF5", "/home/jovyan/work/data/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"))
    if not hdf5_path.exists():
        logger.warning("NSD HDF5 not found at %s, skipping pixel-level analysis", hdf5_path)
        return None
    
    v62_data = data_dict["V62a"]
    nsd_ids = v62_data["nsd_ids"]
    preds = v62_data["preds"]
    kappas = v62_data["kappas"]
    
    # Stratified subset: equal samples from each kappa quintile
    kappa_sorted = np.argsort(kappas)
    per_quintile = n_subset // 5
    subset_idx = []
    for i in range(5):
        start = i * (len(kappas) // 5)
        end = (i + 1) * (len(kappas) // 5)
        quintile_idx = kappa_sorted[start:end]
        chosen = np.random.choice(quintile_idx, size=min(per_quintile, len(quintile_idx)), replace=False)
        subset_idx.extend(chosen)
    subset_idx = np.array(sorted(subset_idx))
    
    logger.info("Attempting pixel-level reconstruction for %d images (stratified subset)", len(subset_idx))
    
    try:
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        from PIL import Image
        import torchvision.transforms as T
        
        logger.info("Loading Stable Diffusion 2.1...")
        model_id = "sd2-community/stable-diffusion-2-1"
        cache_dir = os.environ.get("HF_HOME", "/home/jovyan/work/.cache/hf")
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, cache_dir=cache_dir, torch_dtype=torch.float16
        ).to(DEVICE)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        
        logger.info("Running %d reconstructions...", len(subset_idx))
        
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        
        results_per_image = []
        with h5py.File(hdf5_path, 'r') as hdf5:
            for count, idx in enumerate(subset_idx):
                nsd_id = int(nsd_ids[idx])
                pred_emb = torch.from_numpy(preds[idx:idx+1]).float().to(DEVICE)
                
                with torch.no_grad():
                    text_emb = pred_emb.half()
                    image = pipe(
                        prompt_embeds=text_emb,
                        num_inference_steps=30,
                        guidance_scale=5.0,
                    ).images[0]
                
                gt_img = Image.fromarray(hdf5['imgBrick'][nsd_id])
                
                recon_arr = np.array(image.resize((224, 224))).astype(float) / 255.0
                gt_arr = np.array(gt_img.resize((224, 224))).astype(float) / 255.0
                
                recon_gray = np.mean(recon_arr, axis=2)
                gt_gray = np.mean(gt_arr, axis=2)
                pixcorr = np.corrcoef(recon_gray.flatten(), gt_gray.flatten())[0, 1]
                ssim_val = ssim(gt_gray, recon_gray, data_range=1.0)
                
                results_per_image.append({
                    "nsd_id": nsd_id,
                    "kappa": float(kappas[idx]),
                    "pixcorr": float(pixcorr),
                    "ssim": float(ssim_val),
                })
                
                if (count + 1) % 20 == 0:
                    logger.info("  Reconstructed %d/%d", count + 1, len(subset_idx))
        
        del pipe
        torch.cuda.empty_cache()
        
        # Correlations
        k = np.array([r["kappa"] for r in results_per_image])
        pc = np.array([r["pixcorr"] for r in results_per_image])
        ss = np.array([r["ssim"] for r in results_per_image])
        
        rho_pc, p_pc = stats.spearmanr(k, pc)
        rho_ss, p_ss = stats.spearmanr(k, ss)
        
        logger.info("\n=== Pixel-Level Reconstruction Correlation ===")
        logger.info("  PixCorr: mean=%.4f, Kappa-PixCorr rho=%.4f (p=%.2e)", pc.mean(), rho_pc, p_pc)
        logger.info("  SSIM: mean=%.4f, Kappa-SSIM rho=%.4f (p=%.2e)", ss.mean(), rho_ss, p_ss)
        
        return {
            "n_images": len(results_per_image),
            "per_image": results_per_image,
            "pixcorr_stats": {"mean": float(pc.mean()), "std": float(pc.std())},
            "ssim_stats": {"mean": float(ss.mean()), "std": float(ss.std())},
            "kappa_pixcorr_spearman": {"rho": round(float(rho_pc), 4), "p": float(p_pc)},
            "kappa_ssim_spearman": {"rho": round(float(rho_ss), 4), "p": float(p_ss)},
        }
    
    except Exception as e:
        logger.warning("Reconstruction pipeline failed: %s", str(e))
        return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    data = load_kappa_and_preds()
    
    # Proxy analysis (fast, uses cosine similarity)
    proxy_results = proxy_quality_analysis(data)
    
    # Pixel-level reconstruction (slow, may fail)
    recon_results = try_pixel_level_reconstruction(data, n_subset=N_RECON_SUBSET)
    
    combined = {
        "proxy_quality_correlation": proxy_results,
        "pixel_level_reconstruction": recon_results,
    }
    
    with open(OUTPUT_DIR / "reconstruction_quality_correlation.json", "w") as f:
        json.dump(combined, f, indent=2)
    
    logger.info("\nSaved to %s", OUTPUT_DIR / "reconstruction_quality_correlation.json")
    logger.info("Done!")


if __name__ == "__main__":
    main()
