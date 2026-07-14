#!/bin/bash
echo "=== BETAS ==="
for s in subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08; do
    count=$(ls /home/jovyan/work/data/nsd/nsddata_betas/ppdata/${s}/func1pt8mm/betas_fithrf_GLMdenoise_RR/*.nii.gz 2>/dev/null | wc -l)
    echo "${s}: ${count} beta sessions"
done
echo "=== ROI ==="
for s in subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08; do
    count=$(ls /home/jovyan/work/data/nsd/nsddata/ppdata/${s}/func1pt8mm/roi/*.nii.gz 2>/dev/null | wc -l)
    echo "${s}: ${count} roi files"
done
echo "=== PREEXTRACT ==="
for s in subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08; do
    f="/home/jovyan/work/FMRI2images/cache/preextracted/subject=${s}/fmri_features.npy"
    if [ -f "$f" ]; then
        sz=$(stat -c %s "$f" 2>/dev/null || echo "?")
        echo "${s}: EXISTS (${sz} bytes)"
    else
        echo "${s}: MISSING"
    fi
done
echo "=== INDEX ==="
for s in subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08; do
    f="/home/jovyan/work/FMRI2images/data/indices/nsd_index/subject=${s}/index.parquet"
    if [ -f "$f" ]; then
        echo "${s}: OK"
    else
        echo "${s}: MISSING"
    fi
done
echo "=== CLIP CACHE ==="
ls -la /home/jovyan/work/FMRI2images/outputs/clip_cache/ 2>/dev/null || echo "MISSING"
echo "=== DISK ==="
df -h /home/jovyan/work | tail -1
