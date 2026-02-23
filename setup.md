Step 1 -- Clone and checkout (~1 min)
cd /home/jovyan/workgit clone https://github.com/toniIepure25/FMRI2images.gitcd FMRI2imagesgit checkout dirbrain-vmf-uacfg
(Use HTTPS unless you have SSH keys configured on the pod. If you do have SSH keys, use the git@github.com: URL.)
Step 2 -- Set up environment (~5 min)
# Copy the JupyterHub-specific .env (already configured for /home/jovyan/work)cp .env.jupyterhub .env# Load variables into your shell (set -a makes them auto-exported)set -a && source .env && set +a# Install the project + all dependencies into the base conda envpip install -e ".[train,diffusion]"# Also install awscli for data downloadpip install awscli
The set -a && source .env && set +a pattern is the correct way to load a .env file. After this, every variable in .env is exported to child processes. You should run this line at the start of every new terminal session, or add it to your ~/.bashrc.
Important: The Makefile already reads .env automatically (line 7: -include .env), so all make commands will see the variables even without sourcing. You only need to source manually when running Python scripts directly.
Step 3 -- Preflight check (~30 sec)
make preflight
This checks Python version, CUDA, disk space, required packages, and environment variables. Fix any FAIL items before continuing.
Step 4 -- Download NSD data (~3-8 hours depending on network)
mkdir -p /home/jovyan/work/data/nsd# Download betas for 4 subjects (~40 GB each = ~160 GB total)for subj in subj01 subj02 subj05 subj07; do  aws s3 sync --no-sign-request \    s3://natural-scenes-dataset/nsddata_betas/ppdata/${subj}/func1pt8mm/betas_fithrf_GLMdenoise_RR/ \    /home/jovyan/work/data/nsd/nsddata_betas/ppdata/${subj}/func1pt8mm/betas_fithrf_GLMdenoise_RR/done# Download stimuli (~16 GB) + metadataaws s3 cp --no-sign-request \  s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 \  /home/jovyan/work/data/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5aws s3 cp --no-sign-request \  s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.csv \  /home/jovyan/work/data/nsd/nsddata/experiments/nsd/nsd_stim_info_merged.csv# Download ROI masks for all 4 subjectsfor subj in subj01 subj02 subj05 subj07; do  aws s3 sync --no-sign-request \    s3://natural-scenes-dataset/nsddata/ppdata/${subj}/ \    /home/jovyan/work/data/nsd/nsddata/ppdata/${subj}/done
You can run this in a JupyterLab terminal and leave it running. The 1 TiB PVC you have is more than enough.
Step 5 -- Data preparation (~1-3 hours)
for subj in subj01 subj02 subj05 subj07; do  make index SUBJECT=$subj  make preprocess SUBJECT=$subjdonemake clip-cachemake download-sd
Step 6 -- Run all experiments (~15-30 hours total)
Option A -- Training only (then run reconstruction/evaluation separately):
make ablation SUBJECTS="subj01 subj02 subj05 subj07" GPU=0
Option B -- Full pipeline (training + reconstruction + evaluation + aggregation):
make full-pipeline SUBJECTS="subj01 subj02 subj05 subj07" GPU=0
I recommend Option B since it does everything end-to-end. Run it in a terminal with nohup or tmux/screen so it survives if your browser disconnects:
nohup make full-pipeline SUBJECTS="subj01 subj02 subj05 subj07" GPU=0 \  > full_pipeline.log 2>&1 &tail -f full_pipeline.log
Step 7 -- Check results
All outputs will be in:
experimental_results/  B0_deterministic/subj01/metrics/summary.json  B0_deterministic/subj02/metrics/summary.json  ...  N4_full_system/subj07/metrics/summary.json  ablation_summary.csv    <-- paper table  ablation_summary.tex    <-- LaTeX for paper
