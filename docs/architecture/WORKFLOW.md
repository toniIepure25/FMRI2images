# JupyterHub Workflow Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      JUPYTERHUB/HPC ENVIRONMENT                         │
│  • No sudo access                                                       │
│  • No systemctl                                                         │
│  • Python 3.11.2 + pip                                                  │
│  • NVIDIA A100 GPU + CUDA 12.2                                          │
│  • /bigdata/userhome/students/<USER>/                                   │
└─────────────────────────────────────────────────────────────────────────┘

                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        INITIAL SETUP (Once)                             │
│                                                                          │
│  1. cp .env.jupyterhub .env  ──────────────> Configure paths           │
│                                                                          │
│  2. run=true source initialSetup.sh  ─────> Set shell env (if needed)  │
│                                                                          │
│  3. ./setup.sh  ─────────────────────────────> Create venv + install     │
│     • Creates ./.venv/                                                   │
│     • Installs PyTorch (CUDA 12.2)                                      │
│     • Installs all dependencies                                         │
│                                                                          │
│  4. source .venv/bin/activate  ──────────────> Activate environment     │
│                                                                          │
│  5. make preflight  ─────────────────────────> Verify everything        │
│     ✓ Python + packages                                                 │
│     ✓ GPU/CUDA                                                           │
│     ✓ Disk space                                                         │
│     ✓ Permissions                                                        │
└─────────────────────────────────────────────────────────────────────────┘

                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     SESSION WORKFLOW (Every Login)                      │
│                                                                          │
│  source .venv/bin/activate  ───────────────> Activate venv             │
└─────────────────────────────────────────────────────────────────────────┘

                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        EXPERIMENT EXECUTION                             │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ Option 1: Short/Interactive Run                                │    │
│  │                                                                 │    │
│  │  python scripts/training/train.py --config config.yaml         │    │
│  │                                                                 │    │
│  │  • Runs in foreground                                          │    │
│  │  • See output in real-time                                     │    │
│  │  • Ctrl+C to stop                                              │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ Option 2: Long Run (tmux - Recommended)                        │    │
│  │                                                                 │    │
│  │  bash scripts/tmux_run.sh config.yaml my_session               │    │
│  │                                                                 │    │
│  │  • Runs in persistent tmux session                             │    │
│  │  • Survives disconnect                                         │    │
│  │  • Ctrl+B, D to detach                                         │    │
│  │  • tmux attach -t my_session to reattach                       │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ Option 3: Background (nohup)                                   │    │
│  │                                                                 │    │
│  │  bash scripts/nohup_run.sh config.yaml                         │    │
│  │                                                                 │    │
│  │  • Runs in background                                          │    │
│  │  • Survives logout                                             │    │
│  │  • tail -f logs/*.log to monitor                               │    │
│  │  • kill <PID> to stop                                          │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ Option 4: Batch (Sweep)                                        │    │
│  │                                                                 │    │
│  │  bash scripts/run_sweep.sh configs/experiments/*.yaml          │    │
│  │                                                                 │    │
│  │  • Runs multiple experiments sequentially                      │    │
│  │  • Tracks success/failure                                      │    │
│  │  • --continue-on-error for resilience                          │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘

                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      EXPERIMENT EXECUTION FLOW                          │
│                                                                          │
│  python scripts/training/train.py --config config.yaml                  │
│         │                                                                │
│         ├─> Load .env                                                   │
│         ├─> Check venv active                                           │
│         ├─> Run preflight checks                                        │
│         │      ✓ Python, packages                                       │
│         │      ✓ GPU/CUDA                                               │
│         │      ✓ Disk space                                             │
│         │      ✓ Permissions                                            │
│         │                                                                │
│         └─> python -m src.train --config config.yaml                    │
│                    │                                                     │
│                    ├─> Load config                                      │
│                    ├─> Set seeds (reproducibility)                      │
│                    ├─> Get git info                                     │
│                    ├─> Create run directory                             │
│                    │      runs/<timestamp>_<name>/                      │
│                    │         ├── config.yaml                            │
│                    │         ├── git_info.json                          │
│                    │         ├── environment/  (snapshot)               │
│                    │         ├── checkpoints/                           │
│                    │         ├── logs/                                  │
│                    │         ├── metrics/                               │
│                    │         └── outputs/                               │
│                    │                                                     │
│                    ├─> Setup logging                                    │
│                    ├─> Initialize model                                 │
│                    ├─> Training loop                                    │
│                    │      • Save checkpoints                            │
│                    │      • Log metrics (JSONL)                         │
│                    │      • Track progress (tqdm)                       │
│                    │      • Resume support                              │
│                    │                                                     │
│                    └─> Write SUCCESS or FAILED                          │
└─────────────────────────────────────────────────────────────────────────┘

                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT STRUCTURE                              │
│                                                                          │
│  runs/20260112_143022_my_experiment/                                    │
│    ├── config.yaml              # Exact config used                     │
│    ├── git_info.json            # Git commit, branch, changes           │
│    ├── environment/             # Complete snapshot                     │
│    │   ├── environment.json     # All-in-one                            │
│    │   ├── requirements_frozen.txt  # Python packages                   │
│    │   ├── system_info.json     # OS, CPU, RAM                          │
│    │   ├── gpu_info.json        # GPU, CUDA                             │
│    │   └── env_vars.json        # Environment variables                 │
│    ├── checkpoints/                                                      │
│    │   ├── checkpoint_epoch_1.pt                                        │
│    │   ├── checkpoint_epoch_5.pt                                        │
│    │   └── best_model.pt        # Best performing                       │
│    ├── logs/                                                             │
│    │   └── train.log            # Full training log                     │
│    ├── metrics/                                                          │
│    │   ├── metrics.jsonl        # Per-step metrics                      │
│    │   └── summary.json         # Final summary                         │
│    ├── outputs/                 # Generated outputs (optional)          │
│    └── SUCCESS                  # Or FAILED                             │
│                                                                          │
│  This structure ensures:                                                │
│    ✓ Full reproducibility                                               │
│    ✓ Easy comparison between runs                                       │
│    ✓ Publication-grade provenance                                       │
└─────────────────────────────────────────────────────────────────────────┘

                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      MONITORING & DEBUGGING                             │
│                                                                          │
│  GPU:           nvidia-smi                                              │
│                 watch -n 1 nvidia-smi                                   │
│                                                                          │
│  Logs:          tail -f logs/train.log                                  │
│                 cat runs/*/logs/train.log                               │
│                                                                          │
│  Metrics:       cat runs/*/metrics/summary.json                         │
│                 tail runs/*/metrics/metrics.jsonl                       │
│                                                                          │
│  tmux:          tmux ls                                                 │
│                 tmux attach -t <session>                                │
│                                                                          │
│  Disk:          df -h /bigdata                                          │
│                                                                          │
│  Processes:     ps aux | grep python                                    │
└─────────────────────────────────────────────────────────────────────────┘

                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     KEY DESIGN PRINCIPLES                               │
│                                                                          │
│  1. Reproducibility First                                               │
│     • Fixed seeds with deterministic operations                         │
│     • Git tracking (commit hash, uncommitted changes)                   │
│     • Environment snapshots (packages, system, GPU)                     │
│     • Config versioning                                                 │
│                                                                          │
│  2. Fail-Fast Validation                                                │
│     • Preflight checks before every run                                 │
│     • Clear error messages                                              │
│     • Disk space validation                                             │
│     • Permission tests                                                  │
│                                                                          │
│  3. HPC-Friendly                                                        │
│     • No sudo required                                                  │
│     • No systemctl dependencies                                         │
│     • Virtual environment isolation                                     │
│     • Persistent sessions (tmux)                                        │
│     • Background execution (nohup)                                      │
│                                                                          │
│  4. Production Quality                                                  │
│     • Professional logging                                              │
│     • Structured metrics (JSONL)                                        │
│     • Checkpoint management                                             │
│     • Resume capability                                                 │
│     • Error handling                                                    │
│                                                                          │
│  5. Developer Experience                                                │
│     • Simple commands                                                   │
│     • Clear documentation                                               │
│     • Quick reference card                                              │
│     • Helpful error messages                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## File Organization

```
Bachelor V2/
│
├── 📄 Configuration
│   ├── .env.jupyterhub          # JupyterHub-specific
│   ├── .env.example             # Generic template
│   └── configs/experiments/     # Experiment configs
│
├── Scripts
│   ├── setup.sh                           # Automated setup
│   ├── scripts/training/train.py          # Training entrypoint
│   ├── scripts/evaluation/                # Evaluation scripts
│   ├── scripts/orchestration/             # Batch run scripts
│   └── scripts/utils/                     # Utilities (preflight, doctor, etc.)
│
├── Documentation
│   ├── docs/guides/SETUP.md              # Setup guide
│   ├── docs/guides/JUPYTERHUB_REFERENCE.md # Quick commands
│   └── docs/guides/RUNNING_EXPERIMENTS.md  # Experiment guide
│
└── Outputs (Generated)
    ├── .venv/                  # Virtual environment
    ├── outputs/                # Experiment outputs
    └── logs/                   # Log files
```

## Command Flow Diagram

```
User Command
    │
    ▼
┌─────────────────────┐
│  Training Script    │  (scripts/training/train.py)
│  • Load .env        │
│  • Check venv       │
│  • Run preflight    │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  make preflight     │
│  • Validate env     │
│  • Check GPU        │
│  • Check disk       │
│  • Check perms      │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  src/train.py       │
│  • Load config      │
│  • Set seeds        │
│  • Get git info     │
│  • Create run dir   │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  snapshot_env.sh    │
│  • Capture pkgs     │
│  • Capture system   │
│  • Capture GPU      │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Training Loop      │
│  • Train model      │
│  • Save ckpts       │
│  • Log metrics      │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Run Complete       │
│  • SUCCESS marker   │
│  • Summary saved    │
└─────────────────────┘
```

---

**This architecture ensures your experiments are:**
- ✅ Reproducible
- ✅ Traceable
- ✅ Resumable
- ✅ Production-ready
- ✅ HPC-compatible
