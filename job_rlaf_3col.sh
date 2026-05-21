#!/bin/bash
#SBATCH -J rlaf_3col
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH -t 72:00:00
#SBATCH --output=slurm_rlaf_3col_%j.out
#SBATCH --error=slurm_rlaf_3col_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# ─── Snellius RLAF Training: ArieNet (no COOC), 3-coloring ────────────────
# Paper settings: GRPO, 2000 iterations, glucose solver
# Run without co-occurrence graph extension
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths (adjust if your layout differs) ─────────────────────────────────
NSNET_DIR="$HOME/Thesis2/nsnet"
RLAF_DIR="$HOME/Thesis2/RLAF"

# Dataset: adjust these glob patterns / directory lists to match where you
# extracted your 3-coloring tarball.  Use the same train/val split as the paper.
TRAIN_PATH="dataRLAF/training/3col/**/*.cnf"
VAL_PATH="dataRLAF/validation/3col/**/*.cnf"

# Conda environment name (from environment.yml)
CONDA_ENV="NSNetArie"

# ── Weights & Biases: force offline mode (no interactive prompt) ───────────
export WANDB_MODE=offline

# ── Activate conda ─────────────────────────────────────────────────────────
# If conda is not on PATH after this, adjust the path below to your
# conda/miniconda installation (e.g. $HOME/miniconda3/etc/profile.d/conda.sh)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ── Change to the nsnet working directory ─────────────────────────────────
cd "$NSNET_DIR"

# ── Print environment info for debugging ──────────────────────────────────
echo "===== Job started: $(date) ====="
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no GPU info')"
echo "CPU cores available: $SLURM_CPUS_PER_TASK"
echo "Conda env: $CONDA_ENV  (Python: $(python --version))"
echo "NSNET_DIR: $NSNET_DIR"
echo "RLAF_DIR:  $RLAF_DIR"
echo "==========================================="

# ── Train (paper settings, no COOC) ───────────────────────────────────────
# Key settings matching the RLAF paper:
#   method=grpo              GRPO policy-gradient training
#   use_cooc=false           No literal co-occurrence edges
#   training.iterations=2000 2000 training iterations
#   training.cnf_per_iter=100  100 CNF instances per iteration
#   training.num_samples=40    40 policy samples per CNF (GRPO group size)
#   training.steps_per_iter=50 50 gradient steps per iteration
#   training.clip_ratio=0.2    PPO clip ratio
#   training.kl_penalty=0.1    KL divergence penalty weight (paper default)
#   solver.solver=glucose      Glucose CDCL solver (paper default for 3col)
#   solver.num_workers=16      Parallel solver processes (tune to available cores)
#   solver.solver_dir          Path to built glucose binary
#   solver.params.cpu-lim=60   60-second per-instance CPU limit for glucose
#   solver.params.rnd-freq=0.0 No random branching frequency (paper default)
#   solver.params.K=0.1        Glucose restart multiplier (paper default)

python train_arienet_rlaf.py \
    -n ArieNet_3col_noCOOC \
    method=grpo \
    use_cooc=false \
    training.iterations=2000 \
    training.cnf_per_iter=100 \
    training.num_samples=40 \
    training.steps_per_iter=50 \
    training.clip_ratio=0.2 \
    training.kl_penalty=0.1 \
    training.use_amp=true \
    training.accum_steps=1 \
    training.target_stat=decisions \
    solver.solver=glucose \
    solver.solver_dir="$RLAF_DIR" \
    solver.num_workers=16 \
    "solver.params={cpu-lim: 60, rnd-freq: 0.0, K: 0.1}" \
    "dataset.train_path=$TRAIN_PATH" \
    "dataset.val_path=$VAL_PATH" \
    dataset.num_process_workers=4 \
    loader.batch_size=20 \
    loader.num_workers=0 \
    optim.lr=5e-5 \
    optim.weight_decay=0.0 \
    scale_sigma=0.1 \
    val_interval=5 \
    seed=0 \
    wandb.project=nsnet-rlaf \
    wandb.name=ArieNet_3col_noCOOC

echo "===== Job finished: $(date) ====="
