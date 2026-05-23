#!/bin/bash
#SBATCH -J rlaf_3col_cooc
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH -t 00:15:00
#SBATCH --output=slurm_rlaf_3col_cooc_%j.out
#SBATCH --error=slurm_rlaf_3col_cooc_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# ─── Snellius RLAF Training: ArieNetCooc (with COOC), 3-coloring ──────────
# Paper settings: GRPO, 2000 iterations, glucose solver
# Run WITH co-occurrence graph extension (ArieNetRLAFCooc)
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths (adjust if your layout differs) ─────────────────────────────────
NSNET_DIR="$HOME/thesis/ArieNetScriptie"

# Dataset: adjust these glob patterns to match where you extracted your tarball.
TRAIN_PATH="../3col/**/*.cnf"
VAL_PATH="../3col_val/**/*.cnf"

# Conda environment name (from environment.yml)
CONDA_ENV="NSNetArie"

# ── Weights & Biases: force offline mode (no interactive prompt) ───────────
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Activate conda ─────────────────────────────────────────────────────────
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
echo "==========================================="

# ── Train (paper settings, with COOC) ─────────────────────────────────────
# use_cooc=true switches the model to ArieNetRLAFCooc, which adds a
# literal co-occurrence message-passing step.  The dataset class will also
# apply unit propagation before building BPG objects.
#
# All GRPO/solver hyperparameters are identical to the no-COOC run so that
# the two runs are directly comparable.

python train_arienet_rlaf.py \
    -n ArieNet_3col_COOC \
    method=grpo \
    use_cooc=true \
    training.iterations=2000 \
    training.cnf_per_iter=100 \
    training.num_samples=40 \
    training.steps_per_iter=50 \
    training.clip_ratio=0.2 \
    training.kl_penalty=0.1 \
    training.use_amp=true \
    training.accum_steps=2 \
    training.target_stat=decisions \
    solver.solver=glucose \
    solver.num_workers=16 \
    "dataset.train_path=$TRAIN_PATH" \
    "dataset.val_path=$VAL_PATH" \
    dataset.num_process_workers=4 \
    loader.batch_size=10 \
    loader.num_workers=0 \
    optim.lr=5e-5 \
    optim.weight_decay=0.0 \
    scale_sigma=0.1 \
    val_interval=5 \
    ckpt_interval=100 \
    seed=42 \
    wandb.project=nsnet-rlaf \
    wandb.name=ArieNet_3col_COOC

echo "===== Job finished: $(date) ====="
