#!/bin/bash
#SBATCH -J rlaf_crypto_nolsp
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH -t 4:00:00
#SBATCH --output=slurm_rlaf_crypto_nolsp_%j.out
#SBATCH --error=slurm_rlaf_crypto_nolsp_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# ─── Snellius RLAF Training: no COOC, no local-satisfaction features, crypto ─
# Same as job_rlaf_crypto.sh but with model.no_precomputed_local_sat=true.
# The model receives zeros for the local-satisfaction-percentage edge feature
# at every round.  No co-occurrence edges.
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
NSNET_DIR="$HOME/thesis/ArieNetScriptie"

TRAIN_PATH="/scratch-shared/shoffman/data/data/training/crypto/**/*.cnf"
VAL_PATH="/scratch-shared/shoffman/data/data/validation/crypto/**/*.cnf"

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

# ── Train (no COOC, no precomputed local sat, glucose) ────────────────────
python -u train_arienet_rlaf.py \
    -n ArieNet_crypto_noLSP \
    method=grpo \
    use_cooc=false \
    model.no_precomputed_local_sat=true \
    training.iterations=2000 \
    training.cnf_per_iter=100 \
    training.num_samples=40 \
    training.steps_per_iter=25 \
    training.clip_ratio=0.2 \
    training.kl_penalty=0.1 \
    training.use_amp=true \
    training.accum_steps=4 \
    training.target_stat=decisions \
    solver.solver=glucose \
    solver.num_workers=16 \
    "dataset.train_path=$TRAIN_PATH" \
    "dataset.val_path=$VAL_PATH" \
    dataset.num_process_workers=4 \
    loader.batch_size=5 \
    loader.num_workers=0 \
    optim.lr=1e-4 \
    optim.weight_decay=0.0 \
    scale_sigma=0.1 \
    val_interval=5 \
    ckpt_interval=100 \
    seed=42 \
    wandb.project=nsnet-rlaf \
    wandb.name=ArieNet_crypto_noLSP

echo "===== Job finished: $(date) ====="
