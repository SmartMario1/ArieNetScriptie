#!/bin/bash
#SBATCH -J rlaf_test
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH -t 00:10:00
#SBATCH --output=slurm_rlaf_test_%j.out
#SBATCH --error=slurm_rlaf_test_%j.err

# ─── Quick sanity-check job (1 iteration, matches local-PC settings) ──────
# Purpose: verify the pipeline runs end-to-end on Snellius:
#   - conda env loads correctly
#   - glucose_weighted binary executes on this node
#   - BPG graph cache builds
#   - one full GRPO iteration completes (sample → solve → gradient step)
#
# Settings intentionally match the local-PC run that worked:
#   - no cpu-lim (solver.params={})     so hard instances don't block
#   - num_samples=20, cnf_per_iter=50   smaller workload
#   - num_workers=4                     matches local PC
#   - use_cooc=false                    baseline model, less memory
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

NSNET_DIR="$HOME/thesis/ArieNetScriptie"
TRAIN_PATH="../3col/**/*.cnf"
VAL_PATH="../3col_val/**/*.cnf"
CONDA_ENV="NSNetArie"

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$NSNET_DIR"

echo "===== Job started: $(date) ====="
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no GPU info')"
echo "Python: $(python --version)"
echo "==========================================="

# Time a single iteration with local-PC-equivalent settings.
# Watch the "Solved X formulas in Ys" line — it tells us per-formula speed.
python train_arienet_rlaf.py \
    -n rlaf_test \
    method=grpo \
    use_cooc=false \
    training.iterations=1 \
    training.cnf_per_iter=50 \
    training.num_samples=20 \
    training.steps_per_iter=10 \
    training.clip_ratio=0.2 \
    training.kl_penalty=0.1 \
    training.use_amp=true \
    training.accum_steps=5 \
    solver.solver=glucose \
    solver.num_workers=4 \
    "solver.params={}" \
    "dataset.train_path=$TRAIN_PATH" \
    "dataset.val_path=$VAL_PATH" \
    dataset.num_process_workers=2 \
    loader.batch_size=2 \
    loader.num_workers=0 \
    optim.lr=5e-5 \
    scale_sigma=0.1 \
    val_interval=1 \
    seed=42 \
    wandb.project=nsnet-rlaf \
    wandb.name=rlaf_test

echo "===== Job finished: $(date) ====="
