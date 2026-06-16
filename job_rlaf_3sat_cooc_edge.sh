#!/bin/bash
#SBATCH -J rlaf_3sat_cooc_edge
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH -t 24:00:00
#SBATCH --output=slurm_rlaf_3sat_cooc_edge_%j.out
#SBATCH --error=slurm_rlaf_3sat_cooc_edge_%j.err

# ─── Snellius RLAF Training: ArieNetCoocEdge, glucose solver ───────────────
# Uses use_cooc_edge=true → ArieNetRLAFCoocEdge, where the L2L co-occurrence
# step is a full cycle member: c2l → L2L → l2c → c2l
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

NSNET_DIR="$HOME/thesis/ArieNetScriptie"
TRAIN_PATH="/scratch-shared/shoffman/data/data/training/3sat/**/*.cnf"
VAL_PATH="/scratch-shared/shoffman/data/data/validation/3sat/**/*.cnf"
CONDA_ENV="NSNetArie"

export WANDB_MODE=offline

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$NSNET_DIR"

echo "===== Job started: $(date) ====="
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no GPU info')"
echo "CPU cores available: $SLURM_CPUS_PER_TASK"
echo "Conda env: $CONDA_ENV  (Python: $(python --version))"
echo "NSNET_DIR: $NSNET_DIR"
echo "==========================================="

python -u train_arienet_rlaf.py \
    -n ArieNet_3sat_COOCEdge \
    method=grpo \
    use_cooc=false \
    use_cooc_edge=true \
    combine_mode=add \
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
    optim.lr=5e-5 \
    optim.weight_decay=0.0 \
    scale_sigma=0.1 \
    val_interval=5 \
    ckpt_interval=100 \
    seed=42 \
    wandb.project=nsnet-rlaf \
    wandb.name=ArieNet_3sat_COOCEdge

echo "===== Job finished: $(date) ====="
