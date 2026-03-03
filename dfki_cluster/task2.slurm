#!/bin/bash
#SBATCH --job-name=obfuscation
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --mem-per-cpu=10G
#SBATCH --output=out/task2_%j.out
#SBATCH --error=out/task2_%j.err
#SBATCH --ntasks=1
#SBATCH --time=3-00:00
#SBATCH --partition=RTXA6000,RTX3090,V100-32GB,batch

set -euo pipefail

# Host paths (netscratch)
REPO=/netscratch/bfallah/indic-language-identification
HF_HOME=/netscratch/bfallah/.cache/huggingface
TMPDIR=/netscratch/bfallah/tmp/${SLURM_JOB_ID}
IMAGE=/netscratch/bfallah/enroot/indic-slid-cu118.sqsh

mkdir -p "$REPO/out" "$HF_HOME/datasets" "$HF_HOME/hub" "$TMPDIR"

# Required secrets must exist on host so they're exported into container
: "${HF_TOKEN:?HF_TOKEN is not set. Export it before sbatch submission.}"
: "${WANDB_API_KEY:?WANDB_API_KEY is not set. Export it before sbatch submission.}"

# Optional but strongly recommended: keep enroot unpack/cache on netscratch (more space, fewer weird failures)
export ENROOT_CACHE_PATH=/netscratch/bfallah/enroot/cache
export ENROOT_DATA_PATH=/netscratch/bfallah/enroot/data
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH"

srun \
  --container-image="$IMAGE" \
  --container-workdir=/app \
  --container-mounts="$REPO:/app,$HF_HOME:/root/.cache/huggingface,$TMPDIR:/tmp" \
  --export=ALL \
  bash -lc '
    set -euo pipefail

    echo "Inside container:"
    which python
    python -c "import torch, torchaudio; print(\"torch\", torch.__version__); print(\"torchaudio\", torchaudio.__version__)"

    # Quick SoX check (fail fast if broken)
    python - <<PY
import torch
from torchaudio.sox_effects import apply_effects_tensor
x = torch.zeros(1, 16000)
y, sr = apply_effects_tensor(x, 16000, [["rate","16000"]])
print("SoX OK", y.shape, sr)
PY

    cd /app
    python train_model.py --config configs/task2_ablation1_timbre.yaml
  '