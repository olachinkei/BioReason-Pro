#!/usr/bin/env bash
#SBATCH --job-name=eval_gen5_a1
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH -o /mnt/home/%u/BioReason-Pro/runtime_logs/coreweave/eval_gen5_a1_%j.log
#
# Download gen5-a1-rmix artifact, copy go_embedding.pt, and run holdout eval
#
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"

if [ -f .env ]; then
  set -a; source .env; set +a
fi
if [ -f configs/disease_benchmark/wandb_registry_paths.env ]; then
  set -a; source configs/disease_benchmark/wandb_registry_paths.env; set +a
fi

source "$PROJECT_ROOT/.venv-gpu/bin/activate"

export WANDB_ENTITY="${WANDB_ENTITY:-wandb-healthcare}"
export WANDB_PROJECT="${WANDB_PROJECT:-bioreason-pro}"
export WEAVE_PROJECT="${WEAVE_PROJECT:-${WANDB_ENTITY}/${WANDB_PROJECT}}"

ARTIFACT="wandb-healthcare/bioreason-pro/train-rl-gen5-a1-rmix-2node-step40-val40-best:v1"
LOCAL_DIR="/mnt/data/kkamata+cwb607/BioReason-Pro/data/artifacts/models/gen5_a1_rmix/inference_export"
BASE_MODEL="/mnt/data/kkamata+cwb607/BioReason-Pro/data/artifacts/models/bioreason_pro_rl_paper"

echo "=== Downloading artifact ==="
python -c "
import wandb, os, shutil
api = wandb.Api(timeout=120)
art = api.artifact('${ARTIFACT}')
dl_dir = art.download(root='/mnt/data/kkamata+cwb607/BioReason-Pro/data/artifacts/models/gen5_a1_rmix_raw')
print(f'Downloaded to: {dl_dir}')
src = os.path.join(dl_dir, 'inference_export')
dst = '${LOCAL_DIR}'
if os.path.isdir(src):
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        s = os.path.join(src, f)
        d = os.path.join(dst, f)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    print(f'Copied inference_export to {dst}')
else:
    print('No inference_export subdir, files should be at root')
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(dl_dir):
        s = os.path.join(dl_dir, f)
        d = os.path.join(dst, f)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
"

echo "=== Copying go_embedding.pt from base model ==="
cp "${BASE_MODEL}/go_embedding.pt" "${LOCAL_DIR}/go_embedding.pt"

echo "=== Verifying required files ==="
for f in config.json go_embedding.pt go_projection.pt protein_projection.pt protein_model/pytorch_model.bin; do
  if [ -f "${LOCAL_DIR}/${f}" ]; then
    echo "  OK: ${f}"
  else
    echo "  MISSING: ${f}"
  fi
done

echo "=== Running holdout evaluation ==="
python scripts/run_registered_eval.py \
  --target gen5-a1-rmix \
  --split test \
  --reasoning-prompt-style paper_native_tight \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --weave-project "$WEAVE_PROJECT" \
  --keep-local-eval-outputs \
  --continue-on-error

echo "=== Evaluation complete ==="
