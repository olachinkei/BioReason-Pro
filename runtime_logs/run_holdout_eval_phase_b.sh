#!/usr/bin/env bash
#SBATCH --job-name=eval_holdout
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH -o /mnt/home/%u/BioReason-Pro/runtime_logs/coreweave/eval_holdout_%j.log
#
# Hold-out (test split, ~400 proteins) evaluation.
#
# Evaluates (holdout-available group):
#   1. bioreason-pro-rl-paper  (baseline)
#   2. paper-native-tight-step10 (run aldhlmrz, best early checkpoint)
#   3. paper-native-tight-step20 (run aldhlmrz, final checkpoint)
#
# Usage:
#   sbatch runtime_logs/run_holdout_eval_phase_b.sh
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

TARGET_GROUP="${TARGET_GROUP:-holdout-available}"

echo "=========================================="
echo "Hold-out evaluation: $TARGET_GROUP"
echo "=========================================="

python scripts/run_registered_eval.py \
  --target-group "$TARGET_GROUP" \
  --split test \
  --reasoning-prompt-style paper_native_tight \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --weave-project "$WEAVE_PROJECT" \
  --keep-local-eval-outputs \
  --continue-on-error

echo "=========================================="
echo "Hold-out evaluation complete."
echo "=========================================="
