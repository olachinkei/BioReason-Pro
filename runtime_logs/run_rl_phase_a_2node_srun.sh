#!/usr/bin/env bash
#SBATCH --mem=0
#SBATCH -o /mnt/home/%u/BioReason-Pro/runtime_logs/coreweave/train_rl_phase_a_2node_srun_%j.log
#
# Phase A (A0/A1/A2/A3) launcher on 2 nodes x 8 GPUs.
#
# Structurally identical to runtime_logs/run_rl_paper_tight_2node_srun.sh
# (Kei's known-good 2-node launcher on commit 19a442f), with two deltas:
#
#   1. Accepts an ABLATION env var (A0|A1|A2|A3), which it propagates into
#      the srun'd worker shells so scripts/sh_train_protein_grpo_phase_a.sh
#      can pick it up.
#   2. Invokes scripts/sh_train_protein_grpo_phase_a.sh instead of
#      scripts/sh_train_protein_grpo.sh, so the per-ablation
#      --reward_mode / --disease_loss_weight / --ablation_tag / --wandb_tags
#      flag set is applied. The phase_a wrapper exec's the production
#      launcher internally, so all distributed/rollout/vLLM contracts are
#      preserved exactly as Kei's known-good run.
#
# Usage:
#
#   ABLATION=A0 sbatch runtime_logs/run_rl_phase_a_2node_srun.sh
#   ABLATION=A1 sbatch runtime_logs/run_rl_phase_a_2node_srun.sh
#
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"

ABLATION="${ABLATION:-}"
if [ -z "$ABLATION" ]; then
  echo "Error: set ABLATION=A0|A1|A2|A3 before sbatch." >&2
  exit 1
fi

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

if [ -f configs/disease_benchmark/wandb_registry_paths.env ]; then
  set -a
  source configs/disease_benchmark/wandb_registry_paths.env
  set +a
fi

source "$PROJECT_ROOT/.venv-gpu/bin/activate"

export WANDB_ENTITY="${WANDB_ENTITY:-wandb-healthcare}"
export WANDB_PROJECT="${WANDB_PROJECT:-bioreason-pro-custom}"
export BASE_WANDB_PROJECT="${BASE_WANDB_PROJECT:-$WANDB_PROJECT}"
export WEAVE_PROJECT="${WEAVE_PROJECT:-${WANDB_ENTITY}/${WANDB_PROJECT}}"
export REGISTRY_ENV_FILE="${REGISTRY_ENV_FILE:-configs/disease_benchmark/wandb_registry_paths.env}"
export REASONING_PROMPT_STYLE="${REASONING_PROMPT_STYLE:-paper_native_tight}"
export NNODES="${NNODES:-2}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export QUERIES_PER_STEP="${QUERIES_PER_STEP:-8}"
export ROLLOUTS_PER_QUERY="${ROLLOUTS_PER_QUERY:-24}"
export OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU="${OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU:-6}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10000}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.35}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export VLLM_SWAP_SPACE_GB="${VLLM_SWAP_SPACE_GB:-0}"
export ROLLOUT_LOGPROB_MICROBATCH_SIZE="${ROLLOUT_LOGPROB_MICROBATCH_SIZE:-4}"
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
export MASTER_PORT="${MASTER_PORT:-29511}"
export ABLATION

srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 bash -lc '
  set -euo pipefail
  cd '"$PROJECT_ROOT"'
  source "'"$PROJECT_ROOT"'"/.venv-gpu/bin/activate
  export WANDB_ENTITY="'"$WANDB_ENTITY"'"
  export WANDB_PROJECT="'"$WANDB_PROJECT"'"
  export BASE_WANDB_PROJECT="'"$BASE_WANDB_PROJECT"'"
  export WEAVE_PROJECT="'"$WEAVE_PROJECT"'"
  export REGISTRY_ENV_FILE="'"$REGISTRY_ENV_FILE"'"
  export REASONING_PROMPT_STYLE="'"$REASONING_PROMPT_STYLE"'"
  export NNODES="'"$NNODES"'"
  export GPUS_PER_NODE="'"$GPUS_PER_NODE"'"
  export QUERIES_PER_STEP="'"$QUERIES_PER_STEP"'"
  export ROLLOUTS_PER_QUERY="'"$ROLLOUTS_PER_QUERY"'"
  export OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU="'"$OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU"'"
  export GRADIENT_ACCUMULATION_STEPS="'"$GRADIENT_ACCUMULATION_STEPS"'"
  export MAX_NEW_TOKENS="'"$MAX_NEW_TOKENS"'"
  export VLLM_GPU_MEMORY_UTILIZATION="'"$VLLM_GPU_MEMORY_UTILIZATION"'"
  export VLLM_MAX_MODEL_LEN="'"$VLLM_MAX_MODEL_LEN"'"
  export VLLM_MAX_NUM_SEQS="'"$VLLM_MAX_NUM_SEQS"'"
  export VLLM_SWAP_SPACE_GB="'"$VLLM_SWAP_SPACE_GB"'"
  export ROLLOUT_LOGPROB_MICROBATCH_SIZE="'"$ROLLOUT_LOGPROB_MICROBATCH_SIZE"'"
  export MASTER_ADDR="'"$MASTER_ADDR"'"
  export MASTER_PORT="'"$MASTER_PORT"'"
  export ABLATION="'"$ABLATION"'"
  export NODE_RANK="$SLURM_NODEID"
  export HOSTFILE=""
  bash scripts/sh_train_protein_grpo_phase_a.sh --max_steps 20 --validation_every_n_steps 5 --save_every_n_steps 10
'
