#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"

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
export WANDB_PROJECT="${WANDB_PROJECT:-bioreason-pro}"
export BASE_WANDB_PROJECT="${BASE_WANDB_PROJECT:-$WANDB_PROJECT}"
export WEAVE_PROJECT="${WEAVE_PROJECT:-${WANDB_ENTITY}/${WANDB_PROJECT}}"
export REGISTRY_ENV_FILE="${REGISTRY_ENV_FILE:-configs/disease_benchmark/wandb_registry_paths.env}"
export REASONING_PROMPT_STYLE="${REASONING_PROMPT_STYLE:-paper_native_tight}"
export CHECKPOINT_ARTIFACT_NAME="${CHECKPOINT_ARTIFACT_NAME:-train-rl-paper-native-tight-2node}"
export CHECKPOINT_ARTIFACT_ALIASES="${CHECKPOINT_ARTIFACT_ALIASES:-latest,paper-native-tight,2node}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-rl-paper-native-tight-2node-srun-$(date +%Y%m%d-%H%M%S)}"
export OUTPUT_DIR="${OUTPUT_DIR:-data/artifacts/models/train_rl_output/paper_native_tight_2node_srun}"
export NNODES="${NNODES:-2}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export QUERIES_PER_STEP="${QUERIES_PER_STEP:-8}"
export ROLLOUTS_PER_QUERY="${ROLLOUTS_PER_QUERY:-24}"
export OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU="${OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU:-6}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10000}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.35}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
export ROLLOUT_LOGPROB_MICROBATCH_SIZE="${ROLLOUT_LOGPROB_MICROBATCH_SIZE:-4}"
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
export MASTER_PORT="${MASTER_PORT:-29511}"

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
  export CHECKPOINT_ARTIFACT_NAME="'"$CHECKPOINT_ARTIFACT_NAME"'"
  export CHECKPOINT_ARTIFACT_ALIASES="'"$CHECKPOINT_ARTIFACT_ALIASES"'"
  export WANDB_RUN_NAME="'"$WANDB_RUN_NAME"'"
  export OUTPUT_DIR="'"$OUTPUT_DIR"'"
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
  export ROLLOUT_LOGPROB_MICROBATCH_SIZE="'"$ROLLOUT_LOGPROB_MICROBATCH_SIZE"'"
  export MASTER_ADDR="'"$MASTER_ADDR"'"
  export MASTER_PORT="'"$MASTER_PORT"'"
  export NODE_RANK="$SLURM_NODEID"
  export HOSTFILE=""
  bash scripts/sh_train_protein_grpo.sh --max_steps 20 --validation_every_n_steps 5 --save_every_n_steps 10
'
