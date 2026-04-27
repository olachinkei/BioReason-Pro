#!/usr/bin/env bash
#SBATCH --mem=0
#SBATCH -o /mnt/data/%u/BioReason-Pro/runtime_logs/coreweave/train_rl_phase_a_a1_paper_2node_%j.log
#
# Phase A1 paper-exact launcher: 2 nodes x 8 GPUs, 8 queries x 24 rollouts,
# 10k generation tokens, and per-aspect IA-F1 reward.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
export PROJECT_ROOT

default_runtime_root() {
  if [ -n "${BIOREASON_RUNTIME_ROOT:-}" ]; then
    printf '%s\n' "$BIOREASON_RUNTIME_ROOT"
    return 0
  fi
  if [ -d "/mnt/data" ] && [ -n "${USER:-}" ]; then
    printf '/mnt/data/%s/BioReason-Pro\n' "$USER"
    return 0
  fi
  printf '%s\n' "$PROJECT_ROOT"
}

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

REGISTRY_ENV_FILE="${REGISTRY_ENV_FILE:-configs/disease_benchmark/wandb_registry_paths.env}"
if [ -f "$REGISTRY_ENV_FILE" ]; then
  set -a
  source "$REGISTRY_ENV_FILE"
  set +a
fi

source "$PROJECT_ROOT/.venv-gpu/bin/activate"

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"

export BIOREASON_RUNTIME_ROOT="${BIOREASON_RUNTIME_ROOT:-$(default_runtime_root)}"
export BIOREASON_ARTIFACTS_ROOT="${BIOREASON_ARTIFACTS_ROOT:-${BIOREASON_RUNTIME_ROOT}/data/artifacts}"
export BIOREASON_CACHE_ROOT="${BIOREASON_CACHE_ROOT:-${BIOREASON_RUNTIME_ROOT}/cache}"
export WANDB_DIR="${WANDB_DIR:-${BIOREASON_RUNTIME_ROOT}/wandb}"
export WEAVE_SERVER_CACHE_DIR="${WEAVE_SERVER_CACHE_DIR:-${WANDB_DIR}/weave_server_cache}"
export HF_HOME="${HF_HOME:-${BIOREASON_CACHE_ROOT}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${BIOREASON_CACHE_ROOT}/xdg_config}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${BIOREASON_CACHE_ROOT}/triton}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${XDG_CONFIG_HOME}/vllm}"
mkdir -p "$BIOREASON_ARTIFACTS_ROOT" "$BIOREASON_CACHE_ROOT" "$WANDB_DIR" "$WEAVE_SERVER_CACHE_DIR" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$XDG_CONFIG_HOME" "$TRITON_CACHE_DIR" "$VLLM_CONFIG_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1

export WANDB_ENTITY="${WANDB_ENTITY:-wandb-healthcare}"
export WANDB_PROJECT="${WANDB_PROJECT:-bioreason-pro-custom}"
export BASE_WANDB_PROJECT="${BASE_WANDB_PROJECT:-$WANDB_PROJECT}"
export WEAVE_PROJECT="${WEAVE_PROJECT:-${WANDB_ENTITY}/${WANDB_PROJECT}}"
export DATA_BUNDLE="${DATA_BUNDLE:-main_production}"
export REGISTRY_ENV_FILE

export ABLATION=A1
export REASONING_PROMPT_STYLE=paper_native_tight
export NNODES=2
export GPUS_PER_NODE=8
export QUERIES_PER_STEP=8
export ROLLOUTS_PER_QUERY=24
export OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU=6
export GRADIENT_ACCUMULATION_STEPS=2
export MAX_NEW_TOKENS=10000
export ROLLOUT_MAX_NEW_TOKENS=10000
export ROLLOUT_WORKER_GENERATE_TIMEOUT_S=1500
export ROLLOUT_WORKER_STARTUP_RETRY_COUNT=3
export ROLLOUT_WORKER_STARTUP_RETRY_SLEEP_S=15
export ROLLOUT_LOGPROB_MICROBATCH_SIZE=4
export MAX_LOSS_COMPLETION_TOKENS=0
export VLLM_GPU_MEMORY_UTILIZATION=0.35
export VLLM_MAX_MODEL_LEN=32768
export VLLM_MAX_NUM_SEQS=32
export VLLM_CPU_OFFLOAD_GB=0
export VLLM_SWAP_SPACE_GB=0
export VLLM_ENFORCE_EAGER=true
export VLLM_ENABLE_SLEEP_MODE=false
export VLLM_SLEEP_LEVEL=1
export VLLM_ATTENTION_BACKEND="${BIOREASON_VLLM_ATTENTION_BACKEND:-XFORMERS}"
export VLLM_WORKER_MULTIPROC_METHOD="${BIOREASON_VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V1="${BIOREASON_VLLM_USE_V1:-0}"

export MAX_STEPS=20
export VALIDATION_NUM_PROTEINS=200
export VALIDATION_EVERY_N_STEPS=5
export SAVE_EVERY_N_STEPS=10

export WANDB_RUN_NAME="${WANDB_RUN_NAME:-rl-phase-a-A1-paper-2node-${TS}}"
export OUTPUT_DIR="${BIOREASON_ARTIFACTS_ROOT}/models/train_rl_output_phase_a_a1_paper_2node_${TS}"
export CHECKPOINT_ARTIFACT_NAME="${CHECKPOINT_ARTIFACT_NAME:-train-rl-output-phase-a-a1-paper-2node}"
export CHECKPOINT_ARTIFACT_ALIASES="${CHECKPOINT_ARTIFACT_ALIASES:-latest,phase-a-a1,paper-2node}"
export CHECKPOINT_EXPORT_ONLY="${CHECKPOINT_EXPORT_ONLY:-true}"
export EXECUTION_ID="${EXECUTION_ID:-${SLURM_JOB_ID:-local}-${TS}}"
export SYNC_ROOT="${SYNC_ROOT:-}"
export RESUME_FROM_EXPORT_ARTIFACT="${RESUME_FROM_EXPORT_ARTIFACT:-}"
export RESUME_MODE="${RESUME_MODE:-warm}"
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
export MASTER_PORT="${MASTER_PORT:-29511}"

echo "[phase-a-paper] launching A1 exact with paper values: nodes=$NNODES, gpus_per_node=$GPUS_PER_NODE, queries=$QUERIES_PER_STEP, rollouts=$ROLLOUTS_PER_QUERY, local_rollouts_per_rank=12, max_new_tokens=$MAX_NEW_TOKENS, optimizer_micro_batch=$OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU, grad_accum=$GRADIENT_ACCUMULATION_STEPS"

srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 bash -lc '
  set -euo pipefail
  cd "$PROJECT_ROOT"
  source "$PROJECT_ROOT/.venv-gpu/bin/activate"
  export NODE_RANK="$SLURM_NODEID"
  export HOSTFILE=""
  bash scripts/sh_train_protein_grpo.sh \
    --max_steps "$MAX_STEPS" \
    --validation_num_proteins "$VALIDATION_NUM_PROTEINS" \
    --validation_every_n_steps "$VALIDATION_EVERY_N_STEPS" \
    --save_every_n_steps "$SAVE_EVERY_N_STEPS" \
    --reward_mode per_aspect_ia_f1 \
    --disease_loss_weight 1.0 \
    --ablation_tag phase-a-A1 \
    --wandb_tags phase-a-disease-pilot per-aspect-ia-f1 paper-2node-exact \
    --trace_rollouts_to_weave true \
    --weave_trace_budget 64 \
    --weave_trace_full_group_count 4 \
    --weave_trace_full_rollouts_per_group 24
'
