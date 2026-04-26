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

_CALLER_WANDB_PROJECT="${WANDB_PROJECT:-}"
_CALLER_WANDB_ENTITY="${WANDB_ENTITY:-}"
_CALLER_BASE_WANDB_PROJECT="${BASE_WANDB_PROJECT:-}"
_CALLER_WEAVE_PROJECT="${WEAVE_PROJECT:-}"

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

if [ -n "$_CALLER_WANDB_PROJECT" ]; then
  export WANDB_PROJECT="$_CALLER_WANDB_PROJECT"
fi
if [ -n "$_CALLER_WANDB_ENTITY" ]; then
  export WANDB_ENTITY="$_CALLER_WANDB_ENTITY"
fi
if [ -n "$_CALLER_BASE_WANDB_PROJECT" ]; then
  export BASE_WANDB_PROJECT="$_CALLER_BASE_WANDB_PROJECT"
fi
if [ -n "$_CALLER_WEAVE_PROJECT" ]; then
  export WEAVE_PROJECT="$_CALLER_WEAVE_PROJECT"
fi

export WANDB_ENTITY="${WANDB_ENTITY:-wandb-healthcare}"
export WANDB_PROJECT="${WANDB_PROJECT:-bioreason-pro}"
export BIOREASON_WANDB_RUN_ID="${BIOREASON_WANDB_RUN_ID:-}"
export BIOREASON_WANDB_RESUME="${BIOREASON_WANDB_RESUME:-}"
export BASE_WANDB_PROJECT="${BASE_WANDB_PROJECT:-$WANDB_PROJECT}"
export WEAVE_PROJECT="${WEAVE_PROJECT:-${WANDB_ENTITY}/${WANDB_PROJECT}}"
export REGISTRY_ENV_FILE="${REGISTRY_ENV_FILE:-configs/disease_benchmark/wandb_registry_paths.env}"
export BASE_CHECKPOINT="${BASE_CHECKPOINT:-wandb-healthcare/bioreason-pro/bioreason-pro-rl:latest}"
export TEMPORAL_SPLIT_ARTIFACT="${TEMPORAL_SPLIT_ARTIFACT:-wandb-healthcare/bioreason-pro/disease-temporal-split:production}"
export DATASET_ARTIFACT="${DATASET_ARTIFACT:-wandb-healthcare/bioreason-pro/disease-temporal-reasoning:production}"
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
export MAX_STEPS="${MAX_STEPS:-20}"
export VALIDATION_EVERY_N_STEPS="${VALIDATION_EVERY_N_STEPS:-5}"
export VALIDATION_NUM_PROTEINS="${VALIDATION_NUM_PROTEINS:-10}"
export SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-10}"
export REWARD_WEIGHTS="${REWARD_WEIGHTS:-}"
export DISEASE_WEIGHTING_MODE="${DISEASE_WEIGHTING_MODE:-uniform_fallback}"
export TRACE_ROLLOUTS_TO_WEAVE="${TRACE_ROLLOUTS_TO_WEAVE:-true}"
export WEAVE_TRACE_BUDGET="${WEAVE_TRACE_BUDGET:-64}"
export WEAVE_TRACE_FULL_GROUP_COUNT="${WEAVE_TRACE_FULL_GROUP_COUNT:-4}"
export WEAVE_TRACE_FULL_ROLLOUTS_PER_GROUP="${WEAVE_TRACE_FULL_ROLLOUTS_PER_GROUP:-24}"
export ROLLOUT_GENERATE_TIMEOUT_SECONDS="${ROLLOUT_GENERATE_TIMEOUT_SECONDS:-1200}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export LIGHTWEIGHT_PROFILE="${LIGHTWEIGHT_PROFILE:-false}"
export ALLOW_LIGHTWEIGHT_MULTI_NODE="${ALLOW_LIGHTWEIGHT_MULTI_NODE:-false}"
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
export MASTER_PORT="${MASTER_PORT:-29511}"
export ABLATION

lightweight_flag="$(printf '%s' "$LIGHTWEIGHT_PROFILE" | tr '[:upper:]' '[:lower:]')"
allow_lightweight_multi_node_flag="$(printf '%s' "$ALLOW_LIGHTWEIGHT_MULTI_NODE" | tr '[:upper:]' '[:lower:]')"
if [ "$lightweight_flag" = "1" ] || [ "$lightweight_flag" = "true" ] || [ "$lightweight_flag" = "yes" ]; then
  if [ "$NNODES" != "1" ] && [ "$allow_lightweight_multi_node_flag" != "1" ] && [ "$allow_lightweight_multi_node_flag" != "true" ] && [ "$allow_lightweight_multi_node_flag" != "yes" ]; then
    echo "Warning: LIGHTWEIGHT_PROFILE requested with NNODES=$NNODES; ignoring unless ALLOW_LIGHTWEIGHT_MULTI_NODE=true."
  else
    # Low-memory profile for one-node debugging/baselines: fewer generations,
    # and a microbatch contract compatible with rollouts=5.
    export ROLLOUTS_PER_QUERY="${LIGHTWEIGHT_ROLLOUTS_PER_QUERY:-5}"
    export OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU="${LIGHTWEIGHT_OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU:-5}"
    export GRADIENT_ACCUMULATION_STEPS="${LIGHTWEIGHT_GRADIENT_ACCUMULATION_STEPS:-1}"
    export MAX_NEW_TOKENS="${LIGHTWEIGHT_MAX_NEW_TOKENS:-4096}"
    export VLLM_MAX_NUM_SEQS="${LIGHTWEIGHT_VLLM_MAX_NUM_SEQS:-16}"
    echo "Info: LIGHTWEIGHT_PROFILE enabled (rollouts=${ROLLOUTS_PER_QUERY}, micro_batch=${OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU}, grad_accum=${GRADIENT_ACCUMULATION_STEPS})."
  fi
fi

srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 bash -lc '
  set -euo pipefail
  cd '"$PROJECT_ROOT"'
  source "'"$PROJECT_ROOT"'"/.venv-gpu/bin/activate
  export WANDB_ENTITY="'"$WANDB_ENTITY"'"
  export WANDB_PROJECT="'"$WANDB_PROJECT"'"
  export BIOREASON_WANDB_RUN_ID="'"$BIOREASON_WANDB_RUN_ID"'"
  export BIOREASON_WANDB_RESUME="'"$BIOREASON_WANDB_RESUME"'"
  export BASE_WANDB_PROJECT="'"$BASE_WANDB_PROJECT"'"
  export WEAVE_PROJECT="'"$WEAVE_PROJECT"'"
  export REGISTRY_ENV_FILE="'"$REGISTRY_ENV_FILE"'"
  export BASE_CHECKPOINT="'"$BASE_CHECKPOINT"'"
  export TEMPORAL_SPLIT_ARTIFACT="'"$TEMPORAL_SPLIT_ARTIFACT"'"
  export DATASET_ARTIFACT="'"$DATASET_ARTIFACT"'"
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
  export MAX_STEPS="'"$MAX_STEPS"'"
  export VALIDATION_EVERY_N_STEPS="'"$VALIDATION_EVERY_N_STEPS"'"
  export VALIDATION_NUM_PROTEINS="'"$VALIDATION_NUM_PROTEINS"'"
  export SAVE_EVERY_N_STEPS="'"$SAVE_EVERY_N_STEPS"'"
  export REWARD_WEIGHTS="'"$REWARD_WEIGHTS"'"
  export DISEASE_WEIGHTING_MODE="'"$DISEASE_WEIGHTING_MODE"'"
  export TRACE_ROLLOUTS_TO_WEAVE="'"$TRACE_ROLLOUTS_TO_WEAVE"'"
  export WEAVE_TRACE_BUDGET="'"$WEAVE_TRACE_BUDGET"'"
  export WEAVE_TRACE_FULL_GROUP_COUNT="'"$WEAVE_TRACE_FULL_GROUP_COUNT"'"
  export WEAVE_TRACE_FULL_ROLLOUTS_PER_GROUP="'"$WEAVE_TRACE_FULL_ROLLOUTS_PER_GROUP"'"
  export ROLLOUT_GENERATE_TIMEOUT_SECONDS="'"$ROLLOUT_GENERATE_TIMEOUT_SECONDS"'"
  export NCCL_TIMEOUT="'"$NCCL_TIMEOUT"'"
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="'"$TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"'"
  export MASTER_ADDR="'"$MASTER_ADDR"'"
  export MASTER_PORT="'"$MASTER_PORT"'"
  export ABLATION="'"$ABLATION"'"
  export NODE_RANK="$SLURM_NODEID"
  export HOSTFILE=""
  bash scripts/sh_train_protein_grpo_phase_a.sh \
    --trace_rollouts_to_weave "$TRACE_ROLLOUTS_TO_WEAVE" \
    --weave_trace_budget "$WEAVE_TRACE_BUDGET" \
    --weave_trace_full_group_count "$WEAVE_TRACE_FULL_GROUP_COUNT" \
    --weave_trace_full_rollouts_per_group "$WEAVE_TRACE_FULL_ROLLOUTS_PER_GROUP" \
    --max_steps "$MAX_STEPS" \
    --validation_every_n_steps "$VALIDATION_EVERY_N_STEPS" \
    --validation_num_proteins "$VALIDATION_NUM_PROTEINS" \
    --save_every_n_steps "$SAVE_EVERY_N_STEPS"
'
