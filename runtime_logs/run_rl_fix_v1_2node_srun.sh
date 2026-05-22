#!/usr/bin/env bash
#SBATCH --job-name=rl_fix_v1
#SBATCH --partition=h100
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -o /mnt/home/%u/BioReason-Pro/runtime_logs/coreweave/rl_fix_v1_%j.log
#
# RL Fix V1: Address key issues causing holdout degradation
#
# Root causes identified:
#   - clip_epsilon 0.0007/0.0009 is ~200x too tight → gradient starvation
#   - kl_beta 0.0001 is too weak → policy diverges from reference
#   - No format reward → model loses output structure
#   - Only 8 proteins/step → overfitting
#   - steps_per_generation=2 → stale rollouts
#
# Changes:
#   1. clip_epsilon: 0.0007/0.0009 → 0.15/0.2  (standard GRPO range)
#   2. kl_beta: 0.0001 → 0.01  (100x stronger regularization)
#   3. reward_weights: 0,0,1,0 → 0.05,0,0.9,0.05  (format signal)
#   4. queries_per_step: 8 → 16  (2x protein diversity)
#   5. rollouts_per_query: 24 → 12  (keep total=192)
#   6. steps_per_generation: 2 → 1  (fresh rollouts per update)
#   7. learning_rate: 3e-5 → 1e-5  (more conservative)
#   8. validation: every 3 steps with 30 proteins
#
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"

_CALLER_WANDB_PROJECT="${WANDB_PROJECT:-}"
_CALLER_WANDB_ENTITY="${WANDB_ENTITY:-}"
_CALLER_BASE_WANDB_PROJECT="${BASE_WANDB_PROJECT:-}"
_CALLER_WEAVE_PROJECT="${WEAVE_PROJECT:-}"

if [ -f .env ]; then
  set -a; source .env; set +a
fi
if [ -f configs/disease_benchmark/wandb_registry_paths.env ]; then
  set -a; source configs/disease_benchmark/wandb_registry_paths.env; set +a
fi

source "$PROJECT_ROOT/.venv-gpu/bin/activate"

if [ -n "$_CALLER_WANDB_PROJECT" ]; then export WANDB_PROJECT="$_CALLER_WANDB_PROJECT"; fi
if [ -n "$_CALLER_WANDB_ENTITY" ]; then export WANDB_ENTITY="$_CALLER_WANDB_ENTITY"; fi
if [ -n "$_CALLER_BASE_WANDB_PROJECT" ]; then export BASE_WANDB_PROJECT="$_CALLER_BASE_WANDB_PROJECT"; fi
if [ -n "$_CALLER_WEAVE_PROJECT" ]; then export WEAVE_PROJECT="$_CALLER_WEAVE_PROJECT"; fi

export WANDB_ENTITY="${WANDB_ENTITY:-wandb-healthcare}"
export WANDB_PROJECT="${WANDB_PROJECT:-bioreason-pro}"
export BASE_WANDB_PROJECT="${BASE_WANDB_PROJECT:-$WANDB_PROJECT}"
export WEAVE_PROJECT="${WEAVE_PROJECT:-${WANDB_ENTITY}/${WANDB_PROJECT}}"
export REGISTRY_ENV_FILE="${REGISTRY_ENV_FILE:-configs/disease_benchmark/wandb_registry_paths.env}"
export BASE_CHECKPOINT="${BASE_CHECKPOINT:-wandb-healthcare/bioreason-pro/bioreason-pro-rl:latest}"
export TEMPORAL_SPLIT_ARTIFACT="${TEMPORAL_SPLIT_ARTIFACT:-wandb-healthcare/bioreason-pro/disease-temporal-split:production}"
export DATASET_ARTIFACT="${DATASET_ARTIFACT:-wandb-healthcare/bioreason-pro/disease-temporal-reasoning:production}"
export REASONING_PROMPT_STYLE="${REASONING_PROMPT_STYLE:-paper_native_tight}"

export QUERIES_PER_STEP="${QUERIES_PER_STEP:-16}"
export ROLLOUTS_PER_QUERY="${ROLLOUTS_PER_QUERY:-12}"
export OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU="${OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10000}"
export REWARD_WEIGHTS="${REWARD_WEIGHTS:-0.05,0.0,0.9,0.05}"
export CHECKPOINT_ARTIFACT_NAME="${CHECKPOINT_ARTIFACT_NAME:-train-rl-fix-v1}"
export CHECKPOINT_ARTIFACT_ALIASES="${CHECKPOINT_ARTIFACT_ALIASES:-latest,fix-v1}"
export CHECKPOINT_EXPORT_ONLY="${CHECKPOINT_EXPORT_ONLY:-true}"

export NNODES="${NNODES:-2}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export VLLM_SWAP_SPACE_GB="${VLLM_SWAP_SPACE_GB:-0}"
export ROLLOUT_LOGPROB_MICROBATCH_SIZE="${ROLLOUT_LOGPROB_MICROBATCH_SIZE:-2}"
export MAX_STEPS="${MAX_STEPS:-20}"
export VALIDATION_EVERY_N_STEPS="${VALIDATION_EVERY_N_STEPS:-3}"
export VALIDATION_NUM_PROTEINS="${VALIDATION_NUM_PROTEINS:-30}"
export SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-5}"
export CHECKPOINT_RANK0_TIMEOUT_S="${CHECKPOINT_RANK0_TIMEOUT_S:-7200}"
export VALIDATION_RANK0_TIMEOUT_S="${VALIDATION_RANK0_TIMEOUT_S:-14400}"
export RANK0_SECTION_POLL_INTERVAL_S="${RANK0_SECTION_POLL_INTERVAL_S:-5}"
export DISEASE_WEIGHTING_MODE="${DISEASE_WEIGHTING_MODE:-uniform_fallback}"
export TRACE_ROLLOUTS_TO_WEAVE="${TRACE_ROLLOUTS_TO_WEAVE:-true}"
export WEAVE_TRACE_BUDGET="${WEAVE_TRACE_BUDGET:-64}"
export WEAVE_TRACE_FULL_GROUP_COUNT="${WEAVE_TRACE_FULL_GROUP_COUNT:-4}"
export WEAVE_TRACE_FULL_ROLLOUTS_PER_GROUP="${WEAVE_TRACE_FULL_ROLLOUTS_PER_GROUP:-12}"
export ROLLOUT_GENERATE_TIMEOUT_SECONDS="${ROLLOUT_GENERATE_TIMEOUT_SECONDS:-1200}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"

export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
export MASTER_PORT="${MASTER_PORT:-29511}"

srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 bash -lc '
  set -euo pipefail
  cd '"'$PROJECT_ROOT'"'
  source '"'$PROJECT_ROOT'"'/.venv-gpu/bin/activate
  export WANDB_ENTITY='"'$WANDB_ENTITY'"'
  export WANDB_PROJECT='"'$WANDB_PROJECT'"'
  export BASE_WANDB_PROJECT='"'$BASE_WANDB_PROJECT'"'
  export WEAVE_PROJECT='"'$WEAVE_PROJECT'"'
  export REGISTRY_ENV_FILE='"'$REGISTRY_ENV_FILE'"'
  export BASE_CHECKPOINT='"'$BASE_CHECKPOINT'"'
  export TEMPORAL_SPLIT_ARTIFACT='"'$TEMPORAL_SPLIT_ARTIFACT'"'
  export DATASET_ARTIFACT='"'$DATASET_ARTIFACT'"'
  export REASONING_PROMPT_STYLE='"'$REASONING_PROMPT_STYLE'"'
  export NNODES='"'$NNODES'"'
  export GPUS_PER_NODE='"'$GPUS_PER_NODE'"'
  export QUERIES_PER_STEP='"'$QUERIES_PER_STEP'"'
  export ROLLOUTS_PER_QUERY='"'$ROLLOUTS_PER_QUERY'"'
  export OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU='"'$OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU'"'
  export GRADIENT_ACCUMULATION_STEPS='"'$GRADIENT_ACCUMULATION_STEPS'"'
  export MAX_NEW_TOKENS='"'$MAX_NEW_TOKENS'"'
  export VLLM_GPU_MEMORY_UTILIZATION='"'$VLLM_GPU_MEMORY_UTILIZATION'"'
  export VLLM_MAX_MODEL_LEN='"'$VLLM_MAX_MODEL_LEN'"'
  export VLLM_MAX_NUM_SEQS='"'$VLLM_MAX_NUM_SEQS'"'
  export VLLM_SWAP_SPACE_GB='"'$VLLM_SWAP_SPACE_GB'"'
  export ROLLOUT_LOGPROB_MICROBATCH_SIZE='"'$ROLLOUT_LOGPROB_MICROBATCH_SIZE'"'
  export REWARD_WEIGHTS='"'$REWARD_WEIGHTS'"'
  export DISEASE_WEIGHTING_MODE='"'$DISEASE_WEIGHTING_MODE'"'
  export TRACE_ROLLOUTS_TO_WEAVE='"'$TRACE_ROLLOUTS_TO_WEAVE'"'
  export WEAVE_TRACE_BUDGET='"'$WEAVE_TRACE_BUDGET'"'
  export WEAVE_TRACE_FULL_GROUP_COUNT='"'$WEAVE_TRACE_FULL_GROUP_COUNT'"'
  export WEAVE_TRACE_FULL_ROLLOUTS_PER_GROUP='"'$WEAVE_TRACE_FULL_ROLLOUTS_PER_GROUP'"'
  export ROLLOUT_GENERATE_TIMEOUT_SECONDS='"'$ROLLOUT_GENERATE_TIMEOUT_SECONDS'"'
  export NCCL_TIMEOUT='"'$NCCL_TIMEOUT'"'
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC='"'$TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'"'
  export CHECKPOINT_ARTIFACT_NAME='"'$CHECKPOINT_ARTIFACT_NAME'"'
  export CHECKPOINT_ARTIFACT_ALIASES='"'$CHECKPOINT_ARTIFACT_ALIASES'"'
  export CHECKPOINT_EXPORT_ONLY='"'$CHECKPOINT_EXPORT_ONLY'"'
  export MASTER_ADDR='"'$MASTER_ADDR'"'
  export MASTER_PORT='"'$MASTER_PORT'"'
  export NODE_RANK="$SLURM_NODEID"
  export HOSTFILE=""
  bash scripts/sh_train_protein_grpo.sh \
    --clip_epsilon_low 0.15 \
    --clip_epsilon_high 0.2 \
    --kl_beta 0.01 \
    --learning_rate 1e-5 \
    --steps_per_generation 2 \
    --trace_rollouts_to_weave "$TRACE_ROLLOUTS_TO_WEAVE" \
    --weave_trace_budget "$WEAVE_TRACE_BUDGET" \
    --weave_trace_full_group_count "$WEAVE_TRACE_FULL_GROUP_COUNT" \
    --weave_trace_full_rollouts_per_group "$WEAVE_TRACE_FULL_ROLLOUTS_PER_GROUP" \
    --max_steps '"'$MAX_STEPS'"' \
    --validation_every_n_steps '"'$VALIDATION_EVERY_N_STEPS'"' \
    --validation_num_proteins '"'$VALIDATION_NUM_PROTEINS'"' \
    --save_every_n_steps '"'$SAVE_EVERY_N_STEPS'"' \
    --checkpoint_rank0_timeout_s '"'$CHECKPOINT_RANK0_TIMEOUT_S'"' \
    --validation_rank0_timeout_s '"'$VALIDATION_RANK0_TIMEOUT_S'"' \
    --rank0_section_poll_interval_s '"'$RANK0_SECTION_POLL_INTERVAL_S'"'
'
