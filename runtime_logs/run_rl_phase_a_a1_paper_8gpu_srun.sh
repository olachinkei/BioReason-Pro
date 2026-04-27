#!/usr/bin/env bash
#SBATCH --mem=0
#SBATCH -o /mnt/data/%u/BioReason-Pro/runtime_logs/coreweave/train_rl_phase_a_a1_paper_8gpu_%j.log
#
# Phase A1 paper-exact launcher: 1 node x 8 GPUs, 8 queries x 24 rollouts,
# 10k generation tokens, and per-aspect IA-F1 reward.
#
# This script intentionally forces the paper-shape hyperparameters instead of
# inheriting externally exported lightweight overrides such as MAX_NEW_TOKENS=256
# or ROLLOUTS_PER_QUERY=1. Use this for apples-to-apples A0 vs A1 checks.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"

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

export WANDB_ENTITY="${WANDB_ENTITY:-wandb-healthcare}"
export WANDB_PROJECT="${WANDB_PROJECT:-bioreason-pro-custom}"
export BASE_WANDB_PROJECT="${BASE_WANDB_PROJECT:-$WANDB_PROJECT}"
export WEAVE_PROJECT="${WEAVE_PROJECT:-${WANDB_ENTITY}/${WANDB_PROJECT}}"
export DATA_BUNDLE="${DATA_BUNDLE:-main_production}"

# Force the paper-equivalent A1 training shape. Do not change these through
# sbatch --export; edit this script intentionally if the comparison target changes.
export ABLATION=A1
export REASONING_PROMPT_STYLE=paper_native_tight
export NNODES=1
export GPUS_PER_NODE=8
export QUERIES_PER_STEP=8
export ROLLOUTS_PER_QUERY=24
export OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU=12
export GRADIENT_ACCUMULATION_STEPS=2
export MAX_NEW_TOKENS=10000
export ROLLOUT_MAX_NEW_TOKENS=10000
export STEPS_PER_GENERATION=2
export NUM_ITERATIONS=1
export MAX_LOSS_COMPLETION_TOKENS=0
export ROLLOUT_LOGPROB_MICROBATCH_SIZE=4
export ROLLOUT_WORKER_GENERATE_TIMEOUT_S=1500
export ROLLOUT_WORKER_STARTUP_RETRY_COUNT=3
export ROLLOUT_WORKER_STARTUP_RETRY_SLEEP_S=15
export ROLLOUT_WORKER_VLLM_PORT_BASE=39000
export ROLLOUT_WORKER_VLLM_PORT_STRIDE=32
export ROLLOUT_WORKER_VLLM_HOST_IP=127.0.0.1
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
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1

export MAX_STEPS=20
export VALIDATION_NUM_PROTEINS=200
export VALIDATION_EVERY_N_STEPS=5
export SAVE_EVERY_N_STEPS=10

export WANDB_RUN_NAME="${WANDB_RUN_NAME:-rl-phase-a-A1-paper-8gpu-${TS}}"
export OUTPUT_DIR="${BIOREASON_ARTIFACTS_ROOT}/models/train_rl_output_phase_a_a1_paper_8gpu_${TS}"
export CHECKPOINT_ARTIFACT_NAME="${CHECKPOINT_ARTIFACT_NAME:-train-rl-output-phase-a-a1-paper-8gpu}"
export CHECKPOINT_ARTIFACT_ALIASES="${CHECKPOINT_ARTIFACT_ALIASES:-latest,phase-a-a1,paper-8gpu}"
export CHECKPOINT_EXPORT_ONLY="${CHECKPOINT_EXPORT_ONLY:-false}"
export EXECUTION_ID="${EXECUTION_ID:-${SLURM_JOB_ID:-local}-${TS}}"
export SYNC_ROOT="${SYNC_ROOT:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
MODEL_SOURCE_RESOLVER="${MODEL_SOURCE_RESOLVER:-scripts/materialize_model_source.py}"
DATA_BUNDLE_RESOLVER="${DATA_BUNDLE_RESOLVER:-scripts/materialize_data_bundle.py}"
DATA_MANIFEST_PATH="${DATA_MANIFEST_PATH:-configs/disease_benchmark/data_registry.json}"
GO_OBO_PATH="${GO_OBO_PATH:-bioreason2/dataset/go-basic.obo}"

BASE_CHECKPOINT="${BASE_CHECKPOINT:-${BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH:-}}"
BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-${BIOREASON_ARTIFACTS_ROOT}/models/bioreason_pro_rl_paper}"
if [ -z "$BASE_CHECKPOINT" ]; then
  echo "Error: BASE_CHECKPOINT is not set. Set BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH or BASE_CHECKPOINT." >&2
  exit 1
fi

RESOLVED_BASE_MODEL_DIR=$("$PYTHON_BIN" "$MODEL_SOURCE_RESOLVER" \
  --wandb-registry-path "$BASE_CHECKPOINT" \
  --local-dir "$BASE_CHECKPOINT_DIR" \
  --required-path config.json \
  --required-path tokenizer_config.json \
  --required-path protein_projection.pt \
  --required-path protein_model/pytorch_model.bin)

CAFA5_DATASET="${CAFA5_DATASET:-$("$PYTHON_BIN" "$DATA_BUNDLE_RESOLVER" --data-manifest-path "$DATA_MANIFEST_PATH" --data-bundle "$DATA_BUNDLE" --asset-key reasoning_dataset --print-field local_dir)}"
DATASET_NAME="${DATASET_NAME:-$("$PYTHON_BIN" "$DATA_BUNDLE_RESOLVER" --data-manifest-path "$DATA_MANIFEST_PATH" --data-bundle "$DATA_BUNDLE" --asset-key reasoning_dataset --print-field dataset_name)}"
REASONING_DATASET_NAME="${REASONING_DATASET_NAME:-$DATASET_NAME}"
TEMPORAL_SPLIT_ARTIFACT="${TEMPORAL_SPLIT_ARTIFACT:-$("$PYTHON_BIN" "$DATA_BUNDLE_RESOLVER" --data-manifest-path "$DATA_MANIFEST_PATH" --data-bundle "$DATA_BUNDLE" --asset-key temporal_split_artifact --print-field wandb_registry_path)}"
DATASET_ARTIFACT="${DATASET_ARTIFACT:-$("$PYTHON_BIN" "$DATA_BUNDLE_RESOLVER" --data-manifest-path "$DATA_MANIFEST_PATH" --data-bundle "$DATA_BUNDLE" --asset-key reasoning_dataset --print-field wandb_registry_path)}"
if [ -z "${IA_FILE_PATH:-}" ]; then
  IA_DIR=$("$PYTHON_BIN" "$DATA_BUNDLE_RESOLVER" --data-manifest-path "$DATA_MANIFEST_PATH" --data-bundle "$DATA_BUNDLE" --asset-key ia_file --print-field local_dir)
  IA_FILE_PATH="${IA_DIR}/IA.txt"
fi

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29511}"

echo "[phase-a-paper] launching A1 with paper values: nodes=${NNODES}, gpus_per_node=${GPUS_PER_NODE}, queries=${QUERIES_PER_STEP}, rollouts=${ROLLOUTS_PER_QUERY}, max_new_tokens=${MAX_NEW_TOKENS}, optimizer_micro_batch=${OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU}, grad_accum=${GRADIENT_ACCUMULATION_STEPS}" >&2

exec "$DEEPSPEED_BIN" \
  --num_nodes "$NNODES" \
  --num_gpus "$GPUS_PER_NODE" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  train_protein_grpo.py \
  --text_model_name "$RESOLVED_BASE_MODEL_DIR" \
  --base_checkpoint "$BASE_CHECKPOINT" \
  --cafa5_dataset "$CAFA5_DATASET" \
  --dataset_config "$DATASET_NAME" \
  --reasoning_dataset_config "$REASONING_DATASET_NAME" \
  --reasoning_dataset_name "$REASONING_DATASET_NAME" \
  --validation_num_proteins "$VALIDATION_NUM_PROTEINS" \
  --temporal_split_artifact "$TEMPORAL_SPLIT_ARTIFACT" \
  --dataset_artifact "$DATASET_ARTIFACT" \
  --go_obo_path "$GO_OBO_PATH" \
  --ia_file_path "$IA_FILE_PATH" \
  --benchmark_version "213 -> 221 -> 225 -> 228" \
  --queries_per_step "$QUERIES_PER_STEP" \
  --rollouts_per_query "$ROLLOUTS_PER_QUERY" \
  --steps_per_generation "$STEPS_PER_GENERATION" \
  --num_iterations "$NUM_ITERATIONS" \
  --train_start_release 213 \
  --train_end_release 221 \
  --dev_end_release 225 \
  --test_end_release 228 \
  --target_num_nodes "$NNODES" \
  --target_gpus_per_node "$GPUS_PER_NODE" \
  --rollout_backend subprocess \
  --rollout_worker_start_method spawn \
  --optimizer_micro_batch_size_per_gpu "$OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --rollout_max_new_tokens "$ROLLOUT_MAX_NEW_TOKENS" \
  --rollout_worker_generate_timeout_s "$ROLLOUT_WORKER_GENERATE_TIMEOUT_S" \
  --rollout_worker_startup_retry_count "$ROLLOUT_WORKER_STARTUP_RETRY_COUNT" \
  --rollout_worker_startup_retry_sleep_s "$ROLLOUT_WORKER_STARTUP_RETRY_SLEEP_S" \
  --rollout_worker_vllm_port_base "$ROLLOUT_WORKER_VLLM_PORT_BASE" \
  --rollout_worker_vllm_port_stride "$ROLLOUT_WORKER_VLLM_PORT_STRIDE" \
  --rollout_worker_vllm_host_ip "$ROLLOUT_WORKER_VLLM_HOST_IP" \
  --reasoning_prompt_style "$REASONING_PROMPT_STYLE" \
  --rollout_logprob_microbatch_size "$ROLLOUT_LOGPROB_MICROBATCH_SIZE" \
  --max_loss_completion_tokens "$MAX_LOSS_COMPLETION_TOKENS" \
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
  --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
  --vllm_cpu_offload_gb "$VLLM_CPU_OFFLOAD_GB" \
  --vllm_swap_space_gb "$VLLM_SWAP_SPACE_GB" \
  --vllm_enforce_eager "$VLLM_ENFORCE_EAGER" \
  --vllm_enable_sleep_mode "$VLLM_ENABLE_SLEEP_MODE" \
  --vllm_sleep_level "$VLLM_SLEEP_LEVEL" \
  --vllm_attention_backend "$VLLM_ATTENTION_BACKEND" \
  --vllm_worker_multiproc_method "$VLLM_WORKER_MULTIPROC_METHOD" \
  --vllm_use_v1 "$VLLM_USE_V1" \
  --output_dir "$OUTPUT_DIR" \
  --checkpoint_artifact_name "$CHECKPOINT_ARTIFACT_NAME" \
  --checkpoint_artifact_aliases "$CHECKPOINT_ARTIFACT_ALIASES" \
  --checkpoint_export_only "$CHECKPOINT_EXPORT_ONLY" \
  --resume_mode warm \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" \
  --run_name "$WANDB_RUN_NAME" \
  --weave_project "$WEAVE_PROJECT" \
  --execution_id "$EXECUTION_ID" \
  --reward_mode per_aspect_ia_f1 \
  --disease_loss_weight 1.0 \
  --ablation_tag phase-a-A1 \
  --wandb_tags phase-a-disease-pilot per-aspect-ia-f1 paper-8gpu-exact \
  --trace_rollouts_to_weave true \
  --weave_trace_budget 64 \
  --weave_trace_full_group_count 4 \
  --weave_trace_full_rollouts_per_group 24 \
  --max_steps "$MAX_STEPS" \
  --validation_every_n_steps "$VALIDATION_EVERY_N_STEPS" \
  --save_every_n_steps "$SAVE_EVERY_N_STEPS"
