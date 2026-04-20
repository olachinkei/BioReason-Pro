#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

default_runtime_root() {
  if [ -n "${BIOREASON_RUNTIME_ROOT:-}" ]; then
    printf '%s\n' "$BIOREASON_RUNTIME_ROOT"
    return 0
  fi
  if [ -d "/mnt/data" ] && [ -n "${USER:-}" ]; then
    printf '/mnt/data/%s/BioReason-Pro\n' "$USER"
    return 0
  fi
  pwd
}

source_env_file_without_overrides() {
  local env_file="$1"
  local raw_line line key existing_value
  [ -f "$env_file" ] || return 0

  while IFS= read -r raw_line || [ -n "$raw_line" ]; do
    line="${raw_line#"${raw_line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [ -n "$line" ] || continue
    [[ "$line" == \#* ]] && continue
    [[ "$line" == export\ * ]] && line="${line#export }"
    [[ "$line" == *=* ]] || continue
    key="${line%%=*}"
    key="${key%"${key##*[![:space:]]}"}"
    existing_value="${!key:-}"
    if [ -n "$existing_value" ]; then
      continue
    fi
    eval "export $line"
  done < "$env_file"
}

as_bool() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|t|yes|y) return 0 ;;
    *) return 1 ;;
  esac
}

has_preflight_only() {
  local idx=1
  local total=$#
  while [ "$idx" -le "$total" ]; do
    local arg="${!idx}"
    if [ "$arg" = "--preflight_only" ]; then
      idx=$((idx + 1))
      if [ "$idx" -le "$total" ]; then
        as_bool "${!idx}" && return 0
      else
        return 0
      fi
    fi
    idx=$((idx + 1))
  done
  return 1
}

require_env() {
  local key="$1"
  local value="${!key:-}"
  if [ -z "$value" ]; then
    echo "Error: $key is required for the exact 2-node launch contract."
    exit 1
  fi
}

maybe_generate_slurm_hostfile() {
  local hostfile_path
  if [ -n "$HOSTFILE" ] || [ -z "${SLURM_JOB_NODELIST:-}" ]; then
    return 0
  fi
  if ! command -v scontrol >/dev/null 2>&1; then
    return 0
  fi
  hostfile_path="${HOSTFILE_AUTO_PATH:-"$(pwd)/runtime_logs/deepspeed_hosts.${SLURM_JOB_ID:-$$}.txt"}"
  mkdir -p "$(dirname "$hostfile_path")"
  scontrol show hostnames "$SLURM_JOB_NODELIST" | awk -v slots="$GPUS_PER_NODE" '{print $1 " slots=" slots}' > "$hostfile_path"
  if [ -s "$hostfile_path" ]; then
    HOSTFILE="$hostfile_path"
    export HOSTFILE
    echo "Info: auto-generated DeepSpeed hostfile at $HOSTFILE from SLURM_JOB_NODELIST."
  fi
}

REGISTRY_ENV_FILE=${REGISTRY_ENV_FILE:-"configs/disease_benchmark/wandb_registry_paths.env"}
if [ -f "$REGISTRY_ENV_FILE" ]; then
  source_env_file_without_overrides "$REGISTRY_ENV_FILE"
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

BIOREASON_RUNTIME_ROOT="$(default_runtime_root)"
BIOREASON_ARTIFACTS_ROOT="${BIOREASON_ARTIFACTS_ROOT:-${BIOREASON_RUNTIME_ROOT}/data/artifacts}"
BIOREASON_CACHE_ROOT="${BIOREASON_CACHE_ROOT:-${BIOREASON_RUNTIME_ROOT}/cache}"
WANDB_DIR="${WANDB_DIR:-${BIOREASON_RUNTIME_ROOT}/wandb}"
WEAVE_SERVER_CACHE_DIR="${WEAVE_SERVER_CACHE_DIR:-${WANDB_DIR}/weave_server_cache}"
HF_HOME="${HF_HOME:-${BIOREASON_CACHE_ROOT}/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "$BIOREASON_ARTIFACTS_ROOT" "$BIOREASON_CACHE_ROOT" "$WANDB_DIR" "$WEAVE_SERVER_CACHE_DIR" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
export BIOREASON_RUNTIME_ROOT BIOREASON_ARTIFACTS_ROOT BIOREASON_CACHE_ROOT
export WANDB_DIR WEAVE_SERVER_CACHE_DIR HF_HOME TRANSFORMERS_CACHE HF_DATASETS_CACHE

PYTHON_BIN=${PYTHON_BIN:-python}
DEEPSPEED_BIN=${DEEPSPEED_BIN:-deepspeed}
MODEL_SOURCE_RESOLVER=${MODEL_SOURCE_RESOLVER:-"scripts/materialize_model_source.py"}
DATA_BUNDLE_RESOLVER=${DATA_BUNDLE_RESOLVER:-"scripts/materialize_data_bundle.py"}
DATA_MANIFEST_PATH=${DATA_MANIFEST_PATH:-"configs/disease_benchmark/data_registry.json"}
DATA_BUNDLE=${DATA_BUNDLE:-"main_production"}

WANDB_PROJECT=${WANDB_PROJECT:-"${BASE_WANDB_PROJECT:-bioreason-pro}"}
BASE_WANDB_PROJECT=${BASE_WANDB_PROJECT:-"$WANDB_PROJECT"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
WEAVE_PROJECT=${WEAVE_PROJECT:-""}
if [ -z "$WEAVE_PROJECT" ] && [ -n "$WANDB_ENTITY" ]; then
  WEAVE_PROJECT="${WANDB_ENTITY}/${WANDB_PROJECT}"
fi

BENCHMARK_VERSION=${BENCHMARK_VERSION:-"213 -> 221 -> 225 -> 228"}
TRAIN_START_RELEASE=${TRAIN_START_RELEASE:-213}
TRAIN_END_RELEASE=${TRAIN_END_RELEASE:-221}
DEV_END_RELEASE=${DEV_END_RELEASE:-225}
TEST_END_RELEASE=${TEST_END_RELEASE:-228}

BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH:-}"}
BASE_CHECKPOINT_DIR=${BASE_CHECKPOINT_DIR:-"${BIOREASON_ARTIFACTS_ROOT}/models/bioreason_pro_rl_paper"}
CAFA5_DATASET=${CAFA5_DATASET:-""}
DATASET_NAME=${DATASET_NAME:-""}
REASONING_DATASET_NAME=${REASONING_DATASET_NAME:-""}
TEMPORAL_SPLIT_ARTIFACT=${TEMPORAL_SPLIT_ARTIFACT:-""}
DATASET_ARTIFACT=${DATASET_ARTIFACT:-""}
IA_FILE_PATH=${IA_FILE_PATH:-""}
GO_OBO_PATH=${GO_OBO_PATH:-"bioreason2/dataset/go-basic.obo"}
CHECKPOINT_ARTIFACT_NAME=${CHECKPOINT_ARTIFACT_NAME:-"train-rl-output"}
CHECKPOINT_ARTIFACT_ALIASES=${CHECKPOINT_ARTIFACT_ALIASES:-"latest"}
CHECKPOINT_EXPORT_ONLY=${CHECKPOINT_EXPORT_ONLY:-false}
EXECUTION_ID=${EXECUTION_ID:-"${SLURM_JOB_ID:-local}-$(date -u +%Y%m%d%H%M%S)"}
SYNC_ROOT=${SYNC_ROOT:-""}
RESUME_FROM_EXPORT_ARTIFACT=${RESUME_FROM_EXPORT_ARTIFACT:-""}
RESUME_MODE=${RESUME_MODE:-warm}
OUTPUT_DIR=${OUTPUT_DIR:-"${BIOREASON_ARTIFACTS_ROOT}/models/train_rl_output"}
ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-"subprocess"}
ROLLOUT_WORKER_START_METHOD=${ROLLOUT_WORKER_START_METHOD:-"spawn"}
QUERIES_PER_STEP=${QUERIES_PER_STEP:-8}
ROLLOUTS_PER_QUERY=${ROLLOUTS_PER_QUERY:-24}
OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU=${OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU:-6}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-10000}
ROLLOUT_MAX_NEW_TOKENS=${ROLLOUT_MAX_NEW_TOKENS:-$MAX_NEW_TOKENS}
ROLLOUT_WORKER_GENERATE_TIMEOUT_S=${ROLLOUT_WORKER_GENERATE_TIMEOUT_S:-1500}
ROLLOUT_WORKER_STARTUP_RETRY_COUNT=${ROLLOUT_WORKER_STARTUP_RETRY_COUNT:-3}
ROLLOUT_WORKER_STARTUP_RETRY_SLEEP_S=${ROLLOUT_WORKER_STARTUP_RETRY_SLEEP_S:-15}
ROLLOUT_WORKER_VLLM_PORT_BASE=${ROLLOUT_WORKER_VLLM_PORT_BASE:-39000}
ROLLOUT_WORKER_VLLM_PORT_STRIDE=${ROLLOUT_WORKER_VLLM_PORT_STRIDE:-32}
ROLLOUT_WORKER_VLLM_HOST_IP=${ROLLOUT_WORKER_VLLM_HOST_IP:-127.0.0.1}
REASONING_PROMPT_STYLE=${REASONING_PROMPT_STYLE:-paper_native_tight}
ROLLOUT_LOGPROB_MICROBATCH_SIZE=${ROLLOUT_LOGPROB_MICROBATCH_SIZE:-4}
MAX_LOSS_COMPLETION_TOKENS=${MAX_LOSS_COMPLETION_TOKENS:-0}
VALIDATION_NUM_PROTEINS=${VALIDATION_NUM_PROTEINS:-200}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.35}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-32768}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-256}
VLLM_CPU_OFFLOAD_GB=${VLLM_CPU_OFFLOAD_GB:-0}
VLLM_SWAP_SPACE_GB=${VLLM_SWAP_SPACE_GB:-4}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-true}
VLLM_ENABLE_SLEEP_MODE=${VLLM_ENABLE_SLEEP_MODE:-false}
VLLM_SLEEP_LEVEL=${VLLM_SLEEP_LEVEL:-1}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-${BIOREASON_VLLM_ATTENTION_BACKEND:-XFORMERS}}
VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-${BIOREASON_VLLM_WORKER_MULTIPROC_METHOD:-spawn}}
VLLM_USE_V1=${VLLM_USE_V1:-${BIOREASON_VLLM_USE_V1:-0}}

export VLLM_ATTENTION_BACKEND
export VLLM_WORKER_MULTIPROC_METHOD
export VLLM_USE_V1

NNODES=${NNODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
HOSTFILE=${HOSTFILE:-""}
MASTER_ADDR=${MASTER_ADDR:-""}
MASTER_PORT=${MASTER_PORT:-""}
NODE_RANK=${NODE_RANK:-""}

if [ "$NNODES" != "2" ]; then
  echo "Error: exact production launch requires NNODES=2 (got $NNODES)."
  exit 1
fi
if [ "$GPUS_PER_NODE" != "8" ]; then
  echo "Error: exact production launch requires GPUS_PER_NODE=8 (got $GPUS_PER_NODE)."
  exit 1
fi

if [ -z "$BASE_CHECKPOINT" ]; then
  echo "Error: BASE_CHECKPOINT is not set. Set BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH or BASE_CHECKPOINT."
  exit 1
fi

RESOLVED_BASE_MODEL_DIR=$("$PYTHON_BIN" "$MODEL_SOURCE_RESOLVER" \
  --wandb-registry-path "$BASE_CHECKPOINT" \
  --local-dir "$BASE_CHECKPOINT_DIR" \
  --required-path config.json \
  --required-path tokenizer_config.json \
  --required-path protein_projection.pt \
  --required-path protein_model/pytorch_model.bin)
if [ -z "$RESOLVED_BASE_MODEL_DIR" ]; then
  echo "Error: failed to resolve base checkpoint from $BASE_CHECKPOINT"
  exit 1
fi

if [ -z "$CAFA5_DATASET" ]; then
  CAFA5_DATASET=$("$PYTHON_BIN" "$DATA_BUNDLE_RESOLVER" \
    --data-manifest-path "$DATA_MANIFEST_PATH" \
    --data-bundle "$DATA_BUNDLE" \
    --asset-key reasoning_dataset \
    --print-field local_dir)
fi
if [ -z "$DATASET_NAME" ]; then
  DATASET_NAME=$("$PYTHON_BIN" "$DATA_BUNDLE_RESOLVER" \
    --data-manifest-path "$DATA_MANIFEST_PATH" \
    --data-bundle "$DATA_BUNDLE" \
    --asset-key reasoning_dataset \
    --print-field dataset_name)
fi
if [ -z "$REASONING_DATASET_NAME" ]; then
  REASONING_DATASET_NAME="$DATASET_NAME"
fi
if [ -z "$TEMPORAL_SPLIT_ARTIFACT" ]; then
  TEMPORAL_SPLIT_ARTIFACT=$("$PYTHON_BIN" "$DATA_BUNDLE_RESOLVER" \
    --data-manifest-path "$DATA_MANIFEST_PATH" \
    --data-bundle "$DATA_BUNDLE" \
    --asset-key temporal_split_artifact \
    --print-field wandb_registry_path)
fi
if [ -z "$DATASET_ARTIFACT" ]; then
  DATASET_ARTIFACT=$("$PYTHON_BIN" "$DATA_BUNDLE_RESOLVER" \
    --data-manifest-path "$DATA_MANIFEST_PATH" \
    --data-bundle "$DATA_BUNDLE" \
    --asset-key reasoning_dataset \
    --print-field wandb_registry_path)
fi
if [ -z "$IA_FILE_PATH" ]; then
  IA_DIR=$("$PYTHON_BIN" "$DATA_BUNDLE_RESOLVER" \
    --data-manifest-path "$DATA_MANIFEST_PATH" \
    --data-bundle "$DATA_BUNDLE" \
    --asset-key ia_file \
    --print-field local_dir)
  IA_FILE_PATH="${IA_DIR}/IA.txt"
fi

if [ ! -d "$CAFA5_DATASET" ]; then
  echo "Error: CAFA5_DATASET must point to a materialized local dataset directory (got $CAFA5_DATASET)."
  exit 1
fi
if [ ! -f "$IA_FILE_PATH" ]; then
  echo "Error: IA_FILE_PATH must point to an existing file (got $IA_FILE_PATH)."
  exit 1
fi
if [ ! -f "$GO_OBO_PATH" ]; then
  echo "Error: GO_OBO_PATH must point to an existing file (got $GO_OBO_PATH)."
  exit 1
fi

TRAIN_ARGS=(
  --text_model_name "$RESOLVED_BASE_MODEL_DIR"
  --base_checkpoint "$BASE_CHECKPOINT"
  --cafa5_dataset "$CAFA5_DATASET"
  --dataset_config "$DATASET_NAME"
  --reasoning_dataset_config "$REASONING_DATASET_NAME"
  --reasoning_dataset_name "$REASONING_DATASET_NAME"
  --validation_num_proteins "$VALIDATION_NUM_PROTEINS"
  --temporal_split_artifact "$TEMPORAL_SPLIT_ARTIFACT"
  --dataset_artifact "$DATASET_ARTIFACT"
  --go_obo_path "$GO_OBO_PATH"
  --ia_file_path "$IA_FILE_PATH"
  --benchmark_version "$BENCHMARK_VERSION"
  --queries_per_step "$QUERIES_PER_STEP"
  --rollouts_per_query "$ROLLOUTS_PER_QUERY"
  --train_start_release "$TRAIN_START_RELEASE"
  --train_end_release "$TRAIN_END_RELEASE"
  --dev_end_release "$DEV_END_RELEASE"
  --test_end_release "$TEST_END_RELEASE"
  --target_num_nodes "$NNODES"
  --target_gpus_per_node "$GPUS_PER_NODE"
  --rollout_backend "$ROLLOUT_BACKEND"
  --rollout_worker_start_method "$ROLLOUT_WORKER_START_METHOD"
  --optimizer_micro_batch_size_per_gpu "$OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --rollout_max_new_tokens "$ROLLOUT_MAX_NEW_TOKENS"
  --rollout_worker_generate_timeout_s "$ROLLOUT_WORKER_GENERATE_TIMEOUT_S"
  --rollout_worker_startup_retry_count "$ROLLOUT_WORKER_STARTUP_RETRY_COUNT"
  --rollout_worker_startup_retry_sleep_s "$ROLLOUT_WORKER_STARTUP_RETRY_SLEEP_S"
  --rollout_worker_vllm_port_base "$ROLLOUT_WORKER_VLLM_PORT_BASE"
  --rollout_worker_vllm_port_stride "$ROLLOUT_WORKER_VLLM_PORT_STRIDE"
  --rollout_worker_vllm_host_ip "$ROLLOUT_WORKER_VLLM_HOST_IP"
  --reasoning_prompt_style "$REASONING_PROMPT_STYLE"
  --rollout_logprob_microbatch_size "$ROLLOUT_LOGPROB_MICROBATCH_SIZE"
  --max_loss_completion_tokens "$MAX_LOSS_COMPLETION_TOKENS"
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --vllm_max_model_len "$VLLM_MAX_MODEL_LEN"
  --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS"
  --vllm_cpu_offload_gb "$VLLM_CPU_OFFLOAD_GB"
  --vllm_swap_space_gb "$VLLM_SWAP_SPACE_GB"
  --vllm_enforce_eager "$VLLM_ENFORCE_EAGER"
  --vllm_enable_sleep_mode "$VLLM_ENABLE_SLEEP_MODE"
  --vllm_sleep_level "$VLLM_SLEEP_LEVEL"
  --vllm_attention_backend "$VLLM_ATTENTION_BACKEND"
  --vllm_worker_multiproc_method "$VLLM_WORKER_MULTIPROC_METHOD"
  --vllm_use_v1 "$VLLM_USE_V1"
  --output_dir "$OUTPUT_DIR"
  --checkpoint_artifact_name "$CHECKPOINT_ARTIFACT_NAME"
  --checkpoint_artifact_aliases "$CHECKPOINT_ARTIFACT_ALIASES"
  --checkpoint_export_only "$CHECKPOINT_EXPORT_ONLY"
  --resume_mode "$RESUME_MODE"
  --wandb_project "$WANDB_PROJECT"
)

if [ -n "$WANDB_ENTITY" ]; then
  TRAIN_ARGS+=(--wandb_entity "$WANDB_ENTITY")
fi
if [ -n "${WANDB_RUN_NAME:-}" ]; then
  TRAIN_ARGS+=(--run_name "$WANDB_RUN_NAME")
fi
if [ -n "$WEAVE_PROJECT" ]; then
  TRAIN_ARGS+=(--weave_project "$WEAVE_PROJECT")
fi
if [ -n "$EXECUTION_ID" ]; then
  TRAIN_ARGS+=(--execution_id "$EXECUTION_ID")
fi
if [ -n "$SYNC_ROOT" ]; then
  TRAIN_ARGS+=(--sync_root "$SYNC_ROOT")
fi
if [ -n "$RESUME_FROM_EXPORT_ARTIFACT" ]; then
  TRAIN_ARGS+=(--resume_from_export_artifact "$RESUME_FROM_EXPORT_ARTIFACT")
fi

if has_preflight_only "$@"; then
  exec "$PYTHON_BIN" train_protein_grpo.py "${TRAIN_ARGS[@]}" "$@"
fi

maybe_generate_slurm_hostfile

if [ -n "$HOSTFILE" ]; then
  if [ ! -f "$HOSTFILE" ]; then
    echo "Error: HOSTFILE does not exist: $HOSTFILE"
    exit 1
  fi
  if [ -n "${NODE_RANK:-}" ]; then
    require_env MASTER_ADDR
    require_env MASTER_PORT
    exec "$DEEPSPEED_BIN" \
      --hostfile "$HOSTFILE" \
      --no_ssh \
      --master_addr "$MASTER_ADDR" \
      --master_port "$MASTER_PORT" \
      --node_rank "$NODE_RANK" \
      train_protein_grpo.py \
      "${TRAIN_ARGS[@]}" \
      "$@"
  fi
  exec "$DEEPSPEED_BIN" \
    --hostfile "$HOSTFILE" \
    train_protein_grpo.py \
    "${TRAIN_ARGS[@]}" \
    "$@"
fi

require_env MASTER_ADDR
require_env MASTER_PORT
require_env NODE_RANK

exec "$DEEPSPEED_BIN" \
  --no_ssh \
  --num_nodes "$NNODES" \
  --num_gpus "$GPUS_PER_NODE" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --node_rank "$NODE_RANK" \
  train_protein_grpo.py \
  "${TRAIN_ARGS[@]}" \
  "$@"
