#!/usr/bin/env bash
set -euo pipefail

cd ~/BioReason-Pro

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

source .venv-gpu/bin/activate

export WANDB_ENTITY="${WANDB_ENTITY:-wandb-healthcare}"
export WANDB_PROJECT="${WANDB_PROJECT:-bioreason-pro}"
export WEAVE_PROJECT="${WEAVE_PROJECT:-${WANDB_ENTITY}/${WANDB_PROJECT}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-rl-paper-context-postrebuild-smoke}"
export CHECKPOINT_ARTIFACT_NAME="${CHECKPOINT_ARTIFACT_NAME:-train-rl-postrebuild-smoke}"
export CHECKPOINT_ARTIFACT_ALIASES="${CHECKPOINT_ARTIFACT_ALIASES:-postrebuild-smoke}"
export OUTPUT_DIR="${OUTPUT_DIR:-data/artifacts/models/train_rl_output/postrebuild_smoke}"
export QUERIES_PER_STEP="${QUERIES_PER_STEP:-1}"
export ROLLOUTS_PER_QUERY="${ROLLOUTS_PER_QUERY:-4}"
export MAX_STEPS="${MAX_STEPS:-1}"
export STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-1}"
export OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU="${OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export TARGET_NUM_NODES="${TARGET_NUM_NODES:-1}"
export TARGET_GPUS_PER_NODE="${TARGET_GPUS_PER_NODE:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export ROLLOUT_LOGPROB_MICROBATCH_SIZE="${ROLLOUT_LOGPROB_MICROBATCH_SIZE:-2}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.35}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
export VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-0}"
export VLLM_SWAP_SPACE_GB="${VLLM_SWAP_SPACE_GB:-4}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}"
export VLLM_ENABLE_SLEEP_MODE="${VLLM_ENABLE_SLEEP_MODE:-false}"

BASE_CHECKPOINT="${BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH}"
BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-data/artifacts/models/bioreason_pro_rl_paper}"
DATA_BUNDLE="${DATA_BUNDLE:-main_production}"

RESOLVED_BASE_MODEL_DIR="$(python scripts/materialize_model_source.py \
  --wandb-registry-path "${BASE_CHECKPOINT}" \
  --local-dir "${BASE_CHECKPOINT_DIR}" \
  --required-path config.json \
  --required-path tokenizer_config.json \
  --required-path protein_projection.pt \
  --required-path protein_model/pytorch_model.bin)"

CAFA5_DATASET="$(python scripts/materialize_data_bundle.py --data-bundle "${DATA_BUNDLE}" --asset-key reasoning_dataset --print-field local_dir)"
DATASET_NAME="$(python scripts/materialize_data_bundle.py --data-bundle "${DATA_BUNDLE}" --asset-key reasoning_dataset --print-field dataset_name)"
TEMPORAL_SPLIT_ARTIFACT="$(python scripts/materialize_data_bundle.py --data-bundle "${DATA_BUNDLE}" --asset-key temporal_split_artifact --print-field wandb_registry_path)"
DATASET_ARTIFACT="$(python scripts/materialize_data_bundle.py --data-bundle "${DATA_BUNDLE}" --asset-key reasoning_dataset --print-field wandb_registry_path)"
IA_DIR="$(python scripts/materialize_data_bundle.py --data-bundle "${DATA_BUNDLE}" --asset-key ia_file --print-field local_dir)"
IA_FILE_PATH="${IA_DIR}/IA.txt"

python train_protein_grpo.py \
  --debug_single_process true \
  --text_model_name "${RESOLVED_BASE_MODEL_DIR}" \
  --base_checkpoint "${BASE_CHECKPOINT}" \
  --cafa5_dataset "${CAFA5_DATASET}" \
  --dataset_config "${DATASET_NAME}" \
  --reasoning_dataset_config "${DATASET_NAME}" \
  --reasoning_dataset_name "${DATASET_NAME}" \
  --temporal_split_artifact "${TEMPORAL_SPLIT_ARTIFACT}" \
  --dataset_artifact "${DATASET_ARTIFACT}" \
  --go_obo_path bioreason2/dataset/go-basic.obo \
  --ia_file_path "${IA_FILE_PATH}" \
  --queries_per_step "${QUERIES_PER_STEP}" \
  --rollouts_per_query "${ROLLOUTS_PER_QUERY}" \
  --max_steps "${MAX_STEPS}" \
  --steps_per_generation "${STEPS_PER_GENERATION}" \
  --optimizer_micro_batch_size_per_gpu "${OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --target_num_nodes "${TARGET_NUM_NODES}" \
  --target_gpus_per_node "${TARGET_GPUS_PER_NODE}" \
  --rollout_backend subprocess \
  --rollout_worker_start_method spawn \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --rollout_logprob_microbatch_size "${ROLLOUT_LOGPROB_MICROBATCH_SIZE}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
  --vllm_max_num_seqs "${VLLM_MAX_NUM_SEQS}" \
  --vllm_cpu_offload_gb "${VLLM_CPU_OFFLOAD_GB}" \
  --vllm_swap_space_gb "${VLLM_SWAP_SPACE_GB}" \
  --vllm_enforce_eager "${VLLM_ENFORCE_EAGER}" \
  --vllm_enable_sleep_mode "${VLLM_ENABLE_SLEEP_MODE}" \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint_artifact_name "${CHECKPOINT_ARTIFACT_NAME}" \
  --checkpoint_artifact_aliases "${CHECKPOINT_ARTIFACT_ALIASES}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_project "${WANDB_PROJECT}" \
  --run_name "${WANDB_RUN_NAME}" \
  --weave_project "${WEAVE_PROJECT}"
