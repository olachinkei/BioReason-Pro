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
  --queries_per_step 1 \
  --rollouts_per_query 4 \
  --max_steps 1 \
  --steps_per_generation 1 \
  --optimizer_micro_batch_size_per_gpu 1 \
  --gradient_accumulation_steps 4 \
  --target_num_nodes 1 \
  --target_gpus_per_node 1 \
  --rollout_backend subprocess \
  --rollout_worker_start_method spawn \
  --max_new_tokens 2048 \
  --rollout_logprob_microbatch_size 2 \
  --vllm_gpu_memory_utilization 0.35 \
  --vllm_max_model_len 32768 \
  --vllm_max_num_seqs 16 \
  --vllm_cpu_offload_gb 0 \
  --vllm_swap_space_gb 4 \
  --vllm_enforce_eager true \
  --vllm_enable_sleep_mode false \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint_artifact_name "${CHECKPOINT_ARTIFACT_NAME}" \
  --checkpoint_artifact_aliases "${CHECKPOINT_ARTIFACT_ALIASES}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_project "${WANDB_PROJECT}" \
  --run_name "${WANDB_RUN_NAME}" \
  --weave_project "${WEAVE_PROJECT}"
