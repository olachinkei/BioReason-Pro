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

mkdir -p runtime_logs/coreweave

export WANDB_ENTITY="${WANDB_ENTITY:-wandb-healthcare}"
export WANDB_PROJECT="${WANDB_PROJECT:-bioreason-pro}"
export DATA_BUNDLE="${DATA_BUNDLE:-main_production}"
export BASE_CHECKPOINT="${BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH}"
export BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-data/artifacts/models/bioreason_pro_rl_paper}"
export PROBE_SPLIT="${PROBE_SPLIT:-train}"
export PROBE_INDEX="${PROBE_INDEX:-0}"
export PROBE_MAX_NEW_TOKENS="${PROBE_MAX_NEW_TOKENS:-768}"
export PROBE_INCLUDE_HF="${PROBE_INCLUDE_HF:-true}"
export PROBE_OUTPUT_JSON="${PROBE_OUTPUT_JSON:-runtime_logs/coreweave/probe_rl_prompt_variants_${SLURM_JOB_ID:-manual}.json}"

RESOLVED_BASE_MODEL_DIR="$(python scripts/materialize_model_source.py \
  --wandb-registry-path "${BASE_CHECKPOINT}" \
  --local-dir "${BASE_CHECKPOINT_DIR}" \
  --required-path config.json \
  --required-path tokenizer_config.json \
  --required-path protein_projection.pt \
  --required-path protein_model/pytorch_model.bin)"

CAFA5_DATASET="$(python scripts/materialize_data_bundle.py --data-bundle "${DATA_BUNDLE}" --asset-key reasoning_dataset --print-field local_dir)"
DATASET_NAME="$(python scripts/materialize_data_bundle.py --data-bundle "${DATA_BUNDLE}" --asset-key reasoning_dataset --print-field dataset_name)"
IA_DIR="$(python scripts/materialize_data_bundle.py --data-bundle "${DATA_BUNDLE}" --asset-key ia_file --print-field local_dir)"
IA_FILE_PATH="${IA_DIR}/IA.txt"

python scripts/probe_rl_prompt_variants.py \
  --split "${PROBE_SPLIT}" \
  --index "${PROBE_INDEX}" \
  --probe-max-new-tokens "${PROBE_MAX_NEW_TOKENS}" \
  --include-hf "${PROBE_INCLUDE_HF}" \
  --output-json "${PROBE_OUTPUT_JSON}" \
  --text_model_name "${RESOLVED_BASE_MODEL_DIR}" \
  --base_checkpoint "${BASE_CHECKPOINT}" \
  --cafa5_dataset "${CAFA5_DATASET}" \
  --dataset_config "${DATASET_NAME}" \
  --reasoning_dataset_config "${DATASET_NAME}" \
  --reasoning_dataset_name "${DATASET_NAME}" \
  --go_obo_path bioreason2/dataset/go-basic.obo \
  --ia_file_path "${IA_FILE_PATH}" \
  --target_num_nodes 1 \
  --target_gpus_per_node 1 \
  --queries_per_step 1 \
  --rollouts_per_query 4 \
  --debug_single_process true \
  --wandb_mode disabled
