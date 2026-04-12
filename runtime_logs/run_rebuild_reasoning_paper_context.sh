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

SOURCE_METADATA_DIR="${SOURCE_METADATA_DIR:-data/artifacts/datasets/paper_context_source_metadata_v1/213_221_225_228}"
SOURCE_METADATA_DATASET_DIR="${SOURCE_METADATA_DATASET_DIR:-${SOURCE_METADATA_DIR}/hf_dataset}"
REASONING_OUTPUT_DIR="${REASONING_OUTPUT_DIR:-data/artifacts/datasets/disease_temporal_hc_reasoning_v2/213_221_225_228}"
OLD_REASONING_DIR="${OLD_REASONING_DIR:-data/artifacts/datasets/disease_temporal_hc_reasoning_v1/213_221_225_228}"
INTERPRO_WORKERS="${INTERPRO_WORKERS:-8}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
SOURCE_METADATA_LIMIT="${SOURCE_METADATA_LIMIT:-0}"

SPLIT_DIR="$(python scripts/materialize_data_bundle.py --asset-key temporal_split_artifact)"
echo "Resolved temporal split dir: ${SPLIT_DIR}"

SOURCE_METADATA_CMD=(
  python scripts/build_paper_context_source_metadata.py
  --split-dir "${SPLIT_DIR}"
  --output-dir "${SOURCE_METADATA_DIR}"
  --resume
  --interpro-workers "${INTERPRO_WORKERS}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
)

if [ "${SOURCE_METADATA_LIMIT}" != "0" ]; then
  SOURCE_METADATA_CMD+=(--limit "${SOURCE_METADATA_LIMIT}")
fi

echo "Running source metadata build..."
"${SOURCE_METADATA_CMD[@]}"

echo "Running reasoning dataset rebuild + W&B upload..."
python scripts/run_temporal_split_artifact_pipeline.py \
  --variant main \
  --build-datasets \
  --upload-to-wandb \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-project "${WANDB_PROJECT}" \
  --reasoning-dir "${REASONING_OUTPUT_DIR}" \
  --source-metadata-local-dir "${SOURCE_METADATA_DATASET_DIR}"

echo "Removing stale local reasoning dataset dir: ${OLD_REASONING_DIR}"
rm -rf "${OLD_REASONING_DIR}"

echo "Done. Production family 'disease-temporal-reasoning' should now point at the rebuilt artifact."
