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
mkdir -p runtime_logs

export WANDB_ENTITY=wandb-healthcare
export BASE_WANDB_PROJECT=bioreason-pro-custom
export WANDB_PROJECT="$BASE_WANDB_PROJECT"
export EXPECTED_WANDB_ENTITY=wandb-healthcare
export EXPECTED_WANDB_PROJECT=bioreason-pro-custom
export WEAVE_PROJECT="${WEAVE_PROJECT:-${WANDB_ENTITY}/${BASE_WANDB_PROJECT}}"
export VERIFY_TAG="${VERIFY_TAG:-verify40}"
export DATA_BUNDLE=main_production
export VALIDATION_SUBSET_SIZE=100
export VALIDATION_SUBSET_STRATEGY=stratified_aspect_profile

export TRAIN_PARTITION=h100
export TRAIN_NUM_GPUS=1
export TRAIN_CPUS_PER_TASK=16
export TRAIN_MEM=128G
export TRAIN_TIME_LIMIT=04:00:00

export STAGE2_MAX_EPOCHS=1
export STAGE2_LIMIT_TRAIN_BATCHES=0.01
export STAGE2_LIMIT_VAL_BATCHES=1.0
export STAGE2_VAL_CHECK_INTERVAL=0.2
export STAGE2_BATCH_SIZE=1
export STAGE2_GRADIENT_ACCUMULATION_STEPS=4
export MAX_LENGTH_TEXT=2048
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STAGE2_LOG_EVERY_N_STEPS=10
export STAGE2_SAMPLE_GENERATION_EVERY_N_STEPS=5
export STAGE2_CHECKPOINT_EVERY_N_TRAIN_STEPS=500

export TRAIN_JOB_NAME="${SFT_TRAIN_JOB_NAME:-bioreason-sft-${VERIFY_TAG}}"
export WANDB_RUN_NAME_S2="${WANDB_RUN_NAME_S2:-sft-${VERIFY_TAG}}"
export STAGE2_MODEL_ARTIFACT_NAME="${STAGE2_MODEL_ARTIFACT_NAME:-train-sft-output-${VERIFY_TAG}}"
export STAGE2_CHECKPOINT_ARTIFACT_NAME="${STAGE2_CHECKPOINT_ARTIFACT_NAME:-sft-${VERIFY_TAG}-checkpoints}"

bash scripts/sh_train_protein_qwen_staged.sh

SFT_REF=$(python scripts/resolve_logged_artifact_ref.py \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-project "$BASE_WANDB_PROJECT" \
  --run-name "$WANDB_RUN_NAME_S2" \
  --artifact-name "$STAGE2_MODEL_ARTIFACT_NAME")
echo "Resolved SFT artifact: $SFT_REF"
export BIOREASON_TRAIN_SFT_MODEL_REGISTRY_PATH="$SFT_REF"

python scripts/run_registered_eval.py \
  --target train-sft-output \
  --data-bundle main_production \
  --registry-env-file configs/disease_benchmark/wandb_registry_paths.env \
  --split test \
  --use-srun \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-project "$BASE_WANDB_PROJECT" \
  --weave-project "$WEAVE_PROJECT" \
  --dataset-cache-dir data/artifacts/hf_cache \
  --structure-dir data/structures \
  --go-obo-path bioreason2/dataset/go-basic.obo \
  --output-root "data/evals/${VERIFY_TAG}/sft_test"

export TRAIN_JOB_NAME="${RL_TRAIN_JOB_NAME:-bioreason-rl-${VERIFY_TAG}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-rl-${VERIFY_TAG}}"
export CHECKPOINT_ARTIFACT_NAME="${CHECKPOINT_ARTIFACT_NAME:-train-rl-output-${VERIFY_TAG}}"
export MAX_STEPS="${MAX_STEPS:-50}"
export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-100}"
export EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-10}"
export SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-25}"
export MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-8}"

bash scripts/sh_train_protein_grpo.sh

RL_REF=$(python scripts/resolve_logged_artifact_ref.py \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-project "$BASE_WANDB_PROJECT" \
  --run-name "$WANDB_RUN_NAME" \
  --artifact-name "$CHECKPOINT_ARTIFACT_NAME")
echo "Resolved RL artifact: $RL_REF"
export BIOREASON_TRAIN_RL_MODEL_REGISTRY_PATH="$RL_REF"

python scripts/run_registered_eval.py \
  --target train-rl-output \
  --data-bundle main_production \
  --registry-env-file configs/disease_benchmark/wandb_registry_paths.env \
  --split test \
  --use-srun \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-project "$BASE_WANDB_PROJECT" \
  --weave-project "$WEAVE_PROJECT" \
  --dataset-cache-dir data/artifacts/hf_cache \
  --structure-dir data/structures \
  --go-obo-path bioreason2/dataset/go-basic.obo \
  --output-root "data/evals/${VERIFY_TAG}/rl_test"
