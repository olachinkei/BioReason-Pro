#!/bin/bash

# ===================================================================================================
# SLURM Configuration
# Adjust these to match your cluster. Example values shown.
# ===================================================================================================
# #SBATCH --job-name=train_protein_grpo
# #SBATCH --partition=your_gpu_partition
# #SBATCH --time=12:00:00
# #SBATCH --gpus=1
# #SBATCH --cpus-per-task=16
# #SBATCH --mem=256gb
# #SBATCH --output=train_protein_grpo_%j_%x.out
# #SBATCH --error=train_protein_grpo_%j_%x.err

set -euo pipefail

cd "$(dirname "$0")/.."

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

REGISTRY_ENV_FILE=${REGISTRY_ENV_FILE:-"configs/disease_benchmark/wandb_registry_paths.env"}
if [ -f "$REGISTRY_ENV_FILE" ]; then
  source_env_file_without_overrides "$REGISTRY_ENV_FILE"
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

unset SLURM_TRES_PER_TASK

as_bool() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|t|yes|y) return 0 ;;
    *) return 1 ;;
  esac
}

MODEL_SOURCE_RESOLVER=${MODEL_SOURCE_RESOLVER:-"scripts/materialize_model_source.py"}
DATA_BUNDLE_RESOLVER=${DATA_BUNDLE_RESOLVER:-"scripts/materialize_data_bundle.py"}
DATA_MANIFEST_PATH=${DATA_MANIFEST_PATH:-"configs/disease_benchmark/data_registry.json"}
DATA_BUNDLE=${DATA_BUNDLE:-"main_production"}
SFT_TO_HF_CONVERTER=${SFT_TO_HF_CONVERTER:-"bioreason2/utils/save_unsloth_ckpt.py"}

BASE_WANDB_PROJECT=${BASE_WANDB_PROJECT:-"${WANDB_PROJECT:-bioreason-pro-custom}"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
WANDB_MODE=${WANDB_MODE:-""}
EXPECTED_WANDB_ENTITY=${EXPECTED_WANDB_ENTITY:-"wandb-healthcare"}
EXPECTED_WANDB_PROJECT=${EXPECTED_WANDB_PROJECT:-"bioreason-pro-custom"}
ALLOW_WANDB_PROJECT_MISMATCH=${ALLOW_WANDB_PROJECT_MISMATCH:-"False"}
WEAVE_PROJECT=${WEAVE_PROJECT:-""}
TRAIN_PARTITION=${TRAIN_PARTITION:-"h100"}
TRAIN_NUM_GPUS=${TRAIN_NUM_GPUS:-1}
TRAIN_CPUS_PER_TASK=${TRAIN_CPUS_PER_TASK:-16}
TRAIN_MEM=${TRAIN_MEM:-"128G"}
TRAIN_TIME_LIMIT=${TRAIN_TIME_LIMIT:-"12:00:00"}
TRAIN_JOB_NAME=${TRAIN_JOB_NAME:-"bioreason-rl"}
TRAIN_USE_SRUN=${TRAIN_USE_SRUN:-"True"}

if ! as_bool "$ALLOW_WANDB_PROJECT_MISMATCH"; then
  if [ -n "$WANDB_ENTITY" ] && [ "$WANDB_ENTITY" != "$EXPECTED_WANDB_ENTITY" ]; then
    echo "Error: WANDB_ENTITY=$WANDB_ENTITY does not match expected $EXPECTED_WANDB_ENTITY"
    exit 1
  fi
  if [ -n "$BASE_WANDB_PROJECT" ] && [ "$BASE_WANDB_PROJECT" != "$EXPECTED_WANDB_PROJECT" ]; then
    echo "Error: BASE_WANDB_PROJECT=$BASE_WANDB_PROJECT does not match expected $EXPECTED_WANDB_PROJECT"
    exit 1
  fi
fi

MODEL_CACHE_ROOT=${MODEL_CACHE_ROOT:-"data/artifacts/models"}
TRAIN_SFT_SOURCE_DIR=${TRAIN_SFT_SOURCE_DIR:-"${MODEL_CACHE_ROOT}/train_sft_output_source"}
TRAIN_SFT_HF_DIR=${TRAIN_SFT_HF_DIR:-"${MODEL_CACHE_ROOT}/train_sft_output_hf"}
PAPER_RL_MODEL_DIR=${PAPER_RL_MODEL_DIR:-"${MODEL_CACHE_ROOT}/bioreason_pro_rl_paper"}
RL_OUTPUT_ROOT=${RL_OUTPUT_ROOT:-"${MODEL_CACHE_ROOT}/train_rl_output"}
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-"${BIOREASON_DATASET_CACHE_DIR:-data/artifacts/hf_cache}"}
CACHE_DIR=${CACHE_DIR:-"data/artifacts/cache"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"${BIOREASON_STRUCTURE_DIR:-data/structures}"}
GO_EMBEDDINGS_PATH=${GO_EMBEDDINGS_PATH:-"${BIOREASON_GO_EMBEDDINGS_PATH:-}"}
GO_OBO_PATH=${GO_OBO_PATH:-"${BIOREASON_GO_OBO_PATH:-bioreason2/dataset/go-basic.obo}"}

CAFA5_DATASET=${CAFA5_DATASET:-""}
REASONING_DATASET_NAME=${REASONING_DATASET_NAME:-"disease_temporal_hc_reasoning_v1"}
INTERPRO_DATASET_NAME=${INTERPRO_DATASET_NAME:-"interpro_metadata"}

BENCHMARK_VERSION=${BENCHMARK_VERSION:-"213 -> 221 -> 225 -> 228"}
TEMPORAL_SPLIT_ARTIFACT=${TEMPORAL_SPLIT_ARTIFACT:-"${BIOREASON_MAIN_TEMPORAL_SPLIT_REGISTRY_PATH:-}"}
DATASET_CONFIG=${DATASET_CONFIG:-"disease_temporal_hc_reasoning_v1"}
REASONING_DATASET_CONFIG=${REASONING_DATASET_CONFIG:-"disease_temporal_hc_reasoning_v1"}
DATASET_ARTIFACT=${DATASET_ARTIFACT:-"${BIOREASON_MAIN_REASONING_DATASET_REGISTRY_PATH:-}"}
SHORTLIST_MODE=${SHORTLIST_MODE:-"high-confidence"}
SHORTLIST_QUERY=${SHORTLIST_QUERY:-"reviewed:true AND organism_id:9606 AND cc_disease:* AND (xref:mim-* OR xref:orphanet-*) AND (go_exp:* OR go_ida:* OR go_ipi:* OR go_igi:* OR go_imp:* OR go_iep:* OR go_ic:* OR go_tas:*)"}
TRAIN_START_RELEASE=${TRAIN_START_RELEASE:-213}
TRAIN_END_RELEASE=${TRAIN_END_RELEASE:-221}
DEV_END_RELEASE=${DEV_END_RELEASE:-225}
TEST_END_RELEASE=${TEST_END_RELEASE:-228}
JOB_TIME_LIMIT=${JOB_TIME_LIMIT:-"12:00:00"}

PRIMARY_BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${BIOREASON_TRAIN_SFT_MODEL_REGISTRY_PATH:-}"}
PAPER_RL_CHECKPOINT=${PAPER_RL_CHECKPOINT:-"${BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH:-}"}
ALLOW_PAPER_RL_ABLATION=${ALLOW_PAPER_RL_ABLATION:-"false"}
RESUME_FROM_RAW_CHECKPOINT=${RESUME_FROM_RAW_CHECKPOINT:-""}

PROTEIN_MODEL_NAME=${PROTEIN_MODEL_NAME:-"esm3_sm_open_v1"}
MAX_LENGTH_TEXT=${MAX_LENGTH_TEXT:-10000}
MAX_LENGTH_PROTEIN=${MAX_LENGTH_PROTEIN:-2000}
PROTEIN_EMBEDDING_LAYER=${PROTEIN_EMBEDDING_LAYER:-37}
GO_HIDDEN_DIM=${GO_HIDDEN_DIM:-512}
GO_NUM_GAT_LAYERS=${GO_NUM_GAT_LAYERS:-3}
GO_NUM_HEADS=${GO_NUM_HEADS:-8}
GO_NUM_REDUCED_EMBEDDINGS=${GO_NUM_REDUCED_EMBEDDINGS:-200}
GO_EMBEDDING_DIM=${GO_EMBEDDING_DIM:-2560}
UNIFIED_GO_ENCODER=${UNIFIED_GO_ENCODER:-"True"}
PROTEIN_MODEL_FINETUNE=${PROTEIN_MODEL_FINETUNE:-"False"}
TRAIN_PROJECTOR=${TRAIN_PROJECTOR:-"False"}
TRAIN_GO_MODULES=${TRAIN_GO_MODULES:-"False"}

USE_QLORA=${USE_QLORA:-"True"}
BNB_4BIT_COMPUTE_DTYPE=${BNB_4BIT_COMPUTE_DTYPE:-"bfloat16"}
BNB_4BIT_QUANT_TYPE=${BNB_4BIT_QUANT_TYPE:-"nf4"}
BNB_4BIT_USE_DOUBLE_QUANT=${BNB_4BIT_USE_DOUBLE_QUANT:-"True"}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
SFT_CONVERSION_LORA_RANK=${SFT_CONVERSION_LORA_RANK:-128}
SFT_CONVERSION_LORA_ALPHA=${SFT_CONVERSION_LORA_ALPHA:-256}
SFT_CONVERSION_LORA_DROPOUT=${SFT_CONVERSION_LORA_DROPOUT:-0.05}

LEARNING_RATE=${LEARNING_RATE:-5e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
MAX_STEPS=${MAX_STEPS:-200}
MAX_EPOCHS=${MAX_EPOCHS:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:--1}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-100}
EVAL_SAMPLE_STRATEGY=${EVAL_SAMPLE_STRATEGY:-"stratified_aspect_profile"}
EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-25}
SAVE_EVERY_N_STEPS=${SAVE_EVERY_N_STEPS:-50}
MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:-8}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
WEAVE_TRACE_BUDGET=${WEAVE_TRACE_BUDGET:-64}

NUM_GENERATIONS=${NUM_GENERATIONS:-4}
MIN_NEW_TOKENS=${MIN_NEW_TOKENS:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-768}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
DO_SAMPLE=${DO_SAMPLE:-"True"}
KL_BETA=${KL_BETA:-0.02}
REWARD_FUNCS=${REWARD_FUNCS:-"strict_format,reasoning_presence,go_overlap,answer_nonempty"}
REWARD_WEIGHTS=${REWARD_WEIGHTS:-""}

if [ "${CHECKPOINT_ARTIFACT_NAME+x}" = "x" ]; then
  CHECKPOINT_ARTIFACT_NAME="${CHECKPOINT_ARTIFACT_NAME}"
else
  CHECKPOINT_ARTIFACT_NAME="train-rl-output"
fi
CHECKPOINT_ARTIFACT_ALIASES=${CHECKPOINT_ARTIFACT_ALIASES:-"latest,213.221.225.228"}

PREP_COMMAND=()
TRAIN_COMMAND=()
OPTIONAL_GO_EMBEDDINGS_ARG=()
if as_bool "$TRAIN_USE_SRUN"; then
  PREP_COMMAND+=(
    srun
    --job-name "${TRAIN_JOB_NAME}-prep"
    --partition "$TRAIN_PARTITION"
    --nodes 1
    --ntasks-per-node 1
    --gpus "$TRAIN_NUM_GPUS"
    --cpus-per-task "$TRAIN_CPUS_PER_TASK"
    --mem "$TRAIN_MEM"
    --time "$TRAIN_TIME_LIMIT"
  )
  TRAIN_COMMAND+=(
    srun
    --job-name "$TRAIN_JOB_NAME"
    --partition "$TRAIN_PARTITION"
    --nodes 1
    --ntasks-per-node 1
    --gpus "$TRAIN_NUM_GPUS"
    --cpus-per-task "$TRAIN_CPUS_PER_TASK"
    --mem "$TRAIN_MEM"
    --time "$TRAIN_TIME_LIMIT"
  )
fi

if [ -n "$GO_EMBEDDINGS_PATH" ]; then
  OPTIONAL_GO_EMBEDDINGS_ARG=(--precomputed_embeddings_path "$GO_EMBEDDINGS_PATH")
fi

resolve_artifact_dir() {
  local registry_path="$1"
  local local_dir="$2"

  python "$MODEL_SOURCE_RESOLVER" \
    --wandb-registry-path "$registry_path" \
    --local-dir "$local_dir"
}

resolve_hf_model_dir() {
  local registry_path="$1"
  local local_dir="$2"

  python "$MODEL_SOURCE_RESOLVER" \
    --wandb-registry-path "$registry_path" \
    --local-dir "$local_dir" \
    --required-path config.json
}

PAPER_RL_MODEL_RESOLVED=""
BASE_CHECKPOINT_REF=""
RESOLVED_BASE_MODEL_DIR=""
ABLATION_FROM_PAPER_RL="False"

if [ -n "$PRIMARY_BASE_CHECKPOINT" ]; then
  echo "--- Resolving canonical RL init checkpoint from train-sft-output artifact"
  RESOLVED_TRAIN_SFT_DIR=$(resolve_artifact_dir "$PRIMARY_BASE_CHECKPOINT" "$TRAIN_SFT_SOURCE_DIR")
  if [ -f "$RESOLVED_TRAIN_SFT_DIR/config.json" ]; then
    RESOLVED_BASE_MODEL_DIR="$RESOLVED_TRAIN_SFT_DIR"
    BASE_CHECKPOINT_REF="$PRIMARY_BASE_CHECKPOINT"
    echo "--- Using HF-ready train-sft-output artifact at $RESOLVED_BASE_MODEL_DIR"
  else
    SFT_CKPT_PATH=$(find "$RESOLVED_TRAIN_SFT_DIR" -maxdepth 2 -type f -name "*best*.ckpt" | sort | tail -n 1)
    if [ -z "${SFT_CKPT_PATH:-}" ] || [ ! -f "$SFT_CKPT_PATH" ]; then
      SFT_CKPT_PATH="$RESOLVED_TRAIN_SFT_DIR/last.ckpt"
    fi
    if [ ! -f "$SFT_CKPT_PATH" ]; then
      SFT_CKPT_PATH=$(find "$RESOLVED_TRAIN_SFT_DIR" -maxdepth 2 -type f -name "*.ckpt" | sort | tail -n 1)
    fi

    if [ -z "${SFT_CKPT_PATH:-}" ] || [ ! -f "$SFT_CKPT_PATH" ]; then
      echo "Error: train-sft-output artifact does not contain config.json or a .ckpt checkpoint"
      exit 1
    fi

    if [ -z "$PAPER_RL_CHECKPOINT" ]; then
      echo "Error: BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH is required to convert train-sft-output into HF format"
      exit 1
    fi

    echo "--- Resolving comparison model needed for SFT checkpoint conversion"
    PAPER_RL_MODEL_RESOLVED=$(resolve_hf_model_dir "$PAPER_RL_CHECKPOINT" "$PAPER_RL_MODEL_DIR")

    if [ -f "$TRAIN_SFT_HF_DIR/config.json" ]; then
      echo "--- Reusing converted HF train-sft-output at $TRAIN_SFT_HF_DIR"
    else
      if [ -e "$TRAIN_SFT_HF_DIR" ]; then
        echo "Error: $TRAIN_SFT_HF_DIR exists but does not look like a complete HF model directory"
        exit 1
      fi

      echo "--- Converting train-sft-output checkpoint to HF format"
      "${PREP_COMMAND[@]}" python "$SFT_TO_HF_CONVERTER" \
        --checkpoint_path "$SFT_CKPT_PATH" \
        --save_dir "$TRAIN_SFT_HF_DIR" \
        --text_model_name "$PAPER_RL_MODEL_RESOLVED" \
        --protein_model_name "$PROTEIN_MODEL_NAME" \
        --cache_dir "$CACHE_DIR" \
        --max_length_text "$MAX_LENGTH_TEXT" \
        --max_length_protein "$MAX_LENGTH_PROTEIN" \
        --lora_rank "$SFT_CONVERSION_LORA_RANK" \
        --lora_alpha "$SFT_CONVERSION_LORA_ALPHA" \
        --lora_dropout "$SFT_CONVERSION_LORA_DROPOUT" \
        --protein_embedding_layer "$PROTEIN_EMBEDDING_LAYER" \
        --go_obo_path "$GO_OBO_PATH" \
        --go_hidden_dim "$GO_HIDDEN_DIM" \
        --go_num_gat_layers "$GO_NUM_GAT_LAYERS" \
        --go_num_heads "$GO_NUM_HEADS" \
        --go_num_reduced_embeddings "$GO_NUM_REDUCED_EMBEDDINGS" \
        --go_embedding_dim "$GO_EMBEDDING_DIM" \
        --unified_go_encoder "$UNIFIED_GO_ENCODER" \
        --protein_model_finetune "$PROTEIN_MODEL_FINETUNE" \
        "${OPTIONAL_GO_EMBEDDINGS_ARG[@]}"
    fi

    RESOLVED_BASE_MODEL_DIR="$TRAIN_SFT_HF_DIR"
    BASE_CHECKPOINT_REF="$PRIMARY_BASE_CHECKPOINT"
    echo "--- Using converted HF train-sft-output at $RESOLVED_BASE_MODEL_DIR"
  fi
elif as_bool "$ALLOW_PAPER_RL_ABLATION"; then
  if [ -z "$PAPER_RL_CHECKPOINT" ]; then
    echo "Error: BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH is required for paper-RL ablation"
    exit 1
  fi
  echo "--- No train-sft-output artifact configured; using paper RL ablation path"
  PAPER_RL_MODEL_RESOLVED=$(resolve_hf_model_dir "$PAPER_RL_CHECKPOINT" "$PAPER_RL_MODEL_DIR")
  RESOLVED_BASE_MODEL_DIR="$PAPER_RL_MODEL_RESOLVED"
  BASE_CHECKPOINT_REF="$PAPER_RL_CHECKPOINT"
  ABLATION_FROM_PAPER_RL="True"
else
  echo "Error: BIOREASON_TRAIN_SFT_MODEL_REGISTRY_PATH is not set. Run SFT first or set ALLOW_PAPER_RL_ABLATION=true."
  exit 1
fi

if [ ! -f "$RESOLVED_BASE_MODEL_DIR/config.json" ]; then
  echo "Error: RL init model directory is missing config.json: $RESOLVED_BASE_MODEL_DIR"
  exit 1
fi

if [ -z "$CAFA5_DATASET" ]; then
  echo "--- Resolving reasoning dataset source for RL from W&B Artifact"
  CAFA5_DATASET=$(python "$DATA_BUNDLE_RESOLVER" \
    --data-manifest-path "$DATA_MANIFEST_PATH" \
    --data-bundle "$DATA_BUNDLE" \
    --asset-key reasoning_dataset \
    --print-field local_dir)
fi
if [ -z "$CAFA5_DATASET" ]; then
  echo "Error: failed to resolve reasoning dataset source for RL"
  exit 1
fi
echo "--- Reasoning dataset materialized at $CAFA5_DATASET"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RL_RUN_FAMILY="rl-sft"
if as_bool "$ALLOW_PAPER_RL_ABLATION" && [ -z "$PRIMARY_BASE_CHECKPOINT" ]; then
  RL_RUN_FAMILY="rl-paper"
fi
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"${RL_RUN_FAMILY}-${TIMESTAMP}"}
OUTPUT_DIR="${RL_OUTPUT_ROOT}/${WANDB_RUN_NAME}"
mkdir -p "$RL_OUTPUT_ROOT"

echo "--- RL init checkpoint ref: $BASE_CHECKPOINT_REF"
echo "--- RL init model dir: $RESOLVED_BASE_MODEL_DIR"
echo "--- RL output dir: $OUTPUT_DIR"

RESUME_ARGS=()
if [ -n "$RESUME_FROM_RAW_CHECKPOINT" ]; then
  RESUME_ARGS=(--resume_from_raw_checkpoint "$RESUME_FROM_RAW_CHECKPOINT")
fi

stdbuf -oL -eL "${TRAIN_COMMAND[@]}" python train_protein_grpo.py \
  --run_name "$WANDB_RUN_NAME" \
  --wandb_project "$BASE_WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_mode "$WANDB_MODE" \
  --weave_project "$WEAVE_PROJECT" \
  --weave_trace_budget "$WEAVE_TRACE_BUDGET" \
  --benchmark_version "$BENCHMARK_VERSION" \
  --temporal_split_artifact "$TEMPORAL_SPLIT_ARTIFACT" \
  --dataset_config "$DATASET_CONFIG" \
  --reasoning_dataset_config "$REASONING_DATASET_CONFIG" \
  --dataset_artifact "$DATASET_ARTIFACT" \
  --shortlist_query "$SHORTLIST_QUERY" \
  --shortlist_mode "$SHORTLIST_MODE" \
  --train_start_release "$TRAIN_START_RELEASE" \
  --train_end_release "$TRAIN_END_RELEASE" \
  --dev_end_release "$DEV_END_RELEASE" \
  --test_end_release "$TEST_END_RELEASE" \
  --base_checkpoint "$BASE_CHECKPOINT_REF" \
  --model_artifact "$CHECKPOINT_ARTIFACT_NAME" \
  --job_time_limit "$JOB_TIME_LIMIT" \
  --text_model_name "$RESOLVED_BASE_MODEL_DIR" \
  --protein_model_name "$PROTEIN_MODEL_NAME" \
  --cache_dir "$CACHE_DIR" \
  --go_obo_path "$GO_OBO_PATH" \
  --structure_dir "$STRUCTURE_DIR" \
  --dataset_cache_dir "$DATASET_CACHE_DIR" \
  --dataset_type cafa5 \
  --cafa5_dataset "$CAFA5_DATASET" \
  --cafa5_dataset_name "$REASONING_DATASET_NAME" \
  --reasoning_dataset_name "$REASONING_DATASET_NAME" \
  --interpro_dataset_name "$INTERPRO_DATASET_NAME" \
  --go_gpt_predictions_column go_pred \
  --include_ground_truth_in_final_answer False \
  --add_uniprot_summary True \
  --is_swissprot False \
  --include_go_defs False \
  --interpro_in_prompt True \
  --ppi_in_prompt True \
  --predict_interpro False \
  --include_protein_function_summary True \
  --split_go_aspects False \
  --max_length_text "$MAX_LENGTH_TEXT" \
  --max_length_protein "$MAX_LENGTH_PROTEIN" \
  --protein_embedding_layer "$PROTEIN_EMBEDDING_LAYER" \
  --go_hidden_dim "$GO_HIDDEN_DIM" \
  --go_num_gat_layers "$GO_NUM_GAT_LAYERS" \
  --go_num_heads "$GO_NUM_HEADS" \
  --go_num_reduced_embeddings "$GO_NUM_REDUCED_EMBEDDINGS" \
  --go_embedding_dim "$GO_EMBEDDING_DIM" \
  --unified_go_encoder "$UNIFIED_GO_ENCODER" \
  --protein_model_finetune "$PROTEIN_MODEL_FINETUNE" \
  --train_projector "$TRAIN_PROJECTOR" \
  --train_go_modules "$TRAIN_GO_MODULES" \
  --use_qlora "$USE_QLORA" \
  --bnb_4bit_compute_dtype "$BNB_4BIT_COMPUTE_DTYPE" \
  --bnb_4bit_quant_type "$BNB_4BIT_QUANT_TYPE" \
  --bnb_4bit_use_double_quant "$BNB_4BIT_USE_DOUBLE_QUANT" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --train_batch_size "$TRAIN_BATCH_SIZE" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --max_steps "$MAX_STEPS" \
  --max_epochs "$MAX_EPOCHS" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --max_train_samples "$MAX_TRAIN_SAMPLES" \
  --max_eval_samples "$MAX_EVAL_SAMPLES" \
  --eval_sample_strategy "$EVAL_SAMPLE_STRATEGY" \
  --eval_every_n_steps "$EVAL_EVERY_N_STEPS" \
  --save_every_n_steps "$SAVE_EVERY_N_STEPS" \
  --max_eval_batches "$MAX_EVAL_BATCHES" \
  --max_grad_norm "$MAX_GRAD_NORM" \
  --num_generations "$NUM_GENERATIONS" \
  --min_new_tokens "$MIN_NEW_TOKENS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --do_sample "$DO_SAMPLE" \
  --kl_beta "$KL_BETA" \
  --reward_funcs "$REWARD_FUNCS" \
  --reward_weights "$REWARD_WEIGHTS" \
  --output_dir "$OUTPUT_DIR" \
  --checkpoint_artifact_name "$CHECKPOINT_ARTIFACT_NAME" \
  --checkpoint_artifact_aliases "$CHECKPOINT_ARTIFACT_ALIASES" \
  --ablation_from_paper_rl "$ABLATION_FROM_PAPER_RL" \
  "${OPTIONAL_GO_EMBEDDINGS_ARG[@]}" \
  "${RESUME_ARGS[@]}"
