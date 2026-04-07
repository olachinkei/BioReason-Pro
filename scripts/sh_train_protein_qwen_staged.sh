#!/bin/bash

# ===================================================================================================
# SLURM Configuration
# Adjust these to match your cluster. Example values shown.
# ===================================================================================================
# #SBATCH --job-name=train_protein_qwen_staged
# #SBATCH --partition=your_gpu_partition
# #SBATCH --time=12:00:00
# #SBATCH --gpus=8
# #SBATCH --ntasks-per-node=4
# #SBATCH --nodes=2
# #SBATCH --cpus-per-task=16
# #SBATCH --mem=256gb
# #SBATCH --output=train_protein_qwen_staged_%j_%x.out
# #SBATCH --error=train_protein_qwen_staged_%j_%x.err


# Run from project root
cd "$(dirname "$0")/.."

REGISTRY_ENV_FILE=${REGISTRY_ENV_FILE:-"configs/disease_benchmark/wandb_registry_paths.env"}

if [ -f "$REGISTRY_ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$REGISTRY_ENV_FILE"
fi

# ===================================================================================================
# Environment Setup
# Set these to your conda environment and project root.
# ===================================================================================================
# export PATH=/path/to/your/conda/envs/bin:$PATH     # e.g., /home/user/miniconda/envs/bio/bin
# ROOT_DIR=/path/to/BioReason-Pro                     # e.g., /home/user/BioReason-Pro
# cd $ROOT_DIR

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

unset SLURM_TRES_PER_TASK
# ===================================================================================================

as_bool() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|t|yes|y) return 0 ;;
    *) return 1 ;;
  esac
}



# ===================================================================================================
# Shared Configuration
# ===================================================================================================
BASE_WANDB_PROJECT=${BASE_WANDB_PROJECT:-"${WANDB_PROJECT:-bioreason-pro-finetune}"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
EXPECTED_WANDB_ENTITY=${EXPECTED_WANDB_ENTITY:-"wandb-healthcare"}
EXPECTED_WANDB_PROJECT=${EXPECTED_WANDB_PROJECT:-"bioreason-pro-custom"}
ALLOW_WANDB_PROJECT_MISMATCH=${ALLOW_WANDB_PROJECT_MISMATCH:-"False"}
WEAVE_PROJECT=${WEAVE_PROJECT:-""}
WEAVE_TRACE_BUDGET=${WEAVE_TRACE_BUDGET:-64}
if [ -z "$WEAVE_PROJECT" ] && [ -n "$WANDB_ENTITY" ]; then
  WEAVE_PROJECT="${WANDB_ENTITY}/${BASE_WANDB_PROJECT}"
fi

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
TEXT_MODEL_NAME="Qwen/Qwen3-4B-Thinking-2507"
EXPERIMENT_NAME="reasoning-sft"
MODEL_SOURCE_RESOLVER=${MODEL_SOURCE_RESOLVER:-"scripts/materialize_model_source.py"}
DATA_BUNDLE_RESOLVER=${DATA_BUNDLE_RESOLVER:-"scripts/materialize_data_bundle.py"}
DATA_MANIFEST_PATH=${DATA_MANIFEST_PATH:-"configs/disease_benchmark/data_registry.json"}
DATA_BUNDLE=${DATA_BUNDLE:-"main_production"}
TRAIN_PARTITION=${TRAIN_PARTITION:-"h100"}
TRAIN_NUM_GPUS=${TRAIN_NUM_GPUS:-1}
TRAIN_NUM_NODES=${TRAIN_NUM_NODES:-1}
TRAIN_CPUS_PER_TASK=${TRAIN_CPUS_PER_TASK:-16}
TRAIN_MEM=${TRAIN_MEM:-"128G"}
TRAIN_TIME_LIMIT=${TRAIN_TIME_LIMIT:-"12:00:00"}
TRAIN_JOB_NAME=${TRAIN_JOB_NAME:-"bioreason-sft-stage2"}
TRAIN_USE_SRUN=${TRAIN_USE_SRUN:-"True"}
TRAIN_EXCLUSIVE=${TRAIN_EXCLUSIVE:-"True"}

# --- Paths: Set these to your local directories ---
BASE_CHECKPOINT_DIR=${BASE_CHECKPOINT_DIR:-"data/artifacts/models/bioreason_pro_rl_paper"}
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-"data/artifacts/hf_cache"}
CACHE_DIR=${CACHE_DIR:-"data/artifacts/cache"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"data/structures"}
GO_EMBEDDINGS_PATH=${GO_EMBEDDINGS_PATH:-"${BIOREASON_GO_EMBEDDINGS_PATH:-}"}
GO_OBO_PATH=${GO_OBO_PATH:-"bioreason2/dataset/go-basic.obo"}
WEAVE_SERVER_CACHE_DIR=${WEAVE_SERVER_CACHE_DIR:-"${CACHE_DIR}/weave_server_cache"}
mkdir -p "$WEAVE_SERVER_CACHE_DIR"
export WEAVE_SERVER_CACHE_DIR

# --- Dataset Configuration ---
CAFA5_DATASET=${CAFA5_DATASET:-""}
STAGE1_DATASET_NAME=${STAGE1_DATASET_NAME:-"disease_temporal_hc_reasoning_v1"}
STAGE2_DATASET_NAME=${STAGE2_DATASET_NAME:-"disease_temporal_hc_reasoning_v1"}
STAGE2_DATASET_WEIGHTS="1"
REASONING_DATASET_NAME=${REASONING_DATASET_NAME:-"disease_temporal_hc_reasoning_v1"}
GO_GPT_PREDICTIONS_COLUMN="go_pred"
INCLUDE_GROUND_TRUTH_IN_FINAL_ANSWER=False
ADD_UNIPROT_SUMMARY=True
IS_SWISSPROT=False
VALIDATION_SUBSET_SIZE=${VALIDATION_SUBSET_SIZE:-100}
VALIDATION_SUBSET_STRATEGY=${VALIDATION_SUBSET_STRATEGY:-"stratified_aspect_profile"}
STAGE2_BATCH_SIZE=${STAGE2_BATCH_SIZE:-4}
STAGE2_GRADIENT_ACCUMULATION_STEPS=${STAGE2_GRADIENT_ACCUMULATION_STEPS:-1}
STAGE2_MAX_EPOCHS=${STAGE2_MAX_EPOCHS:-10}
STAGE2_LEARNING_RATE=${STAGE2_LEARNING_RATE:-1e-4}
STAGE2_WARMUP_RATIO=${STAGE2_WARMUP_RATIO:-0.05}
STAGE2_VAL_CHECK_INTERVAL=${STAGE2_VAL_CHECK_INTERVAL:-0.2}
STAGE2_EARLY_STOPPING_PATIENCE=${STAGE2_EARLY_STOPPING_PATIENCE:-2}
STAGE2_EARLY_STOPPING_MIN_DELTA=${STAGE2_EARLY_STOPPING_MIN_DELTA:-0.0}
STAGE2_LIMIT_TRAIN_BATCHES=${STAGE2_LIMIT_TRAIN_BATCHES:-1.0}
STAGE2_LIMIT_VAL_BATCHES=${STAGE2_LIMIT_VAL_BATCHES:-1.0}
STAGE2_LOG_EVERY_N_STEPS=${STAGE2_LOG_EVERY_N_STEPS:-10}
STAGE2_SAMPLE_GENERATION_EVERY_N_STEPS=${STAGE2_SAMPLE_GENERATION_EVERY_N_STEPS:-500}
STAGE2_RUN_LABEL=${STAGE2_RUN_LABEL:-""}
USE_UNSLOTH=${USE_UNSLOTH:-False}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}

# --- Benchmark / Tracking Configuration ---
BENCHMARK_VERSION="213 -> 221 -> 225 -> 228"
TEMPORAL_SPLIT_ARTIFACT=${TEMPORAL_SPLIT_ARTIFACT:-"${BIOREASON_MAIN_TEMPORAL_SPLIT_REGISTRY_PATH:-}"}
DATASET_CONFIG="disease_temporal_hc_reasoning_v1"
REASONING_DATASET_CONFIG="disease_temporal_hc_reasoning_v1"
DATASET_ARTIFACT=${DATASET_ARTIFACT:-"${BIOREASON_MAIN_REASONING_DATASET_REGISTRY_PATH:-}"}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH:-}"}
SHORTLIST_MODE="high-confidence"
SHORTLIST_QUERY="reviewed:true AND organism_id:9606 AND cc_disease:* AND (xref:mim-* OR xref:orphanet-*) AND (go_exp:* OR go_ida:* OR go_ipi:* OR go_igi:* OR go_imp:* OR go_iep:* OR go_ic:* OR go_tas:*)"
TRAIN_START_RELEASE=213
TRAIN_END_RELEASE=221
DEV_END_RELEASE=225
TEST_END_RELEASE=228
JOB_TIME_LIMIT="12:00:00"

# --- Model Configuration ---
MAX_LENGTH_TEXT=${MAX_LENGTH_TEXT:-4096}
MAX_LENGTH_PROTEIN=${MAX_LENGTH_PROTEIN:-2000}
LORA_RANK=128
LORA_ALPHA=256
LORA_DROPOUT=0.05
ESM_LAYER=37
NUM_WORKERS=${NUM_WORKERS:-4}

INTERPRO_IN_PROMPT=True
PREDICT_INTERPRO=False
PPI_IN_PROMPT=True

RESOLVED_BASE_MODEL_DIR=""
if [ -z "$BASE_CHECKPOINT" ]; then
  echo "Error: BASE_CHECKPOINT is not set. Set BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH first."
  exit 1
fi

echo "--- Resolving pretuning comparison model source for SFT from W&B Registry"
RESOLVED_BASE_MODEL_DIR=$(python "$MODEL_SOURCE_RESOLVER" \
  --wandb-registry-path "$BASE_CHECKPOINT" \
  --local-dir "$BASE_CHECKPOINT_DIR" \
  --required-path config.json)
if [ -z "$RESOLVED_BASE_MODEL_DIR" ]; then
  echo "Error: failed to resolve base model source from registry"
  exit 1
fi
TEXT_MODEL_NAME="$RESOLVED_BASE_MODEL_DIR"
echo "--- Pretuning comparison model materialized at $RESOLVED_BASE_MODEL_DIR"

if [ -z "$CAFA5_DATASET" ]; then
  echo "--- Resolving reasoning dataset source for SFT from W&B Artifact"
  CAFA5_DATASET=$(python "$DATA_BUNDLE_RESOLVER" \
    --data-manifest-path "$DATA_MANIFEST_PATH" \
    --data-bundle "$DATA_BUNDLE" \
    --asset-key reasoning_dataset \
    --print-field local_dir)
fi
if [ -z "$CAFA5_DATASET" ]; then
  echo "Error: failed to resolve reasoning dataset source"
  exit 1
fi
echo "--- Reasoning dataset materialized at $CAFA5_DATASET"

BASE_MODEL_PROJECTOR_WEIGHTS_PATH="${RESOLVED_BASE_MODEL_DIR:+$RESOLVED_BASE_MODEL_DIR/protein_projection.pt}"
BASE_MODEL_GO_PROJECTOR_WEIGHTS_PATH="${RESOLVED_BASE_MODEL_DIR:+$RESOLVED_BASE_MODEL_DIR/go_projection.pt}"
BASE_MODEL_GO_ENCODER_WEIGHTS_PATH="${RESOLVED_BASE_MODEL_DIR:+$RESOLVED_BASE_MODEL_DIR/go_encoder.pt}"
RUN_STAGE1=${RUN_STAGE1:-"False"}

BASE_COMMAND=()
if as_bool "$TRAIN_USE_SRUN"; then
  BASE_COMMAND+=(
    srun
    --job-name "$TRAIN_JOB_NAME"
    --partition "$TRAIN_PARTITION"
    --nodes "$TRAIN_NUM_NODES"
    --ntasks-per-node 1
    --gpus "$TRAIN_NUM_GPUS"
    --cpus-per-task "$TRAIN_CPUS_PER_TASK"
    --mem "$TRAIN_MEM"
    --time "$TRAIN_TIME_LIMIT"
  )
  if as_bool "$TRAIN_EXCLUSIVE"; then
    BASE_COMMAND+=(--exclusive)
  fi
fi

BASE_COMMAND+=(
    python train_protein_llm.py
    --cache_dir "$CACHE_DIR"
    --wandb_entity "$WANDB_ENTITY"
    --wandb_job_type train_sft
    --benchmark_version "$BENCHMARK_VERSION"
    --temporal_split_artifact "$TEMPORAL_SPLIT_ARTIFACT"
    --dataset_config "$DATASET_CONFIG"
    --reasoning_dataset_config "$REASONING_DATASET_CONFIG"
    --dataset_artifact "$DATASET_ARTIFACT"
    --shortlist_query "$SHORTLIST_QUERY"
    --shortlist_mode "$SHORTLIST_MODE"
    --train_start_release "$TRAIN_START_RELEASE"
    --train_end_release "$TRAIN_END_RELEASE"
    --dev_end_release "$DEV_END_RELEASE"
    --test_end_release "$TEST_END_RELEASE"
    --base_checkpoint "$BASE_CHECKPOINT"
    --job_time_limit "$JOB_TIME_LIMIT"
    --text_model_name "$TEXT_MODEL_NAME"
    --protein_model_name esm3_sm_open_v1
    --strategy ddp_find_unused_parameters_false
    --use_qlora False
    --use_unsloth "$USE_UNSLOTH"
    --num_gpus "$TRAIN_NUM_GPUS"
    --batch_size 4
    --num_nodes "$TRAIN_NUM_NODES"
    --gradient_accumulation_steps 1
    --model_type protein-llm
    --dataset_type cafa5
    --cafa5_dataset "$CAFA5_DATASET"
    --reasoning_dataset_name "$REASONING_DATASET_NAME"
    --go_gpt_predictions_column "$GO_GPT_PREDICTIONS_COLUMN"
    --include_ground_truth_in_final_answer "$INCLUDE_GROUND_TRUTH_IN_FINAL_ANSWER"
    --add_uniprot_summary "$ADD_UNIPROT_SUMMARY"
    --is_swissprot "$IS_SWISSPROT"
    --dataset_cache_dir "$DATASET_CACHE_DIR"
    --cache_dir "$CACHE_DIR"
    --structure_dir "$STRUCTURE_DIR"
    --val_split_ratio 0.1
    --validation_subset_size "$VALIDATION_SUBSET_SIZE"
    --validation_subset_strategy "$VALIDATION_SUBSET_STRATEGY"
    --max_length_protein "$MAX_LENGTH_PROTEIN"
    --max_length_text "$MAX_LENGTH_TEXT"
    --lora_rank "$LORA_RANK"
    --lora_alpha "$LORA_ALPHA"
    --lora_dropout "$LORA_DROPOUT"
    --protein_model_finetune False
    --protein_embedding_layer "$ESM_LAYER"
    --go_model_finetune True
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --go_obo_path "$GO_OBO_PATH"
    --go_hidden_dim 512
    --go_num_gat_layers 3
    --go_num_heads 8
    --go_num_reduced_embeddings 200
    --go_embedding_dim 2560
    --unified_go_encoder True
    --return_answer_in_batch False
    --num_workers "$NUM_WORKERS"
    --weight_decay 0.01
    --seed 23
    --save_top_k 1
    --include_go_defs False
    --interpro_dataset_name interpro_metadata
    --split_go_aspects False
    --interpro_in_prompt "$INTERPRO_IN_PROMPT"
    --predict_interpro "$PREDICT_INTERPRO"
    --ppi_in_prompt "$PPI_IN_PROMPT"
    --include_protein_function_summary True
    --min_go_mf_freq 1
    --min_go_bp_freq 1
    --min_go_cc_freq 1
    --apply_go_filtering_to_val_test False
    --log_every_n_steps "$STAGE2_LOG_EVERY_N_STEPS"
    --enable_sample_generation True
    --verbose_sample_generation False
    --weave_project "$WEAVE_PROJECT"
    --weave_trace_budget "$WEAVE_TRACE_BUDGET"
    --val_check_interval 0.2
    --debug False
)

if [ -n "$GO_EMBEDDINGS_PATH" ]; then
  BASE_COMMAND+=(--precomputed_embeddings_path "$GO_EMBEDDINGS_PATH")
fi
# ===================================================================================================



# ===================================================================================================
# --- Optional Stage 1: Warm-up (Projector + GO Training) ---
# ===================================================================================================
PROJECTOR_WEIGHTS_PATH="$BASE_MODEL_PROJECTOR_WEIGHTS_PATH"
GO_PROJECTOR_WEIGHTS_PATH="$BASE_MODEL_GO_PROJECTOR_WEIGHTS_PATH"
GO_ENCODER_WEIGHTS_PATH="$BASE_MODEL_GO_ENCODER_WEIGHTS_PATH"

if as_bool "$RUN_STAGE1"; then
  echo "--- Starting optional Stage 1: Projector Training"

  RUN_NAME_S1_DIR="${BASE_WANDB_PROJECT}-$(basename ${TEXT_MODEL_NAME})-stage1"
  TIMESTAMP_S1=$(date +%Y%m%d-%H%M%S)
  STAGE1_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR}/${RUN_NAME_S1_DIR}-${TIMESTAMP_S1}"
  WANDB_RUN_NAME_S1_DEFAULT="stage1-$(basename ${TEXT_MODEL_NAME})-${EXPERIMENT_NAME}"
  WANDB_RUN_NAME_S1="${WANDB_RUN_NAME_S1:-${WANDB_RUN_NAME:-$WANDB_RUN_NAME_S1_DEFAULT}}"
  STAGE1_CHECKPOINT_ARTIFACT_NAME="${STAGE1_CHECKPOINT_ARTIFACT_NAME:-${CHECKPOINT_ARTIFACT_NAME:-${WANDB_RUN_NAME_S1}-checkpoints}}"
  mkdir -p $STAGE1_CHECKPOINT_DIR

  stdbuf -oL -eL "${BASE_COMMAND[@]}" \
    ${BASE_MODEL_PROJECTOR_WEIGHTS_PATH:+--projector_checkpoint_path "$BASE_MODEL_PROJECTOR_WEIGHTS_PATH"} \
    ${BASE_MODEL_GO_PROJECTOR_WEIGHTS_PATH:+--go_projection_checkpoint_path "$BASE_MODEL_GO_PROJECTOR_WEIGHTS_PATH"} \
    ${BASE_MODEL_GO_ENCODER_WEIGHTS_PATH:+--go_encoder_checkpoint_path "$BASE_MODEL_GO_ENCODER_WEIGHTS_PATH"} \
    --run_name "${WANDB_RUN_NAME_S1}" \
    --checkpoint_artifact_name "${STAGE1_CHECKPOINT_ARTIFACT_NAME}" \
    --cafa5_dataset_name $STAGE1_DATASET_NAME \
    --training_stage 1 \
    --max_epochs 1 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --text_model_finetune False \
    --checkpoint_dir $STAGE1_CHECKPOINT_DIR \
    --wandb_project "${BASE_WANDB_PROJECT}-stage1"

  PROJECTOR_WEIGHTS_PATH="$STAGE1_CHECKPOINT_DIR/projector_weights.pt"
  if [ ! -f "$PROJECTOR_WEIGHTS_PATH" ]; then
    echo "Error: Stage 1 failed. Projector weights not found at $PROJECTOR_WEIGHTS_PATH"
    exit 1
  fi

  GO_PROJECTOR_WEIGHTS_PATH="$STAGE1_CHECKPOINT_DIR/go_projection_weights.pt"
  GO_ENCODER_WEIGHTS_PATH="$STAGE1_CHECKPOINT_DIR/go_encoder_weights.pt"
  if [ ! -f "$GO_PROJECTOR_WEIGHTS_PATH" ] || [ ! -f "$GO_ENCODER_WEIGHTS_PATH" ]; then
    echo "Error: Stage 1 failed. GO projector or GO encoder weights not found"
    exit 1
  fi

  echo "--- Stage 1 Complete. Projector weights saved to $PROJECTOR_WEIGHTS_PATH"
  echo "--- Stage 1 Complete. GO projector weights saved to $GO_PROJECTOR_WEIGHTS_PATH"
  echo "--- Stage 1 Complete. GO encoder weights saved to $GO_ENCODER_WEIGHTS_PATH"
else
  echo "--- Skipping Stage 1. Stage 2 will warm-start directly from comparison model projector / GO weights"
fi
# ===================================================================================================



# ===================================================================================================
# --- Stage 2: Full Model Fine-tuning ---
# ===================================================================================================
echo "--- Starting Stage 2: Full Model Fine-tuning"

RUN_NAME_S2_DIR_DEFAULT="${BASE_WANDB_PROJECT}-$(basename ${TEXT_MODEL_NAME})-${EXPERIMENT_NAME}-stage2"
WANDB_RUN_NAME_S2_DEFAULT="stage2-$(basename ${TEXT_MODEL_NAME})-${EXPERIMENT_NAME}"
if [ -n "$STAGE2_RUN_LABEL" ]; then
  RUN_NAME_S2_DIR_DEFAULT="${RUN_NAME_S2_DIR_DEFAULT}-${STAGE2_RUN_LABEL}"
  WANDB_RUN_NAME_S2_DEFAULT="${WANDB_RUN_NAME_S2_DEFAULT}-${STAGE2_RUN_LABEL}"
fi
RUN_NAME_S2_DIR="${RUN_NAME_S2_DIR:-$RUN_NAME_S2_DIR_DEFAULT}"
STAGE2_CHECKPOINT_DIR="${STAGE2_CHECKPOINT_DIR:-${BASE_CHECKPOINT_DIR}/${RUN_NAME_S2_DIR}}"
WANDB_RUN_NAME_S2="${WANDB_RUN_NAME_S2:-${WANDB_RUN_NAME:-$WANDB_RUN_NAME_S2_DEFAULT}}"
STAGE2_CHECKPOINT_ARTIFACT_NAME="${STAGE2_CHECKPOINT_ARTIFACT_NAME:-${CHECKPOINT_ARTIFACT_NAME:-${WANDB_RUN_NAME_S2}-checkpoints}}"
mkdir -p $STAGE2_CHECKPOINT_DIR

CKPT_ARG=""
LATEST_CKPT="${STAGE2_CHECKPOINT_DIR}/last.ckpt"

if [ -e "$LATEST_CKPT" ]; then
    echo "   Found existing checkpoint at $LATEST_CKPT, resuming training"
    CKPT_ARG="--ckpt_path $LATEST_CKPT"
else
    echo "   No existing checkpoint found, starting fresh training"
    CKPT_ARG=""
fi

stdbuf -oL -eL "${BASE_COMMAND[@]}" \
    --run_name "${WANDB_RUN_NAME_S2}" \
    --checkpoint_artifact_name "${STAGE2_CHECKPOINT_ARTIFACT_NAME}" \
    --cafa5_dataset_name $STAGE2_DATASET_NAME \
    --training_stage 2 \
    --max_epochs "$STAGE2_MAX_EPOCHS" \
    --learning_rate "$STAGE2_LEARNING_RATE" \
    --warmup_ratio "$STAGE2_WARMUP_RATIO" \
    --text_model_finetune True \
    --projector_checkpoint_path $PROJECTOR_WEIGHTS_PATH \
    --go_projection_checkpoint_path $GO_PROJECTOR_WEIGHTS_PATH \
    --go_encoder_checkpoint_path $GO_ENCODER_WEIGHTS_PATH \
    --checkpoint_dir $STAGE2_CHECKPOINT_DIR \
    --every_n_train_steps "$STAGE2_SAMPLE_GENERATION_EVERY_N_STEPS" \
    --batch_size "$STAGE2_BATCH_SIZE" \
    --gradient_accumulation_steps "$STAGE2_GRADIENT_ACCUMULATION_STEPS" \
    --val_check_interval "$STAGE2_VAL_CHECK_INTERVAL" \
    --early_stopping_patience "$STAGE2_EARLY_STOPPING_PATIENCE" \
    --early_stopping_min_delta "$STAGE2_EARLY_STOPPING_MIN_DELTA" \
    --limit_train_batches "$STAGE2_LIMIT_TRAIN_BATCHES" \
    --limit_val_batches "$STAGE2_LIMIT_VAL_BATCHES" \
    --wandb_project "${BASE_WANDB_PROJECT}" \
    $CKPT_ARG
# ===================================================================================================
