# PLAN

This PLAN is a direct translation of [specification.md](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/domain/specification/busiless-rules/specification.md) and [RESEARCH_README.md](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/RESEARCH_README.md) into an execution plan.  
The purpose is to **eliminate ambiguity about what is done and what to do next**.

## 0. Current Status

### 0.1 Adopted Assumptions

The fixed assumptions are as follows.

- benchmark version: `train=213->225, future=225->228, dev=200, holdout=400`
- benchmark alias: `213.221.225.228`
- primary dataset: `disease_temporal_hc_reasoning_v1`
- comparison model: `bioreason-pro-rl-paper`
- source of truth: W&B Artifact ref
- local filesystem: scratch

### 0.2 Progress Summary

| Item | Status | Current State |
|---|---|---|
| Specification cleanup | Complete | Final spec unified in [specification.md](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/domain/specification/busiless-rules/specification.md) |
| Execution procedure cleanup | Complete | Current runbook is [RESEARCH_README.md](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/RESEARCH_README.md) |
| Temporal split artifact creation | Complete | `wandb-healthcare/bioreason-pro-custom/disease-temporal-split:production` |
| Reasoning dataset creation | Complete | `wandb-healthcare/bioreason-pro-custom/disease-temporal-reasoning:production` |
| Comparison model artifact finalized | Complete | `wandb-healthcare/bioreason-pro-custom/bioreason-pro-rl:production` |
| CoreWeave execution flow cleanup | Complete | `srun`-based execution, remote env, artifact resolution, 1-sample smoke verified |
| Comparison model validation eval | Needs re-run | Re-run on the fixed 200-protein dev split |
| RL | Needs re-run | Re-run directly from `bioreason-pro-rl-paper` with the fixed 200-protein dev split |

### 0.3 Immediate Next Steps

The next execution target is to **rebuild the benchmark with merged train + fixed dev/holdout, then verify `comparison-family` on the 200-protein dev split**.  
After that, review dev metrics and proceed directly to RL from `bioreason-pro-rl-paper`; final reported values will come from a separate `test` eval run on the 400-protein holdout split.

## 1. Data Preparation

Status: **Complete**

### 1.1 Tasks

Perform the following on local Mac.

1. Build temporal split artifact
2. Pass split sanity check
3. Build reasoning dataset
4. Upload to W&B Artifact

The high-level entry point is fixed to [run_temporal_split_artifact_pipeline.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/scripts/run_temporal_split_artifact_pipeline.py).

### 1.2 Execution Commands

```bash
cd /Users/keisuke/Project/learning/drug_discovery/BioReason-Pro

set -a
source .env
set +a

uv venv .venv-mac-data --python 3.11
source .venv-mac-data/bin/activate
uv pip install -r requirements/uv-local-data.txt

uv run --active python scripts/run_temporal_split_artifact_pipeline.py \
  --variant main \
  --shortlist-mode high-confidence \
  --use-shell-filter \
  --build-datasets \
  --upload-to-wandb \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-project "$WANDB_PROJECT"
```

### 1.3 Completion Criteria

Considered complete when the following are satisfied.

- `split_validation.time_order_valid == true`
- `split_validation.protein_disjoint_valid == true`
- temporal split artifact uploaded to W&B
- reasoning dataset artifact uploaded to W&B

### 1.4 Current Artifacts

- temporal split artifact
  - `wandb-healthcare/bioreason-pro-custom/disease-temporal-split:production`
- reasoning dataset artifact
  - `wandb-healthcare/bioreason-pro-custom/disease-temporal-reasoning:production`

## 2. GPU Access

Status: **Complete**

### 2.1 Tasks

Submit jobs via `srun` from the CoreWeave SUNK login node.  
Do not assume manual access to GPU nodes.

### 2.2 Execution Procedure

1. SSH into login node
2. `rsync` code only from local
3. Create `uv` environment on CoreWeave side
4. Prepare `wandb_registry_paths.env` and `wandb_asset_sources.env`

### 2.3 Execution Commands

SSH:

```bash
ssh -o IdentitiesOnly=yes kkamata+cwb607@sunk.cwb607-training.coreweave.app
```

rsync:

```bash
cd /Users/keisuke/Project/learning/drug_discovery

rsync -av --delete \
  --exclude 'data/artifacts/' \
  --exclude '.venv*/' \
  BioReason-Pro/ \
  kkamata+cwb607@sunk.cwb607-training.coreweave.app:~/BioReason-Pro/
```

CoreWeave environment setup:

```bash
cd ~/BioReason-Pro

uv venv .venv-gpu --python 3.11
source .venv-gpu/bin/activate

uv sync
uv pip install esm --no-deps
uv pip install flash-attn --no-build-isolation --no-cache-dir
uv pip install unsloth
uv run --active wandb login
```

env preparation:

```bash
cd ~/BioReason-Pro

set -a
source .env
set +a

cp configs/disease_benchmark/wandb_registry_paths.env.example \
  configs/disease_benchmark/wandb_registry_paths.env

cp configs/disease_benchmark/wandb_asset_sources.env.example \
  configs/disease_benchmark/wandb_asset_sources.env
```

### 2.4 Completion Criteria

This phase is complete when the following are satisfied.

- Can access login node
- Latest code is in `~/BioReason-Pro`
- `.venv-gpu` is created
- `wandb_registry_paths.env` contains:
  - `BIOREASON_MAIN_TEMPORAL_SPLIT_REGISTRY_PATH`
  - `BIOREASON_MAIN_REASONING_DATASET_REGISTRY_PATH`
  - `BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH`

## 3. Comparison Model Evaluation

Status: **Re-run after fixes**

### 3.1 Purpose

Evaluate the pre-tuning comparison model `bioreason-pro-rl-paper` on the currently adopted benchmark using the `validation` split.  
`validation` is the fixed **200-protein dev split** carved deterministically from the future pool, not an on-the-fly subset.

### 3.2 Evaluation Targets

- `comparison-family`: `bioreason-pro-rl-paper`
- `tuned-family`: `train-rl-output`
- `spec-comparison`: all of the above

At this stage, only `comparison-family` is actually being run.

### 3.3 Execution Commands

```bash
srun \
  --partition <gpu_partition> \
  --account <account_name> \
  --gpus 1 \
  --cpus-per-task 8 \
  --mem 128G \
  --time 12:00:00 \
  bash -lc '
    cd ~/BioReason-Pro &&
    source .venv-gpu/bin/activate &&
    uv run --active python scripts/run_registered_eval.py \
      --target-group comparison-family \
      --data-bundle main_production \
      --split validation \
      --wandb-entity "$WANDB_ENTITY" \
      --wandb-project "$WANDB_PROJECT"
  '
```

### 3.4 Completion Criteria

Complete when the following are visible on W&B.

- `job_type=eval`
- `fmax_mf`
- `fmax_bp`
- `fmax_cc`
- `overall_mean_fmax`
- `eval_summary` table
- `eval_samples` table
- Weave Evaluation record

Validation uses the full dev split of `100` proteins; split construction is fixed to `stratified_aspect_profile`.
Runs missing any of `fmax_mf`, `fmax_bp`, `fmax_cc`, `eval_summary`, `eval_samples`, or Weave Evaluation are not treated as complete.

### 3.5 After This Phase

Review results and proceed directly to RL.  
`test` is not used at this point.

## 4. RL

Status: **Ready, run not yet executed**

### 4.1 Purpose

Perform RL using `bioreason-pro-rl-paper` as the canonical input.

### 4.2 Fixed Rules

- Rollout / reward optimization uses the `train` split
- Checkpoint selection and offline sanity-check use the full `validation` split of `200` proteins
- The `test` split is not used
- `job_type=train_rl`
- Wall time is `12:00:00`
- Canonical starting checkpoint is `bioreason-pro-rl-paper`
- Validation split is fixed to `200` proteins and constructed with `EVAL_SAMPLE_STRATEGY=stratified_aspect_profile`

### 4.3 Execution Commands

```bash
cd ~/BioReason-Pro
source .venv-gpu/bin/activate

bash scripts/sh_train_protein_grpo.sh
```

This wrapper internally calls `srun python train_protein_grpo.py ...`.  
Canonical uses `BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH`.

### 4.4 Completion Criteria

Complete when the following are present.

- `job_type=train_rl`
- reward metrics
- KL metrics
- rollout traces
- output checkpoint artifact

After completion, add the following to `wandb_registry_paths.env`.

```bash
export BIOREASON_TRAIN_RL_MODEL_REGISTRY_PATH="entity/project/train-rl-output:alias"
```

## 5. Final Evaluation

Status: **Not started**

### 6.1 Purpose

Evaluate `spec-comparison` on the `test` split as a separate run and produce the final comparison.

### 6.2 Execution Commands

```bash
srun \
  --partition <gpu_partition> \
  --account <account_name> \
  --gpus 1 \
  --cpus-per-task 8 \
  --mem 128G \
  --time 12:00:00 \
  bash -lc '
    cd ~/BioReason-Pro &&
    source .venv-gpu/bin/activate &&
    uv run --active python scripts/run_registered_eval.py \
      --target-group spec-comparison \
      --data-bundle main_production \
      --split test \
      --wandb-entity "$WANDB_ENTITY" \
      --wandb-project "$WANDB_PROJECT"
  '
```

### 6.3 Completion Criteria

Complete when the following are present on W&B.

- `test` metrics for both comparison model and tuned model
- `fmax_mf`
- `fmax_bp`
- `fmax_cc`
- `eval_summary` table
- `eval_samples` table
- Weave Evaluation

## 6. Next Actions

The nearest next actions are the following two.

1. Set up `.venv-gpu` and `wandb_registry_paths.env` on CoreWeave side
2. Evaluate `comparison-family` on `validation` split
