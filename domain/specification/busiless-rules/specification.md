# Specification

This specification defines **only the operational rules ultimately adopted** for GO function prediction of disease-associated human proteins using BioReason-Pro.  
Investigation processes, comparison versions, candidate filters, and lessons from count-based decisions are separated into [learning-log](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/domain/learning-log) and are not retained in this document.

Assumptions:

- The currently adopted benchmark is the **small-data version**
- The current version is `213 -> 221 -> 225 -> 228`
- train / validation / test maintain **protein-disjoint** and **temporal order**
- Generated artifacts on local filesystem are treated as scratch; **the source of truth is W&B Artifact ref**
- `dev` in this document is synonymous with the split name `validation` in code

## 1. Problem Setting

The problem addressed by this specification is **protein-level GO term prediction and disease-aware reasoning for disease-associated human proteins**.

Fixed questions:

- Is there value in performing additional training on BioReason-Pro's public model for disease-associated human proteins?
- On a temporally independent benchmark, does custom tuning produce better predictions and reasoning than the pre-tuning comparison model?

Not addressed by this specification:

- variant pathogenicity classification
- ClinVar-only supervision
- cross-species disease transfer

## 2. Models and Comparison Targets

### 2.1 Model

**BioReason-Pro** (bowang-lab, bioRxiv 2026-03)

| Element | Description |
|---|---|
| Architecture | ESM3 + GO Graph Encoder + Qwen3-4B |
| Task | Protein sequence -> GO term prediction + reasoning trace generation |
| Training method | SFT -> RL |
| Base training data cutoff | UniProt GOA 2022-11 release |

### 2.2 Comparison Targets

Comparison targets are fixed to the following 3 families.

| Method | Role |
|---|---|
| `bioreason-pro-rl-paper` | Pre-tuning comparison model |
| `train-sft-output` | Output artifact produced by custom `train_sft` run |
| `train-rl-output` | Output artifact produced by custom `train_rl` run |

The only model artifact ref needed initially is `bioreason-pro-rl-paper`.  
`train-sft-output` and `train-rl-output` appear as artifacts after their respective training runs are executed.

## 3. Benchmark

### 3.1 Final Adopted Benchmark

The adopted shortlist query is fixed as follows.

```text
reviewed:true AND organism_id:9606 AND cc_disease:* AND
(xref:mim-* OR xref:orphanet-*) AND
(go_exp:* OR go_ida:* OR go_ipi:* OR go_igi:* OR
 go_imp:* OR go_iep:* OR go_ic:* OR go_tas:*)
```

The adopted benchmark version is fixed as follows.

| Version | Train proteins | Train unique labels | Validation proteins | Test proteins |
|---|---:|---:|---:|---:|
| `213 -> 221 -> 225 -> 228` | 1,245 | 2,773 | 590 | 875 |

### 3.2 Strict Split Rules

Fixed rules:

1. `train -> validation -> test` must always follow temporal order
2. The same protein must not appear in multiple splits
3. Each protein is assigned exactly once to the split where its first new label appears
4. New labels are defined at the `(DB_ID, GO_ID, Aspect)` granularity
5. Differences in evidence code alone do not count as separate labels

Split definitions:

- train: `213 -> 221`
- validation: `221 -> 225`
- test: `225 -> 228`

Mandatory conditions for split validation:

- `split_validation.time_order_valid == true`
- `split_validation.protein_disjoint_valid == true`

Any run where even one protein overlap is found is deemed invalid and will not be adopted.

### 3.3 Independence Determination

Independence is determined by **GOA archive release diffs**.  
The following are not adopted:

- Independence determination by `annotation_date`
- Independence determination by UniProt REST `date_modified`
- Designs that make `ClinVar` cross-reference a mandatory condition of the main filter

### 3.4 Source-of-Truth Artifacts

The source-of-truth artifacts for the current version are fixed as follows.

- temporal split artifact: `wandb-healthcare/bioreason-pro-custom/disease-temporal-split:production`
- reasoning dataset artifact: `wandb-healthcare/bioreason-pro-custom/disease-temporal-reasoning:production`
- comparison model artifact: `wandb-healthcare/bioreason-pro-custom/bioreason-pro-rl:production`

Local build / download results are scratch and are not the source of truth.

## 4. Data Specification

### 4.1 Evidence Codes Used

The GO evidence codes used are fixed as follows.

- `EXP`
- `IDA`
- `IPI`
- `IGI`
- `IMP`
- `IEP`
- `IC`
- `TAS`

In GAF files, only rows where `DB == UniProtKB` and `DB_Type == protein` are targeted.

### 4.2 Reasoning Dataset

The primary dataset is the reasoning dataset only.  
The config name is fixed to `disease_temporal_hc_reasoning_v1`.

Required columns:

| Column | Required | Purpose |
|---|---|---|
| `protein_id` | Yes | Primary key |
| `sequence` | Yes | Model input |
| `organism` | Yes | Prompt generation |
| `go_bp` | Yes | BP ground truth labels |
| `go_mf` | Yes | MF ground truth labels |
| `go_cc` | Yes | CC ground truth labels |
| `reasoning` | Yes | SFT reasoning |
| `final_answer` | Yes | SFT answer |
| `protein_function` | Yes | UniProt summary context |
| `go_pred` | Yes | Pre-computed GO-GPT predictions |
| `interpro_formatted` | Optional | InterPro context |
| `ppi_formatted` | Optional | PPI context |

Split names are fixed to `train` / `validation` / `test`.  
Even if optional columns do not exist, the columns themselves are retained and filled with empty strings.

### 4.3 SFT / RL Split Usage Rules

Fixed rules:

- SFT / RL / eval use **the same benchmark split**
- SFT uses the `train` split of the reasoning dataset for training
- SFT checkpoint selection uses a **100-sample stratified subset** deterministically drawn from `validation`
- RL rollout / reward optimization uses `train`
- RL checkpoint selection and offline sanity-check use a **100-sample stratified subset** deterministically drawn from `validation`
- `test` is reserved exclusively for final evaluation and is not used for training signals
- When creating derived datasets for RL, source data comes from the `train` split only
- Stratified subsets are created using `stratified_aspect_profile`, preserving `go_aspect` and label-profile

## 5. Data Preparation

### 5.1 Orchestration

The high-level entry point for data preparation is fixed to `scripts/run_temporal_split_artifact_pipeline.py`.  
This orchestration performs at least the following:

- temporal split artifact build
- sanity check
- reasoning dataset build
- W&B Artifact upload
- pipeline status recording

Low-level entry points:

- temporal split artifact build: `scripts/build_disease_temporal_split_artifact.py`
- reasoning dataset build: `scripts/build_disease_benchmark_datasets.py`

### 5.2 Required Artifacts

The required artifacts for the temporal split artifact are as follows.

- `summary.json`
- `report.md`
- `train_assigned_labels.tsv`
- `dev_assigned_labels.tsv`
- `test_assigned_labels.tsv`
- `*_assigned_propagated.tsv`
- `pipeline_status.json`

Additional files for supplementary analysis may be generated, but none beyond the above are considered required artifacts.

### 5.3 Storage Policy

Fixed rules:

- Local output is treated as scratch and is not assumed to be permanently stored
- After successful upload to W&B Artifact, local scratch may be cleaned up
- Downstream eval / SFT / RL resolve from W&B Artifact ref, not from local directories
- Unnecessary datasets and supplementary files are not generated

## 6. Data and Model Upload to W&B

### 6.1 Common Principles

W&B is the standard for experiment tracking and artifact lineage.  
At minimum, the following must be recorded in run config for each phase:

- `job_type`
- `benchmark_version`
- `temporal_split_artifact`
- `dataset_config`
- `reasoning_dataset_config`
- `dataset_artifact`
- `shortlist_query`
- `shortlist_mode`
- `train_start_release`
- `train_end_release`
- `dev_end_release`
- `test_end_release`
- `base_checkpoint`
- `model_artifact`
- `seed`
- `learning_rate`
- `batch_size`
- `gradient_accumulation_steps`
- `num_train_epochs`
- `job_time_limit`

### 6.2 Dataset Upload

Fixed rules:

- Datasets are registered as W&B Artifacts
- Dataset artifact version / alias is recorded in run config
- Downstream eval / SFT / RL resolve datasets from W&B Artifact ref

### 6.3 Model Upload

Fixed rules:

- Model checkpoints are registered as W&B Artifacts
- Model artifact version / alias is recorded in run config
- The comparison model is fixed to `bioreason-pro-rl-paper`
- `bioreason-pro-rl-paper` is materialized once from the public Hugging Face source `wanglab/bioreason-pro-rl` and pinned as a W&B model artifact
- `train_sft` and `train_rl` outputs are registered as output artifacts of their respective runs

### 6.4 Artifact Ref Manifest

For execution on CoreWeave, the repo's artifact ref manifest is the entry point, rather than passing arbitrary local directories each time.

Fixed files:

- data-bundle manifest: `configs/disease_benchmark/data_registry.json`
- evaluation-target manifest: `configs/disease_benchmark/eval_target_registry.json`
- asset publish manifest: `configs/disease_benchmark/artifact_publish_registry.json`
- artifact ref env template: `configs/disease_benchmark/wandb_registry_paths.env.example`
- source env template: `configs/disease_benchmark/wandb_asset_sources.env.example`

Fixed rules:

- Refs passed to manifests use `entity/project/artifact_name:alias` format
- The only model ref that must be explicitly prepared initially is `BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH`
- Refs for `train-sft-output` and `train-rl-output` are determined as artifacts after their respective training runs complete

## 7. Evaluation

### 7.1 Eval Phase

The logical phase name for evaluation is fixed to `eval`.  
The high-level entry point is `scripts/run_registered_eval.py`; the low-level entry points are `eval.py` and `scripts/sh_eval.sh`.  
W&B runs are started with `wandb.init(..., job_type="eval")`.  
Weave is started with `weave.init()`.

Split usage:

- Development comparisons, ablations, and checkpoint comparisons use `validation`
- `validation` runs use a deterministic **100-sample stratified subset**
- Final reported values come from a separate `test` run
- `test` runs use the full split

### 7.2 Quantitative Evaluation

The primary metric is **F_max**. Required namespaces are:

- MF
- BP
- CC

Evaluation targets:

- `bioreason-pro-rl-paper`
- `train-sft-output`
- `train-rl-output` if available

Target families:

- `comparison-family`: `bioreason-pro-rl-paper`
- `tuned-family`: `train-sft-output`, `train-rl-output`
- `spec-comparison`: all of the above

Each metric is saved via `wandb.log()`, with at minimum `fmax_mf`, `fmax_bp`, `fmax_cc`.

### 7.3 Saved Artifacts and Tracking

- Inference functions are traced with Weave
- test (final reported values)
    - Executed as a **separate run** after SFT / RL
    - Uses the full `test` split
    - Evaluation is performed using Weave's Evaluation Logger
    - Runs where Weave logging fails are not treated as successful
    - Weave is used, but simultaneously the evaluation summary is saved as a **W&B Table with 1 evaluated target = 1 row**
        - Columns: model_name, split, benchmark_version, <accuracy metrics follow in subsequent columns>
    - Sample-level results are saved as a **W&B Table with 1 sample = 1 row**
    - For reasoning tasks, `reasoning_full`, `final_answer`, `intermediate_trace` are retained
    - JSON results, summary exports, and sample exports are not version-managed as W&B Artifacts. The basic philosophy is to not hold results locally
    - Local eval output is treated as scratch and may be cleaned up by default after successful W&B save

- Development comparisons, ablations, and checkpoint comparisons use `validation`
    - Use a deterministic **100-sample stratified subset**
    - Only metrics are needed in this case
    - Runs where `fmax_mf`, `fmax_bp`, `fmax_cc` cannot be saved are not treated as successful

## 8. Training for SFT

### 8.1 train_sft Phase

The logical phase name for SFT is fixed to `train_sft`.  
Entry points are `train_protein_llm.py` and `scripts/sh_train_protein_qwen_staged.sh`.  
W&B runs are started with `wandb.init(..., job_type="train_sft")`.

### 8.2 Input and Strict Rules

Input:

- `disease_temporal_hc_reasoning_v1`
- `bioreason-pro-rl-paper` checkpoint artifact

Fixed rules:

- Training uses the `train` split of the reasoning dataset
- Checkpoint selection uses a **100-sample stratified subset** deterministically drawn from `validation`
- The `test` split is not used for SFT training
- The pre-tuning comparison model is materialized from W&B Artifact ref
- Canonical SFT execution is **stage 2 only**
- In canonical mode, projector / GO module weights from the comparison model are used as-is for warm-start
- Stage 1 projector warm-up is treated only as a fallback for training instability or as an ablation
- Validation subset strategy is fixed to `stratified_aspect_profile`
- Checkpoint selection validation subset must show `selected=100` in logs
- Train / validation metrics are saved via `wandb.log()`
- Sample table is saved as a W&B Table
- Output checkpoint is registered as a W&B Artifact

### 8.3 Execution Conditions

Fixed rules:

- Training requires GPU
- Dataset config is supplied assuming `dataset_type=cafa5`
- Maximum wall time for training jobs is 12 hours
- Time limit at submission is `12:00:00`
- Operations assume checkpoint / resume

## 9. Training for RL

### 9.1 train_rl Phase

The logical phase name for RL is fixed to `train_rl`.  
Entry points are fixed to `train_protein_grpo.py` and `scripts/sh_train_protein_grpo.sh`.  
W&B runs are started with `wandb.init(..., job_type="train_rl")`.

### 9.2 Input and Strict Rules

Input:

- Canonically, the `train-sft-output` checkpoint
- Reward configuration
- The same benchmark definition

Fixed rules:

- Rollout / reward optimization uses the `train` split of the benchmark
- Checkpoint selection and offline sanity-check use a **100-sample stratified subset** deterministically drawn from `validation`
- The `test` split is not used for RL training
- When deriving RL datasets, source data comes from the `train` split only
- Canonical input is the `train-sft-output` artifact
- If the `train-sft-output` artifact contains only raw Lightning checkpoints, convert to HF model before starting RL
- Starting RL directly from `bioreason-pro-rl-paper` is treated as an ablation, used only when necessary
- Validation subset strategy is fixed to `stratified_aspect_profile`
- Reward metrics, KL metrics, and training stability indicators are saved via `wandb.log()`
- Rollout traces are saved with Weave
- RL output checkpoint is registered as a W&B Artifact

### 9.3 Execution Conditions

Fixed rules:

- RL uses the same benchmark version
- Maximum wall time for training jobs is 12 hours
- Time limit at submission is `12:00:00`
- Operations assume checkpoint / resume
- Local output directory is treated as scratch; the source of truth is W&B Artifact ref
