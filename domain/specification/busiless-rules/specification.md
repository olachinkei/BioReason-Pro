# Specification

This specification defines **only the operational rules ultimately adopted** for GO function prediction of disease-associated human proteins using BioReason-Pro.  
Investigation processes, comparison versions, candidate filters, and lessons from count-based decisions are separated into [learning-log](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/domain/learning-log) and are not retained in this document.

Assumptions:

- The currently adopted benchmark is the **small-data version**
- The current version uses GOA anchors `213 -> 221 -> 225 -> 228`, but the operational split is `train=213->225`, `dev=200 from 225->228`, `holdout=400 from 225->228`
- `train`, `dev`, `holdout`, and `reserve` maintain **protein-disjointness**
- `dev`, `holdout`, and `reserve` are future-only partitions drawn from `225 -> 228`
- Generated artifacts on local filesystem are treated as scratch; **the source of truth is W&B Artifact ref**
- `dev` in this document is synonymous with the split name `validation` in code
- The fixed W&B project name is `bioreasoning-pro` (`WANDB_PROJECT=bioreasoning-pro`)

## 1. Problem Setting

The problem addressed by this specification is **protein-level GO term prediction and disease-aware reasoning for disease-associated human proteins**.

Fixed questions:

- Is there value in performing additional RL tuning on BioReason-Pro's public model for disease-associated human proteins?
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
| `bioreason-pro-rl-paper` | Pre-tuning comparison model and canonical RL starting checkpoint |
| `train-rl-output` | Output artifact produced by custom `train_rl` run |

The only model artifact ref needed initially is `bioreason-pro-rl-paper`.  
`train-rl-output` appears as an artifact after RL training is executed.

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

| Version | Train definition | Dev proteins | Holdout proteins | Reserve proteins |
|---|---|---:|---:|---:|
| `213 -> 221 -> 225 -> 228` | merge(`213 -> 221`, `221 -> 225`) | 200 | 400 | remainder of `225 -> 228` |

### 3.2 Strict Split Rules

Fixed rules:

1. `train` is the merged past window `213 -> 225`
2. `dev` and `holdout` are drawn only from the future pool `225 -> 228`
3. The same protein must not appear in multiple splits
4. Each protein is assigned exactly once to the split where its first new label appears
5. New labels are defined at the `(DB_ID, GO_ID, Aspect)` granularity
6. Differences in evidence code alone do not count as separate labels

Split definitions:

- train: merge(`213 -> 221`, `221 -> 225`)
- validation (`dev`): deterministic protein-disjoint stratified **200-protein** partition of `225 -> 228`
- test (`holdout`): deterministic protein-disjoint stratified **400-protein** partition of the remaining `225 -> 228`
- reserve: unused remainder of `225 -> 228`; not used for training signals or final reported values

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

- temporal split artifact: `wandb-healthcare/bioreasoning-pro/disease-temporal-split:production`
- reasoning dataset artifact: `wandb-healthcare/bioreasoning-pro/disease-temporal-reasoning:production`
- comparison model artifact: `wandb-healthcare/bioreasoning-pro/bioreason-pro-rl:production`

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

### 4.3 RL / Eval Split Usage Rules

Fixed rules:

- RL / eval use **the same benchmark split**
- The `validation` split itself is fixed to **200 proteins**
- RL rollout / reward optimization uses `train`
- RL checkpoint selection and offline sanity-check use the full `validation` split
- `test` is reserved exclusively for final evaluation and is not used for training signals
- When creating derived datasets for RL, source data comes from the `train` split only
- `validation` and `test` are created using deterministic `stratified_aspect_profile`, preserving `go_aspect` and label-profile

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
- Downstream eval / RL resolve from W&B Artifact ref, not from local directories
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
- Downstream eval / RL resolve datasets from W&B Artifact ref

### 6.3 Model Upload

Fixed rules:

- Model checkpoints are registered as W&B Artifacts
- Model artifact version / alias is recorded in run config
- The comparison model is fixed to `bioreason-pro-rl-paper`
- `bioreason-pro-rl-paper` is materialized once from the public Hugging Face source `wanglab/bioreason-pro-rl` and pinned as a W&B model artifact
- `train_rl` outputs are registered as output artifacts of their respective runs

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
- Refs for `train-rl-output` are determined as artifacts after RL training runs complete

## 7. Evaluation

### 7.1 Eval Phase

The logical phase name for evaluation is fixed to `eval`.  
The high-level entry point is `scripts/run_registered_eval.py`; the low-level entry points are `eval.py` and `scripts/sh_eval.sh`.  
`wandb.init(..., job_type="eval")` and `weave.init(...)` are executed **before any sample inference begins**.

Split usage:

- Development comparisons, ablations, and checkpoint comparisons use `validation`
- `validation` runs use the full **200-protein dev split**
- Final reported values come from a separate `test` run
- `test` runs use the full **400-protein holdout split**

### 7.2 Quantitative Evaluation

The primary metric is **F_max**. Required namespaces are:

- MF
- BP
- CC

Evaluation targets:

- `bioreason-pro-rl-paper`
- `train-rl-output` if available

Target families:

- `comparison-family`: `bioreason-pro-rl-paper`
- `tuned-family`: `train-rl-output`
- `spec-comparison`: all of the above

Each metric is saved via `wandb.log()`, with at minimum `fmax_mf`, `fmax_bp`, `fmax_cc`.

### 7.3 Saved Artifacts and Tracking

Common fixed rules for both `validation` and `test`:

- Each sample inference is traced with `@weave.op`
- Evaluation is recorded with Weave's **Evaluation Logger**, following W&B's imperative evaluation logging pattern (`log_prediction`, `log_score`, `finish`, `log_summary`)
- `wandb.init(..., job_type="eval")` and `weave.init(...)` are executed before sample processing
- A one-row **`eval_summary` W&B Table** is saved for each eval run
- A one-sample-per-row **`eval_samples` W&B Table** is saved for each eval run
- For reasoning tasks, `reasoning_full`, `final_answer`, and `intermediate_trace` are retained in sample-level records
- `ev.log_summary(...)` must include at minimum:
  - `fmax_mf`
  - `fmax_bp`
  - `fmax_cc`
  - `overall_mean_fmax`
- The one-row `eval_summary` W&B Table must include at minimum:
  - `model_name`
  - `split`
  - `benchmark_version`
  - `fmax_mf`
  - `fmax_bp`
  - `fmax_cc`
  - `overall_mean_fmax`
- Runs missing W&B metric logging, `eval_summary`, `eval_samples`, or Weave Evaluation logging are not treated as successful
- JSON results, summary exports, and sample exports are not version-managed as W&B Artifacts
- Local eval output is treated as scratch and may be cleaned up after successful W&B save

Split-specific rules:

- `validation`
  - Uses the full `validation` split of **200 proteins**
  - Used for development comparisons, ablations, and checkpoint comparisons

- `test`
  - Uses the full `test` split of **400 proteins**
  - Executed as a **separate run** after RL
  - Provides the final reported values

## 8. Training for RL

### 8.1 train_rl Phase

The logical phase name for RL is fixed to `train_rl`.  
Entry points are fixed to `train_protein_grpo.py` and `scripts/sh_train_protein_grpo.sh`.  
W&B runs are started with `wandb.init(..., job_type="train_rl")`.

### 8.2 Input and Strict Rules

Input:

- Canonically, the `bioreason-pro-rl-paper` checkpoint
- Reward configuration
- The same benchmark definition

Fixed rules:

- Rollout / reward optimization uses the `train` split of the benchmark
- Checkpoint selection and offline sanity-check use the full `validation` split of **200 proteins**
- The `test` split is not used for RL training
- When deriving RL datasets, source data comes from the `train` split only
- Canonical input is the `bioreason-pro-rl-paper` artifact
- Validation split construction is fixed to deterministic `stratified_aspect_profile`
- Reward metrics, KL metrics, and training stability indicators are saved via `wandb.log()`
- Rollout traces are saved with Weave
- RL output checkpoint is registered as a W&B Artifact

### 8.3 Execution Conditions

Fixed rules:

- RL uses the same benchmark version
- Maximum wall time for training jobs is 12 hours
- Time limit at submission is `12:00:00`
- Operations assume checkpoint / resume
- Local output directory is treated as scratch; the source of truth is W&B Artifact ref

## Note. BioReason-Pro RL Reference

The reference algorithm for BioReason-Pro RL is **DR-GRPO** rather than plain GRPO.  
Operationally, this means the RL implementation should follow the following paper-level principles:

- Sequence-level importance sampling correction, not token-level correction
- Advantage defined as `(reward - group_mean_reward) / (global_batch_std + eps_std)`
- Reward extracted from the generated **reasoning trace via GO-term regex matching**, i.e. regex over the model's generated trace rather than a final-answer-only block, matching the Appendix description of the paper
- GO predictions propagated through `is_a` and `part_of` before scoring
- Reward aligned to **IA-weighted F1 / Fmax_w** with an explicit IA file; paper-faithful RL should fail closed rather than silently falling back
- Asymmetric Clip-Higher behavior with `epsilon_low=7e-4` and `epsilon_high=9e-4`
- Small KL anchor (`beta=1e-4`) and Dr. GRPO style length-bias mitigation via fixed-length normalization

Paper reference values retained as the canonical comparison point:

- group size `G=24`
- steps per generation `=2`
- inner optimization iterations `=1`
- LoRA `rank=16`, `alpha=32`, `dropout=0.05`
- AdamW `lr=3e-5`, `beta1=0.9`, `beta2=0.999`, `eps=1e-8`, `weight_decay=0`
- scheduler `cosine`, `warmup_ratio=0.03`
- sampling `temperature=1.0`, `top_k=20`, `top_p=0.95`, `min_p=0`, `repetition_penalty=1.0`
- prompt length reference `512`
- completion length reference `10,000`
- RL steps reference `1,200`
- seed `42`

Operational adaptation for the current CoreWeave continuation-tuning setup:

- The paper values above are the reference, and the canonical continuation path should keep `max_new_tokens=10000` so rollout failures are diagnosed from full generations rather than from an artificially short cap
- When running the current single-node 8 GPU continuation setup, prefer preserving the paper's `B/G=8` unique proteins per step by keeping `num_generations=24` and setting `per_device_train_batch_size=1` so that `world_size=8` yields `global_unique_proteins_per_step=8`, then reduce `max_steps` and/or `max_new_tokens` before reducing group diversity
- Any deviation from the paper reference, especially `num_generations`, `max_new_tokens`, `train_batch_size`, `max_steps`, and IA availability, must be explicit in W&B config
- `reward_weights` and all RL-control parameters (`loss_type`, clip epsilons, reward scaling mode, IS cap, scheduler settings, KL beta, and final-answer-only reward mode) must remain visible in W&B config for each RL run
- The paper states that GO term identifiers are extracted from the **final answer block** of each generated trace, so the canonical RL reward path should default to `reward_prediction_source=final_answer`; `reasoning_trace` should be treated as an explicit ablation only
- The paper-faithful `paper_compact` prompt should be **answer-first** at the end of generation: keep reasoning brief, emit a single structured `GO_SUMMARY` block, and stop immediately after `GO_SUMMARY_END`; do not ask for a UniProt-style prose summary in the RL continuation prompt
- In the current codebase, `final_answer` means the post-`</think>` answer region and may contain a structured `GO_SUMMARY` block; `structured_go_summary` is the stricter ablation, not the default paper-faithful path
- The paper's prompt-length reference `512` applies to the **text context only**; protein residue embeddings and GO graph embeddings are separate multimodal inputs and should not be counted against that text budget
- RL text prompts should therefore stay close to the paper's compact slot structure rather than carrying full free-form metadata dumps
- The preferred RL text slots are: `organism`, `interpro_annotations`, `ppi_partners`, `go_mf_speculations`, `go_bp_speculations`, `go_cc_speculations`, and `focus_aspect`
- `go_*_speculations` should be rendered per aspect as `GO:XXXXXXX (term name)` strings, matching Table S19, rather than GO IDs alone
- `ppi_partners` should stay close to the paper's data pipeline convention of top-10 high-confidence STRING partners, not an open-ended dump
- In practical terms, the raw **text context** should remain compact and close to the paper slots above; note that the implementation expands `<|protein_pad|>` and `<|go_graph_pad|>` into many tokenizer-visible placeholders before replacement, so trainer-side `prompt_len` is not the same quantity as the paper's `max prompt length = 512`
- The paper-faithful RL prompt path should therefore exclude long prose fields such as UniProt summaries, function summaries, localization text, and other metadata dumps unless an explicit ablation is being run
- The paper-faithful RL prompt path should not explicitly request prose endings such as `Summarize in UniProt format.`; it should encourage the model to stop once the structured GO answer is complete
- Optional fields such as UniProt prose summaries, extended function descriptions, or other long metadata should be excluded from the paper-faithful RL prompt path unless they are being tested as an explicit ablation
- The first comparison between `SFT -> RL` and `paper RL -> continuation RL` should keep the objective, reward extraction, sampling controls, and memory controls identical so that the initialization checkpoint is the only changed variable
- If `paper RL -> continuation RL` shows collapsed within-group reward variance or low rollout diversity, follow-up tuning may shorten `max_steps` and/or strengthen the KL anchor, but that should be recorded as a separate ablation rather than the first A/B comparison
- For the current implementation, paper-faithful single-node 8 GPU runs should log both per-device and global batch semantics in W&B config: `per_device_train_batch_size`, `world_size`, `global_unique_proteins_per_step`, and `global_num_trajectories_per_step`
- Because the current code samples `G` rollouts per protein prompt, the paper's published `per-device batch size=6` is not treated as a strict implementation target; the canonical invariant is instead `G=24`, `B/G=8`, and therefore `global_num_trajectories_per_step=192`
- Rollout acceleration should first come from caching frozen multimodal inputs and frozen-reference log-probs, not from reducing `num_generations`; lowering group diversity is a secondary fallback only after cache-based acceleration has been tried
- If distributed rollout generation must temporarily fall back to sequential mode for CUDA stability, that should be treated as a runtime deviation from the paper's efficient training setup and must be visible in W&B config or run notes
- For single-node multi-GPU jobs, keep the runtime allocator and NCCL settings explicit in the launcher (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512`, `NCCL_ASYNC_ERROR_HANDLING=1`, `NCCL_CUMEM_ENABLE=0`, and related single-node defaults) so paper-faithful hyperparameters are not confounded by avoidable runtime-memory drift
- Smoke runs should also prefer the full `max_new_tokens=10000` cap when the goal is rollout diagnosis; if a shorter cap is used for systems triage, record it as an explicit smoke-only deviation
- RL observability should log not only reward values but also rollout stop behavior: `stop_reason` (`summary_end`, `eos`, `max_tokens`, `unknown`), `max_new_tokens_hit_rate`, `has_go_summary_end`, and `first_go_summary_token_idx` so long generations can be distinguished from late-but-valid structured answers
