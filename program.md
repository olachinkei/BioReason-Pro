# BioReason-Pro Senpai Target

BioReason-Pro predicts Gene Ontology terms for proteins with a multimodal reasoning LLM. This target is a focused Senpai research loop for improving the RL recipe without changing the benchmark split or evaluation metric.

## Mission

Find training changes that improve validation `overall_mean_fmax` over the frozen paper RL checkpoint. The validation metric is for steering and ranking Senpai PRs. Test/holdout evaluation is not part of the autonomous gate and should only be run when a human researcher asks for final confirmation.

Primary metric: `overall_mean_fmax`, higher is better.

## Codebase

- **Primary editable entrypoint:** `train.py`. All training behavior must remain here.
- **Editable for focused experiments:** `bioreason2/` model, dataset, reward, prompt, and tracking helpers.
- **Evaluation code:** `eval.py`, `evals/`, `scripts/sh_eval.sh`, and `scripts/run_registered_eval.py`. Edit only for bug fixes or metric/logging reliability.
- **Registry helpers:** `scripts/materialize_data_bundle.py`, `scripts/materialize_model_source.py`, `scripts/resolve_logged_artifact_ref.py`, and `bioreason2/utils/research_registry.py`.
- **Protected benchmark contract:** `configs/disease_benchmark/data_registry.json`, `configs/disease_benchmark/eval_target_registry.json`, and `bioreason2/dataset/go-basic.obo`. Do not change splits, artifact refs, hidden labels, or metric definitions in experiment PRs.
- **Dependency files:** `pyproject.toml` and `uv.lock`. Add packages only in the same PR that uses them.

## Data And Baseline

Data and model assets are materialized from W&B Artifact refs. Local files are scratch. Runtime outputs must stay under `/mnt/data/$USER/BioReason-Pro` on CoreWeave.

Required source refs are documented in `BASELINE.md` and may also be supplied through `configs/disease_benchmark/wandb_registry_paths.env`.

## Running

Default Senpai run:

```bash
cd "$PROBLEM_DIR" && python train.py \
  --wandb_name "$STUDENT_NAME/<hypothesis-slug>" \
  --wandb_group "<hypothesis-or-pr>"
```

Debug smoke:

```bash
cd "$PROBLEM_DIR" && python train.py \
  --gate_steps 1 \
  --continue_steps 0 \
  --max_val_samples 2 \
  --wandb_name "smoke/senpai"
```

The target is fixed to 1 node x 8 GPU. The training flow is baseline validation, 5-step gate training, validation, and continuation to 20 total steps only if the gate beats baseline.

## W&B Metrics

Required comparison metric:

- `overall_mean_fmax`: primary validation metric, higher is better.

Useful health metrics:

- `train/reward_mean`
- `train/reward_nonzero_rate`
- `train/format_valid_rate`
- `train/final_answer_tag_rate`
- `train/valid_rollouts`
- `train/filtered_rollouts`
- `timing/*`
- `system/*`

Preserve existing metric names unless a PR explicitly changes logging.

## Benchmark Integrity

Do not modify the validation/test split, artifact manifests, GO ontology, IA weighting file, metric calculation, or hidden labels to make an experiment look better. Do not run test/holdout as part of the 5-step gate. Do not cherry-pick a continuation unless the 5-step validation gate strictly improves on baseline.

Missing or NaN `overall_mean_fmax` is a failed gate.

## Results Contract

Every terminal student result must include this single-line marker:

```markdown
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<run-id>"],"primary_metric":{"name":"overall_mean_fmax","value":<number>},"test_metric":{"name":"overall_mean_fmax","value":<number>}}
```

The PR comment must also include the exact command, baseline metric, step-5 validation metric, continuation metric if run, peak memory or OOM status, W&B run IDs, and a short explanation of what happened.

## Roles

Research is coordinated through GitHub PRs with advisor/student roles. See `instructions/prompt-advisor.md` and `instructions/prompt-student.md`.
