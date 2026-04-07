# 2026-04-07: Organizing the Comparison Model as `bioreason-pro-rl-paper`

## Background

- The public paper model and custom-trained tuned models were conflated across README / specification / PLAN / manifests
- The name `bioreason-pro-base` did not accurately represent the "public model before custom tuning" that we actually want to compare against
- `BLAST / Diamond` and `ESM standalone` remained in documents as comparison targets, but at this point there were no confirmed reusable public Artifact refs or readily available prediction sources in the repo

## Decision

1. The pre-tuning comparison model is fixed to `bioreason-pro-rl-paper`
2. `bioreason-pro-rl-paper` refers to the public Hugging Face source `wanglab/bioreason-pro-rl`
3. The actual W&B ref used is `wandb-healthcare/bioreason-pro-custom/bioreason-pro-rl:production` for now
4. `BLAST / Diamond` and `ESM standalone` are removed from the current scope of specification / PLAN / RESEARCH_README
5. The comparison model and the `train_sft` / `train_rl` outputs generated later are managed under separate names

## Updated Policy

- Comparison targets:
  - `bioreason-pro-rl-paper`
  - `train_sft` output
  - `train_rl` output
- Target groups:
  - `comparison-family`
  - `tuned-family`
  - `spec-comparison`
- SFT initial checkpoint:
  - `BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH` is used as the default input

## Files Updated

- `domain/specification/busiless-rules/specification.md`
- `domain/specification/PLAN.md`
- `RESEARCH_README.md`
- `configs/disease_benchmark/eval_target_registry.json`
- `configs/disease_benchmark/artifact_publish_registry.json`
- `configs/disease_benchmark/wandb_registry_paths.env`
- `configs/disease_benchmark/wandb_registry_paths.env.example`
- `configs/disease_benchmark/wandb_asset_sources.env`
- `configs/disease_benchmark/wandb_asset_sources.env.example`
- `scripts/sh_train_protein_qwen_staged.sh`

## Notes

- The W&B Artifact family itself remains `bioreason-pro-rl` at this point, but the logical name within the repo is treated as `bioreason-pro-rl-paper`
- This is an organizational measure to avoid conflating the "public paper model" with "custom RL outputs created later"
