# 2026-04-09 Paper-Native Continuation Implementation

## Summary
- Introduced a first-class split between `paper_native` continuation and `repo_structured` continuation.
- Moved the default RL continuation contract back toward the paper/native checkpoint behavior instead of the custom `GO_SUMMARY` schema.
- Kept the structured `paper_compact` path as an explicit ablation/debugging mode rather than the canonical default.

## What Changed

### 1. Continuation modes are now explicit
- `continuation_mode=paper_native`
  - default prompt style resolves to `paper_native`
  - default reward extraction resolves to `reasoning_trace`
  - default sampling resolves to `checkpoint_native`
  - custom `GO_SUMMARY_END` stopping is disabled
- `continuation_mode=repo_structured`
  - default prompt style resolves to `paper_compact`
  - reward extraction can remain `final_answer` / `structured_go_summary`
  - custom tagged-summary stopping remains available

### 2. Paper-native prompt was restored
- Added a freer-form `paper_native` reasoning prompt that keeps compact evidence slots:
  - organism
  - InterPro annotations
  - PPI partners
  - GO term hypotheses per aspect
- This path no longer requires the model to emit a custom `<|GO_SUMMARY_START|> ... <|GO_SUMMARY_END|>` block.
- The `paper_compact` prompt remains for structured ablations.

### 3. Sampling contract is now separated from explicit overrides
- Added `sampling_contract`.
- `paper_native` defaults to `checkpoint_native`, meaning the local checkpoint's packaged `generation_config` should be the source of truth unless an explicit ablation overrides it.
- This is intended to reduce mismatch between the paper RL checkpoint and the current continuation launcher.

### 4. Wrapper and smoke launcher now follow the mode split
- Wrapper defaults now prefer:
  - `CONTINUATION_MODE=paper_native`
  - `REASONING_PROMPT_STYLE=auto`
  - `SAMPLING_CONTRACT=auto`
  - `REWARD_PREDICTION_SOURCE=auto`
- In `paper_native`, `INCLUDE_PROTEIN_FUNCTION_SUMMARY` defaults back to `True`.
- The smoke launcher was updated to inherit the same semantics.

### 5. Tracking/spec/tests were updated
- W&B tracking now records:
  - `continuation_mode`
  - `sampling_contract`
- Specification now states that:
  - canonical continuation is `paper_native`
  - `paper_compact` is a structured ablation
  - `reasoning_trace` is the canonical RL reward source for `paper_native`

## Why This Matters
- Previous failures were hard to interpret because the paper RL checkpoint was being continued under a custom schema that likely was not checkpoint-native.
- The most important cleanup here is conceptual: paper-native continuation and repo-structured continuation are no longer silently mixed.

## Remaining Risk
- This does not solve the DDP/loss-path OOM by itself.
- It only makes the next experiments interpretable by aligning prompt/reward/sampling contracts with the checkpoint more faithfully.

## Next Recommended Step
- Run a `paper_native` **no-update audit** on the paper RL checkpoint:
  - optimizer steps disabled or skipped
  - observe rollouts only
  - inspect reasoning trace vs final answer boundaries
  - verify whether checkpoint-native sampling and prompt contract produce more coherent outputs before resuming RL continuation
