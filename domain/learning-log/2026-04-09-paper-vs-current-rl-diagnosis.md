# 2026-04-09: Paper vs. Current RL Diagnosis

## Background

We re-read [bioreasoning-pro.pdf](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/domain/specification/bioreasoning-pro.pdf) after multiple RL continuation attempts showed a repeated pattern:

- some rollouts ended naturally in a few hundred tokens
- some rollouts ran into the very long range, including `8780` and `10000`
- paper-RL continuation looked unstable even when the checkpoint itself appeared healthy

The working question for this note was not "why is RL hard in general?" but rather:

- are we continuing the paper checkpoint under the same input/output contract that the paper actually used?

## Main Finding

The strongest explanation is now:

- **the current continuation path is mixing together multiple incompatible contracts**

More specifically:

- the paper's RL reward contract
- the paper's inference / evaluation extraction contract
- our repo's custom `GO_SUMMARY` contract

are not the same thing, but our current implementation and specification partially treat them as if they were interchangeable.

## What the Paper Actually Says

### 1. RL reward extraction and inference extraction are not the same

The paper makes a distinction that is easy to miss:

- In the RL formulation, GO term identifiers are extracted from the **reasoning trace** and then scored with propagated IA-weighted F1 / `Fmax_w`
- At inference / evaluation time, GO term identifiers are extracted from the **final answer block**

This means the paper itself intentionally uses:

- **train reward surface**: reasoning trace
- **eval extraction surface**: final answer block

This is not a contradiction in the paper. It is a train-vs-eval design choice.

### 2. The paper checkpoint was not trained on our custom `GO_SUMMARY` schema

Appendix `C.2` shows a freer-form inference prompt:

- the model reasons step by step
- it considers InterPro, GO hypotheses, and PPI
- it then provides a summary in the final answer
- it explicitly says "Summarize in UniProt format"

That is different from our current repo adaptation, which asks for:

- brief reasoning
- a single structured `GO_SUMMARY` block
- early stop
- no UniProt-style prose

So when the paper RL checkpoint produces prose-heavy or long-form outputs under our continuation runs, that is not necessarily evidence that the checkpoint is bad.
It may simply mean we are asking it to behave under a schema it was not originally trained to satisfy.

## Where the Current Repo Drifted

### 1. Reward default drift

At different points in the repo and specification, we have treated both of these as canonical:

- `reasoning_trace`
- `final_answer`

That is too loose.

The paper-faithful interpretation should be:

- RL reward default: `reasoning_trace`
- evaluation / prediction extraction default: `final_answer`

### 2. Prompt drift

The current `paper_compact` prompt is not just "compact paper C.2".
It is a genuine adaptation with different behavioral pressure:

- answer-first
- schema-constrained
- early stop
- anti-prose

That can still be useful as an ablation, but it should not be described as if it were the native paper continuation setting.

### 3. Stop-condition drift

Current continuation logic strongly prefers a structured summary boundary.
The paper checkpoint likely expects a more natural free-form ending behavior, followed by a final answer section.

This means our stop behavior and reward parser may both be steering the model away from the format it already knows.

## How the Observed Rollouts Fit This Diagnosis

Across recent paper-RL smoke runs, the rollouts were not uniformly broken.

Observed behavior included:

- shorter completions in the low hundreds of tokens
- mid-length completions in the several-hundred-token range
- occasional very long completions near the `10000` cap

This is important.

If the model were simply "runaway" in a uniform way, we would expect most or all completions to saturate.
Instead, we see a **mixed regime**:

- some outputs are able to finish naturally
- some outputs drift into long prose

That is exactly what we would expect from a checkpoint being pushed through a partially mismatched output contract.

## Revised Interpretation

The current evidence suggests:

1. The paper RL checkpoint is probably not the primary problem
2. The larger problem is that we are continuing it under a different prompt / reward / stop contract than the one it was trained with
3. Our custom structured-summary mode may still be useful, but it should be treated as a repo-level ablation, not the default paper-faithful mode

## Working Policy Going Forward

For **paper-faithful continuation**:

- reward extraction should default to `reasoning_trace`
- evaluation extraction should stay on `final_answer`
- prompt behavior should stay closer to Appendix `C.2`
- custom `GO_SUMMARY` control should be treated as an explicit adaptation or ablation

For **repo-adapted continuation**:

- custom `GO_SUMMARY` can still be used
- but we should stop assuming that failures under this schema automatically reflect the quality of the paper checkpoint itself

## Implication

The most likely semantic root cause is:

- **schema mismatch between the paper RL checkpoint and our current continuation contract**

This should be considered the primary hypothesis before making stronger claims such as:

- "paper RL continuation does not work"
- "the checkpoint has poor rollout behavior"
- "the model itself is the source of the instability"

## Files Most Relevant To This Diagnosis

- [train_protein_grpo.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/train_protein_grpo.py)
- [bioreason2/dataset/prompts/cafa5.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/bioreason2/dataset/prompts/cafa5.py)
- [bioreason2/dataset/cafa5/load.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/bioreason2/dataset/cafa5/load.py)
- [domain/specification/busiless-rules/specification.md](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/domain/specification/busiless-rules/specification.md)
