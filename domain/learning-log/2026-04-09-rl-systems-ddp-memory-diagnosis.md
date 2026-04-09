# 2026-04-09: RL Systems Diagnosis from the DDP / Memory Angle

## Background

A different way to look at the same failures is to temporarily ignore the prompt semantics and ask a narrower question:

- if rollout generation works, where exactly does the training stack break?

This note records that systems-side diagnosis from recent single-node 8 GPU smoke runs, especially the paper-RL continuation path.

## Main Finding

The dominant systems failure is:

- **not rollout generation itself**
- but the **loss-computation path after rollout generation**

In other words, the semantic problem and the systems problem are both real, but they happen at different stages:

- rollout behavior is a prompt / reward / schema problem
- training crashes are a DDP / memory / loss-forward problem

## Evidence From Recent Runs

### 1. Paper-RL smoke runs can generate many rollouts before crashing

In the paper-RL smoke path, we observed:

- rollout lengths in the low hundreds of tokens
- some much longer completions
- in later runs, full-length behavior up to the `10000` cap

So the pipeline is able to:

- launch DDP
- materialize the dataset
- load the paper RL checkpoint
- generate rollouts across ranks

The crash happens later.

### 2. The earlier distributed timeout was not the final root cause

We previously saw a small `ALLREDUCE Numel=3` timeout.
That was a real bug, but after moving control-plane collectives away from the main NCCL path, the failure mode changed.

This is useful because it exposed the next bottleneck:

- **OOM in the loss path**

### 3. The true heavy path is loss forward on the DDP-wrapped model

The strongest evidence came from the completed-but-failed paper-RL smoke run `1ioxjzs1`:

- rollout generation had already succeeded far enough to produce many completions
- the first big OOM happened during batched loss computation
- the stack included DDP internals such as `_DDPSink.apply`
- the sequential fallback then also OOMed in model attention

This suggests two things:

- the DDP-wrapped model is expensive even before the model's own attention peak
- our "sequential fallback" is not a truly lightweight fallback if it still uses the same wrapped forward path

## The Systems Interpretation

### 1. We are not using the same runtime stack as the paper

The paper's RL stack is closer to:

- `DeepSpeed`
- `vLLM` colocate
- paper-scale runtime assumptions

Our current continuation stack is:

- single-node DDP
- local forward passes for current / old / ref policy scoring
- extra control logic and rollout filtering layered on top

That means even a semantically correct prompt / reward setup can still fail for purely systems reasons.

### 2. Long-rollout observability and long-rollout training are different problems

We intentionally moved to:

- `max_new_tokens = 10000`

because the shorter caps made it impossible to see whether the model could eventually produce the full answer.

That was the right observability move.

But it also creates a second problem:

- a rollout can be worth observing
- while still being too expensive to include in the loss computation

The introduction of `max_loss_completion_tokens` was an attempt to separate these two concerns:

- observe long rollouts
- train only on shorter ones

That helps, but it is also a sign that the training stack is not yet robust enough for the full paper-style rollout regime.

### 3. Filtering long rollouts is useful, but it is already a runtime deviation

Once we drop long rollouts from the loss path after generation, we are no longer doing a clean paper-faithful update.

The run may still be valuable for debugging, but interpretation changes:

- reward and rollout diversity are observed on one set of trajectories
- optimization is applied to a smaller filtered subset

This is not necessarily wrong for engineering purposes, but it must be treated as a deliberate runtime adaptation.

## What This Angle Suggests About the Root Cause

From the systems perspective, the leading cause of failure is:

- **DDP loss forward on long multimodal rollouts is too memory-expensive in the current implementation**

This is distinct from the prompt / reward diagnosis.

Even if we fixed the schema mismatch perfectly, the current stack would still be at risk of crashing if it had to backprop through too many long trajectories at once.

## Why the Current Fallback Is Still Not Enough

The fallback logic improves reliability, but it does not fully solve the structural issue:

- the batched loss path can OOM
- the sequential fallback still uses the heavy model forward
- the model is still wrapped in DDP
- the same long multimodal prefixes still have to be scored

So we should not think of the present fallback as "safe mode".
It is more accurate to think of it as:

- **a less bad version of the same expensive computation**

## Revised Working Hypothesis

The current continuation failures should be split into two categories:

### Category A: semantic mismatch

- paper checkpoint continued under a custom schema
- reward / stop / prompt contracts are not aligned with paper C.2 plus paper RL reward behavior

### Category B: systems mismatch

- paper-scale rollout lengths are being processed by a training stack that is not paper-scale in the same way
- DDP loss forward remains the main memory hotspot

The key lesson is that these two categories are interacting, but they are not the same problem.

## Practical Implication

When a paper-RL continuation run fails, we should avoid over-interpreting it as evidence that:

- the checkpoint is bad
- the prompt is necessarily wrong
- the reward is necessarily wrong

Sometimes the run is failing because:

- the rollout was generated successfully
- but the training stack could not score or backprop through it

That is a systems bottleneck, not a semantic one.

## Files Most Relevant To This Diagnosis

- [train_protein_grpo.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/train_protein_grpo.py)
- [bioreason2/models/protein_llm.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/bioreason2/models/protein_llm.py)
- [scripts/sh_train_protein_grpo.sh](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/scripts/sh_train_protein_grpo.sh)
- [runtime_logs/run_rl_ddp_single_node_smoke.sh](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/runtime_logs/run_rl_ddp_single_node_smoke.sh)
- [domain/specification/busiless-rules/specification.md](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/domain/specification/busiless-rules/specification.md)
