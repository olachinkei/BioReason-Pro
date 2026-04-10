## Summary

The current RL slowdown is not primarily caused by the nominal rollout hyperparameters alone. The larger issue is that the current implementation executes rollouts in a very different systems configuration from the BioReason-Pro paper.

The paper uses:

- DR-GRPO
- group size `G=24`
- effective batch size `B=192`
- unique proteins per step `B/G=8`
- `Lmax=10000`
- `DeepSpeed + vLLM (colocate)` for rollout generation

By contrast, the current code path performs rollout generation in a much more serial way under distributed execution.

## Evidence From The Paper

Key paper references:

- RL objective and sequence-level ratio are in Equation 4.18-4.24.
- The paper explicitly states that RL training runs with `DeepSpeed + vLLM (colocate)` across 8 H100 GPUs.
- Table S18 reports:
  - `G=24`
  - `steps_per_generation=2`
  - `B=192`
  - `B/G=8`
  - `temperature=1.0`
  - `top_k=20`
  - `top_p=0.95`
  - `Lmax=10000`

This implies that the intended rollout backend is not "one protein, one rollout at a time" in Python loops. It is a colocated high-throughput generation system.

## Evidence From The Upstream Repo

The upstream inference path in `predict.py` is already much closer to a batched generation design than the current RL path:

- `predict.py` builds a batch of prompts.
- It batches protein sequences.
- It calls `ProteinLLMModel.generate(...)` once per batch.

Important defaults in upstream inference:

- `batch_size=4`
- `max_new_tokens=5000`
- deterministic greedy by default for inference (`temperature=0.0`)

So even the public inference code expects multimodal generation to be batched at the prompt level. The current RL path falls back far below that.

## What The Current RL Implementation Actually Does

In the current trainer:

- each train batch is iterated example-by-example
- for each example, `generate_rollouts_for_example(...)` is called
- under distributed mode, rollout generation is forced into `sequential mode for stability`
- this means `model.generate(...)` is called repeatedly inside a Python loop for each rollout index

This is the key line of divergence:

- paper intent: many rollouts are generated through a colocated generation backend
- current code: rollouts are generated sequentially per example when distributed mode is enabled

As a result, the runtime behavior observed in audit runs is:

- many rollouts take hundreds to low thousands of tokens
- some rollouts still run away to `8035` or `10000`
- one step can take on the order of `45-55 minutes`

This is not consistent with the spirit of the paper's rollout system, even if some scalar hyperparameters match.

## Why One Rollout Is So Slow

The dominant reasons are:

1. Rollouts are generated sequentially in DDP mode.

Instead of a colocated generation server producing many responses efficiently, the code loops over `rollout_idx` and repeatedly calls `model.generate(...)`.

2. The multimodal prefix is still heavy.

Even with multimodal caching, each rollout still carries:

- residue-level protein embeddings
- 200 GO graph embeddings
- a long textual context

3. Long completions are allowed all the way to `10000`.

This is correct for observability and paper-faithful upper bound behavior, but it makes a serial rollout backend extremely expensive.

4. The loss path is even heavier than the generation path.

Generation may finish, but later scoring paths still need:

- current policy log-probs
- reference policy log-probs
- sequence-level ratios
- attention over prompt + completion

That is where the observed OOMs are happening.

## What Will Likely Break If We Try To Make Rollouts More Paper-Like

If we try to move toward paper-like batched / colocated rollouts, the following risks are most likely:

### 1. Batched multimodal generation will trigger the same failures that already appeared in DDP

The previous DDP rollout path had to fall back to sequential generation for stability. If we re-enable large batched rollout generation without redesigning the multimodal input packing path, the likely failures are:

- CUDA device-side asserts
- bad shape / indexing interactions in `batch_idx_map`
- invalid batched multimodal cache expansion

In other words, the current code has not yet demonstrated that "24 rollouts in one generate call" is stable under the multimodal path.

### 2. Memory blow-up in scoring is still the main blocker

Even if rollout generation becomes fast, the training step still needs to score long completions.

The observed OOMs happened in:

- Qwen attention during loss forward
- after generation had already succeeded

So a faster rollout backend alone will not solve the training bottleneck. The current scoring path still scales badly with long completions.

### 3. `10000` token observations are incompatible with naive full-backprop scoring

Current observations show that some rollouts naturally terminate around `600-1400` tokens, but some continue to `8035` or `10000`.

If we try to batch those long sequences directly into:

- current log-prob computation
- ref log-prob computation
- KL calculation

we should expect OOM unless the scoring path is redesigned.

### 4. DDP introduces synchronization and tail-latency sensitivity

Even when generation works, the distributed job is sensitive to stragglers.

With mixed rollout lengths:

- one rank can finish a short rollout quickly
- another rank can still be generating or scoring a `10000` token completion

This creates pressure on collective synchronization and can easily re-surface hangs or timeouts unless the control flow is carefully separated.

### 5. The current implementation still duplicates expensive work

Although multimodal cache support now exists, the code still pays significant repeated costs across:

- rollout generation
- old-policy scoring
- ref-policy scoring
- current-policy scoring

The generation path and the scoring path are not yet architected like a colocated service that cheaply reuses prepared prefixes.

## Most Important Interpretation

At this point, the main mismatch is not:

- "paper says `Lmax=10000` but current code uses the wrong number"

The main mismatch is:

- "paper assumes a rollout backend that can make `G=24, B/G=8, Lmax=10000` operationally feasible"
- "current code still executes those semantics through a much more serial and memory-heavy implementation"

This explains why the current rollout wall time is far too high even when the nominal hyperparameters look paper-like.

## Recommended Next Direction

Before trying to make the rollout backend more paper-like, the following should be treated as prerequisites:

1. Separate generation scalability from scoring scalability.

Generation and loss scoring should be treated as two different systems problems.

2. Keep `max_new_tokens=10000` for observability, but do not assume all observed tokens must be scored naively.

3. Redesign the scoring path for long completions before re-enabling aggressive batched rollout generation.

4. Treat "batched 24-rollout generation" as a separate systems milestone, not as a free consequence of DDP.

## Bottom Line

The current rollout slowdown is best explained as a rollout backend mismatch:

- the paper uses a colocated high-throughput rollout stack
- the current implementation still executes rollouts too sequentially and scores them too expensively

So if we try to "match the paper" by just preserving `G=24` and `Lmax=10000`, the likely result is more time spent and more OOMs unless the backend design is changed.
