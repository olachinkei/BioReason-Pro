# Zero-Based RL Redesign For 2-3 Nodes x 8 H100 80GB

Date: 2026-04-10

## Goal

Re-think the RL path from zero base using:

- [specification.md](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/domain/specification/busiless-rules/specification.md)
- [specification_rl.md](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/domain/specification/busiless-rules/specification_rl.md)
- the current trainer snapshot:
  [2026-04-10-train_protein_grpo_snapshot.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/domain/learning-log/2026-04-10-train_protein_grpo_snapshot.py)

Assume access to **2-3 nodes**, each with **8x H100 80GB**.

## Source-of-Truth Constraints

From the current specifications, the canonical paper-faithful RL target is:

- DR-GRPO
- `G = 24`
- `B = 192`
- `B/G = 8` unique proteins per step
- `Lmax = 10000`
- `steps_per_generation = 2`
- `num_iterations = 1`
- `KL beta = 1e-4`
- `epsilon_low = 7e-4`
- `epsilon_high = 9e-4`
- rollout sampling at `temperature=1.0`, `top_k=20`, `top_p=0.95`
- intended runtime stack: `DeepSpeed + vLLM (colocate)`

The paper semantics are:

1. Sample 8 proteins from the train pool.
2. Generate 24 responses for each protein.
3. Score all 192 trajectories.
4. Compute batch-global reward std.
5. Take 2 optimizer steps from the same rollout batch.
6. Refresh `old_policy`.

## What The Current Trainer Snapshot Suggests

The current trainer is useful as a systems-debugging baseline, but it is not yet a faithful implementation of the target execution model.

Observed issues from prior debugging:

- rollout generation tends to fall back to sequential mode under distributed execution
- long responses can be generated successfully, but scoring / loss is still the main OOM site
- prompt/reward schema has drifted between `paper_native` and `repo_structured` modes
- batching is not yet aligned with the paper's intended `8 proteins x 24 responses` backend behavior

## Zero-Based Target Architecture

### 1. Separate Generation From Scoring

Use two clearly different runtime planes:

- **Generation plane**
  - backed by `vLLM`
  - optimized for high-throughput rollout generation
  - owns old-policy sampling
- **Training plane**
  - backed by `DeepSpeed`
  - optimized for forward/backward scoring and LoRA updates
  - owns current-policy and ref-policy scoring plus optimizer steps

This separation matches the paper much better than using one monolithic DDP trainer for everything.

### 2. Canonical Step Shape

Each canonical RL step should be organized as:

- global batch of 8 unique proteins
- 24 sampled responses per protein
- 192 trajectories total
- scoring and reward computed on all 192 trajectories

Do not define the outer loop as "one example at a time, then 24 rollouts." The paper semantics are batch-first, not protein-first sequential execution.

### 3. Node Allocation Assumption

Under 2-3 nodes x 8 H100 80GB, the most plausible paper-faithful layout is:

- **2-node minimum**
  - Node A: rollout generation service(s), vLLM colocate
  - Node B: DeepSpeed training/scoring
- **3-node preferred**
  - Node A: rollout generation
  - Node B: current-policy / optimizer
  - Node C: heavy scoring support or evaluation / spare capacity for overlap

If exact colocate behavior is difficult, the key is still to stop treating rollout generation as a per-example sequential operation inside the trainer.

## First-Principles Risks

### Risk 1. Scoring, Not Generation, Is The Immediate OOM Site

Past failures show that generation can proceed, even with long outputs, but OOM occurs later during current/ref/old log-prob scoring and loss computation.

Implication:

- the redesign must optimize scoring memory first
- generation-only speedups are not enough

### Risk 2. 10000-Token Observation And 10000-Token Backprop Are Different Problems

The current direction of "let the model fully reveal its behavior" is good for diagnosis. But a faithful implementation still needs a principled way to score long trajectories without blowing up memory.

Implication:

- keep `Lmax=10000` for rollout observation
- redesign the scoring backend so it can process long completions safely

### Risk 3. Prompt / Reward Contract Drift

If the paper checkpoint is continued under a different prompt or answer schema than its native contract, long or unstable outputs may reflect schema mismatch rather than true model weakness.

Implication:

- `paper_native` must remain the baseline contract
- `repo_structured` should be treated as an explicit ablation

### Risk 4. DDP Alone Is Not The Paper Runtime

Even if DDP runs stably, it does not automatically recover the throughput properties of `DeepSpeed + vLLM (colocate)`.

Implication:

- do not interpret "DDP is stable" as "paper-faithful rollout backend achieved"

## Proposed Rebuild Order

### Phase A. Make The Spec Boundary Explicit

- Keep `paper_native` as the only canonical continuation mode for the paper checkpoint.
- Keep `repo_structured` as a separate experimental mode.
- Make the trainer log which contract is active on every run.

### Phase B. Rebuild The Rollout Backend Around Batch-First Semantics

- Input to one rollout call should be 8 proteins, not 1 protein.
- Generation backend should materialize 24 responses per protein without the trainer looping over proteins one by one.
- Preserve the mapping from the 192 trajectories back to 8 protein groups for reward centering.

### Phase C. Rebuild Scoring For Long Trajectories

Needed redesign goals:

- score trajectories in chunks
- avoid duplicating full attention-state cost across current / old / ref where possible
- precompute and reuse multimodal prefix work aggressively
- keep reward observation length independent from backprop memory strategy

### Phase D. Only Then Restore Full RL Updates

Before optimizer-on training:

- run no-update audit on the rebuilt backend
- verify rollout timing
- verify reward non-zero rate
- verify stop / end-pattern diagnostics

## Expected Success Criteria

The redesign is successful only if:

- one RL step really means 8 proteins and 192 trajectories
- rollout generation is not dominated by per-example sequential loops
- long trajectories can be observed without immediate scoring OOM
- `paper_native` continuation can complete at least one full optimizer step under the intended contract

## Practical Conclusion

The next real milestone should not be "patch the current DDP trainer a little more."

It should be:

**rebuild the RL runtime so that batch-first rollout generation and long-trajectory scoring are first-class, using the paper contract as the baseline and the current trainer snapshot only as a debugging reference.**
