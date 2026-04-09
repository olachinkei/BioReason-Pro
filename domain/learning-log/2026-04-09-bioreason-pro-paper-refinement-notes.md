# 2026-04-09: BioReason-Pro Paper Refinement Notes

## Background

We revisited [bioreasoning-pro.pdf](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/domain/specification/bioreasoning-pro.pdf) as the source of truth for the current RL continuation work.

The main trigger was a mismatch between the intended BioReason-Pro setup and observed behavior in our runs:

- RL rollouts frequently hit the completion ceiling
- `prompt_len` grew far beyond the paper's `512` text-budget reference
- reward extraction and prompt structure had drifted away from what the paper described

This note records the main lessons learned from re-reading the paper and how they should shape our implementation and experiments.

## Key Learnings

### 1. The paper mixes three distinct prompt regimes

The paper describes at least three prompt/evaluation regimes that should not be conflated:

- **Reasoning data generation prompt** in Appendix `C.1`
  - Rich synthetic-trace generation prompt for GPT-5
  - Includes broader metadata such as UniProt function summary and subcellular localization
  - Produces long structured traces with `<|REASONING|>` and `<|FINAL_ANSWER|>`
- **BioReason-Pro inference prompt** in Appendix `C.2`
  - Runtime model prompt for BioReason-Pro itself
  - Includes protein pads, GO graph pads, organism, InterPro, PPI, and initial GO speculations
  - Focuses the model toward one or more GO aspects
- **RL objective / reward path**
  - Reward is described as GO-term extraction from the generated reasoning trace, followed by GO propagation and IA-weighted scoring

Practical lesson:

- We should not copy the `C.1` synthetic data prompt into the RL inference path
- RL prompt shaping should follow `C.2`, not the richer SFT-trace generation prompt

### 2. `max prompt length = 512` is the text budget, not the full multimodal budget

Table `S18` states `Max prompt length = 512 tokens`.

From `C.2`, the actual runtime input contains:

- protein pad tokens for residue embeddings
- GO graph pad tokens for graph embeddings
- text context slots

Practical lesson:

- The `512` reference applies to the tokenizer-visible text context
- Protein embeddings and GO graph embeddings are separate multimodal inputs and should not be counted against that text budget
- If `prompt_len` climbs into the thousands on the paper-faithful RL path, we should treat that as a bug or prompt-drift signal

### 3. The inference prompt in `C.2` is compact, but not as minimal as "GO IDs only"

Appendix `C.2` shows the inference slots explicitly:

- `organism`
- `interpro_annotations`
- `ppi_partners`
- `go_mf_speculations`
- `go_bp_speculations`
- `go_cc_speculations`
- `focus_aspect`

It also specifies that GO speculations are rendered as:

- `GO:XXXXXXX (term name)`

and that PPI partners are a compact list of names.

Practical lesson:

- A paper-faithful prompt should be compact, but not overly compressed into bare GO IDs with lost aspect structure
- GO speculations should stay grouped by aspect
- Focus aspect should use the aspect names used in the prompt template, not ad hoc shorthand

### 4. PPI context should stay bounded

The Methods section states that the data pipeline keeps the **top 10 high-confidence STRING partners**.

Practical lesson:

- The RL prompt path should not dump arbitrary long PPI context
- A top-10 style cap is a better paper-aligned default than larger open-ended lists

### 5. Reward extraction in the paper is closer to "reasoning trace" than "final answer only"

The paper text around RL and the appendix description indicate that GO terms are extracted from the generated reasoning trace with regex and then scored using propagated IA-weighted F1 / `Fmax_w`.

Practical lesson:

- A strictly final-answer-only reward is not paper-faithful
- The paper-faithful mode should treat the model's generated trace as the reward extraction surface
- This does not eliminate the value of structured summaries; it only changes the reward source

### 6. The output-format adaptation is still a repo-level choice

The paper's public examples and synthetic-trace data use richer final summaries than our current codebase output format.
Our repo currently relies on structured `GO_SUMMARY` blocks and uses them operationally in downstream code.

Practical lesson:

- It is acceptable to keep the repo's structured `GO_SUMMARY` output format as a practical adaptation
- But we should label it clearly as an implementation adaptation rather than claiming it is the exact paper output schema

### 7. The RL reference settings are clear, but our environment is smaller

Table `S18` provides a clean RL reference point:

- `loss_type = DR-GRPO`
- `G = 24`
- `steps_per_generation = 2`
- `num_iterations = 1`
- `beta = 1e-4`
- `epsilon_low = 7e-4`
- `epsilon_high = 9e-4`
- sequence-level IS correction with cap `2`
- `lr = 3e-5`
- `lora_rank = 16`
- `lora_alpha = 32`
- `lora_dropout = 0.05`
- `temperature = 1.0`
- `top_k = 20`
- `top_p = 0.95`
- `min_p = 0`
- `repetition_penalty = 1.0`
- `max_completion_length = 10000`
- `max_prompt_length = 512`
- `max_steps = 1200`

Practical lesson:

- These should remain the canonical comparison values
- In our single-job continuation setting, reducing `max_steps` or runtime-oriented limits is more acceptable than collapsing group diversity too early
- Any deviation from the paper reference must remain visible in W&B config

## Changes Adopted From This Re-read

The following decisions now have stronger paper support:

- Reward defaults should use `reasoning_trace` extraction in paper-faithful mode
- `reward_final_answer_only` should not be the default for paper-faithful RL
- RL prompt compaction should follow `C.2` rather than an improvised minimal schema
- The paper-faithful prompt path should exclude long prose fields such as UniProt summaries, function summaries, and localization text
- GO speculations should stay aspect-separated and preserve `GO:ID (term name)` formatting when available
- PPI prompt context should remain capped near the paper's top-10 convention

## Open Questions

### 1. "Reasoning trace" reward vs. structured summary reward

The paper-faithful interpretation points toward reasoning-trace extraction.
However, structured-summary-only reward may still be a useful ablation for robustness and controllability.

Current policy:

- paper-faithful default: `reasoning_trace`
- alternative ablation: `structured_go_summary`

### 2. Why are rollouts still hitting the length ceiling?

Even after removing some optional prose, some runs still showed long completions.

Possible remaining causes:

- prompt drift beyond the paper `C.2` shape
- stop-condition mismatch
- training dynamics causing runaway generation
- the model still attempting to produce richer SFT-style outputs under RL

This should be treated as an algorithm-and-prompt debugging issue, not just a sampling issue.

### 3. Paper SFT prompt vs. paper inference prompt

The paper intentionally uses a richer synthetic prompt for trace generation than for runtime inference.
This means it is normal for the SFT data format and the inference prompt to differ.

Practical implication:

- We should avoid "making RL look like the SFT trace-generation prompt" just because the SFT data is richer

## Working Policy Going Forward

For paper-faithful RL continuation experiments:

- Use the `C.2` inference prompt structure as the prompt source of truth
- Keep text prompt content within the paper's compact slot family
- Treat `prompt_len >> 512` as a regression signal
- Keep reward extraction in `reasoning_trace` mode unless an explicit ablation says otherwise
- Keep all paper-reference RL controls visible in W&B config
- Record every intentional deviation from Table `S18` as an explicit runtime adaptation, not as silent drift

## Files Most Affected By These Lessons

- [train_protein_grpo.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/train_protein_grpo.py)
- [scripts/sh_train_protein_grpo.sh](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/scripts/sh_train_protein_grpo.sh)
- [bioreason2/dataset/cafa5/load.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/bioreason2/dataset/cafa5/load.py)
- [bioreason2/dataset/prompts/cafa5.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/bioreason2/dataset/prompts/cafa5.py)
- [bioreason2/utils/tracking.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/bioreason2/utils/tracking.py)
- [specification.md](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/domain/specification/busiless-rules/specification.md)
