# 2026-04-09: Checkpoint-Native Decoding Contract Diagnosis

## Background

After writing down the semantic mismatch and the DDP / memory-path diagnosis, we investigated the problem from a third angle:

- what does the **paper RL checkpoint itself** appear to expect at decode time?

This angle focuses on the packaged checkpoint artifacts and the current run configs, rather than on the paper text or the DDP memory path alone.

The goal was to answer:

- are we asking the checkpoint to emit a format that is actually native to the checkpoint?
- are we decoding it with the same sampling defaults that the checkpoint carries?

## Main Finding

The strongest checkpoint-side mismatch is:

- **our current continuation path depends heavily on `GO_SUMMARY` markers**
- but the paper RL checkpoint does **not** appear to treat those markers as native tokenizer special tokens

At the same time:

- the packaged checkpoint carries a generation config with `temperature = 0.6`
- while our run configs explicitly request `temperature = 1.0`

This creates a second mismatch:

- the run config says one thing
- the checkpoint carries another
- and the logs show the two are still interacting

## Evidence

### 1. The paper RL checkpoint does not package `GO_SUMMARY` markers as tokenizer special tokens

From the local checkpoint artifacts:

- [bioreason_pro_rl_paper/tokenizer_config.json](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/data/artifacts/models/bioreason_pro_rl_paper/tokenizer_config.json)
- [bioreason_pro_rl_paper/special_tokens_map.json](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/data/artifacts/models/bioreason_pro_rl_paper/special_tokens_map.json)

we confirmed:

- `<|protein_pad|>` exists
- `<|go_graph_pad|>` exists
- `<|im_start|>` / `<|im_end|>` exist
- `<think>` / `</think>` exist in the tokenizer config
- **`<|GO_SUMMARY_START|>` does not exist**
- **`<|GO_SUMMARY_END|>` does not exist**

This matters because the current continuation logic relies on those markers in multiple places:

- prompt format
- reward parsing
- stop-reason interpretation
- observability metrics

But from the checkpoint artifact alone, those markers do not look like native boundaries that the model was explicitly tokenizer-aware of.

## 2. The runtime code only adds pad tokens, not `GO_SUMMARY` markers

[special_tokens.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/bioreason2/models/special_tokens.py) is explicit:

- only `<|protein_pad|>`
- and `<|go_graph_pad|>`

are added to tokenizer vocabulary.

[protein_llm.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/bioreason2/models/protein_llm.py) then does:

- `self.text_tokenizer.add_special_tokens({"additional_special_tokens": all_special_tokens})`
- `self.text_model.resize_token_embeddings(len(self.text_tokenizer))`

But `all_special_tokens` only contains the two multimodal pad tokens.

So our current runtime does **not** make `GO_SUMMARY_START/END` into real added special tokens for the paper RL checkpoint.

That means the repo currently treats `GO_SUMMARY` as a critical structured contract, while the checkpoint sees it as ordinary text fragments rather than a native added-token boundary.

## 3. The checkpoint packages a different decoding default than our runs request

The paper RL checkpoint's packaged [generation_config.json](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/data/artifacts/models/bioreason_pro_rl_paper/generation_config.json) contains:

- `do_sample = true`
- `temperature = 0.6`
- `top_k = 20`
- `top_p = 0.95`

The same is true for the local `bioreason_pro_rl` and `bioreason_pro_sft` artifacts.

However, our paper-RL smoke runs recorded in W&B used:

- `temperature = 1`
- `top_p = 0.95`
- `top_k = 20`
- `max_new_tokens = 10000`
- `reward_prediction_source = final_answer`
- `reasoning_prompt_style = paper_compact`

For example:

- `1ioxjzs1`
- `wbxyiy83`
- `pc1blp5e`

all show `temperature = 1` in W&B config.

So even before we consider prompt schema mismatch, there is already a **checkpoint-native decoding contract mismatch**:

- checkpoint package default: `0.6`
- run config: `1.0`

This lines up with the repeated notice in the logs:

- ``generation_config` default values have been modified to match model-specific defaults: {'temperature': 0.6}``

That warning should be taken seriously.
It means sampling behavior is not yet fully cleanly controlled from one place.

## 4. The current paper-RL smoke runs are still not paper-faithful at the checkpoint contract level

Even after the recent prompt / systems work, the current paper-RL smoke path still combines:

- `reward_prediction_source = final_answer`
- `reasoning_prompt_style = paper_compact`
- custom `GO_SUMMARY` enforcement

This means we are not just deviating from the paper abstractly.
We are also deviating from what the packaged checkpoint itself appears to have been set up to do by default.

The checkpoint artifact seems natively aligned with:

- chat-template style prompting
- multimodal pad tokens
- generic Qwen chat boundaries
- standard sampling defaults around `temperature = 0.6`

It does **not** natively advertise:

- `GO_SUMMARY` markers as tokenizer-added boundaries

## Why This Matters For the Rollout Symptoms

This angle helps explain the mixed rollout behavior:

- some completions stop naturally in a few hundred tokens
- some continue much longer
- some never satisfy the structured-summary logic that our repo wants

That pattern is consistent with a checkpoint that:

- knows how to produce free-form chat-style reasoning and answer text
- but is being asked to satisfy a custom marker-based schema that is not native to the checkpoint vocabulary contract

In other words:

- the model may not be "ignoring" our schema out of failure
- it may simply not have been packaged to treat that schema as a first-class decoding interface

## Revised Interpretation

From this checkpoint-native angle, the likely issue is not only:

- prompt mismatch
- reward mismatch
- systems mismatch

but also:

- **tokenizer / generation-config mismatch with the checkpoint's native decoding contract**

This is a more specific and practical statement than simply saying "the prompt is off."

## Practical Implication

Before interpreting paper-RL continuation quality too strongly, we should separate two cases:

### Case A: checkpoint-native continuation

- use the checkpoint in a way that matches its packaged decoding contract as closely as possible
- do not lean on non-native marker boundaries as if they were intrinsic to the checkpoint

### Case B: repo-native structured continuation

- custom `GO_SUMMARY` format
- custom stop logic
- custom parsing expectations

This may still be useful, but it should be labeled as:

- a repo adaptation
- not the checkpoint's native continuation setting

## Bottom Line

This third angle suggests:

- the paper RL checkpoint is being continued under a decoding contract that is not fully native to the checkpoint itself

The most concrete signs are:

- `GO_SUMMARY` markers are not packaged as tokenizer special tokens
- runtime only adds multimodal pad tokens, not summary markers
- packaged generation config prefers `temperature = 0.6`
- our runs request `temperature = 1.0`
- logs show those two decoding defaults are still colliding

So this is not just a paper-vs-spec issue.
It is also a **checkpoint-vs-runtime contract issue**.

## Files Most Relevant To This Diagnosis

- [data/artifacts/models/bioreason_pro_rl_paper/generation_config.json](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/data/artifacts/models/bioreason_pro_rl_paper/generation_config.json)
- [data/artifacts/models/bioreason_pro_rl_paper/tokenizer_config.json](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/data/artifacts/models/bioreason_pro_rl_paper/tokenizer_config.json)
- [data/artifacts/models/bioreason_pro_rl_paper/special_tokens_map.json](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/data/artifacts/models/bioreason_pro_rl_paper/special_tokens_map.json)
- [bioreason2/models/special_tokens.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/bioreason2/models/special_tokens.py)
- [bioreason2/models/protein_llm.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/bioreason2/models/protein_llm.py)
- [train_protein_grpo.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/train_protein_grpo.py)
- [runtime_logs/run_rl_ddp_single_node_smoke.sh](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/runtime_logs/run_rl_ddp_single_node_smoke.sh)
