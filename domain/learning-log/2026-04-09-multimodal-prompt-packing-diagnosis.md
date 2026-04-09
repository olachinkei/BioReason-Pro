# 2026-04-09: Multimodal Prompt Packing Diagnosis

## Background

Another angle is the processor itself.
Even if the prompt text is paper-aligned, the model only sees the packed sequence after chat templating, multimodal placeholder expansion, and truncation.

## Main Finding

The current processor does **not** keep the paper text budget separate from the multimodal budget.

Instead:

- the chat template injects `Protein: <|protein_pad|>` and `GO graph: <|go_graph_pad|>` into the user turn
- the processor expands those placeholders before tokenization
- the tokenizer then truncates the combined sequence

This means long protein or GO placeholder expansion can interfere with which textual instructions survive.

## Evidence

### 1. Chat template mixes multimodal pads into the text stream

[qwen3_4b_chat_template.jinja2](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/bioreason2/models/pl/qwen3_4b_chat_template.jinja2) inserts:

- `Protein: <|protein_pad|>`
- `GO graph: <|go_graph_pad|>`

directly inside the user message.

### 2. Processor expands those pads before tokenization

[processing_pl.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/bioreason2/models/pl/processing_pl.py) replaces:

- each `<|protein_pad|>` with a repeated placeholder span based on sequence length
- each `<|go_graph_pad|>` with `num_go_tokens` repeated placeholders

and only then calls the tokenizer.

### 3. Truncation is applied after packing

The same processor tokenizes with:

- `max_length = max_length_text + num_go_tokens + max_length_protein + 2`
- `truncation = True`

So truncation happens on the packed multimodal-plus-text sequence, not on a pure text-context slice.

## Why This Matters

Our current `paper_compact` prompt puts the most important schema instructions near the end of the user prompt:

- required final answer format
- `GO_SUMMARY` example
- begin the final answer with `GO_SUMMARY_START`

Because truncation is right-sided after multimodal packing, these are exactly the instructions that are most likely to be lost first on long examples.

## Interpretation

This suggests a fourth root cause candidate:

- some rollout instability may come from **processor-level prompt packing and truncation**
- not just from the checkpoint, reward, or DDP systems path

In short:

- the model may fail to follow the schema because it did not reliably receive the full schema tail after multimodal packing

## Files

- [bioreason2/models/pl/processing_pl.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/bioreason2/models/pl/processing_pl.py)
- [bioreason2/models/pl/qwen3_4b_chat_template.jinja2](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/bioreason2/models/pl/qwen3_4b_chat_template.jinja2)
- [bioreason2/dataset/prompts/cafa5.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/bioreason2/dataset/prompts/cafa5.py)
- [bioreason2/dataset/cafa5/load.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/bioreason2/dataset/cafa5/load.py)
