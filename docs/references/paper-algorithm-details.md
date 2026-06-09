# Paper Algorithm Details And Local Audit

This note records the algorithmic details checked against the BioReason-Pro
paper and maps them to this Senpai branch. It is a reference/audit note, not an
architecture document.

## Sources Checked

- Full paper: https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1.full
- DOI: https://doi.org/10.64898/2026.03.19.712954
- Accessible full-text mirror used during review:
  https://www.researchgate.net/publication/402932507_BioReason-Pro_Advancing_Protein_Function_Prediction_with_Multimodal_Biological_Reasoning
- Upstream inference repository:
  https://github.com/bowang-lab/BioReason-Pro

Checked on 2026-06-09.

## Paper Algorithm

### Data And Inputs

The paper builds a UniProt-derived protein dataset with experimental GO
annotations, organism metadata, InterPro domain annotations, STRING protein
interaction context, and PDB structure references. Protein sequences are capped
at 2,000 residues for BioReason-Pro. The paper's temporal holdout trains on
pre-November 2022 data and evaluates on later proteins that gained experimental
annotations in the holdout window.

The model prompt is assembled from:

- organism,
- InterPro identifiers, names, and residue spans,
- GO-GPT predictions,
- optional protein-protein interaction partners.

At inference time, GO identifiers are extracted from the generated final-answer
block and propagated through the GO hierarchy before CAFA-style scoring.

### GO-GPT

GO-GPT is a separate autoregressive transformer for GO term prediction. It uses
frozen ESM2 protein embeddings, an organism embedding, and previously generated
GO terms to model ontology order, hierarchy, and cross-aspect dependencies. The
paper uses GO-GPT outputs as a key input to BioReason-Pro.

This branch does not package GO-GPT training or inference. It consumes
precomputed GO-GPT predictions through the dataset column configured by
`--go_gpt_predictions_column`, defaulting to `go_pred`.

### BioReason-Pro SFT

The paper's SFT stage starts from Qwen3-4B-Thinking, uses frozen ESM3-1B protein
embeddings from layer 37, trains protein projection, GO projection, and GO graph
encoder components, then adds LoRA on the language model. The SFT table reports
max protein length 2,000 residues and max text length 10,000 tokens. The
synthetic reasoning traces are generated from a GPT-5 prompt that asks for
`<|REASONING|>` and `<|FINAL_ANSWER|>` sections.

This Senpai branch is not an SFT reproduction branch. It starts RL from a
materialized SFT/RL-compatible checkpoint and keeps data-generation code only so
the public data artifact can be rebuilt and inspected.

### BioReason-Pro RL

The paper's RL configuration is DR-GRPO starting from the SFT epoch-8 checkpoint
with vLLM-based rollout generation in colocate mode. Key reported settings:

- group size: 24 rollouts per query,
- unique proteins per step: 8,
- total trajectories per step: 192,
- steps per generation: 2,
- inner optimization iterations: 1,
- sequence-level importance sampling correction with cap 2,
- clipping epsilon low/high: `7e-4` / `9e-4`,
- KL beta: `1e-4`,
- LoRA rank/alpha/dropout: `16` / `32` / `0.05`,
- max prompt length: 512 tokens,
- max completion length: 10,000 tokens,
- effective batch size: 192 over 8 H100 GPUs.

### Evaluation

The paper reports CAFA-style per-aspect Fmax and IA-weighted Fmax. The official
CAFA toolkit propagates predictions and ground truth through the GO hierarchy
and applies IA weighting where available. Fmax is a global-threshold metric, so
it is not naturally a per-protein decomposable quantity.

## Local Implementation Mapping

| Paper Concept | Local Surface | Audit Status |
| --- | --- | --- |
| ESM3 protein length cap at 2,000 | `train.py --max_length_protein`; dataset collate | Matches paper. |
| ESM3 layer 37 | `train.py --protein_embedding_layer` | Matches paper. |
| GO graph encoder hidden/layers/heads/reduced embeddings | `train.py` defaults and `bioreason2/models/go_graph_encoder.py` | Matches paper defaults. |
| GO-GPT context | `go_pred` column, `bioreason2/dataset/cafa5/load.py` | Consumes precomputed predictions; GO-GPT itself is external. |
| InterPro and PPI prompt slots | `bioreason2/dataset/cafa5/load.py`; `scripts/data_generation/build_reasoning_dataset.py` | Present. |
| Structured reasoning/final-answer tags | Prompt formatting and `train.py` GO extraction | Present, with tolerant fallback for malformed close tags. |
| DR-GRPO RL loss | `train.py` algorithm spec and policy loss | Matches reported RL knobs in backend mode. |
| CAFA Fmax evaluation | `eval.py`, `evals/cafa_evals.py` | Matches the paper's validation framing. |
| Senpai gate | `train.py` top-level Senpai orchestration | Local extension, not in paper. |

## Algorithm Audit

No fatal algorithmic contradiction was found in the branch's core RL path. The
important caveat is that this branch is a Senpai screening target, not a
full paper-reproduction target. The Senpai wrapper now keeps the paper rollout
shape, while using a slightly shorter completion budget for memory headroom.

### Looks Correct

- The backend RL defaults preserve the paper's DR-GRPO constants: 8 queries,
  24 rollouts, 192 trajectories, clipping values, KL beta, LoRA rank/alpha, and
  completion length.
- The Senpai wrapper launches the backend with the paper rollout shape:
  8 queries, 24 rollouts per query, 6 optimizer microbatch, and 4 accumulation
  steps.
- The 24 rollouts are generated through a smaller active vLLM window by default:
  `SENPAI_VLLM_MAX_NUM_SEQS=8`. The trainer submits rollout chunks
  sequentially, waiting for a chunk to finish before using the freed slots for
  the next chunk.
- The local prompt path includes the paper's main biological context slots:
  organism, InterPro, PPI, and GO-GPT predictions.
- GO IDs are extracted from the final-answer region and propagated through the
  GO DAG before reward/evaluation calculations.
- Validation uses CAFA-style Fmax aliases, with `overall_mean_fmax` as the
  Senpai primary metric.
- `--max_length_text 512` is appropriate for this RL trainer because the RL
  table reports max prompt length 512. The SFT table's 10,000-token text length
  is a different stage.

### Intentional Senpai Deviations

- Senpai mode defaults `SENPAI_MAX_NEW_TOKENS=8192`,
  `SENPAI_VLLM_MAX_MODEL_LEN=12288`, and active vLLM slots of 8. The completion
  budget is below the paper's 10,000-token setting but near enough to preserve
  long reasoning while reducing OOM risk.
- The branch is fixed to 1 node x 8 GPU. The paper reports 8 H100 GPUs across 2
  nodes; the algorithmic world size is still 8 ranks, but the hardware topology
  differs.
- The local benchmark artifact is the Senpai disease temporal benchmark. It
  should not be compared numerically to the paper's public temporal holdout
  unless the split/artifact is rebuilt to match the paper exactly.

### Algorithmic Risks To Track

- The training reward is a per-completion surrogate: IA-weighted F1 after GO
  propagation, optionally with per-aspect variants. The paper discusses Fmax as
  the GO evaluation objective, but Fmax is global-thresholded and not directly
  decomposable per rollout. The local surrogate is reasonable for RL, but it is
  not identical to paper-level Fmax.
- The 5-step gate is useful for cheap screening but should not be treated as a
  final training result.
- This branch consumes GO-GPT predictions but does not regenerate them. If the
  `go_pred` source changes, the experiment changes even if `train.py` does not.
- The paper mentions broader evidence sources such as structure and subcellular
  localization in the full evaluation context. The local training prompt path is
  centered on organism, InterPro, PPI, and GO-GPT slots; any experiment depending
  on explicit localization or structural fields needs a dataset/prompt audit.

## Practical Recommendation

For Senpai PRs, keep the current single-node defaults and judge candidates only
against the frozen local baseline using `overall_mean_fmax`. The defaults keep
the paper rollout shape and use an approximately 8k-token completion budget. To
push completion length all the way to the paper's 10k setting, override:

```bash
SENPAI_MAX_NEW_TOKENS=10000 \
SENPAI_VLLM_MAX_MODEL_LEN=12288 \
python train.py --wandb_name "<name>" --wandb_group "<group>"
```

The 10k completion setting is expected to require more memory headroom than the
default Senpai gate and may need further CoreWeave tuning.
