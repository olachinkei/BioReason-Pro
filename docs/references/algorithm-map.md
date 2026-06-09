# Algorithm And Source Map

This map links external algorithm references to local implementation surfaces.
For scientific details, start from the full paper. For runtime behavior in this
branch, read the local files.

## External To Local

| Concept | External Reference | Local Surface |
| --- | --- | --- |
| BioReason-Pro method and evaluation | `docs/references/bioreason-pro-paper.md`, `docs/references/paper-algorithm-details.md` | `train.py`, `eval.py`, `evals/` |
| Upstream inference path | `docs/references/upstream-inference.md` | Not packaged here; local evaluation uses `eval.py` and `scripts/sh_eval.sh`. |
| GO-GPT predictions as prompt context | Full paper and upstream inference repo | Dataset columns such as `go_pred`; prompt formatting in `bioreason2/dataset/cafa5/load.py`. |
| Biological context slots | Full paper | `interpro_formatted`, `ppi_formatted`, and related prompt formatting in `bioreason2/dataset/cafa5/load.py`. |
| Reasoning trace construction | Full paper | `bioreason2/dataset/cafa5/processor.py` and `scripts/data_generation/build_reasoning_dataset.py`. |
| RL screening loop | Local Senpai contract | `train.py`, `program.md`, `docs/design-docs/senpai-screening-contract.md`. |
| CAFA-style validation metric | Full paper and local contract | `eval.py`, `evals/`, `train.py`, `BASELINE.md`. |
| Disease temporal split artifact | Local reproducibility code | `scripts/data_generation/build_temporal_split_artifact.py`. |
| IA weights | CAFA-style metric framing | `scripts/data_generation/build_ia_weights.py`, `evals/`, `train.py`. |
| CoreWeave runtime boundary | Local operational decision | `docs/runbooks/coreweave-implementation.md`, `docs/adr/0003-coreweave-runtime-boundary.md`. |

## Paper Audit Summary

Use `paper-algorithm-details.md` for the section-level algorithm notes. The
current finding is that the backend RL path and Senpai wrapper match the paper's
DR-GRPO rollout shape. The Senpai wrapper uses an approximately 8k-token
completion budget instead of the paper's 10k budget, and runs the 24 rollouts
through a smaller active vLLM window that waits for freed slots.

## Local To External

| Local File | Use External Reference When |
| --- | --- |
| `train.py` | You need the paper's rationale for SFT/RL choices or evaluation framing. |
| `bioreason2/dataset/cafa5/load.py` | You need the paper's description of prompt context and biological reasoning inputs. |
| `bioreason2/dataset/cafa5/processor.py` | You need the paper's framing for reasoning traces and final answers. |
| `eval.py` / `evals/` | You need the paper's benchmark framing, not the local mechanics. |
| `scripts/data_generation/` | You need local reproducibility, artifact layout, or public release data build steps. |

## Non-Goals

- This map does not restate the full algorithm.
- This map does not turn the branch into the upstream inference package.
- This map does not add frontend, product, or DB documentation surfaces.
