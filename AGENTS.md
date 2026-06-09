# Agent Guide

This branch is a compact Senpai target for BioReason-Pro RL experiments.

## Read First

1. `docs/index.md` for the documentation map.
2. `program.md` for the Senpai target contract.
3. `BASELINE.md` for the frozen baseline and metric.
4. `docs/runbooks/coreweave-implementation.md` before CoreWeave work.

## Error Handling References

- For Senpai CUDA OOM, vLLM KV cache OOM, or DeepSpeed memory pressure, read `docs/runbooks/senpai-oom-handling.md`.

## Working Rules

- Keep training behavior in root `train.py`.
- Use `bioreason2/` for focused model, dataset, reward, prompt, and tracking changes.
- Treat `eval.py`, `evals/`, and `scripts/sh_eval.sh` as evaluation infrastructure. Edit only for bug fixes or metric/logging reliability.
- Do not change protected benchmark data, hidden labels, GO ontology, IA weights, metric definitions, or W&B registry refs to make a run look better.
- Keep generated artifacts, caches, W&B state, checkpoints, and local secrets out of the repo.
- Use the global `coreweave-gpu-implementation` skill for live CoreWeave work. Do not add that skill to this repository.

## Documentation Rules

- Add conceptual decisions as ADRs under `docs/adr/`.
- Add operational cluster procedures under `docs/runbooks/`.
- Add data build and benchmark-design notes under `docs/design-docs/`.
- Add source/paper/reference mapping under `docs/references/`.
- Do not add frontend docs, product specs, or DB schema docs unless this repository grows those surfaces.
