# Agent Guide

This branch is a compact Senpai target for BioReason-Pro RL experiments.

## Read First

1. `docs/index.md` for the documentation map.
2. `program.md` for the Senpai target contract.
3. `BASELINE.md` for the frozen baseline and metric.
4. `docs/runbooks/coreweave-implementation.md` before CoreWeave work.
5. `docs/runbooks/wandb-senpai-control-plane.md` before launching the upstream teacher/advisor + student Senpai loop.

## Error Handling References

- For Senpai CUDA OOM, vLLM KV cache OOM, or DeepSpeed memory pressure, read `docs/runbooks/senpai-oom-handling.md`.

## Working Rules

- Keep training behavior in root `train.py`.
- Use `bioreason2/` for focused model, dataset, reward, prompt, and tracking changes.
- Treat `eval.py`, `evals/`, and `scripts/sh_eval.sh` as evaluation infrastructure. Edit only for bug fixes or metric/logging reliability.
- Do not change protected benchmark data, hidden labels, GO ontology, IA weights, metric definitions, or W&B registry refs to make a run look better.
- Keep generated artifacts, caches, W&B state, checkpoints, and local secrets out of the repo.
- Use the global `coreweave-gpu-implementation` skill for live CoreWeave work. Do not add that skill to this repository.
- When the request is to run Senpai itself, with teacher/advisor and student agents coordinated through GitHub issues or PRs, use the upstream `wandb/senpai` control plane via `scripts/launch_wandb_senpai_control_plane.sh`. Do not substitute this repository's direct `k8s/launch.py`; that helper is only for explicit direct training Jobs.
- W&B online tracking is mandatory for Senpai smoke, gate, and full runs. Do not set `WANDB_MODE=offline`, `WANDB_MODE=dryrun`, `WANDB_MODE=disabled`, or pass `--wandb_mode offline` to bypass auth or SDK issues. If W&B returns 401 or fails to initialize, stop the run, upgrade/verify `wandb` and `weave` in the target runtime, fix credentials, and relaunch only after online W&B preflight passes.

## Documentation Rules

- Add conceptual decisions as ADRs under `docs/adr/`.
- Add operational cluster procedures under `docs/runbooks/`.
- Add data build and benchmark-design notes under `docs/design-docs/`.
- Add source/paper/reference mapping under `docs/references/`.
- Do not add frontend docs, product specs, or DB schema docs unless this repository grows those surfaces.
