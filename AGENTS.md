# Agent Guide

This branch is a compact Senpai target for BioReason-Pro RL experiments.

## No-Context Start

If this is a fresh Codex thread with no prior context, do this first:

1. Read this file completely.
2. Read `docs/index.md` to find the right detailed document.
3. Read `program.md` and `BASELINE.md` before changing training behavior.
4. If the user says "run Senpai", "use CoreWeave GPUs", "teacher/student",
   "advisor/student", "GitHub issues/PRs", or "wandb/senpai", read
   `docs/adr/0005-upstream-senpai-control-plane.md` and
   `docs/runbooks/wandb-senpai-control-plane.md` before running anything.
5. If doing live CoreWeave work, read `docs/runbooks/coreweave-implementation.md`
   and use the global `coreweave-gpu-implementation` skill.

Current critical decision: this repository is the Senpai target problem repo,
not the Senpai control-plane runner. The real advisor/teacher + student loop is
upstream `wandb/senpai` and is launched here through:

```bash
scripts/launch_wandb_senpai_control_plane.sh <research-tag>
```

Do not replace that with `python k8s/launch.py` unless the user explicitly asks
for a direct training Job without the upstream advisor/student GitHub workflow.

## Read First

1. `docs/index.md` for the documentation map.
2. `program.md` for the Senpai target contract.
3. `BASELINE.md` for the frozen baseline and metric.
4. `docs/runbooks/coreweave-implementation.md` before CoreWeave work.
5. `docs/runbooks/wandb-senpai-control-plane.md` before launching the upstream teacher/advisor + student Senpai loop.
6. `docs/adr/0005-upstream-senpai-control-plane.md` for the next real-launch
   confirmation checklist.

## What This Repo Is

- Target repo: BioReason-Pro RL experiments for improving validation
  `overall_mean_fmax`.
- Upstream Senpai runner: `https://github.com/wandb/senpai.git`.
- Target GitHub repo used by the runner:
  `https://github.com/olachinkei/BioReason-Pro.git`.
- Primary training entrypoint: root `train.py`.
- Default hardware contract: one CoreWeave node with eight GPUs.
- Runtime outputs belong under `/mnt/data/$USER/BioReason-Pro`, not in git.

## Launch Decision Tree

- Real Senpai run with teacher/advisor + students + GitHub PR labels:
  `scripts/launch_wandb_senpai_control_plane.sh <tag>`.
- Preflight before a real Senpai launch:
  `scripts/launch_wandb_senpai_control_plane.sh <tag> --preflight_only`.
- Dry-run manifests:
  `scripts/launch_wandb_senpai_control_plane.sh <tag> --dry_run`.
- Manual CoreWeave SSH/Slurm work:
  follow `docs/runbooks/coreweave-implementation.md`.
- Optional direct CKS/SUNK training Job only:
  read `docs/runbooks/coreweave-sunk-senpai.md`, then use `k8s/launch.py`.

## Required Real-Launch Checks

Before launching upstream Senpai for real, confirm:

- `~/.secrets/bioreason-senpai.env` or the environment has
  `GITHUB_TOKEN`, `ANTHROPIC_API_KEY`, `EXA_API_KEY`, `WANDB_API_KEY`, and
  usually `HF_TOKEN`.
- `GITHUB_TOKEN` can clone, push branches, open PRs, set labels, and read org
  metadata for the BioReason-Pro target repo.
- W&B online auth works. Offline, dryrun, and disabled W&B modes are forbidden.
- Preflight passes.
- Dry-run manifests include advisor and student Deployments, GitHub/Anthropic/Exa
  launch secrets, `WANDB_API_KEY`, `HF_TOKEN`, `TARGET_REPO_URL`, the intended
  `ADVISOR_BRANCH`, and `WANDB_MODE=online`.

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

## Experiment Contract

- Primary metric: validation `overall_mean_fmax`, higher is better.
- Gate flow: baseline validation, 5 training steps, validation gate, then
  continue to 20 total steps only on strict improvement.
- Missing or NaN `overall_mean_fmax` is a failed gate.
- Do not run test/holdout unless a human explicitly asks for final confirmation.
- Student terminal results must include the single-line `SENPAI-RESULT` marker
  described in `program.md`.

## Existing Implementation State

- `scripts/launch_wandb_senpai_control_plane.sh` is the wrapper for upstream
  `wandb/senpai`.
- `docs/runbooks/wandb-senpai-control-plane.md` is the operational runbook.
- `docs/adr/0005-upstream-senpai-control-plane.md` records the control-plane
  decision and next-launch checklist.
- `k8s/launch.py` and `senpai.yaml` are direct-training helpers only.
- `train.py` rejects offline, dryrun, and disabled W&B modes.

## Documentation Rules

- Add conceptual decisions as ADRs under `docs/adr/`.
- Add operational cluster procedures under `docs/runbooks/`.
- Add data build and benchmark-design notes under `docs/design-docs/`.
- Add source/paper/reference mapping under `docs/references/`.
- Do not add frontend docs, product specs, or DB schema docs unless this repository grows those surfaces.
