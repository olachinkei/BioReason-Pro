# Claude Code Guide

This file is the Claude Code entrypoint for the BioReason-Pro Senpai target.
For Codex-specific rules, `AGENTS.md` is the canonical guide. Follow both files
when working in this repository.

## Start Here With No Context

1. Read `AGENTS.md`.
2. Read `docs/index.md`.
3. Read `program.md` and `BASELINE.md` before changing training behavior.
4. For a real Senpai run, read
   `docs/adr/0005-upstream-senpai-control-plane.md` and
   `docs/runbooks/wandb-senpai-control-plane.md`.
5. For live CoreWeave work, read `docs/runbooks/coreweave-implementation.md`.

## Most Important Distinction

This repository is the target problem repo. It is not the upstream Senpai
runner.

Real Senpai means the upstream `wandb/senpai` advisor/teacher + student control
plane, with GitHub PRs and labels routing work. Launch that workflow through:

```bash
scripts/launch_wandb_senpai_control_plane.sh <research-tag>
```

Do not use `python k8s/launch.py` as a substitute for real Senpai. That file is
only for explicit direct training Jobs that skip the upstream advisor/student
GitHub workflow.

## Real Senpai Launch Checklist

Before launching for real:

1. Confirm secrets outside git, normally `~/.secrets/bioreason-senpai.env`.
2. Required keys: `GITHUB_TOKEN`, `ANTHROPIC_API_KEY`, `EXA_API_KEY`,
   `WANDB_API_KEY`, and usually `HF_TOKEN`.
3. `GITHUB_TOKEN` must work for the BioReason-Pro target repo: clone, push
   branches, open PRs, set labels, and read organization metadata.
4. W&B online tracking is mandatory. Never use offline, dryrun, or disabled W&B
   as a workaround.
5. Run preflight:

   ```bash
   scripts/launch_wandb_senpai_control_plane.sh <tag> --preflight_only
   ```

6. Run dry-run and inspect manifests:

   ```bash
   scripts/launch_wandb_senpai_control_plane.sh <tag> --dry_run
   ```

7. Confirm the dry run has advisor and student Deployments, GitHub/Anthropic/Exa
   launch secrets, `WANDB_API_KEY`, `HF_TOKEN`, `TARGET_REPO_URL`,
   `ADVISOR_BRANCH`, and `WANDB_MODE=online`.

## Experiment Contract

- Primary metric: validation `overall_mean_fmax`, higher is better.
- Flow: baseline validation, 5 training steps, validation gate, then continue to
  20 total steps only if the 5-step candidate strictly beats baseline.
- Missing or NaN `overall_mean_fmax` is a failed gate.
- Do not change protected benchmark data, hidden labels, GO ontology, IA
  weights, metric definitions, or W&B registry refs to make a run look better.
- Do not run test/holdout unless a human explicitly asks for final confirmation.
- Terminal student reports must include the `SENPAI-RESULT` marker from
  `program.md`.

## Safe Editing Surface

- Keep training behavior in root `train.py`.
- Use `bioreason2/` for focused model, dataset, reward, prompt, and tracking
  changes.
- Treat `eval.py`, `evals/`, and `scripts/sh_eval.sh` as evaluation
  infrastructure. Edit only for bug fixes or metric/logging reliability.
- Keep generated artifacts, caches, W&B state, checkpoints, and local secrets
  out of the repo.

## Current Implementation State

- `scripts/launch_wandb_senpai_control_plane.sh` wraps upstream `wandb/senpai`.
- `docs/runbooks/wandb-senpai-control-plane.md` explains the real control-plane
  workflow.
- `docs/adr/0005-upstream-senpai-control-plane.md` records the decision and
  next-launch checklist.
- `k8s/launch.py` and `senpai.yaml` are direct-training helpers only.
- `train.py` rejects offline, dryrun, and disabled W&B modes.
