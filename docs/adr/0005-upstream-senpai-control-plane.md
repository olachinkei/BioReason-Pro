# ADR 0005: Use Upstream wandb/senpai Control Plane for Senpai Runs

## Status

Accepted

## Context

BioReason-Pro is the Senpai target repository. It contains `train.py`,
`program.md`, benchmark contracts, and direct CoreWeave training helpers.

The real Senpai workflow lives in the upstream `wandb/senpai` runner. That
workflow has a teacher/advisor process and student workers coordinated through
GitHub PRs, labels, and W&B. Running this repository's `k8s/launch.py` directly
does not exercise that control loop.

## Decision

When the request is to run Senpai itself, use the upstream `wandb/senpai`
control plane through:

```bash
scripts/launch_wandb_senpai_control_plane.sh <research-tag>
```

Use this repository's `k8s/launch.py` only when the request explicitly asks for
a direct training Job without the upstream teacher/advisor and student loop.

W&B online tracking is mandatory. Do not use `WANDB_MODE=offline`, dryrun, or
disabled W&B as a workaround for auth or SDK failures.

## Next Confirmation Checklist

Before the next real launch, confirm these items in order:

1. Local secrets exist outside the repo, normally
   `~/.secrets/bioreason-senpai.env`.
2. `GITHUB_TOKEN` is valid for the BioReason-Pro target repo and can clone,
   push branches, open PRs, set labels, and read organization metadata.
3. `ANTHROPIC_API_KEY`, `EXA_API_KEY`, and `WANDB_API_KEY` are valid.
4. `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` is present if the target runtime needs
   gated Hugging Face model access.
5. `scripts/launch_wandb_senpai_control_plane.sh <tag> --preflight_only`
   passes.
6. A dry run renders one advisor Deployment and the requested student
   Deployments.
7. The dry-run manifests contain `GITHUB_TOKEN`, `ANTHROPIC_API_KEY`, and
   `EXA_API_KEY` in the launch Secret, plus `WANDB_API_KEY`, `HF_TOKEN`, and
   `HUGGING_FACE_HUB_TOKEN` from `senpai-secrets`.
8. The dry-run manifests set `TARGET_REPO_URL` to the BioReason-Pro target repo,
   `ADVISOR_BRANCH` to the intended advisor branch, and `WANDB_MODE=online`.
9. The intended student count, names, GPU count, CPU, memory, timeout, and PVC
   are correct for the CoreWeave allocation.
10. After launch, GitHub shows advisor-created PRs with routing labels such as
    `student:<name>`, `status:wip`, and `status:review`.
11. W&B shows online runs under the intended entity and project.

## Consequences

The target repo remains compact and focused on training behavior, while the
agent coordination logic stays in upstream `wandb/senpai`.

Future agents should not treat a successful direct `train.py` or
`k8s/launch.py` run as proof that Senpai itself is working. The upstream
advisor/student control plane must be preflighted, dry-run, and launched when
the goal is an actual Senpai run.
