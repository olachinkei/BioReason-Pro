# Upstream wandb/senpai Control Plane

Use this runbook when the desired behavior is the real Senpai loop: a
teacher/advisor pod creates and reviews GitHub PRs, student pods pick up
assigned PRs, edit this target repository, run `train.py`, and report results
back through GitHub and W&B.

For the next-launch confirmation checklist, also read
`docs/adr/0005-upstream-senpai-control-plane.md`.

Do not use this repository's `k8s/launch.py` for that workflow. That helper is
training-only and runs `python train.py` directly. The upstream control plane is
launched through:

```bash
scripts/launch_wandb_senpai_control_plane.sh <research-tag>
```

## What Runs

- Upstream runner repo: `https://github.com/wandb/senpai.git`.
- Target problem repo: `https://github.com/olachinkei/BioReason-Pro.git`.
- Teacher/advisor Deployment: no GPU; creates hypothesis PRs, reviews results,
  merges winners, closes dead ends, and checks human GitHub issues.
- Student Deployments: GPU workers; poll GitHub for assigned PRs, implement the
  hypothesis, run the target command from `program.md`, and comment with the
  `SENPAI-RESULT` marker.
- GitHub PR labels route work: advisor branch label, `status:wip`,
  `status:review`, and `student:<name>`.

## Required Secrets

Keep secrets outside the repository, normally:

```bash
~/.secrets/bioreason-senpai.env
```

Required:

```bash
GITHUB_TOKEN=...
ANTHROPIC_API_KEY=...
EXA_API_KEY=...
WANDB_API_KEY=...
HF_TOKEN=...
```

`GITHUB_TOKEN` must be a token that can clone, push branches, open PRs, label
PRs, and read organization metadata for the target repo. For private org repos,
use a PAT with `repo` and `read:org`.

The wrapper passes credentials as Kubernetes Secrets:

- `GITHUB_TOKEN`, `ANTHROPIC_API_KEY`, and `EXA_API_KEY` go into the upstream
  per-launch Secret consumed by every advisor/student pod.
- `WANDB_API_KEY` and `HF_TOKEN` go into shared `senpai-secrets`.

This means GitHub login is available inside each advisor/student pod. The
upstream entrypoints set `gh repo set-default "$GH_REPO"` for the target repo,
so GitHub issues and PRs are managed against BioReason-Pro, not the runner repo.

## W&B Requirement

W&B online tracking is mandatory. Do not launch with offline, dryrun, or
disabled W&B modes. If W&B returns 401 or cannot initialize, fix credentials or
upgrade/verify `wandb` and `weave` in the target runtime before relaunching.

## Preflight

Sync the local secret file to CoreWeave if launching there:

```bash
./scripts/coreweave_sync_secrets.sh
```

Run the upstream launcher preflight through the wrapper:

```bash
scripts/launch_wandb_senpai_control_plane.sh bioreason-r1 --preflight_only
```

This verifies GitHub token access to the target repo, resolves the target base
branch, checks Anthropic and Exa credentials, and confirms the advisor branch
can be created. W&B is still enforced by the target runtime and by the
mandatory online run policy.

## Dry Run

Render redacted manifests without applying:

```bash
scripts/launch_wandb_senpai_control_plane.sh bioreason-r1 \
  --dry_run \
  --n_students 2 \
  --names frieren,fern
```

Check for:

- an advisor Deployment,
- one student Deployment per name,
- `GITHUB_TOKEN`, `ANTHROPIC_API_KEY`, and `EXA_API_KEY` in the launch Secret,
- `WANDB_API_KEY` from `senpai-secrets`,
- `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` from `senpai-secrets`,
- `TARGET_REPO_URL=https://github.com/olachinkei/BioReason-Pro.git`,
- `ADVISOR_BRANCH=senpai-bioreason`,
- `WANDB_MODE=online`.

## Launch

Start the teacher/advisor plus one student:

```bash
scripts/launch_wandb_senpai_control_plane.sh bioreason-r1
```

Start the teacher/advisor plus multiple students:

```bash
N_STUDENTS=4 STUDENT_NAMES=frieren,fern,tanjiro,nezuko \
  scripts/launch_wandb_senpai_control_plane.sh bioreason-r1
```

Useful overrides:

```bash
ADVISOR_BRANCH=senpai-bioreason-r2 \
GH_HISTORY_SCOPE=fresh \
POLL_INTERVAL_S=60 \
POLL_JITTER_S=10 \
  scripts/launch_wandb_senpai_control_plane.sh bioreason-r2
```

## Monitor

```bash
kubectl get deployments,pods -l app=senpai,research-tag=bioreason-r1
kubectl logs -f deployment/senpai-advisor-bioreason-r1
kubectl logs -f deployment/senpai-bioreason-r1-frieren
```

GitHub should show PRs opened against `ADVISOR_BRANCH`, with routing labels
such as `student:frieren`, `status:wip`, and `status:review`.

## Stop

```bash
kubectl delete deployments,configmaps,secrets -l research-tag=bioreason-r1
```

Do not delete the shared runtime PVC, W&B artifacts, model checkpoints, or data
artifacts unless exact paths have been confirmed as disposable.
