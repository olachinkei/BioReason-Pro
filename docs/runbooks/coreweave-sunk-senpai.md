# CoreWeave CKS/SUNK Senpai Launch

This runbook launches BioReason-Pro Senpai training jobs as Kubernetes Jobs on
CoreWeave Kubernetes Service while the SUNK scheduler asks Slurm to place and
account for the Pods. It complements the SSH/Slurm workflow in
`docs/runbooks/coreweave-implementation.md`.

## Shape

- `senpai.yaml` holds the launch defaults.
- `k8s/launch.py` renders or applies Kubernetes manifests.
- `k8s/student-job.yaml` is the SUNK-scheduled student Job template.
- Each student Job runs root `train.py`; training behavior stays in `train.py`.
- Mutable state, W&B files, Hugging Face cache, temporary files, and
  checkpoints stay under `BIOREASON_RUNTIME_ROOT` on the mounted runtime PVC.

This repository is the Senpai problem package. It does not vendor the full
`wandb/senpai` control-plane repository. Use upstream `wandb/senpai` for the
advisor/student PR loop, and use this repo's `k8s/launch.py` when you want the
BioReason training work itself to run as CKS/SUNK Jobs.

## Prerequisites

Build and publish the image from this repository to a registry visible to CKS:

```bash
docker build -t <registry>/bioreason-pro-senpai:cuda12.6 .
docker push <registry>/bioreason-pro-senpai:cuda12.6
```

Keep deployment secrets in a local `.env` file at the repository root or in a
private path such as `~/.secrets/bioreason-senpai.env`. The repository ignores
`.env`; do not commit it. A local `.env` used for deployment should contain:

```bash
WANDB_API_KEY=...
WANDB_PROJECT=bioreasoning-pro-senpai
HF_TOKEN=...
SENPAI_MAX_NEW_TOKENS=10000
SENPAI_VLLM_MAX_MODEL_LEN=32768
SENPAI_VLLM_MAX_NUM_SEQS=4
```

If you also run the upstream `wandb/senpai` advisor/student control plane from
the same local environment, include the control-plane keys there too:

```bash
GITHUB_TOKEN=...
ANTHROPIC_API_KEY=...
EXA_API_KEY=...
```

Load the local file before creating Kubernetes Secrets:

```bash
set -a
source .env
set +a
```

Create or rotate the shared Kubernetes Secret:

```bash
kubectl create secret generic senpai-secrets \
  --from-literal=wandb-api-key="$WANDB_API_KEY" \
  --from-literal=hf-token="${HF_TOKEN:-}" \
  --dry-run=client -o yaml | kubectl apply -f -
```

Make sure the configured runtime PVC exists:

```bash
kubectl get pvc bioreason-runtime-pvc
```

Find the SUNK scheduler name if the default in `senpai.yaml` is not correct:

```bash
kubectl get pods -l app.kubernetes.io/name=sunk-scheduler -oyaml \
  | yq '.items[0].spec.containers[] | select(.name == "scheduler").args'
```

## Render

Always render first:

```bash
python k8s/launch.py \
  --tag smoke-r1 \
  --image <registry>/bioreason-pro-senpai:cuda12.6 \
  --mode gate \
  --gate_steps 1 \
  --continue_steps 0 \
  --max_val_samples 2 \
  --dry_run
```

Check for these fields before applying:

- `schedulerName` matches the SUNK scheduler.
- `sunk.coreweave.com/partition` points at the desired Slurm partition.
- `sunk.coreweave.com/exclusive` is `none` for a full-node 8 GPU student.
- `terminationGracePeriodSeconds` is below the scheduler kill-wait threshold.
- CPU is request-only; the template intentionally avoids a CPU limit to reduce
  the risk of SUNK static CPU allocation conflicts.
- Senpai keeps the paper rollout token length by default:
  `SENPAI_MAX_NEW_TOKENS=10000`. If vLLM OOMs, reduce
  `SENPAI_VLLM_MAX_NUM_SEQS` from `4` to `2` before reducing token length. See
  `docs/runbooks/senpai-oom-handling.md`.

## Launch

Smoke run:

```bash
python k8s/launch.py \
  --tag smoke-r1 \
  --image <registry>/bioreason-pro-senpai:cuda12.6 \
  --mode gate \
  --gate_steps 1 \
  --continue_steps 0 \
  --max_val_samples 2
```

Full gate:

```bash
python k8s/launch.py \
  --tag bioreason-r1 \
  --image <registry>/bioreason-pro-senpai:cuda12.6 \
  --n_students 4 \
  --names frieren,fern,tanjiro,nezuko
```

Watch the Kubernetes and Slurm views:

```bash
kubectl get jobs,pods -l app=bioreason-senpai,research-tag=bioreason-r1
kubectl describe pod -l app=bioreason-senpai,research-tag=bioreason-r1
kubectl logs -l app=bioreason-senpai,research-tag=bioreason-r1 -f
```

Inside SUNK, the scheduler creates a Slurm placeholder job for each Pod, so
cluster operators can also inspect placement and accounting with Slurm tools.

## Cleanup

Delete a launch when finished:

```bash
kubectl delete jobs,configmaps -l app=bioreason-senpai,research-tag=bioreason-r1
```

Do not delete the runtime PVC or W&B/Hugging Face artifacts unless the exact
paths are confirmed as disposable. Follow the storage cleanup rules in
`docs/runbooks/coreweave-implementation.md`.
