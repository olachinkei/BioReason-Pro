# Runbooks

Operational procedures live here.

- `coreweave-implementation.md`: Default CoreWeave GPU path using SSH, Slurm, runtime paths, smoke tests, and cleanup boundaries.
- `wandb-senpai-control-plane.md`: Upstream `wandb/senpai` teacher/advisor + student GitHub PR control plane.
- `coreweave-sunk-senpai.md`: Optional direct Kubernetes training Job launch on CoreWeave CKS with SUNK/Slurm scheduling when CKS direct training is explicitly requested.
- `senpai-oom-handling.md`: OOM triage while preserving the paper rollout token length.
- `senpai-docker.md`: Docker image build, required external APIs, runtime variables, and smoke/full run commands.

Reusable agent skills stay global and are not copied into this repository.
