# BioReason-Pro Senpai Target

This branch is the compact Senpai target for BioReason-Pro reinforcement-learning experiments.

The only training entrypoint is:

```bash
python train.py --wandb_name "$STUDENT_NAME/<hypothesis-slug>"
```

`train.py` runs the fixed Senpai flow:

1. evaluate the baseline paper checkpoint on validation,
2. train for 5 steps,
3. evaluate validation `overall_mean_fmax`,
4. stop if the 5-step model does not beat baseline,
5. continue to 20 total steps only if the 5-step gate improves.

Read `docs/index.md` first. The experiment contract lives in `program.md`, the upstream Senpai teacher/student control-plane launch lives in `docs/runbooks/wandb-senpai-control-plane.md`, CoreWeave SSH/Slurm GPU operating steps live in `docs/runbooks/coreweave-implementation.md`, and data-generation steps live in `docs/design-docs/data-generation.md`.

For the real GitHub PR-based Senpai loop, launch upstream `wandb/senpai` through:

```bash
scripts/launch_wandb_senpai_control_plane.sh bioreason-r1
```

For manual CoreWeave GPU runs, use the SSH/Slurm workflow in `docs/runbooks/coreweave-implementation.md`. If you specifically want a direct Kubernetes training Job on CoreWeave CKS with SUNK scheduling, render the manifests with:

```bash
python k8s/launch.py --tag smoke-r1 --gate_steps 1 --continue_steps 0 --max_val_samples 2 --dry_run
```

See `docs/runbooks/coreweave-sunk-senpai.md` before applying direct training manifests to the cluster.
