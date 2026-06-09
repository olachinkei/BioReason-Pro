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

Read `docs/index.md` first. The experiment contract lives in `program.md`, CoreWeave operating steps live in `docs/runbooks/coreweave-implementation.md`, and data-generation steps live in `docs/design-docs/data-generation.md`.

For CoreWeave CKS/SUNK launches, render the Kubernetes Job manifests with:

```bash
python k8s/launch.py --tag smoke-r1 --gate_steps 1 --continue_steps 0 --max_val_samples 2 --dry_run
```

See `docs/runbooks/coreweave-sunk-senpai.md` before applying them to the cluster.
