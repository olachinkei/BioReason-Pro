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

Read `program.md` before assigning or running experiments. CoreWeave operating steps live in `docs/coreweave-implementation.md`. Data-generation steps live in `docs/data-generation.md`. Architecture notes are intentionally omitted; agents should infer code structure from the code.
