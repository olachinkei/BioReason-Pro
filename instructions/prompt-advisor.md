# Advisor Overlay

Assign one focused hypothesis per PR. The target is BioReason-Pro RL training with a fixed 5-step validation gate.

Use `program.md` and `BASELINE.md` as the contract. The primary ranking metric is validation `overall_mean_fmax`, higher is better.

When writing PR instructions:

- State exactly what to change in `train.py` or `bioreason2/`.
- Keep the run command as `python train.py --wandb_name "<student>/<slug>" --wandb_group "<pr-or-hypothesis>"`.
- Do not ask students to run test/holdout unless the human researcher explicitly requests final confirmation.
- Treat missing or NaN validation metrics as a failed experiment.
- Merge only terminal results that strictly improve `overall_mean_fmax` over the current baseline.

After a winner, ask for cleanup that makes the winning behavior the default and removes stale switches.
