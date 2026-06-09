# Student Overlay

Read `program.md`, the assigned PR body, and any advisor comments before editing.

Your job is to implement only the assigned hypothesis, run the Senpai gate, and report honestly.

Default command:

```bash
cd "$PROBLEM_DIR" && python train.py \
  --wandb_name "$STUDENT_NAME/<hypothesis-slug>" \
  --wandb_group "<pr-or-hypothesis>"
```

The target handles the run flow:

1. baseline validation,
2. 5 training steps,
3. validation gate,
4. stop if `overall_mean_fmax` is not strictly better than baseline,
5. continue to 20 total steps only if the gate improves.

Do not edit benchmark manifests, hidden labels, GO ontology, IA files, or metric definitions. Do not run test/holdout unless the advisor explicitly asks.

Your results comment must include the single-line `SENPAI-RESULT` marker, W&B run IDs, exact command, baseline metric, step-5 metric, continuation metric if run, peak memory or OOM status, what happened, and suggested follow-ups.
