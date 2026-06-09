# Senpai Screening Contract

## Goal

Find changes that improve validation `overall_mean_fmax` over the frozen paper
RL checkpoint without changing benchmark data or metric definitions.

## Fixed Flow

1. Evaluate the baseline checkpoint on validation.
2. Train the candidate for `--gate_steps`, default 5.
3. Evaluate the gate checkpoint on validation.
4. Stop if the gate metric is missing, NaN, or not strictly better than baseline.
5. Continue for `--continue_steps`, default 15, only when the gate improves.
6. Report the best validation metric reached by the candidate.

## Default Command

```bash
python train.py \
  --wandb_name "$STUDENT_NAME/<hypothesis-slug>" \
  --wandb_group "<hypothesis-or-pr>"
```

## Failure Cases

- Missing `overall_mean_fmax`.
- NaN or infinite metric values.
- Validation subprocess failure before metrics are written.
- OOM or runtime failure.
- Any change that modifies hidden labels, split definitions, GO ontology, IA
  weights, or the metric calculation.

## Related Files

- `train.py`
- `program.md`
- `BASELINE.md`
- `instructions/prompt-advisor.md`
- `instructions/prompt-student.md`
