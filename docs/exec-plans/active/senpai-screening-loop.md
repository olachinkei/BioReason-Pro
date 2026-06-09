# Active Plan: Senpai Screening Loop

## Objective

Use a 5-step training probe to discover promising RL recipe changes without
OOMing one CoreWeave node or spending long runs on weak hypotheses.

## Current Flow

- Baseline validation runs first.
- Candidate trains for 5 steps.
- Validation gate compares `overall_mean_fmax` against baseline.
- Continuation runs only if the gate strictly improves.
- Terminal output includes `SENPAI-RESULT`.

## Implementation Surface

- Primary: `train.py`.
- Evaluation: `eval.py`, `evals/`, `scripts/sh_eval.sh`.
- Runtime operation: `docs/runbooks/coreweave-implementation.md`.

## Acceptance

- `python train.py --help` works locally.
- Contract tests pass.
- CoreWeave smoke does not OOM on one node.
- Gate failure stops cleanly.
- Gate success continues to the configured total step budget.
