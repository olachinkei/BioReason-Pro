# Architecture Map

This file is intentionally a source map, not a full architecture narrative.
Agents should infer implementation details from code.

## Runtime Surfaces

- `train.py`: the only training entrypoint. It owns the Senpai gate, baseline validation, backend DeepSpeed training, validation evaluation, and final `SENPAI-RESULT`.
- `eval.py`, `evals/`, `scripts/sh_eval.sh`: validation and CAFA-style metric computation.
- `bioreason2/`: model, dataset formatting, prompt, reward, registry, and tracking helpers.
- `scripts/data_generation/`: reproducible artifact generation for future public release. These scripts are not training launchers.
- `configs/disease_benchmark/`: W&B artifact registry and benchmark target configuration.
- `instructions/`: advisor/student overlays for Senpai PR assignment and reporting.

## Protected Contracts

- Primary validation metric: `overall_mean_fmax`, higher is better.
- Screening flow: baseline validation, 5-step training, validation gate, then continue only on strict improvement.
- Target hardware: one node with eight GPUs.
- Runtime artifacts belong under `/mnt/data/$USER/BioReason-Pro` on CoreWeave, not in git.

## Documentation Map

Use `docs/index.md` as the canonical documentation map.
