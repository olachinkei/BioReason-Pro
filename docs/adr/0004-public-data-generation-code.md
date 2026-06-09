# 0004: Keep Public Data-Generation Code Outside the Training Entrypoint

## Status

Accepted

## Context

The Senpai branch intentionally has one training entrypoint, `train.py`. At the
same time, future public releases need readable code that explains how the data
artifacts were generated.

## Decision

Keep data-generation code under `scripts/data_generation/` and document its
usage in `docs/design-docs/data-generation.md`. The scripts rebuild generated artifacts but
do not act as training launchers.

## Consequences

- `train.py` remains the only training entrypoint.
- Data-generation code is preserved for reproducibility and publication.
- Generated data stays ignored under `data/artifacts/`.
- Historical notebooks and experiment launch scripts remain excluded.
