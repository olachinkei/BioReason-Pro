# ADR 0002: Use `train.py` As The Only Training Entrypoint

## Status

Accepted

## Context

The previous repository had multiple training scripts and shell wrappers. Senpai students need one command and one place to edit training behavior.

## Decision

All training behavior lives in root `train.py`. Normal execution runs the Senpai gate orchestration. Internal DeepSpeed workers invoke the same file with `--backend_train`.

Historical training entrypoints and launch wrappers are removed from the Senpai branch.

## Consequences

Students have a single editable training surface. The file is larger than a pure wrapper because it also contains the backend RL implementation, but there is no second training script to keep in sync.
