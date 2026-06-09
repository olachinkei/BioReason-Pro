# ADR 0003: Keep CoreWeave Implementation Inside GPU Allocations

## Status

Accepted

## Context

The CoreWeave login node is shared and should not run GPU-heavy imports, installs, training, or evaluation. BioReason-Pro also produces large cache, checkpoint, W&B, and eval artifacts.

## Decision

Use the global `coreweave-gpu-implementation` skill for live work. Keep implementation, installs, tests, training, and eval inside a Slurm GPU allocation. Keep runtime outputs under `/mnt/data/$USER/BioReason-Pro`.

Do not copy the global skill into this repository.

## Consequences

The repo documents operational steps in `docs/runbooks/coreweave-implementation.md`, while reusable cluster behavior stays global. This avoids polluting the Senpai target with local agent skill files.
