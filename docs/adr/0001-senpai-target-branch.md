# ADR 0001: Preserve Research Snapshot Before Senpai Simplification

## Status

Accepted

## Context

The previous branch contained research history, presentations, notebooks, launch scripts, and experimental helpers. Senpai needs a compact target with one obvious training path.

## Decision

Create `snapshot/pre-senpai-20260609` before pruning. Create `senpai-target-20260609` from that snapshot and dedicate it to Senpai.

Ignored runtime state such as `.env`, `.venv*/`, `wandb/`, and `data/artifacts/` is not part of the snapshot commit.

## Consequences

The Senpai branch can aggressively delete historical material. The prior state remains recoverable from the snapshot branch.
