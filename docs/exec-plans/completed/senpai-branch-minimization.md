# Completed Plan: Senpai Branch Minimization

## Summary

The historical research branch was preserved first, then this branch was made
compact for Senpai experiments.

## Completed Work

- Created `snapshot/pre-senpai-20260609`.
- Created `senpai-target-20260609`.
- Consolidated training behavior into root `train.py`.
- Removed historical training entrypoints and launch wrappers.
- Preserved public data-generation scripts under `scripts/data_generation/`.
- Kept runtime artifacts ignored.
- Documented CoreWeave operation without copying the global skill into the repo.

## Decisions

The detailed decision records live in `docs/adr/`.
