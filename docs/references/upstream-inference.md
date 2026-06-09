# Upstream Inference Reference

## Source

- Repository: https://github.com/bowang-lab/BioReason-Pro

## What It Covers

The upstream public repository documents the inference workflow. Its README
describes a single-file prediction path that runs these stages:

1. InterPro domain annotation.
2. GO-GPT Gene Ontology prediction.
3. BioReason-Pro generation.

## Local Relationship

This Senpai branch is not an inference packaging branch. It keeps evaluation and
training infrastructure for RL experiments:

- Training/gate orchestration: `train.py`
- Evaluation: `eval.py`, `evals/`, `scripts/sh_eval.sh`
- Data artifact rebuilds: `scripts/data_generation/`

Use the upstream repository when inference behavior or public prediction UX is
the question. Use this branch when the question is Senpai RL training,
validation, and benchmark gating.
