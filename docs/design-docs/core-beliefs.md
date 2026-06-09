# Core Beliefs

## Optimize The Loop, Not The Story

The target exists to find training changes that improve validation
`overall_mean_fmax`. A good experiment is one that can survive the same fixed
baseline, data bundle, validation split, and metric contract as every other
experiment.

## Five Steps Should Teach Us Something

The first gate is intentionally short. If a hypothesis cannot improve after a
5-step training probe, it should stop and report honestly. Continuation is only
earned by strict validation improvement.

## Keep Training Obvious

`train.py` is the only training entrypoint. Extra launch scripts and historical
experiment wrappers make it harder for Senpai students to know what matters.

## Keep Generated State Out Of Git

Datasets, checkpoints, W&B state, caches, eval scratch, and CoreWeave runtime
outputs are reproducible or external state. They belong under runtime storage,
not in the source tree.

## Preserve Reproducibility

The branch is compact, but public data-generation code stays in the repo so a
future release can explain how benchmark and reasoning artifacts were produced.
