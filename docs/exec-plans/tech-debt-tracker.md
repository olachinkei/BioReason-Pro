# Tech Debt Tracker

## Open

| Item | Why It Matters | Owner |
| --- | --- | --- |
| Confirm online W&B credentials on CoreWeave | Offline smoke works, but real Senpai runs should log to W&B. | Human/Codex |
| Run full data-generation pipeline once from cached sources | The scripts are contract-tested, but the large network-heavy build should be verified before public release. | Codex on CoreWeave |
| Tighten paper-to-code algorithm mapping after final paper review | `docs/references/algorithm-map.md` points to the paper and local surfaces, but section-level mapping can improve after a human reads the final method sections. | Human/Codex |
| Keep CoreWeave runbook fresh | Slurm partitions, mount behavior, and auth can drift. | Human/Codex |

## Not Planned

| Item | Reason |
| --- | --- |
| Frontend documentation | This branch has no frontend. |
| Product specs | This is a research target, not a user-facing product. |
| DB schema docs | The repository has no database schema. |
