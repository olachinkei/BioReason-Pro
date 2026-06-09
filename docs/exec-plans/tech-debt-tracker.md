# Tech Debt Tracker

## Open

| Item | Why It Matters | Owner |
| --- | --- | --- |
| Confirm online W&B credentials on CoreWeave | Offline smoke works, but real Senpai runs should log to W&B. | Human/Codex |
| Run full data-generation pipeline once from cached sources | The scripts are contract-tested, but the large network-heavy build should be verified before public release. | Codex on CoreWeave |
| Decide whether to add a full 10k-token reproduction profile | Senpai defaults keep the paper rollout shape but use an approximately 8k-token completion budget. A separate profile would make 10k-token reproduction explicit. | Human/Codex |
| Audit explicit structure/localization prompt fields before claiming paper reproduction | The Senpai prompt path centers organism, InterPro, PPI, and GO-GPT context. Paper-faithful reproduction may require explicit structure/localization fields depending on the artifact. | Human/Codex |
| Keep CoreWeave runbook fresh | Slurm partitions, mount behavior, and auth can drift. | Human/Codex |

## Not Planned

| Item | Reason |
| --- | --- |
| Frontend documentation | This branch has no frontend. |
| Product specs | This is a research target, not a user-facing product. |
| DB schema docs | The repository has no database schema. |
