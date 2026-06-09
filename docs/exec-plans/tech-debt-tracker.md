# Tech Debt Tracker

## Open

| Item | Why It Matters | Owner |
| --- | --- | --- |
| Confirm online W&B credentials on CoreWeave | Senpai runs must not launch unless online W&B preflight succeeds. | Human/Codex |
| Run full data-generation pipeline once from cached sources | The scripts are contract-tested, but the large network-heavy build should be verified before public release. | Codex on CoreWeave |
| Audit explicit structure/localization prompt fields before claiming paper reproduction | The Senpai prompt path centers organism, InterPro, PPI, and GO-GPT context. Paper-faithful reproduction may require explicit structure/localization fields depending on the artifact. | Human/Codex |
| Keep CoreWeave runbook fresh | Slurm partitions, mount behavior, and auth can drift. | Human/Codex |

## Completed

| Item | Outcome |
| --- | --- |
| Add a full 10k-token reproduction profile | Senpai defaults now use `SENPAI_MAX_NEW_TOKENS=10000`, `SENPAI_VLLM_MAX_MODEL_LEN=32768`, and `SENPAI_VLLM_MAX_NUM_SEQS=4`; OOM fallback reduces vLLM active sequences to 2 before reducing token length. |

## Not Planned

| Item | Reason |
| --- | --- |
| Frontend documentation | This branch has no frontend. |
| Product specs | This is a research target, not a user-facing product. |
| DB schema docs | The repository has no database schema. |
