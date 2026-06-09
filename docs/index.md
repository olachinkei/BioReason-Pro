# Documentation Map

This map is the starting point for humans and coding agents.

## Root Contracts

| Document | Purpose |
| --- | --- |
| `AGENTS.md` | Codex operating rules, no-context startup steps, and doc placement rules. |
| `CLAUDE.md` | Claude Code startup guide mirroring the critical agent rules. |
| `ARCHITECTURE.md` | Thin source map for runtime surfaces. |
| `program.md` | Senpai target contract and experiment loop. |
| `BASELINE.md` | Baseline model, data bundle, metric, and comparison rule. |
| `README.md` | Minimal repository entrypoint. |

## Docs Directory

| Area | Path | Purpose |
| --- | --- | --- |
| Design docs | `docs/design-docs/` | Target principles, data-generation design, and benchmark contracts. |
| Execution plans | `docs/exec-plans/` | Active/completed implementation plans and tech debt. |
| Runbooks | `docs/runbooks/` | Operational procedures for the upstream Senpai control plane, default CoreWeave SSH/Slurm path, optional direct CKS/SUNK launches, and Docker runtime setup. |
| References | `docs/references/` | External papers, upstream repositories, and algorithm-to-code mapping. |
| ADRs | `docs/adr/` | Accepted architectural and process decisions. |
| Generated docs | `docs/generated/` | Reserved for generated docs. No DB schema is needed today. |

## Intentionally Omitted

- `docs/product-specs/`: this is a research training target, not a product surface.
- `docs/FRONTEND.md`: there is no frontend in this Senpai branch.
- `docs/generated/db-schema.md`: there is no database schema.
- Historical notebooks and presentation material: recoverable from the snapshot branch, not part of this target.

## Fast Paths

- Assign or run experiments: `program.md`.
- Understand the fixed baseline: `BASELINE.md`.
- Rebuild data artifacts: `docs/design-docs/data-generation.md`.
- Confirm the next real Senpai launch: `docs/adr/0005-upstream-senpai-control-plane.md`.
- Launch upstream Senpai teacher/student loop: `docs/runbooks/wandb-senpai-control-plane.md`.
- Work on CoreWeave: `docs/runbooks/coreweave-implementation.md`.
- Launch optional direct CKS/SUNK training jobs: `docs/runbooks/coreweave-sunk-senpai.md`.
- Handle Senpai OOMs: `docs/runbooks/senpai-oom-handling.md`.
- Build the Docker image: `docs/runbooks/senpai-docker.md`.
- Find paper/upstream reference mapping: `docs/references/algorithm-map.md`.
