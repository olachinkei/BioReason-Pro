# Documentation Map

This map is the starting point for humans and coding agents.

## Root Contracts

| Document | Purpose |
| --- | --- |
| `AGENTS.md` | Agent operating rules and doc placement rules. |
| `ARCHITECTURE.md` | Thin source map for runtime surfaces. |
| `program.md` | Senpai target contract and experiment loop. |
| `BASELINE.md` | Baseline model, data bundle, metric, and comparison rule. |
| `README.md` | Minimal repository entrypoint. |

## Docs Directory

| Area | Path | Purpose |
| --- | --- | --- |
| Design docs | `docs/design-docs/` | Target principles, data-generation design, and benchmark contracts. |
| Execution plans | `docs/exec-plans/` | Active/completed implementation plans and tech debt. |
| Runbooks | `docs/runbooks/` | Operational procedures such as CoreWeave usage, CKS/SUNK launches, and Docker runtime setup. |
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
- Work on CoreWeave: `docs/runbooks/coreweave-implementation.md`.
- Launch CKS/SUNK jobs: `docs/runbooks/coreweave-sunk-senpai.md`.
- Handle Senpai OOMs: `docs/runbooks/senpai-oom-handling.md`.
- Build the Docker image: `docs/runbooks/senpai-docker.md`.
- Find paper/upstream reference mapping: `docs/references/algorithm-map.md`.
