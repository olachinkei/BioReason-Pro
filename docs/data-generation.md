# Data Generation

This repository keeps the public data-generation code separate from the Senpai
training entrypoint. Training remains rooted at `train.py`; data artifacts are
rebuilt with scripts under `scripts/data_generation/`.

## Outputs

The default `main` pipeline writes:

- `data/artifacts/benchmarks/213_221_225_228/temporal_split/`
  - `train_assigned_labels.tsv`
  - `dev_assigned_labels.tsv`
  - `test_assigned_labels.tsv`
  - `*_assigned_propagated.tsv`
  - `IA.txt`
  - `summary.json`
  - `report.md`
- `data/artifacts/datasets/disease_temporal_hc_reasoning_v2/213_221_225_228/`
  - a Hugging Face `DatasetDict` with `train`, `validation`, and `test`
  - `dataset_summary.json`
  - `build_metadata.json`

`data/artifacts/` is ignored because these are generated runtime artifacts.

## Inputs

The temporal split builder uses:

- UniProt reviewed human disease-protein shortlist from the UniProt REST API.
- Historical GOA human GAF releases from the EBI GOA archive.
- `go-basic.obo`, preferably from `bioreason2/dataset/go-basic.obo`.
- Optional local GAF cache via `--gaf-dir`.
- Optional local shortlist via `--shortlist-tsv`.

The reasoning dataset builder also uses:

- UniProt REST metadata for sequence, organism, protein name, and function text.
- Optional source metadata with `go_pred`, `interpro_formatted`, and `ppi_formatted`.
- Optional InterPro metadata with `interpro_id`, `entry_name`, and `type`.

The strict paper-native dataset should provide source metadata. Use
`--allow-missing-paper-context` only for debug or ablation builds.

## Dry Run

Print the exact commands without downloading or writing large artifacts:

```bash
python scripts/data_generation/run_pipeline.py --variant main --dry-run
```

## Build Main Artifacts

```bash
python scripts/data_generation/run_pipeline.py \
  --variant main \
  --source-metadata-local-dir /path/to/source_metadata_dataset \
  --interpro-metadata-local-dir /path/to/interpro_metadata_dataset
```

For cached external files:

```bash
python scripts/data_generation/run_pipeline.py \
  --variant main \
  --gaf-dir /path/to/goa_gaf_cache \
  --shortlist-tsv /path/to/uniprot_disease_shortlist.tsv \
  --source-metadata-local-dir /path/to/source_metadata_dataset \
  --interpro-metadata-local-dir /path/to/interpro_metadata_dataset
```

## Step By Step

Build only the temporal split:

```bash
python scripts/data_generation/build_temporal_split_artifact.py \
  --output-dir data/artifacts/benchmarks/213_221_225_228/temporal_split \
  --train-start-release 213 \
  --train-end-release 221 \
  --dev-end-release 225 \
  --test-end-release 228
```

Build only the reasoning dataset:

```bash
python scripts/data_generation/build_reasoning_dataset.py \
  --temporal-split-dir data/artifacts/benchmarks/213_221_225_228/temporal_split \
  --reasoning-output-dir data/artifacts/datasets/disease_temporal_hc_reasoning_v2/213_221_225_228 \
  --source-metadata-local-dir /path/to/source_metadata_dataset \
  --interpro-metadata-local-dir /path/to/interpro_metadata_dataset
```

Build only IA weights:

```bash
python scripts/data_generation/build_ia_weights.py \
  --annotations data/artifacts/benchmarks/213_221_225_228/temporal_split/train_assigned_propagated.tsv \
  --obo data/artifacts/benchmarks/213_221_225_228/temporal_split/go-basic.obo \
  --output data/artifacts/benchmarks/213_221_225_228/temporal_split/IA.txt
```

## W&B Upload

The pipeline can upload the temporal split, reasoning dataset, and IA file:

```bash
WANDB_ENTITY=<entity> WANDB_PROJECT=<project> \
python scripts/data_generation/run_pipeline.py \
  --variant main \
  --upload-to-wandb \
  --source-metadata-local-dir /path/to/source_metadata_dataset \
  --interpro-metadata-local-dir /path/to/interpro_metadata_dataset
```

## CoreWeave Notes

Use the global `coreweave-gpu-implementation` skill for cluster access. Keep
runtime outputs and caches under `/mnt/data/$USER/BioReason-Pro` when building
large artifacts on CoreWeave. Do not place generated artifacts or local secrets
inside the repository.
