# Disease Benchmark Research README

この README は、`domain/specification/busiless-rules/specification.md` に沿って、現在の実装をどう実行するかをまとめたものである。  
現在の benchmark 版は `213 -> 221 -> 225 -> 228`、shortlist mode は `high-confidence` で固定している。

## 1. いま実行できる範囲

2026-04-07 時点で、実行できる範囲は次のとおり。

- Step 0 の temporal split 生成
- Step 0 / eval / training tracking の contract test
- `validation` / `test` を切り替えた eval
- eval の local artifact 出力
- eval の W&B / Weave tracking
- SFT 用 training run の W&B `job_type=train_sft` と benchmark lineage 記録
- SFT 用 training checkpoint directory の W&B Artifact 登録

未着手または未完成のもの:

- Step 0 artifact から supervised / reasoning dataset を正式に組み立てる専用 script
- RL 用 dataset 派生 script
- `train_rl` の実行 entry point

## 2. 前提環境

### 2.1 共有前提

- 作業ディレクトリは repo root:

```bash
cd /Users/keisuke/Project/learning/drug_discovery/BioReason-Pro
```

- Step 0 artifact の current implementation 版:

```text
domain/specification/busiless-rules/artifacts/step0_human_ub_20260406
```

- reference sensitivity 版:

```text
domain/specification/busiless-rules/artifacts/step0_human_lb_20260406
```

### 2.2 テスト用環境

contract test は `.venv-step0` を前提にしている。

```bash
.venv-step0/bin/python -m unittest discover -s test -v
```

### 2.3 学習・評価用環境

eval と SFT は GPU 前提で、少なくとも次が入った環境を使う。

- `torch`
- `pytorch_lightning`
- `transformers`
- `wandb`
- `weave`
- `datasets`
- `cafaeval`

## 3. まず最初にやる確認

実装を触ったあとや、別環境へ移した直後は、最初に contract test を通す。

```bash
.venv-step0/bin/python -m unittest discover -s test -v
```

期待する状態:

- Step 0 の split contract が通る
- eval の `validation/test` 切替 contract が通る
- eval sample table の `1 sample = 1 row` contract が通る
- training tracking helper の contract が通る

## 4. Step 0 を再生成する

current implementation 版を再生成する場合:

```bash
.venv-step0/bin/python scripts/step0_disease_temporal_split.py \
  --output-dir domain/specification/busiless-rules/artifacts/step0_human_ub_20260406 \
  --train-start-release 213 \
  --train-end-release 221 \
  --dev-end-release 225 \
  --test-end-release 228 \
  --shortlist-mode high-confidence \
  --use-shell-filter
```

reference sensitivity 版を再生成する場合:

```bash
.venv-step0/bin/python scripts/step0_disease_temporal_split.py \
  --output-dir domain/specification/busiless-rules/artifacts/step0_human_lb_20260406 \
  --train-start-release 214 \
  --train-end-release 221 \
  --dev-end-release 225 \
  --test-end-release 228 \
  --shortlist-mode high-confidence \
  --use-shell-filter
```

Step 0 実行後に確認するファイル:

- `summary.json`
- `report.md`
- `train_assigned_labels.tsv`
- `dev_assigned_labels.tsv`
- `test_assigned_labels.tsv`
- `*_assigned_propagated.tsv`
- `*_assigned_nk_lk.tsv`
- `*_assigned_nk_lk_propagated.tsv`
- `nk_lk_eda.tsv`

特に `summary.json` では次を確認する。

- `split_validation.time_order_valid == true`
- `split_validation.protein_disjoint_valid == true`
- `nk_lk_status`
- split ごとの protein 件数

## 5. eval を実行する

### 5.1 先に設定するもの

`scripts/sh_eval.sh` の上部にある次の値を自分の環境に合わせて埋める。

- `MODEL_PATH`
- `GO_OBO_PATH`
- `GO_EMBEDDINGS_PATH`
- `DATASET_CACHE_DIR`
- `STRUCTURE_DIR`
- `DATASET_NAME`
- `REASONING_DATASET_NAME`

`validation` で開発中の比較を回す場合:

```bash
EVAL_SPLIT=validation \
BENCHMARK_VERSION="213 -> 221 -> 225 -> 228" \
MODEL_NAME="bioreason-pro-base" \
bash scripts/sh_eval.sh
```

`test` で最終報告用の評価を回す場合:

```bash
EVAL_SPLIT=test \
BENCHMARK_VERSION="213 -> 221 -> 225 -> 228" \
MODEL_NAME="bioreason-pro-base" \
bash scripts/sh_eval.sh
```

W&B と Weave も同時に追跡したい場合:

```bash
EVAL_SPLIT=validation \
BENCHMARK_VERSION="213 -> 221 -> 225 -> 228" \
MODEL_NAME="bioreason-pro-base" \
WANDB_PROJECT="bioreason-pro-disease-benchmark" \
WANDB_ENTITY="your-team-or-user" \
WANDB_RUN_NAME="eval-base-validation" \
WANDB_ARTIFACT_NAME="eval-base-validation-artifacts" \
WEAVE_PROJECT="your-team-or-user/bioreason-pro-disease-benchmark" \
WEAVE_EVAL_NAME="eval-base-validation" \
bash scripts/sh_eval.sh
```

### 5.2 eval の出力

`scripts/sh_eval.sh` は既定で次へ出力する。

```text
evals/results
```

主な出力:

- 個別 JSON
- `sample_results.tsv`
- `run_summary.json`
- `evaluation_errors.json`

### 5.3 CAFA metric を集計する

eval の JSON を出したあとに、`evals/cafa_evals.py` で F_max を集計する。

```bash
python evals/cafa_evals.py \
  --input_dir evals/results \
  --ontology /path/to/go-basic.obo \
  --ia_file /path/to/IA.txt \
  --output_dir evals/cafa_metrics \
  --reasoning_mode True \
  --final_answer_only True
```

主に使う出力:

- `evals/cafa_metrics/metrics_summary.json`

この `metrics_summary.json` は、必要なら次回 eval 実行時に `METRICS_SUMMARY_PATH` として渡す。

```bash
EVAL_SPLIT=test \
BENCHMARK_VERSION="213 -> 221 -> 225 -> 228" \
MODEL_NAME="bioreason-pro-sft" \
METRICS_SUMMARY_PATH="evals/cafa_metrics/metrics_summary.json" \
WANDB_PROJECT="bioreason-pro-disease-benchmark" \
WANDB_ENTITY="your-team-or-user" \
WANDB_RUN_NAME="eval-sft-test" \
WANDB_ARTIFACT_NAME="eval-sft-test-artifacts" \
WEAVE_PROJECT="your-team-or-user/bioreason-pro-disease-benchmark" \
WEAVE_EVAL_NAME="eval-sft-test" \
bash scripts/sh_eval.sh
```

## 6. SFT を実行する

### 6.1 先に設定するもの

`scripts/sh_train_protein_qwen_staged.sh` の上部で、少なくとも次を埋める。

- `BASE_CHECKPOINT_DIR`
- `DATASET_CACHE_DIR`
- `CACHE_DIR`
- `STRUCTURE_DIR`
- `GO_EMBEDDINGS_PATH`
- `GO_OBO_PATH`
- `DATASET_ARTIFACT`
- `BASE_CHECKPOINT`

この script には、仕様に必要な tracking metadata をすでに入れてある。

- `--wandb_job_type train_sft`
- `--benchmark_version`
- `--step0_artifact`
- `--dataset_config`
- `--reasoning_dataset_config`
- `--shortlist_mode`
- `--train_start_release`
- `--train_end_release`
- `--dev_end_release`
- `--test_end_release`
- `--job_time_limit 12:00:00`
- `--checkpoint_artifact_name`

### 6.2 実行コマンド

```bash
bash scripts/sh_train_protein_qwen_staged.sh
```

この wrapper は次の 2 段階を回す。

1. Stage 1: projector / GO module warm-up
2. Stage 2: full model fine-tuning

### 6.3 SFT 実行後に見るもの

- checkpoint directory
- W&B run config
- train / validation loss
- sample generation table
- output checkpoint artifact

W&B 上では少なくとも次が見えていることを確認する。

- `job_type=train_sft`
- `benchmark_version`
- `step0_artifact`
- `dataset_config`
- `reasoning_dataset_config`
- `dataset_artifact`
- `model_artifact`
- `job_time_limit=12:00:00`

## 7. RL について

仕様上の `train_rl` は残っているが、repo にはまだ実行 entry point が揃っていない。  
次に進める場合は、まず `PLAN.md` の `3.3`, `3.7`, `4.4` を実装する。

## 8. 実行順の最短版

迷ったら次の順に実行する。

1. contract test

```bash
.venv-step0/bin/python -m unittest discover -s test -v
```

2. Step 0

```bash
.venv-step0/bin/python scripts/step0_disease_temporal_split.py \
  --output-dir domain/specification/busiless-rules/artifacts/step0_human_ub_20260406 \
  --train-start-release 213 \
  --train-end-release 221 \
  --dev-end-release 225 \
  --test-end-release 228 \
  --shortlist-mode high-confidence \
  --use-shell-filter
```

3. eval on validation

```bash
EVAL_SPLIT=validation \
BENCHMARK_VERSION="213 -> 221 -> 225 -> 228" \
MODEL_NAME="bioreason-pro-base" \
bash scripts/sh_eval.sh
```

4. CAFA metric 集計

```bash
python evals/cafa_evals.py \
  --input_dir evals/results \
  --ontology /path/to/go-basic.obo \
  --ia_file /path/to/IA.txt \
  --output_dir evals/cafa_metrics \
  --reasoning_mode True \
  --final_answer_only True
```

5. SFT

```bash
bash scripts/sh_train_protein_qwen_staged.sh
```
