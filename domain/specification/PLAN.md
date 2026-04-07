# PLAN

この PLAN は、[specification.md](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/domain/specification/busiless-rules/specification.md) と [RESEARCH_README.md](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/RESEARCH_README.md) をそのまま実行計画に落としたものである。  
目的は、**何が終わっていて、次に何をやるかを曖昧にしないこと**である。

## 0. 現在地

### 0.1 採用前提

固定前提は次である。

- benchmark version: `213 -> 221 -> 225 -> 228`
- benchmark alias: `213.221.225.228`
- primary dataset: `disease_temporal_hc_reasoning_v1`
- comparison model: `bioreason-pro-rl-paper`
- 正本: W&B Artifact ref
- local filesystem: scratch

### 0.2 進捗サマリ

| 項目 | 状態 | 現状 |
|---|---|---|
| 仕様の整理 | 完了 | final 仕様は [specification.md](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/domain/specification/busiless-rules/specification.md) に統一済み |
| 実行手順の整理 | 完了 | 現行 runbook は [RESEARCH_README.md](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/RESEARCH_README.md) |
| temporal split artifact 作成 | 完了 | `wandb-healthcare/bioreason-pro-custom/disease-temporal-split:production` |
| reasoning dataset 作成 | 完了 | `wandb-healthcare/bioreason-pro-custom/disease-temporal-reasoning:production` |
| comparison model artifact 確定 | 完了 | `wandb-healthcare/bioreason-pro-custom/bioreason-pro-rl:production` |
| CoreWeave 実行フロー整理 | 完了 | `srun` ベースの実行、remote env、artifact 解決、1-sample smoke まで確認済み |
| comparison model の validation 評価 | 実行中 | `comparison-family` の `validation` run を CoreWeave で実行中 |
| SFT | 未着手 | wrapper と tracking は用意済み、run は未実施 |
| RL | 準備済み | `train_protein_grpo.py` と `scripts/sh_train_protein_grpo.sh` は実装済み、run は未実施 |

### 0.3 いま次にやること

次の実行対象は **実行中の `comparison-family` validation を完走させ、W&B 上の結果を確認すること** である。  
その後、SFT に進む。

## 1. データの準備

状態: **完了**

### 1.1 やること

ローカル Mac で次を行う。

1. temporal split artifact を build する
2. split sanity check を通す
3. reasoning dataset を build する
4. W&B Artifact に upload する

高位 entry point は [run_temporal_split_artifact_pipeline.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/scripts/run_temporal_split_artifact_pipeline.py) に固定する。

### 1.2 実行コマンド

```bash
cd /Users/keisuke/Project/learning/drug_discovery/BioReason-Pro

set -a
source .env
set +a

uv venv .venv-mac-data --python 3.11
source .venv-mac-data/bin/activate
uv pip install -r requirements/uv-local-data.txt

uv run --active python scripts/run_temporal_split_artifact_pipeline.py \
  --variant main \
  --shortlist-mode high-confidence \
  --use-shell-filter \
  --build-datasets \
  --upload-to-wandb \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-project "$WANDB_PROJECT"
```

### 1.3 完了条件

次が揃っていれば完了とする。

- `split_validation.time_order_valid == true`
- `split_validation.protein_disjoint_valid == true`
- temporal split artifact が W&B に upload されている
- reasoning dataset artifact が W&B に upload されている

### 1.4 現在の成果物

- temporal split artifact
  - `wandb-healthcare/bioreason-pro-custom/disease-temporal-split:production`
- reasoning dataset artifact
  - `wandb-healthcare/bioreason-pro-custom/disease-temporal-reasoning:production`

## 2. GPU へのアクセス

状態: **完了**

### 2.1 やること

CoreWeave SUNK の login node から `srun` で job を送る。  
GPU node に手で入る前提にはしない。

### 2.2 実行手順

1. login node に SSH する
2. ローカルからコードだけ `rsync` する
3. CoreWeave 側で `uv` 環境を作る
4. `wandb_registry_paths.env` と `wandb_asset_sources.env` を用意する

### 2.3 実行コマンド

SSH:

```bash
ssh -o IdentitiesOnly=yes kkamata+cwb607@sunk.cwb607-training.coreweave.app
```

rsync:

```bash
cd /Users/keisuke/Project/learning/drug_discovery

rsync -av --delete \
  --exclude 'data/artifacts/' \
  --exclude '.venv*/' \
  BioReason-Pro/ \
  kkamata+cwb607@sunk.cwb607-training.coreweave.app:~/BioReason-Pro/
```

CoreWeave 上の環境構築:

```bash
cd ~/BioReason-Pro

uv venv .venv-gpu --python 3.11
source .venv-gpu/bin/activate

uv sync
uv pip install esm --no-deps
uv pip install flash-attn --no-build-isolation --no-cache-dir
uv pip install unsloth
uv run --active wandb login
```

env 準備:

```bash
cd ~/BioReason-Pro

set -a
source .env
set +a

cp configs/disease_benchmark/wandb_registry_paths.env.example \
  configs/disease_benchmark/wandb_registry_paths.env

cp configs/disease_benchmark/wandb_asset_sources.env.example \
  configs/disease_benchmark/wandb_asset_sources.env
```

### 2.4 完了条件

次が揃っていればこのフェーズは完了とする。

- login node に入れる
- `~/BioReason-Pro` に最新コードがある
- `.venv-gpu` が作成済み
- `wandb_registry_paths.env` に次が入っている
  - `BIOREASON_MAIN_TEMPORAL_SPLIT_REGISTRY_PATH`
  - `BIOREASON_MAIN_REASONING_DATASET_REGISTRY_PATH`
  - `BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH`

## 3. 比較モデルの評価

状態: **実行中**

### 3.1 目的

独自 tuning 前の比較モデル `bioreason-pro-rl-paper` を、現在採用している benchmark 上で `validation` split で評価する。

### 3.2 評価対象

- `comparison-family`: `bioreason-pro-rl-paper`
- `tuned-family`: `train-sft-output`, `train-rl-output`
- `spec-comparison`: 上記すべて

この段階で実際に回しているのは `comparison-family` のみである。

### 3.3 実行コマンド

```bash
srun \
  --partition <gpu_partition> \
  --account <account_name> \
  --gpus 1 \
  --cpus-per-task 8 \
  --mem 128G \
  --time 12:00:00 \
  bash -lc '
    cd ~/BioReason-Pro &&
    source .venv-gpu/bin/activate &&
    uv run --active python scripts/run_registered_eval.py \
      --target-group comparison-family \
      --data-bundle main_production \
      --split validation \
      --wandb-entity "$WANDB_ENTITY" \
      --wandb-project "$WANDB_PROJECT"
  '
```

### 3.4 完了条件

W&B 上に次が見えていれば完了とする。

- `job_type=eval`
- `fmax_mf`
- `fmax_bp`
- `fmax_cc`
- `overall_mean_fmax`
- `eval_summary` table
- `eval_samples` table
- eval artifact

2026-04-07 現在、CoreWeave SUNK 上で `comparison-family` の `validation` run を実行中である。

### 3.5 このフェーズが終わったらやること

結果を確認し、SFT に進む。  
この時点では `test` はまだ使わない。

## 4. SFT

状態: **未着手**

### 4.1 目的

`bioreason-pro-rl-paper` を初期値として、reasoning dataset の `train` split を使って SFT を行う。
canonical run は **stage 2 only** とし、comparison model に含まれる projector / GO module 重みをそのまま warm-start として使う。

### 4.2 入力

- temporal split artifact
  - `BIOREASON_MAIN_TEMPORAL_SPLIT_REGISTRY_PATH`
- reasoning dataset artifact
  - `BIOREASON_MAIN_REASONING_DATASET_REGISTRY_PATH`
- comparison model artifact
  - `BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH`

### 4.3 実行コマンド

```bash
cd ~/BioReason-Pro
source .venv-gpu/bin/activate

bash scripts/sh_train_protein_qwen_staged.sh
```

この wrapper は内部で `srun python train_protein_llm.py ...` を呼ぶ。

### 4.4 固定ルール

- 学習には `train` split を使う
- checkpoint selection には `validation` split を使う
- `test` split は使わない
- `job_type=train_sft`
- wall time は `12:00:00`
- canonical は `stage 2 only`
- `RUN_STAGE1=true` は fallback / ablation のときだけ使う

### 4.5 完了条件

W&B 上に次が揃っていれば完了とする。

- `job_type=train_sft`
- train / validation loss
- sample table
- output checkpoint artifact

完了後は `wandb_registry_paths.env` に次を追記する。

```bash
export BIOREASON_TRAIN_SFT_MODEL_REGISTRY_PATH="entity/project/train-sft-output:alias"
```

## 5. RL

状態: **準備済み、run は未実施**

### 5.1 目的

`train-sft-output` を canonical input として RL を行う。

### 5.2 固定ルール

- rollout / reward 最適化には `train` split を使う
- checkpoint selection と offline sanity-check には `validation` split を使う
- `test` split は使わない
- `job_type=train_rl`
- wall time は `12:00:00`
- `bioreason-pro-rl-paper` から直接 RL を始める経路は ablation のみ

### 5.3 実行コマンド

```bash
cd ~/BioReason-Pro
source .venv-gpu/bin/activate

bash scripts/sh_train_protein_grpo.sh
```

この wrapper は内部で `srun python train_protein_grpo.py ...` を呼ぶ。  
canonical には `BIOREASON_TRAIN_SFT_MODEL_REGISTRY_PATH` を使い、raw SFT checkpoint artifact しか無い場合は HF model へ変換してから RL を始める。

### 5.4 完了条件

次が揃っていれば完了とする。

- `job_type=train_rl`
- reward 系 metric
- KL 系 metric
- rollout trace
- output checkpoint artifact

完了後は `wandb_registry_paths.env` に次を追記する。

```bash
export BIOREASON_TRAIN_RL_MODEL_REGISTRY_PATH="entity/project/train-rl-output:alias"
```

## 6. 最終評価

状態: **未着手**

### 6.1 目的

`spec-comparison` を `test` split で評価し、最終比較を出す。

### 6.2 実行コマンド

```bash
srun \
  --partition <gpu_partition> \
  --account <account_name> \
  --gpus 1 \
  --cpus-per-task 8 \
  --mem 128G \
  --time 12:00:00 \
  bash -lc '
    cd ~/BioReason-Pro &&
    source .venv-gpu/bin/activate &&
    uv run --active python scripts/run_registered_eval.py \
      --target-group spec-comparison \
      --data-bundle main_production \
      --split test \
      --wandb-entity "$WANDB_ENTITY" \
      --wandb-project "$WANDB_PROJECT"
  '
```

### 6.3 完了条件

W&B 上に次が揃っていれば完了とする。

- comparison model と tuned model の `test` 指標
- `eval_summary` table
- `eval_samples` table
- 最終 eval artifact

## 7. 次アクション

いま一番近い next action は次の 2 つである。

1. CoreWeave 側で `.venv-gpu` と `wandb_registry_paths.env` を揃える
2. `comparison-family` を `validation` split で評価する
