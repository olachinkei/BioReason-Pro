# Disease Benchmark Research README

この README は、`domain/specification/busiless-rules/specification.md` に沿って、疾患ベンチマークの研究実装をどう進めるかを実行順に整理したものである。  
流れは **データの準備 -> GPU へのアクセス -> いろんなモデルの評価 -> SFT -> RL** とする。

前提は次の 3 点である。

- ローカル Mac では **データの準備だけ** を行う
- 学習と大規模評価は **CoreWeave GPU クラスター** に移ってから行う
- Python 環境は **uv** を前提にする

## 0. 命名と保存先

### 0.1 この README 内での呼び方

この README では次の呼び方を使う。

- `213-start main variant`: `213 -> 221 -> 225 -> 228`
- `214-start comparison variant`: `214 -> 221 -> 225 -> 228`

`214-start comparison variant` という呼び方は **この README 内だけの便宜上の名称** である。  
W&B Artifact 側では、両者は同じ artifact family の別 version / alias として同等に扱う。

### 0.2 ローカル保存先

ローカル生成物は `domain/specification/.../artifacts` には置かず、**`data/artifacts`** 配下に置く。

使うディレクトリは次で固定する。

```text
data/artifacts/
├── benchmarks/
│   ├── 213_221_225_228/
│   │   └── step0/
│   └── 214_221_225_228/
│       └── step0/
├── datasets/
│   ├── disease_temporal_hc_v1/
│   │   └── 213_221_225_228/
│   └── disease_temporal_hc_reasoning_v1/
│       └── 213_221_225_228/
└── eval/
    └── ...
```

### 0.3 W&B Artifact の family, alias, URL

Artifact family は variant 名を family 名に埋め込まず、同じ family の中で alias で区別する。

| 用途 | Artifact family | 主 alias | 補助 alias | URL |
|---|---|---|---|---|
| Step 0 benchmark | `disease-temporal-step0` | `213.221.225.228`, `production` | `214.221.225.228` | `https://wandb.ai/<entity>/<project>/artifacts/dataset/disease-temporal-step0` |
| Supervised dataset | `disease-temporal-supervised` | `213.221.225.228`, `production` | `214.221.225.228` | `https://wandb.ai/<entity>/<project>/artifacts/dataset/disease-temporal-supervised` |
| Reasoning dataset | `disease-temporal-reasoning` | `213.221.225.228`, `production` | `214.221.225.228` | `https://wandb.ai/<entity>/<project>/artifacts/dataset/disease-temporal-reasoning` |

運用ルール:

- `213.221.225.228` はベンチマーク版を表す共通 alias として使う
- 今回主に使う main variant には `production` alias を付ける
- `214.221.225.228` は comparison variant 用 alias として使う
- comparison variant には `production` を付けない

## 1. データの準備

この工程は **ローカル Mac** で行う。

### 1.1 uv でローカル Mac 用の軽量環境を作る

ローカル Mac では GPU 学習用のフル依存は入れず、Step 0 と artifact upload に必要なものだけ入れる。

```bash
cd /Users/keisuke/Project/learning/drug_discovery/BioReason-Pro

uv venv .venv-mac-data --python 3.11
source .venv-mac-data/bin/activate

uv pip install numpy pandas requests datasets cafaeval goatools wandb weave
```

### 1.2 ローカル保存先を作る

```bash
cd /Users/keisuke/Project/learning/drug_discovery/BioReason-Pro

mkdir -p data/artifacts/benchmarks/213_221_225_228/step0
mkdir -p data/artifacts/benchmarks/214_221_225_228/step0
mkdir -p data/artifacts/datasets/disease_temporal_hc_v1/213_221_225_228
mkdir -p data/artifacts/datasets/disease_temporal_hc_reasoning_v1/213_221_225_228
mkdir -p data/artifacts/eval
```

### 1.3 Step 0 を実装する

main variant:

```bash
cd /Users/keisuke/Project/learning/drug_discovery/BioReason-Pro

uv run --active python scripts/step0_disease_temporal_split.py \
  --output-dir data/artifacts/benchmarks/213_221_225_228/step0 \
  --train-start-release 213 \
  --train-end-release 221 \
  --dev-end-release 225 \
  --test-end-release 228 \
  --shortlist-mode high-confidence \
  --use-shell-filter
```

comparison variant:

```bash
cd /Users/keisuke/Project/learning/drug_discovery/BioReason-Pro

uv run --active python scripts/step0_disease_temporal_split.py \
  --output-dir data/artifacts/benchmarks/214_221_225_228/step0 \
  --train-start-release 214 \
  --train-end-release 221 \
  --dev-end-release 225 \
  --test-end-release 228 \
  --shortlist-mode high-confidence \
  --use-shell-filter
```

このコマンドを実行すると、少なくとも次がそれぞれの `step0/` ディレクトリに入る。

- `summary.json`
- `report.md`
- `train_assigned_labels.tsv`
- `dev_assigned_labels.tsv`
- `test_assigned_labels.tsv`
- `*_assigned_propagated.tsv`
- `*_assigned_nk_lk.tsv`
- `*_assigned_nk_lk_propagated.tsv`
- `nk_lk_eda.tsv`

### 1.4 Step 0 の sanity check

少なくとも次を確認する。

- `summary.json` に `split_validation.time_order_valid == true`
- `summary.json` に `split_validation.protein_disjoint_valid == true`
- `summary.json` に split ごとの protein 数が入っている
- `report.md` に split summary table がある

### 1.5 W&B に benchmark / dataset artifact を upload する

まず entity と project を決める。

```bash
export WANDB_ENTITY="<your-entity>"
export WANDB_PROJECT="bioreason-pro-disease-benchmark"
```

次の helper をそのまま shell に貼って使う。

```bash
upload_dir_artifact () {
  ARTIFACT_NAME="$1" \
  LOCAL_DIR="$2" \
  ARTIFACT_ALIASES="$3" \
  BENCHMARK_TAG="$4" \
  STEP0_ARTIFACT="$5" \
  uv run --active python - <<'PY'
import os
import wandb

artifact_name = os.environ["ARTIFACT_NAME"]
local_dir = os.environ["LOCAL_DIR"]
aliases = [x.strip() for x in os.environ["ARTIFACT_ALIASES"].split(",") if x.strip()]
benchmark_tag = os.environ["BENCHMARK_TAG"]
step0_artifact = os.environ["STEP0_ARTIFACT"]

run = wandb.init(
    entity=os.environ["WANDB_ENTITY"],
    project=os.environ["WANDB_PROJECT"],
    job_type="data_prep",
    name=f"upload-{artifact_name}-{benchmark_tag}",
    config={
        "benchmark_tag": benchmark_tag,
        "local_dir": local_dir,
        "step0_artifact": step0_artifact,
    },
)

artifact = wandb.Artifact(
    artifact_name,
    type="dataset",
    metadata={
        "benchmark_tag": benchmark_tag,
        "local_dir": local_dir,
        "step0_artifact": step0_artifact,
    },
)
artifact.add_dir(local_dir)
run.log_artifact(artifact, aliases=aliases)
run.finish()
PY
}
```

main variant の Step 0 benchmark を upload:

```bash
upload_dir_artifact \
  "disease-temporal-step0" \
  "data/artifacts/benchmarks/213_221_225_228/step0" \
  "213.221.225.228,production" \
  "213.221.225.228" \
  "data/artifacts/benchmarks/213_221_225_228/step0"
```

comparison variant の Step 0 benchmark を upload:

```bash
upload_dir_artifact \
  "disease-temporal-step0" \
  "data/artifacts/benchmarks/214_221_225_228/step0" \
  "214.221.225.228" \
  "214.221.225.228" \
  "data/artifacts/benchmarks/214_221_225_228/step0"
```

supervised dataset と reasoning dataset をローカルに生成したら、同じ helper で upload する。

supervised dataset:

```bash
upload_dir_artifact \
  "disease-temporal-supervised" \
  "data/artifacts/datasets/disease_temporal_hc_v1/213_221_225_228" \
  "213.221.225.228,production" \
  "213.221.225.228" \
  "data/artifacts/benchmarks/213_221_225_228/step0"
```

reasoning dataset:

```bash
upload_dir_artifact \
  "disease-temporal-reasoning" \
  "data/artifacts/datasets/disease_temporal_hc_reasoning_v1/213_221_225_228" \
  "213.221.225.228,production" \
  "213.221.225.228" \
  "data/artifacts/benchmarks/213_221_225_228/step0"
```

### 1.6 contract test

ローカル Mac で実装を触ったあとに contract test を回す場合は、別途 test 用 uv 環境を作る。

```bash
cd /Users/keisuke/Project/learning/drug_discovery/BioReason-Pro

uv venv .venv-test --python 3.11
source .venv-test/bin/activate
uv pip install torch numpy pandas requests datasets cafaeval goatools colorama wandb weave

uv run --active python -m unittest discover -s test -v
```

## 2. GPU へのアクセス

評価と学習は **CoreWeave GPU クラスター** に移ってから行う。

### 2.1 login

```bash
ssh -o IdentitiesOnly=yes kkamata+cwb607@sunk.cwb607-training.coreweave.app
```

注意:

- 長時間 job を login node で回さない
- 学習・評価は GPU node 上で回す
- 終了後は node を解放する

### 2.2 ローカル Mac からコードと data を送る

ローカル Mac 側で実行する。

```bash
cd /Users/keisuke/Project/learning/drug_discovery

rsync -av --delete \
  BioReason-Pro/ \
  kkamata+cwb607@sunk.cwb607-training.coreweave.app:~/BioReason-Pro/
```

### 2.3 CoreWeave 上で uv 環境を作る

CoreWeave 側ではフル実装用の環境を作る。

```bash
cd ~/BioReason-Pro

uv venv .venv-gpu --python 3.11
source .venv-gpu/bin/activate

uv sync
uv pip install esm --no-deps
uv pip install flash-attn --no-build-isolation --no-cache-dir
uv pip install unsloth
```

### 2.4 GPU node に入る

実際の partition 名や account 名は自分の環境に合わせて置き換える。

```bash
srun \
  --partition <gpu_partition> \
  --account <account_name> \
  --gpus 1 \
  --cpus-per-task 8 \
  --mem 128G \
  --time 12:00:00 \
  --pty bash
```

## 3. いろんなモデルの評価

この工程は **CoreWeave の GPU node 上** で行う。

評価対象の基本セット:

- `bioreason-pro-base`
- `bioreason-pro-sft`
- `bioreason-pro-rl`

### 3.1 先に設定するもの

`scripts/sh_eval.sh` は、次の値を **環境変数で上書きできる**。

- `MODEL_PATH`
- `GO_OBO_PATH`
- `GO_EMBEDDINGS_PATH`
- `DATASET_CACHE_DIR`
- `STRUCTURE_DIR`
- `DATASET_NAME`
- `REASONING_DATASET_NAME`
- `EVALS_DIR`

推奨する出力先:

- `data/artifacts/eval/base/validation`
- `data/artifacts/eval/base/test`
- `data/artifacts/eval/sft/validation`
- `data/artifacts/eval/sft/test`
- `data/artifacts/eval/rl/validation`
- `data/artifacts/eval/rl/test`

### 3.2 base model を評価する

```bash
cd ~/BioReason-Pro
source .venv-gpu/bin/activate

mkdir -p data/artifacts/eval/base/validation

EVALS_DIR="data/artifacts/eval/base" \
MODEL_PATH="/path/to/bioreason-pro-base" \
GO_OBO_PATH="/path/to/go-basic.obo" \
GO_EMBEDDINGS_PATH="/path/to/go-embeddings" \
DATASET_CACHE_DIR="/path/to/hf-cache" \
STRUCTURE_DIR="/path/to/structures" \
DATASET_NAME="disease_temporal_hc_reasoning_v1" \
REASONING_DATASET_NAME="disease_temporal_hc_reasoning_v1" \
EVAL_SPLIT=validation \
BENCHMARK_VERSION="213 -> 221 -> 225 -> 228" \
MODEL_NAME="bioreason-pro-base" \
WANDB_PROJECT="bioreason-pro-disease-benchmark" \
WANDB_ENTITY="<your-entity>" \
WANDB_RUN_NAME="eval-base-validation-213.221.225.228" \
WANDB_ARTIFACT_NAME="eval-base-validation-213.221.225.228" \
WEAVE_PROJECT="<your-entity>/bioreason-pro-disease-benchmark" \
WEAVE_EVAL_NAME="eval-base-validation-213.221.225.228" \
bash scripts/sh_eval.sh
```

### 3.3 SFT model を評価する

```bash
cd ~/BioReason-Pro
source .venv-gpu/bin/activate

mkdir -p data/artifacts/eval/sft/test

EVALS_DIR="data/artifacts/eval/sft" \
MODEL_PATH="/path/to/bioreason-pro-sft" \
GO_OBO_PATH="/path/to/go-basic.obo" \
GO_EMBEDDINGS_PATH="/path/to/go-embeddings" \
DATASET_CACHE_DIR="/path/to/hf-cache" \
STRUCTURE_DIR="/path/to/structures" \
DATASET_NAME="disease_temporal_hc_reasoning_v1" \
REASONING_DATASET_NAME="disease_temporal_hc_reasoning_v1" \
EVAL_SPLIT=test \
BENCHMARK_VERSION="213 -> 221 -> 225 -> 228" \
MODEL_NAME="bioreason-pro-sft" \
WANDB_PROJECT="bioreason-pro-disease-benchmark" \
WANDB_ENTITY="<your-entity>" \
WANDB_RUN_NAME="eval-sft-test-213.221.225.228" \
WANDB_ARTIFACT_NAME="eval-sft-test-213.221.225.228" \
WEAVE_PROJECT="<your-entity>/bioreason-pro-disease-benchmark" \
WEAVE_EVAL_NAME="eval-sft-test-213.221.225.228" \
bash scripts/sh_eval.sh
```

### 3.4 RL model を評価する

```bash
cd ~/BioReason-Pro
source .venv-gpu/bin/activate

mkdir -p data/artifacts/eval/rl/test

EVALS_DIR="data/artifacts/eval/rl" \
MODEL_PATH="/path/to/bioreason-pro-rl" \
GO_OBO_PATH="/path/to/go-basic.obo" \
GO_EMBEDDINGS_PATH="/path/to/go-embeddings" \
DATASET_CACHE_DIR="/path/to/hf-cache" \
STRUCTURE_DIR="/path/to/structures" \
DATASET_NAME="disease_temporal_hc_reasoning_v1" \
REASONING_DATASET_NAME="disease_temporal_hc_reasoning_v1" \
EVAL_SPLIT=test \
BENCHMARK_VERSION="213 -> 221 -> 225 -> 228" \
MODEL_NAME="bioreason-pro-rl" \
WANDB_PROJECT="bioreason-pro-disease-benchmark" \
WANDB_ENTITY="<your-entity>" \
WANDB_RUN_NAME="eval-rl-test-213.221.225.228" \
WANDB_ARTIFACT_NAME="eval-rl-test-213.221.225.228" \
WEAVE_PROJECT="<your-entity>/bioreason-pro-disease-benchmark" \
WEAVE_EVAL_NAME="eval-rl-test-213.221.225.228" \
bash scripts/sh_eval.sh
```

### 3.5 F_max を集計する

各評価ディレクトリについて、個別 JSON を出したあとに CAFA metric を集計する。

```bash
cd ~/BioReason-Pro
source .venv-gpu/bin/activate

uv run --active python evals/cafa_evals.py \
  --input_dir data/artifacts/eval/sft/results \
  --ontology /path/to/go-basic.obo \
  --ia_file /path/to/IA.txt \
  --output_dir data/artifacts/eval/sft/cafa_metrics \
  --reasoning_mode True \
  --final_answer_only True
```

その後、必要なら `METRICS_SUMMARY_PATH` を渡して再度 eval tracking を行う。

```bash
cd ~/BioReason-Pro
source .venv-gpu/bin/activate

EVALS_DIR="data/artifacts/eval/sft" \
MODEL_PATH="/path/to/bioreason-pro-sft" \
GO_OBO_PATH="/path/to/go-basic.obo" \
GO_EMBEDDINGS_PATH="/path/to/go-embeddings" \
DATASET_CACHE_DIR="/path/to/hf-cache" \
STRUCTURE_DIR="/path/to/structures" \
DATASET_NAME="disease_temporal_hc_reasoning_v1" \
REASONING_DATASET_NAME="disease_temporal_hc_reasoning_v1" \
EVAL_SPLIT=test \
BENCHMARK_VERSION="213 -> 221 -> 225 -> 228" \
MODEL_NAME="bioreason-pro-sft" \
METRICS_SUMMARY_PATH="data/artifacts/eval/sft/cafa_metrics/metrics_summary.json" \
WANDB_PROJECT="bioreason-pro-disease-benchmark" \
WANDB_ENTITY="<your-entity>" \
WANDB_RUN_NAME="eval-sft-test-213.221.225.228-with-metrics" \
WANDB_ARTIFACT_NAME="eval-sft-test-213.221.225.228-artifacts" \
WEAVE_PROJECT="<your-entity>/bioreason-pro-disease-benchmark" \
WEAVE_EVAL_NAME="eval-sft-test-213.221.225.228-with-metrics" \
bash scripts/sh_eval.sh
```

## 4. SFT

この工程は **CoreWeave の GPU node 上** で行う。

### 4.1 先に埋めるもの

`scripts/sh_train_protein_qwen_staged.sh` の上部で次を埋める。

- `BASE_CHECKPOINT_DIR`
- `DATASET_CACHE_DIR`
- `CACHE_DIR`
- `STRUCTURE_DIR`
- `GO_EMBEDDINGS_PATH`
- `GO_OBO_PATH`
- `DATASET_ARTIFACT`
- `BASE_CHECKPOINT`

特に `STEP0_ARTIFACT` と `DATASET_ARTIFACT` は、ローカル Mac で upload した W&B artifact に合わせる。

推奨値:

- `STEP0_ARTIFACT="disease-temporal-step0:production"`
- `DATASET_ARTIFACT="disease-temporal-reasoning:production"`

### 4.2 実行コマンド

```bash
cd ~/BioReason-Pro
source .venv-gpu/bin/activate

bash scripts/sh_train_protein_qwen_staged.sh
```

この wrapper は次を行う。

1. Stage 1: projector / GO module warm-up
2. Stage 2: full model fine-tuning

### 4.3 実行後に確認するもの

- checkpoint directory
- W&B run config
- train / validation loss
- sample generation table
- output checkpoint artifact

W&B 上で最低限確認するもの:

- `job_type=train_sft`
- `benchmark_version`
- `step0_artifact`
- `dataset_config`
- `reasoning_dataset_config`
- `dataset_artifact`
- `model_artifact`
- `job_time_limit=12:00:00`

## 5. RL

RL は flow 上は最後に置くが、**現時点では repo に実行 entry point がまだ揃っていない**。  
そのため、今すぐ回す対象は Step 0, eval, SFT までである。

RL 着手時の前提だけ先に固定する。

- benchmark version は `213 -> 221 -> 225 -> 228`
- benchmark alias は `213.221.225.228`
- main run にだけ `production` alias を使う
- RL 用 dataset は `train` split 由来のみ
- `validation` は checkpoint selection 用
- `test` は最終評価専用

RL 実装が入ったら、README には次を追加する。

- RL dataset の生成コマンド
- RL rollout の W&B / Weave tracking コマンド
- RL checkpoint artifact の upload / alias 付け

## 6. 最短実行順

迷ったら次の順に進める。

1. ローカル Mac で `uv` 環境を作る
2. `data/artifacts` を作る
3. Step 0 を main variant / comparison variant で実行する
4. Step 0 を W&B Artifact に upload する
5. CoreWeave に repo と data を送る
6. CoreWeave 上で `uv` 環境を作る
7. base / sft / rl を順に評価する
8. SFT を回す
9. RL entry point が入ったら RL に進む
