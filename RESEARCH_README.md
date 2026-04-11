# Disease Benchmark Research README

この README は、[specification.md](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/domain/specification/busiless-rules/specification.md) に沿って、**現在採用する運用だけ**を実行順にまとめたものである。  
流れは **データの準備 -> GPU へのアクセス -> 比較モデルの評価 -> RL -> 最終評価** とする。

前提:

- ローカル Mac では **データの準備だけ**を行う
- 学習と GPU 評価は **CoreWeave SUNK** で行う
- Python 環境は **uv** を前提にする
- local filesystem 上の生成物は scratch とみなし、**正本は W&B Artifact ref**

## 0. 命名と正本

### 0.1 現在採用する benchmark

現在採用する benchmark は次で固定する。

- benchmark version: `train=213->225, future=225->228, dev=200, holdout=400`
- benchmark alias: `213.221.225.228`
- data bundle: `main_production`
- comparison model: `bioreason-pro-rl-paper`
- primary dataset: `disease_temporal_hc_reasoning_v1`
- W&B project: `bioreasoning-pro`

この README でいう `temporal split artifact` は、release 差分、protein-disjoint split、label assignment を固定した benchmark artifact を指す。  
学習や評価で直接読む dataset は、そこから派生した reasoning dataset である。

### 0.2 現在の正本 Artifact ref

現在の正本は次で固定する。

| 用途 | W&B Artifact ref |
|---|---|
| temporal split artifact | `wandb-healthcare/bioreasoning-pro/disease-temporal-split:production` |
| reasoning dataset artifact | `wandb-healthcare/bioreasoning-pro/disease-temporal-reasoning:production` |
| IA file artifact | `wandb-healthcare/bioreasoning-pro/disease-temporal-ia:production` |
| comparison model artifact | `wandb-healthcare/bioreasoning-pro/bioreason-pro-rl-paper:production` |

local の build / download 結果は scratch であり、source-of-truth ではない。

### 0.3 W&B Artifact ref manifest

ここで扱うのは **W&B Artifact ref** である。  
repo 内の JSON は artifact 自体ではなく、**どの W&B Artifact ref を使うかを束ねる manifest** として扱う。

使うファイルは次で固定する。

- data-bundle manifest: [data_registry.json](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/configs/disease_benchmark/data_registry.json)
- evaluation-target manifest: [eval_target_registry.json](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/configs/disease_benchmark/eval_target_registry.json)
- asset publish manifest: [artifact_publish_registry.json](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/configs/disease_benchmark/artifact_publish_registry.json)
- artifact ref env template: [wandb_registry_paths.env.example](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/configs/disease_benchmark/wandb_registry_paths.env.example)
- local env file: [wandb_registry_paths.env](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/configs/disease_benchmark/wandb_registry_paths.env)
- source env template: [wandb_asset_sources.env.example](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/configs/disease_benchmark/wandb_asset_sources.env.example)

役割は次のとおりである。

- `wandb_registry_paths.env`: temporal split artifact, reasoning dataset, comparison model, 学習後の model output の W&B Artifact ref を入れる
- `wandb_asset_sources.env`: Hugging Face repo ID など、publish 前の source を入れる
- `data_registry.json`: `main_production` bundle が使う temporal split artifact と reasoning dataset を束ねる
- `eval_target_registry.json`: `comparison-family`, `tuned-family`, `spec-comparison` の target group を束ねる

Artifact ref は browser URL ではなく、`entity/project/artifact_name:alias` 形式を使う。

初期状態で人が明示的に用意する ref は次の 4 つだけでよい。

| env var | 用途 | 現在の ref |
|---|---|---|
| `BIOREASON_MAIN_TEMPORAL_SPLIT_REGISTRY_PATH` | main temporal split artifact | `wandb-healthcare/bioreasoning-pro/disease-temporal-split:production` |
| `BIOREASON_MAIN_REASONING_DATASET_REGISTRY_PATH` | main reasoning dataset artifact | `wandb-healthcare/bioreasoning-pro/disease-temporal-reasoning:production` |
| `BIOREASON_MAIN_IA_REGISTRY_PATH` | main IA file artifact | `wandb-healthcare/bioreasoning-pro/disease-temporal-ia:production` |
| `BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH` | tuning 前の比較モデル | `wandb-healthcare/bioreasoning-pro/bioreason-pro-rl-paper:production` |

`BIOREASON_TRAIN_RL_MODEL_REGISTRY_PATH` は、RL run の完了後に成果物として決まる。

## 1. データの準備

この工程は **ローカル Mac** で行う。

### 1.1 uv 環境を作る

ローカル Mac 側では、準備済みの uv 用ファイルを使う。

- data prep 用: [uv-local-data.txt](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/requirements/uv-local-data.txt)
- contract test 用: [uv-contract-tests.txt](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/requirements/uv-contract-tests.txt)

```bash
cd /Users/keisuke/Project/learning/drug_discovery/BioReason-Pro

uv venv .venv-mac-data --python 3.11
source .venv-mac-data/bin/activate
uv pip install -r requirements/uv-local-data.txt
```

### 1.2 temporal split artifact と reasoning dataset を一気に作って upload する

まず `.env` を読み込む。

```bash
cd /Users/keisuke/Project/learning/drug_discovery/BioReason-Pro

set -a
source .env
set +a
```

次の 1 本で、temporal split artifact build、sanity check、reasoning dataset build、W&B upload をまとめて回す。

```bash
uv run --active python scripts/run_temporal_split_artifact_pipeline.py \
  --variant main \
  --shortlist-mode high-confidence \
  --use-shell-filter \
  --build-datasets \
  --upload-to-wandb \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-project "$WANDB_PROJECT"
```

この orchestration は少なくとも次を行う。

1. `scripts/build_disease_temporal_split_artifact.py` を実行する
2. `summary.json` と `report.md` を用いて split sanity check を行う
3. `scripts/build_disease_benchmark_datasets.py` で reasoning dataset を作る
4. `disease-temporal-split` と `disease-temporal-reasoning` を W&B Artifact に upload する
5. `pipeline_status.json` を書く

sanity check で必ず通っているべき項目は次である。

- `split_validation.time_order_valid == true`
- `split_validation.protein_disjoint_valid == true`
- `train`, `dev`, `test` の protein 数が `summary.json` に入っている

temporal split artifact の必須成果物は次である。

- `summary.json`
- `report.md`
- `train_assigned_labels.tsv`
- `dev_assigned_labels.tsv`
- `test_assigned_labels.tsv`
- `*_assigned_propagated.tsv`
- `pipeline_status.json`

W&B upload が成功したら、local 側の生成物は scratch とみなし、恒久保存を前提にしない。

### 1.3 実行後に確認すること

確認先は local ではなく **W&B** を正本とする。

最低限、次の 3 つが見えていることを確認する。

- `wandb-healthcare/bioreasoning-pro/disease-temporal-split:production`
- `wandb-healthcare/bioreasoning-pro/disease-temporal-reasoning:production`
- `benchmark_alias=213.221.225.228`

## 2. GPU へのアクセス

評価と学習は **CoreWeave SUNK** で行う。  
GPU node に直接入るのではなく、**login node から `srun` で送る**運用に固定する。

### 2.1 login node に入る

```bash
ssh -o IdentitiesOnly=yes kkamata+cwb607@sunk.cwb607-training.coreweave.app
```

### 2.2 コードだけ送る

ローカル Mac 側で実行する。

```bash
cd /Users/keisuke/Project/learning/drug_discovery

rsync -av --delete \
  --exclude 'data/artifacts/' \
  --exclude '.venv*/' \
  BioReason-Pro/ \
  kkamata+cwb607@sunk.cwb607-training.coreweave.app:~/BioReason-Pro/
```

`data/artifacts` は scratch なので送らない。  
CoreWeave 側では W&B Artifact ref から data と model を取得する。

### 2.3 CoreWeave 側で uv 環境を作る

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

### 2.4 env file を用意する

```bash
cd ~/BioReason-Pro

set -a
source .env
set +a

cp configs/disease_benchmark/wandb_registry_paths.env.example \
  configs/disease_benchmark/wandb_registry_paths.env

cp configs/disease_benchmark/wandb_asset_sources.env.example \
  configs/disease_benchmark/wandb_asset_sources.env

$EDITOR configs/disease_benchmark/wandb_registry_paths.env
$EDITOR configs/disease_benchmark/wandb_asset_sources.env

export BIOREASON_GO_EMBEDDINGS_PATH="/path/to/go-embeddings"
export BIOREASON_IA_FILE_PATH="/path/to/IA.txt"
export BIOREASON_STRUCTURE_DIR="$HOME/BioReason-Pro/data/structures"
export BIOREASON_DATASET_CACHE_DIR="$HOME/BioReason-Pro/data/artifacts/hf_cache"
```

通常この段階で入れるべき Artifact ref は次の 4 つだけでよい。

- `BIOREASON_MAIN_TEMPORAL_SPLIT_REGISTRY_PATH`
- `BIOREASON_MAIN_REASONING_DATASET_REGISTRY_PATH`
- `BIOREASON_MAIN_IA_REGISTRY_PATH`
- `BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH`

### 2.5 比較モデルを一度 W&B Artifact に固定する

`BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH` が既に `wandb-healthcare/bioreasoning-pro/bioreason-pro-rl-paper:production` を指しているなら、この工程はスキップしてよい。  
未 publish の場合だけ、一度 materialize して W&B Artifact に固定する。

```bash
cd ~/BioReason-Pro
source .venv-gpu/bin/activate

uv run --active python scripts/register_research_assets.py \
  --asset bioreason-pro-rl-paper \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-project "$WANDB_PROJECT"
```

`configs/disease_benchmark/wandb_asset_sources.env` には次を入れる。

```bash
export BIOREASON_PRO_RL_HF_REPO="wanglab/bioreason-pro-rl"
```

### 2.6 `srun` の基本形

```bash
srun \
  --partition <gpu_partition> \
  --account <account_name> \
  --gpus 1 \
  --cpus-per-task 8 \
  --mem 128G \
  --time 12:00:00 \
  bash -lc '
    cd ~/BioReason-Pro
    source .venv-gpu/bin/activate
    <your command here>
  '
```

## 3. 比較モデルの評価

高位 entry point は [run_registered_eval.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/scripts/run_registered_eval.py) に固定する。  
低位 wrapper の [sh_eval.sh](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/scripts/sh_eval.sh) は直接触らない前提で進める。

### 3.1 評価対象と split

現在の評価対象は次である。

- `bioreason-pro-rl-paper`
- `train-rl-output` がある場合はそれも含める

target family は manifest 上で次を使う。

- `comparison-family`: `bioreason-pro-rl-paper`
- `tuned-family`: `train-rl-output`
- `spec-comparison`: 上記すべて

split の使い分けは次で固定する。

- 開発中の比較、ablation、checkpoint 比較: `validation`
- `validation` は artifact 上で固定された **200-protein dev split** をそのまま使う
- 最終報告値: separate run の `test`
- `test` は artifact 上で固定された **400-protein holdout split** をそのまま使う

### 3.2 比較モデルを `validation` で評価する

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

### 3.3 1 target だけ評価する

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
      --target bioreason-pro-rl-paper \
      --data-bundle main_production \
      --split validation \
      --wandb-entity "$WANDB_ENTITY" \
      --wandb-project "$WANDB_PROJECT"
  '
```

### 3.4 評価で W&B に保存されるもの

eval run では、`wandb.init(..., job_type="eval")` と `weave.init(...)` を開始時に行う。  
そのうえで各 sample の推論は `@weave.op` で trace し、run の最後に Weave Evaluation を記録する。

`validation` run では artifact 上の **200-protein dev split** をそのまま使う。  
`test` run では artifact 上の **400-protein holdout split** をそのまま使う。

すべての eval run で次を W&B に保存する。

- `fmax_mf`
- `fmax_bp`
- `fmax_cc`
- `overall_mean_fmax`
- `eval_summary` table
- `eval_samples` table
- Weave evaluation record

この一式が揃わなかった run は失敗として扱う。

local eval 出力は scratch とみなし、W&B 保存成功後は既定で cleanup される。  
local に残したいときだけ `--keep-local-eval-outputs` を付ける。

## 4. RL

RL は `train_rl` phase として扱う。  
canonical input は `bioreason-pro-rl-paper` checkpoint である。

entry point は [train_protein_grpo.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/train_protein_grpo.py) と [sh_train_protein_grpo.sh](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/scripts/sh_train_protein_grpo.sh) に固定する。production の公式 runner は [run_registered_train_rl.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/scripts/run_registered_train_rl.py) とする。  
運用ルールは次で固定する。

- RL の rollout / reward 最適化には benchmark の `train` split を使う
- checkpoint selection と offline sanity-check には artifact 上の `validation` split を使う
- `validation` split は fixed **200-protein dev split**
- `test` split は RL 学習に使わない
- RL 用派生 dataset を作る場合も、元データは `train` split のみから作る
- canonical input は `BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH` が指す `bioreason-pro-rl-paper` artifact

`scripts/sh_train_protein_grpo.sh` は wrapper resolves を採用し、既定では次を自動で解決する。

- `BASE_CHECKPOINT` は `BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH`
- `CAFA5_DATASET` は `main_production` bundle の `reasoning_dataset`
- `IA_FILE_PATH` は `main_production` bundle の `ia_file`
- `TEMPORAL_SPLIT_ARTIFACT` と `DATASET_ARTIFACT` は `main_production` bundle

明示 env がある場合は registry env file よりそちらを優先する。`orchestrate_best_sft_to_rl_eval.py` は ablation / 旧系導線であり、本番既定には使わない。

### 4.1 実行コマンド

```bash
cd ~/BioReason-Pro
source .venv-gpu/bin/activate
export REGISTRY_ENV_FILE=configs/disease_benchmark/wandb_registry_paths.env
export WANDB_ENTITY=wandb-healthcare
export WANDB_PROJECT=bioreasoning-pro
export NNODES=2
export GPUS_PER_NODE=8
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500
export NODE_RANK=0

bash scripts/sh_train_protein_grpo.sh --preflight_only true
```

preflight が通ったら、本番 launch は production runner から行う。

```bash
cd ~/BioReason-Pro
source .venv-gpu/bin/activate

python scripts/run_registered_train_rl.py \
  --registry-env-file configs/disease_benchmark/wandb_registry_paths.env \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-project "$WANDB_PROJECT" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  --node-rank "$NODE_RANK"
```

`HOSTFILE` を使う環境では `--hostfile /path/to/hosts` を使う。  
manual launch が必要なときだけ、同じ env を与えた状態で `bash scripts/sh_train_protein_grpo.sh` を直接呼ぶ。  
この wrapper は exact `2 nodes x 8 GPUs` を production 既定とし、通常実行時は内部で `deepspeed train_protein_grpo.py ...` を呼ぶ。  
production default では `queries_per_step=8`, `rollouts_per_query=24`, `gradient_accumulation_steps=2` とし、`16 ranks` を `2 ranks/query` で使う。  
これにより global shape は `8 proteins x 24 rollouts = 192 trajectories` を維持しつつ、hardware shape だけが paper の `8 GPU` から `16 GPU` へ拡張される。  
`run_registered_train_rl.py` が成功した場合は `BIOREASON_TRAIN_RL_MODEL_REGISTRY_PATH` を `configs/disease_benchmark/wandb_registry_paths.env` に自動で追記する。

実GPU向けの既定 runtime adaptation は次である。

- `rollout_backend=subprocess`
- `vllm_enable_sleep_mode=false`
- `max_new_tokens=10000`
- `dataset_num_proc=1` during distributed preprocessing
- `rollout_logprob_microbatch_size=4`

より保守的な smoke run にしたい場合だけ、wrapper 呼び出し前に次の env を上書きしてよい。

```bash
export ROLLOUT_LOGPROB_MICROBATCH_SIZE=2
export MAX_LOSS_COMPLETION_TOKENS=1024
export MAX_NEW_TOKENS=1024
export VLLM_GPU_MEMORY_UTILIZATION=0.25
export VLLM_CPU_OFFLOAD_GB=8
```

`MAX_LOSS_COMPLETION_TOKENS` は rollout 観測長ではなく loss に載せる completion 長の上限であり、`0` のときは full completion をそのまま使う。

### 4.2 RL 後の評価

その後は `spec-comparison` を `test` split の **separate eval run** で評価する。

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

## 5. 最短実行順

迷ったら次の順に進める。

1. ローカル Mac で `uv` 環境を作る
2. `scripts/run_temporal_split_artifact_pipeline.py` を `--variant main --build-datasets --upload-to-wandb` で実行する
3. GPU 環境にコードを送る
4. GPU 環境上で `uv` 環境を作り、`wandb_registry_paths.env` を用意する
5. 必要なら `bioreason-pro-rl-paper` を一度 W&B Artifact に固定する
6. `comparison-family` を `validation` で評価する
7. `bioreason-pro-rl-paper` を初期値にして `bash scripts/sh_train_protein_grpo.sh --preflight_only true` を通す
8. `python scripts/run_registered_train_rl.py ...` で RL を実行し、`BIOREASON_TRAIN_RL_MODEL_REGISTRY_PATH` を更新する
9. `tuned-family` を `validation` で評価する
10. `spec-comparison` を `test` で評価する
