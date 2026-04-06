# PLAN

この PLAN は、[specification.md](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/domain/specification/busiless-rules/specification.md) を実装に落とすための実行順序を整理したものである。  
以後、仕様の正本は `domain/specification/busiless-rules/specification.md` とし、本書はその実装計画として扱う。

## 1. 実装の前提

まず固定してから実装する前提は次である。

- benchmark version は `213 -> 221 -> 225 -> 228`
- shortlist mode は `high-confidence`
- split の意味は `train / validation / test`
- 文中の `dev` は dataset split 名 `validation` と同義
- train / validation / test は protein-disjoint かつ temporal order を守る
- SFT と RL は別 split を作らず、同じ benchmark の `train / validation / test` を共有する
- SFT は reasoning dataset を使う
- RL は同じ `train` split から派生した reward / preference データを使う
- W&B run の `job_type` は `eval`, `train_sft`, `train_rl`
- 学習ジョブの time limit は `12:00:00`

## 2. 先に作るテスト

実装に先立って、仕様のうち既にコード化されている business rule を `test/` に固定する。  
これは実装中に benchmark 定義が崩れないようにするためである。

今回先に固定するテスト対象は次である。

- `scripts/step0_disease_temporal_split.py` の default 引数
- shortlist mode ごとの query
- temporal delta の定義
- earliest appearance による protein-disjoint split
- split integrity validation
- NK / LK 判定
- Step 0 markdown report の必須要素

追加済みのテストファイル:

- [test_step0_disease_temporal_split.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/test/test_step0_disease_temporal_split.py)

実行コマンド:

```bash
.venv-step0/bin/python -m unittest discover -s test -v
```

## 3. 実装順序

### 3.1 Step 0 を current implementation version で固定する

対象ファイル:

- [step0_disease_temporal_split.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/scripts/step0_disease_temporal_split.py)

やること:

- `213 -> 221 -> 225 -> 228` を current implementation version として維持する
- shortlist mode の default を `high-confidence` に保つ
- `summary.json` に `split_validation` を必ず残す
- `nk_lk_eda.tsv` を必ず出力する
- `earliest_split_by_protein.json` を必ず出力する

実行コマンド:

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

受け入れ基準:

- `summary.json` に `split_validation.time_order_valid == true`
- `summary.json` に `split_validation.protein_disjoint_valid == true`
- `report.md` に split summary table がある
- `train_assigned_labels.tsv`, `dev_assigned_labels.tsv`, `test_assigned_labels.tsv` がある
- `nk_lk_eda.tsv` がある

### 3.2 Step 0 artifact から dataset を作る

対象ファイル:

- 新規 script を追加する場合は `scripts/build_disease_temporal_datasets.py`
- dataset loader を追加する場合は `bioreason2/dataset/` 配下

やること:

- Step 0 artifact から supervised dataset を作る
- Step 0 artifact から reasoning dataset を作る
- `train / validation / test` の split 名で保存する
- `protein_id` と `split` が supervised / reasoning dataset で一致することを保証する
- optional field は欠損列にせず空文字列で埋める

固定する dataset config 名:

- `disease_temporal_hc_v1`
- `disease_temporal_hc_reasoning_v1`

最低限入れる列:

- `protein_id`
- `sequence`
- `organism`
- `go_bp`
- `go_mf`
- `go_cc`
- `protein_function`
- `go_pred`
- `interpro_formatted`
- `ppi_formatted`
- reasoning dataset では追加で `reasoning`, `final_answer`

受け入れ基準:

- supervised dataset と reasoning dataset の split ごとの `protein_id` 集合が一致する
- `test` に入っている protein が `train` / `validation` に存在しない
- dataset が `load_cafa5_dataset()` 経由で読める形になっている

### 3.3 RL 用の派生 dataset を作る

対象ファイル:

- 新規 script を追加する場合は `scripts/build_disease_temporal_rl_dataset.py`

やること:

- RL 用の reward / preference dataset は `train` split からのみ作る
- `protein_id` を保持する
- `benchmark_version` を保持する
- 元の reasoning dataset と join 可能な形にする

推奨列:

- `protein_id`
- `split`
- `prompt`
- `response`
- `reward`
- `reward_components`
- `notes`

受け入れ基準:

- RL dataset に `validation` / `test` 由来の sample が入っていない
- `protein_id` で元 dataset に戻れる
- reward の内訳を後から監査できる

### 3.4 W&B と Weave の共通 tracking を入れる

対象ファイル:

- [eval.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/eval.py)
- [train_protein_llm.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/train_protein_llm.py)
- 必要なら共通 helper を `bioreason2/utils/` に追加

やること:

- 各 run で `job_type` を明示する
- dataset artifact / model artifact / step0 artifact を config に残す
- `benchmark_version`, `shortlist_mode`, release anchor を config に残す
- W&B Artifact の alias を run config に残す
- eval では `weave.Evaluation` を使って score と trace を保存する
- RL rollout は Weave trace で追えるようにする

最低限 W&B config に入れるもの:

- `job_type`
- `benchmark_version`
- `step0_artifact`
- `dataset_config`
- `reasoning_dataset_config`
- `dataset_artifact`
- `shortlist_query`
- `shortlist_mode`
- `train_start_release`
- `train_end_release`
- `dev_end_release`
- `test_end_release`
- `base_checkpoint`
- `model_artifact`
- `output_dir`
- `seed`
- `learning_rate`
- `batch_size`
- `gradient_accumulation_steps`
- `num_train_epochs`
- `job_time_limit`

受け入れ基準:

- W&B run page だけで benchmark version と dataset lineage が追える
- sample table から artifact version に戻れる
- eval と RL の reasoning trace が Weave 側から辿れる

### 3.5 eval を仕様どおりに直す

対象ファイル:

- [eval.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/eval.py)
- [sh_eval.sh](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/scripts/sh_eval.sh)

現状のギャップ:

- `eval.py` は現状 `validation` split を固定で読んでいる
- `test` split を最終報告用に明示的に扱えていない
- `wandb.init(job_type="eval")` と Weave logger がまだ実装されていない

やること:

- `--eval_split` のような引数を追加し、`validation` と `test` を切り替えられるようにする
- 開発中の比較は `validation`、最終報告は `test` を使う
- metric を `wandb.log()` する
- summary table を 1 evaluated target 1 row で `wandb.log()` する
- sample-level table を 1 sample 1 row で `wandb.log()` する
- reasoning task では `reasoning_full`, `final_answer`, `intermediate_trace` を保存する
- JSON 結果を Artifact として version 管理する
- `weave.Evaluation` の eval logger でも追跡する

受け入れ基準:

- 同じ checkpoint に対して `validation` と `test` を分けて評価できる
- W&B 上に metric, summary table, sample table が残る
- Weave 側に同一 eval run の追跡が残る

### 3.6 SFT を仕様どおりに直す

対象ファイル:

- [train_protein_llm.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/train_protein_llm.py)
- [sh_train_protein_qwen_staged.sh](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/scripts/sh_train_protein_qwen_staged.sh)

やること:

- `wandb.init(..., job_type="train_sft")` で開始する
- `reasoning dataset` の `train` を学習に使う
- `reasoning dataset` の `validation` を checkpoint selection に使う
- `test` は学習に使わない
- train / validation metric を `wandb.log()` する
- 代表 sample を W&B Table に保存する
- output checkpoint を Artifact に保存する
- registry を使う場合は model artifact を昇格させる
- time limit を `12:00:00` にそろえる

受け入れ基準:

- run config から dataset version と benchmark version が分かる
- best checkpoint と last checkpoint が区別できる
- sample table に `reasoning` と `final_answer` がある

### 3.7 RL を仕様どおりに直す

対象ファイル:

- RL 学習 entry point が未整備なら新規追加
- 既存学習コードを流用する場合は [train_protein_llm.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/train_protein_llm.py) と新規 wrapper script

やること:

- `wandb.init(..., job_type="train_rl")` で開始する
- benchmark の `train` split を rollout / reward 最適化に使う
- `validation` を checkpoint selection と offline sanity-check に使う
- `test` は学習に使わない
- reward, KL, stability metric を `wandb.log()` する
- rollout sample を W&B Table に保存する
- rollout trace を Weave で保存する
- output checkpoint を Artifact に保存する

受け入れ基準:

- RL run から benchmark version, dataset artifact, base or SFT checkpoint が追える
- rollout sample と reward の関係が W&B / Weave の両方で監査できる
- `test` split に由来する sample が RL 学習に混ざっていない

## 4. 実装時に追加するテスト

Step 0 の契約テストに続いて、次を順に追加する。

### 4.1 dataset 化テスト

- supervised / reasoning dataset の split 整合
- `protein_id` 一致
- optional field の空文字埋め
- RL dataset が `train` のみ由来であること

### 4.2 eval テスト

- `validation` / `test` の切替
- summary table の列
- sample-level table の列
- reasoning 途中過程の保存
- Artifact 出力

### 4.3 SFT テスト

- `job_type=train_sft`
- dataset artifact の読み込み
- metric logging
- sample table 生成
- checkpoint artifact 生成

### 4.4 RL テスト

- `job_type=train_rl`
- reward logging
- rollout sample table
- Weave trace 生成
- `test` split 非混入

## 5. 実装の優先順位

迷ったら次の順に進める。

1. Step 0 の再現性を壊さない
2. dataset の split 整合を固める
3. eval を `validation` / `test` 切替可能にする
4. W&B / Weave の lineage を入れる
5. SFT を仕様どおりにする
6. RL を仕様どおりにする

## 6. 最終チェック

実装が終わったら次を順に確認する。

### 6.1 ローカルテスト

```bash
.venv-step0/bin/python -m unittest discover -s test -v
```

### 6.2 Step 0 artifact の確認

- `summary.json`
- `report.md`
- `*_assigned_labels.tsv`
- `*_assigned_propagated.tsv`
- `*_assigned_nk_lk.tsv`
- `nk_lk_eda.tsv`

### 6.3 W&B / Weave の確認

- `eval`, `train_sft`, `train_rl` の `job_type`
- dataset / model artifact の version
- summary table
- sample-level table
- Weave trace / evaluation

### 6.4 split leakage の確認

- train / validation / test の protein overlap が 0
- RL dataset に validation / test 由来 sample が無い
- final report に使う metric は `test` からのみ作られている
