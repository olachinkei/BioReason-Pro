# Specification

本仕様は、BioReason-Pro を用いた疾患関連ヒトタンパク質の GO 機能予測について、2026-04-06 時点で確認できている実測結果と運用ルールを整理したものである。

最初に明記しておくと、現在採用する benchmark は **small-data 版** である。現行版は `213 -> 221 -> 225 -> 228` を用い、train は **1,245 proteins / 2,773 unique labels**、dev は **590 proteins**、test は **875 proteins** である。この点を前提に、データ設計・学習・評価・主張の仕方を定める。

Status:
- Step 0 の実測結果は `213` 版と `214` 版がある
- 現行版 benchmark は `213 -> 221 -> 225 -> 228`
- train / dev / test は protein-disjoint かつ temporal order を守って分割する
- 文中の `dev` は dataset split 名 `validation` と同義である

## 1. 問題設定

上位目的は `domain/foundation/philosophy.md` に従う。  
本仕様で扱う問題は、**疾患関連ヒトタンパク質に対する protein-level GO term prediction と disease-aware reasoning** である。

本仕様で固定する問いは次の 2 点である。

- 疾患関連ヒトタンパク質に対して、BioReason-Pro の既存 checkpoint へ追加学習を行う価値があるか
- 時系列的に独立した benchmark 上で、base より良い予測と reasoning が出るか

本仕様で扱わないものは次である。

- variant pathogenicity classification
- ClinVar 単独 supervision
- cross-species disease transfer

## 2. 使用モデル・比較対象

### 2.1 使用モデル

**BioReason-Pro**（bowang-lab, bioRxiv 2026-03）

| 要素 | 内容 |
|---|---|
| アーキテクチャ | ESM3 + GO Graph Encoder + Qwen3-4B |
| タスク | タンパク質配列 -> GO term 予測 + 推論トレース生成 |
| 学習方式 | SFT（合成推論トレース） -> RL |
| 公開物 | モデル weight、学習データ、推論コード |
| ベース学習データのカットオフ | UniProt GOA 2022-11 リリース |

### 2.2 比較対象

比較対象は次の 4 系統に固定する。

| 手法 | 位置づけ |
|---|---|
| BLAST / Diamond | 配列類似性ベースのベースライン |
| ESM 系単体 | 基盤モデル単体ベースライン |
| BioReason-Pro base | 公開 checkpoint をそのまま使うベースライン |
| tuned model | 疾患特化データで追加学習したモデル |

## 3. ベンチマーク

### 3.1 現在採用する benchmark 版

現在採用する benchmark は、次の high-confidence disease shortlist を使う。

```text
reviewed:true AND organism_id:9606 AND cc_disease:* AND
(xref:mim-* OR xref:orphanet-*) AND
(go_exp:* OR go_ida:* OR go_ipi:* OR go_igi:* OR
 go_imp:* OR go_iep:* OR go_ic:* OR go_tas:*)
```

この shortlist の live count は **5,088 proteins** である。

この値は **現在の UniProt shortlist size** を示す参考値であり、そのまま train / dev / test の件数になるわけではない。  
実際の benchmark 規模は、release 差分、evidence filter、protein-disjoint assignment を通した Step 0 artifact の出力で決まる。

現在採用する split は次である。

| Version | Train proteins | Train unique labels | Dev proteins | Test proteins |
|---|---:|---:|---:|---:|
| `213 -> 221 -> 225 -> 228` | 1,245 | 2,773 | 590 | 875 |

以後、この版を `current implementation version` と呼ぶ。

### 3.2 参考 benchmark 版

reference sensitivity 版として、次も保持する。

| Version | Train proteins | Train unique labels | Dev proteins | Test proteins |
|---|---:|---:|---:|---:|
| `214 -> 221 -> 225 -> 228` | 932 | 1,898 | 662 | 969 |

`214` 版は、件数感度と archive choice の影響を確認するための reference として使う。

### 3.3 broader alternative filter

broader alternative として、`cc_disease:*` を軸にした shortlist も存在する。

```text
reviewed:true AND organism_id:9606 AND cc_disease:* AND
(go_exp:* OR go_ida:* OR go_ipi:* OR go_igi:* OR
 go_imp:* OR go_iep:* OR go_ic:* OR go_tas:*)
```

この broader query の live count は **5,093 proteins** である。  
現時点の benchmark は `MIM/Orphanet` を含む high-confidence 版で進めるが、broader filter は追加 EDA や将来の拡張候補として保持する。

### 3.4 split の基本ルール

この benchmark で守る条件は 2 つである。

1. train / dev / test は **時系列順** に並ぶこと
2. train / dev / test は **同一 protein を共有しない**こと

current implementation version の split は次で固定する。

- train: `213 -> 221`
- dev: `221 -> 225`
- test: `225 -> 228`

時間の流れは必ず `train -> dev -> test` とする。  
train より後の時点で初めて現れた label を dev に入れ、dev より後の時点で初めて現れた label を test に入れる。逆流は許さない。

なお、benchmark の説明では中間 split を `dev` と呼ぶが、dataset config とコード上の split 名は `validation` で統一する。

### 3.5 新規 label の定義

新規 label は `(DB_ID, GO_ID, Aspect)` 単位で定義する。  
evidence code の違いだけでは別 label と数えない。

つまり、

- old release に無い
- new release にはある
- `(protein, GO, aspect)` が新しい

ときだけ、その label を temporal delta に含める。

### 3.6 protein-disjoint ルール

同じ protein を train / dev / test の複数 split に入れてはならない。  
各 protein は、**最初に新規 label が現れた split に 1 回だけ割り当てる**。

例:

- ある protein が `213 -> 221` で初めて新規 GO を持ったら train 専属
- その protein が `221 -> 225` で追加 GO を得ても dev には入れない
- その protein が `225 -> 228` でさらに追加 GO を得ても test には入れない

要するに、train / dev / test は **時系列でも分かれ、protein でも分かれる**。

### 3.7 split validation

Step 0 の artifact は、`summary.json` に次が出力されていることを必須条件とする。

- `split_validation.time_order_valid == true`
- `split_validation.protein_disjoint_valid == true`

protein overlap が 1 件でも見つかった run は不正とみなし、採用しない。

### 3.8 現在使う artifact

current implementation version の source artifact は次に固定する。

- `domain/specification/busiless-rules/artifacts/step0_human_ub_20260406`

reference sensitivity artifact は次である。

- `domain/specification/busiless-rules/artifacts/step0_human_lb_20260406`

## 4. データの仕様

### 4.1 使用する証拠コード

使用する GO evidence code は次に固定する。

- `EXP`
- `IDA`
- `IPI`
- `IGI`
- `IMP`
- `IEP`
- `IC`
- `TAS`

GAF では `DB == UniProtKB` かつ `DB_Type == protein` の行だけを対象にする。

### 4.2 データ規模の解釈

現在の benchmark は small-data 版である。  
ただし、この規模で実装する正当性はある。

理由は次のとおり。

- 追加学習の対象は zero-from-scratch ではなく、既存の BioReason-Pro checkpoint である
- 時系列的に独立した benchmark がある
- `BioReason-Pro base` との直接比較ができる
- 小規模でも、疾患文脈の改善が出るかどうかは検証できる

`3,000 unique labels` は **full-scale study を安心して主張しやすい目安** であり、current implementation version を否定する hard gate ではない。

### 4.3 main label と propagated label

学習用の正解ラベルは `*_assigned_labels.tsv` を基準に作る。  
`*_assigned_propagated.tsv` は次に使う。

- coverage 確認
- benchmark 記録
- F_max 評価時の補助集計

### 4.4 除外する判定軸

以下は main benchmark の定義や独立性判定には使わない。

- `ClinVar` cross-reference を main filter の必須条件にする設計
- `annotation_date` を使って初回付与日を引く設計
- UniProt REST の `date_modified` だけで独立性を判定する設計

### 4.5 dataset config

現行の `train_protein_llm.py` と `eval.py` は `load_cafa5_dataset()` を通す実装になっているため、current implementation split は **CAFA5 互換の Hugging Face dataset config** として提供する。

config 名:

- supervised dataset: `disease_temporal_hc_v1`
- reasoning dataset: `disease_temporal_hc_reasoning_v1`

役割:

- supervised dataset は GO label を直接使う評価、補助学習、非 reasoning 系 ablation の基準データに使う
- reasoning dataset は SFT の教師データ、および reasoning を伴う eval の入力に使う

必要列:

| 列 | 必須 | 用途 |
|---|---|---|
| `protein_id` | Yes | 主キー |
| `sequence` | Yes | モデル入力 |
| `organism` | Yes | prompt 生成 |
| `go_bp` | Yes | BP 正解ラベル |
| `go_mf` | Yes | MF 正解ラベル |
| `go_cc` | Yes | CC 正解ラベル |
| `reasoning` | SFT 用 | assistant reasoning |
| `final_answer` | SFT 用 | assistant answer |
| `protein_function` | Yes | UniProt summary 文脈 |
| `go_pred` | Yes | GO-GPT 予測の事前計算列 |
| `interpro_formatted` | 任意 | InterPro 文脈 |
| `ppi_formatted` | 任意 | PPI 文脈 |

split 名は `train` / `validation` / `test` に固定する。  
任意列が存在しない場合は列自体を省略せず、空文字列で埋める。

RL 用に派生 dataset を作る場合は、少なくとも次の列を持つ形を推奨する。

| 列 | 必須 | 用途 |
|---|---|---|
| `protein_id` | Yes | 元データとの対応付け |
| `split` | Yes | `train` 固定の確認 |
| `prompt` | Yes | rollout 入力 |
| `response` | Yes | policy 出力または候補出力 |
| `reward` | Yes | scalar reward |
| `reward_components` | 任意 | reward 内訳 |
| `notes` | 任意 | 例外・監査メモ |

### 4.6 SFT と RL のデータの分け方

SFT 用と RL 用に benchmark split を別々に作るのではなく、**split 定義は 1 つに固定し、その上で各 phase が使う列と最適化目的を分ける**。

固定ルール:

- `train` / `validation` / `test` の protein membership は SFT / RL / eval で共通にする
- SFT は reasoning dataset の `train` split を使い、`reasoning` と `final_answer` を教師信号として使う
- RL は同じ benchmark version の `train` split を使い、reward に基づいて最適化する
- `validation` は SFT の early stopping と RL の offline sanity-check / checkpoint selection に使う
- `test` は最終評価専用であり、SFT / RL の学習信号には使わない
- supervised dataset と reasoning dataset は `protein_id` と `split` が一致していなければならない
- RL 用の preference / reward dataset を派生生成する場合も、元データは `train` split のみから作る

要するに、SFT と RL は **別の benchmark を持つのではなく、同じ benchmark の train split を別目的で使う**。

### 4.7 NK / LK の仕様

NK / LK は benchmark の補助的な解析軸として保持する。  
current implementation version では、NK / LK を主要判定軸にはしないが、**どれぐらい含まれているかを EDA できる状態**にしておく。

固定ルール:

- NK / LK の bucket は split ごとに保存する
- EDA 用の集計表を保存する
- summary に split ごとの NK / LK 件数を残す
- CAFA5 access がある場合は実カウント、無い場合は status を明記する

保存物:

- `*_assigned_nk_lk.tsv`
- `*_assigned_nk_lk_propagated.tsv`
- `nk_lk_eda.tsv`
- `summary.json` の `nk_lk_status`, `nk_lk_error`, split 別 NK/LK 件数
- `report.md` の split summary table

## 5. データの準備

### 5.1 Step 0

Step 0 の生成スクリプトは `scripts/step0_disease_temporal_split.py` を使う。

current implementation run:

```bash
python scripts/step0_disease_temporal_split.py \
  --output-dir domain/specification/busiless-rules/artifacts/step0_human_ub_20260406 \
  --train-start-release 213 \
  --train-end-release 221 \
  --dev-end-release 225 \
  --test-end-release 228 \
  --shortlist-mode high-confidence \
  --use-shell-filter
```

reference sensitivity run:

```bash
python scripts/step0_disease_temporal_split.py \
  --output-dir domain/specification/busiless-rules/artifacts/step0_human_lb_20260406 \
  --train-start-release 214 \
  --train-end-release 221 \
  --dev-end-release 225 \
  --test-end-release 228 \
  --shortlist-mode high-confidence \
  --use-shell-filter
```

### 5.2 Step 0 の出力

Step 0 では少なくとも次を出力する。

- `summary.json`
- `report.md`
- `train_assigned_labels.tsv`
- `dev_assigned_labels.tsv`
- `test_assigned_labels.tsv`
- `*_assigned_propagated.tsv`
- `*_assigned_nk_lk.tsv`
- `*_assigned_nk_lk_propagated.tsv`
- `nk_lk_eda.tsv`

### 5.3 dataset 化

Step 0 artifact を基に、次の 2 系統の dataset を作る。

- supervised dataset
- reasoning dataset

必要に応じて、RL 用の preference / reward dataset を **train split からのみ**派生生成してよい。  
ただしこの派生 dataset も、元の benchmark version と `protein_id` 対応を保持する。

生成後は `train` / `validation` / `test` に分割し、学習と評価の両方で同じ dataset version を参照する。  
supervised dataset と reasoning dataset は `protein_id` / `split` の対応が完全一致していることを必須とする。

生成物は少なくとも次の単位で version 管理する。

- Step 0 artifact
- supervised dataset artifact
- reasoning dataset artifact
- RL preference / reward dataset artifact を作った場合はその artifact

## 6. データ・モデルのupload for W&B

本仕様では、W&B を experiment tracking と artifact lineage の標準にする。

### 6.1 共通の run config

各 phase で最低限 W&B に記録する config は次とする。

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


### 6.2 dataset upload

固定ルール:

- dataset は W&B Artifact として登録する
- dataset artifact の version / alias を run config に残す
- stable な dataset version は registry に昇格させてよい

### 6.3 model upload

固定ルール:

- model checkpoint は W&B Artifact として登録する
- model artifact の version / alias を run config に残す
- stable な model version は registry に昇格させてよい


## 7. 評価

### 7.1 eval phase

本仕様では、評価の論理フェーズ名を **`eval`** に固定する。  
repo 内の実装 entry point は `eval.py` と `scripts/sh_eval.sh` を使う。  
W&B run は `wandb.init(..., job_type="eval")` で開始する。  
同じ評価 run について、`weave.Evaluation` の eval logger でも trace と score を追跡する。

評価 split の使い分けは次とする。

- 開発中の比較、ablation、checkpoint 比較は `validation`
- 最終報告値は lock 済みの `test`

### 7.2 定量評価

定量評価の主指標は **F_max** とし、次を必須とする。

- MF
- BP
- CC

評価対象:

- `BioReason-Pro base`
- `train_sft` の出力
- `train_rl` の出力がある場合はそれも含める

各 metric は `wandb.log()` で記録する。  
少なくとも split ごとに `fmax_mf`, `fmax_bp`, `fmax_cc` を保存する。

### 7.3 定性評価

定性評価では、少なくとも次の 3 点を確認する。

- 疾患名・経路名の具体的言及
- 変異と機能障害の因果記述
- 相互作用パートナーと疾患との関連記述

### 7.4 eval の保存物

必須事項:

- metric を `wandb.log()` する
- 評価 summary を **1 evaluated target = 1 row** の W&B Table として `wandb.log()` する
- sample-level 結果を **1 sample = 1 row** の W&B Table として `wandb.log()` する
- reasoning task の場合は sample-level table に reasoning 途中過程も含める
- JSON 結果、summary export、sample export は Artifact として version 管理する

summary table の最低列:

- `model_name`
- `dataset_config`
- `split`
- `benchmark_version`
- `fmax_mf`
- `fmax_bp`
- `fmax_cc`
- `macro_note`

sample-level table の最低列:

- `protein_id`
- `split`
- `model_name`
- `prompt`
- `prediction`
- `expected_output`
- `accuracy_or_match_note`
- `reasoning_excerpt`

reasoning task の場合は少なくとも次も保持する。

- `reasoning_full`
- `final_answer`
- `intermediate_trace`

## 8. 学習 for SFT

### 8.1 train_sft phase

本仕様では、SFT の論理フェーズ名を **`train_sft`** に固定する。  
repo 内の実装 entry point は `train_protein_llm.py` と `scripts/sh_train_protein_qwen_staged.sh` を使う。  
W&B run は `wandb.init(..., job_type="train_sft")` で開始する。

### 8.2 train_sft の入力

入力:

- `disease_temporal_hc_v1`
- `disease_temporal_hc_reasoning_v1`
- base checkpoint

使い方:

- 学習には reasoning dataset の `train` split を使う
- checkpoint selection には reasoning dataset の `validation` split を使う
- `test` split は SFT 学習には使わず、最終評価まで保持する
- supervised dataset は label reference と補助的な non-reasoning 比較に使ってよい

### 8.3 train_sft の必須事項

固定ルール:

- dataset を W&B Artifact として登録する
- 学習 run は W&B で追跡する
- train / validation の主要 metric を `wandb.log()` で保存する
- train sample / validation sample の代表例を W&B Table として `wandb.log()` する
- output checkpoint を W&B Artifact として登録する
- registry を使う場合は、この phase の model artifact を registry に昇格させてよい

W&B Table の最低列:

- `protein_id`
- `split`
- `input_summary`
- `reasoning`
- `final_answer`
- `expected_go_bp`
- `expected_go_mf`
- `expected_go_cc`

### 8.4 実行条件

固定ルール:

- 学習は GPU 前提
- `dataset_type=cafa5` 前提で dataset config を供給する
- 学習ジョブの wall time は **最大 12 時間**
- submission 時の time limit は `12:00:00` を使う
- checkpoint / resume 前提で運用する

## 9. 学習 for RL

### 9.1 train_rl phase

本仕様では、RL の論理フェーズ名を **`train_rl`** に固定する。  
phase 名は仕様上維持し、実装時には現行の学習スタックへ対応づけて運用する。  
W&B run は `wandb.init(..., job_type="train_rl")` で開始する。

### 9.2 train_rl の入力

入力:

- base checkpoint または SFT checkpoint
- reward 設定
- 同じ benchmark 定義

使い方:

- rollout / reward 最適化には benchmark の `train` split を使う
- checkpoint selection と offline sanity-check には `validation` split を使う
- `test` split は RL 学習には使わず、最終評価まで保持する
- RL 用 dataset を派生生成する場合も `protein_id` と `benchmark_version` を保持し、SFT と同じ split 境界を崩さない

### 9.3 train_rl の必須事項

固定ルール:

- RL run の config を W&B に保存する
- reward 系 metric、KL 系 metric、学習安定性指標を `wandb.log()` で保存する
- 途中のrolloutをweaveでtraceする
- RL input / reasoning 途中経過 / output sample を W&B Table として `wandb.log()` する
- RL 出力 checkpoint を W&B Artifact として登録する


W&B Table の最低列:

- `protein_id`
- `split`
- `prompt`
- `response`
- `reward`
- `parsed_go_terms`
- `notes`

### 9.4 実行条件

固定ルール:

- RL でも同じ benchmark version を使う
- 学習ジョブの wall time は **最大 12 時間**
- submission 時の time limit は `12:00:00` を使う
- checkpoint / resume 前提で運用する
