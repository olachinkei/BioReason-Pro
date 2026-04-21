# BioReason-Pro: SFT & RL Training with W&B Integration

> **Fork of [bowang-lab/BioReason-Pro](https://github.com/bowang-lab/BioReason-Pro)**
>
> This repository extends the original BioReason-Pro codebase with end-to-end SFT and RL training scripts, multi-node distributed training support, and full Weights & Biases integration for experiment tracking and rollout tracing. It is intended as a **research starter** — the scripts are functional and tested on multi-node H100 clusters, but have not undergone exhaustive hyperparameter tuning. Use them as a starting point for your own experiments.

For the original paper, model checkpoints, inference pipeline, and datasets, see the [upstream repository](https://github.com/bowang-lab/BioReason-Pro).

<br>

## About BioReason-Pro

BioReason-Pro is the first multimodal reasoning LLM for protein function prediction. It integrates ESM3 protein embeddings with biological context (InterPro domains, PPI networks, GO-GPT predictions) to generate structured reasoning traces that predict Gene Ontology (GO) terms. The model achieves 73.6% F_max and is preferred over curated UniProt annotations by human experts in 79% of cases.

For full details, see the [paper on bioRxiv](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1) and the [upstream README](https://github.com/bowang-lab/BioReason-Pro/blob/main/README.md).

<br>

## What's Been Added

### Training Scripts

| Script | Description |
|--------|-------------|
| `train_protein_grpo.py` | Main RL training script using DR-GRPO (Disease-Reward Group Relative Policy Optimization) with DeepSpeed ZeRO-2 and vLLM-based rollout generation |
| `scripts/sh_train_protein_grpo.sh` | RL training launcher (single-node and multi-node) |
| `runtime_logs/run_rl_paper_tight_2node_srun.sh` | Multi-node (2-node) RL training launcher via SLURM `srun` |
| `scripts/sh_train_protein_qwen_staged.sh` | SFT training launcher with staged curriculum |

> **Note:** The SFT script is included for completeness but has not been rigorously validated in this fork. The RL training pipeline (`train_protein_grpo.py`) is the primary focus and has been tested end-to-end on 2-node H100 clusters.

### W&B Integration

- **W&B Models**: Per-step metric tracking including reward mean/std, policy loss, rollout timing, checkpoint timing, and validation metrics. Checkpoint artifacts are logged to the W&B Model Registry with configurable aliases.
- **W&B Weave**: Detailed per-rollout tracing with full prompt/completion text, reward scores per rollout, target GO terms, and predicted GO terms. Traces are structured hierarchically: `train_rl_step` > `train_rl_query` > `train_rl_rollout_generate` + `train_rl_reward_score`.

<br>

## How to Run

### Prerequisites

- Python 3.11+
- CUDA-capable GPUs (H100 80GB recommended)
- SLURM cluster for multi-node training
- W&B account with API key configured (`wandb login`)

### Installation

```bash
git clone https://github.com/olachinkei/BioReason-Pro.git
cd BioReason-Pro

# Create virtual environment
python -m venv .venv-gpu
source .venv-gpu/bin/activate

# Install ESM (must use --no-deps due to transformers/vllm version conflict)
pip install esm --no-deps

# Install package
pip install -e .

# Install flash-attn (requires CUDA; skip if not available — code falls back to sdpa)
pip install flash-attn --no-build-isolation --no-cache-dir
```

### Single-Node RL Training

```bash
bash scripts/sh_train_protein_grpo.sh
```

### Multi-Node RL Training (2-Node via SLURM)

```bash
# Submit a 2-node job with 8 GPUs per node
VALIDATION_EVERY_N_STEPS=5 \
SAVE_EVERY_N_STEPS=1 \
sbatch --nodes=2 --gpus-per-node=8 --partition=h100 \
  --job-name=bioreason-rl-train \
  runtime_logs/run_rl_paper_tight_2node_srun.sh
```

Key environment variables for customization:

| Variable | Default | Description |
|----------|---------|-------------|
| `NNODES` | `2` | Number of nodes |
| `GPUS_PER_NODE` | `8` | GPUs per node |
| `QUERIES_PER_STEP` | `8` | Unique proteins per training step |
| `ROLLOUTS_PER_QUERY` | `24` | Rollouts generated per protein |
| `MAX_NEW_TOKENS` | `10000` | Max generation length per rollout |
| `VALIDATION_EVERY_N_STEPS` | `1` | Validation frequency |
| `SAVE_EVERY_N_STEPS` | `1` | Checkpoint save frequency |
| `WANDB_ENTITY` | `wandb-healthcare` | W&B entity |
| `WANDB_PROJECT` | `bioreason-pro` | W&B project |
| `REASONING_PROMPT_STYLE` | `paper_native_tight` | Prompt template style |

### SFT Training

```bash
bash scripts/sh_train_protein_qwen_staged.sh
```

<br>

## Multi-Node Distributed Training Architecture

The RL training pipeline uses a **DeepSpeed + vLLM colocated** architecture where each GPU runs both a DeepSpeed training worker and a vLLM rollout worker in the same process group. Below is a summary of the key design decisions.

### Launch Strategy: SLURM `srun` + DeepSpeed `--no_ssh`

Multi-node launches use SLURM `srun` to start one task per node, each of which invokes DeepSpeed with `--no_ssh`:

```
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -lc '
  export NODE_RANK="$SLURM_NODEID"
  export MASTER_ADDR="<head-node>"
  export MASTER_PORT=29511
  bash scripts/sh_train_protein_grpo.sh --max_steps 20
'
```

SLURM handles node allocation and process placement; DeepSpeed handles intra-node GPU distribution. `MASTER_ADDR` is resolved from `SLURM_JOB_NODELIST` automatically. This avoids SSH-based launcher issues common with DeepSpeed on managed clusters.

### DeepSpeed ZeRO-2 for Parameter-Efficient Training

The optimizer uses ZeRO Stage 2 (optimizer + gradient partitioning) with BF16 mixed precision. Each GPU holds a full model replica for forward/backward but shards optimizer states and gradients across the world. This keeps memory usage manageable on 80GB H100s while training a multimodal Qwen-based model.

### Colocated vLLM Rollout Workers

Each DeepSpeed rank spawns a **subprocess-based vLLM worker** on its own GPU. The worker runs in a separate process with `CUDA_VISIBLE_DEVICES` pinned to a single device and its own isolated `torch.distributed` group (`RANK=0, WORLD_SIZE=1`). Communication between the training process and the rollout worker happens via Python `multiprocessing.Connection` (pipe-based IPC):

- `generate`: send a prompt, receive completions
- `refresh`: reload model weights from a new checkpoint directory
- `sleep` / `wake_up`: optional vLLM sleep mode to free GPU memory between rollout phases

After each policy update, the training loop exports an inference checkpoint (merging LoRA if applicable) and sends a `refresh` command to each vLLM worker so that rollout generation always uses the latest policy.

### Query-Parallel Rollout Sharding

With 16 GPUs (2 nodes x 8 GPUs) and 8 queries per step, the system assigns a **query-parallel degree** of 2 — meaning every pair of GPUs collaborates on the same protein query. Each GPU in the pair generates `rollouts_per_query / query_parallel_degree` rollouts (e.g., 24 / 2 = 12 rollouts per GPU). This distributes the generation workload evenly while ensuring each query gets the full number of rollouts.

The rewards from each pair are aggregated via a **file-backed scalar collective**: each rank writes its local metrics to a JSON file on shared storage, and the leader rank polls until all participants have reported, then computes the aggregate. This avoids NCCL collectives during the rollout/reward phase (which would conflict with the vLLM subprocess's own CUDA context) and is robust to node-level timing skew.

### Weight Sync: Export + Refresh

After each DeepSpeed optimizer step, rank 0 exports a full inference checkpoint (merging LoRA weights and saving the text model, protein projection, and GO encoder). All ranks then send a `refresh` command to their local vLLM worker, which reloads from the exported checkpoint directory. This keeps the rollout policy in sync with the training policy without requiring shared-memory weight transfer.

<br>

## Current Status & Known Issues

The latest RL training run (20 steps, 2-node H100, DR-GRPO) completed successfully:

**W&B Run**: [rl-paper-native-tight-2node-srun-20260421-044513](https://wandb.ai/wandb-healthcare/bioreason-pro/runs/aldhlmrz)

### Training Metrics (20 Steps)

| Step | Reward | Loss | Step | Reward | Loss |
|------|--------|------|------|--------|------|
| 1 | 0.1625 | -0.000010 | 11 | 0.1332 | -0.000019 |
| 2 | 0.1338 | -0.000035 | 12 | 0.1256 | +0.000002 |
| 3 | 0.1039 | -0.000020 | 13 | 0.1339 | +0.000017 |
| 4 | 0.2232 | -0.000013 | 14 | 0.1781 | +0.000016 |
| 5 | 0.2407 | -0.000014 | 15 | 0.0993 | +0.000026 |
| 6 | 0.1772 | -0.000045 | 16 | 0.0860 | +0.000023 |
| 7 | 0.1581 | -0.000028 | 17 | 0.1648 | +0.000013 |
| 8 | 0.0714 | -0.000017 | 18 | 0.0833 | +0.000035 |
| 9 | 0.1000 | -0.000017 | 19 | 0.1212 | +0.000019 |
| 10 | 0.1260 | -0.000004 | 20 | 0.1235 | +0.000029 |

### Issues Identified via Weave Trace Analysis

1. **`<final_answer>` tag not generated**: The model fails to produce the expected `<final_answer>` tag in its completions. Since the reward function (`reward_prediction_source: final_answer_block`) extracts GO terms exclusively from this tag, missing tags result in reward = 0 regardless of reasoning quality. In some steps (15, 20), all 12 rollouts scored 0.

2. **Tag format mismatch**: When the model does attempt a final answer block, it sometimes uses incorrect formats (e.g., `<|FINAL_ANSWER|>` instead of `<final_answer>`), which the reward parser does not recognize.

3. **Repetition collapse**: In step 15 (protein Q9UKF6), the model generated 25,688 characters of repetitive, degraded text (e.g., `nuc leus`, `end onuclease`) indicating a decoding failure.

4. **Loss sign flip**: Policy loss transitions from negative (steps 1-11, indicating reward-aligned updates) to positive (steps 12-20), suggesting the policy gradient signal has become ineffective or adversarial.

5. **No reward improvement**: Mean reward fluctuates between 0.07-0.24 across all 20 steps with no upward trend. The learning signal is dominated by noise from tag-parsing failures rather than genuine function prediction quality.

### Root Cause

The `paper_native_tight` prompt template likely does not match what the model has learned to output during SFT. The model generates reasoning text but wraps its final answer in a format the reward parser cannot extract from. This means the RL loop receives near-random reward signals and cannot learn effectively.

### Next Steps

- Investigate the `paper_native_tight` prompt template to verify the expected output tag format
- Check what tag format the SFT checkpoint was trained to produce
- Align the reward parser's `final_answer_block` extraction with the model's actual output format
- Consider adding a format reward component that incentivizes correct tag usage

<br>

## Contributors

- **Keisuke Kamata** — Senior Manager, AI Solutions Engineer, Weights & Biases
- **Uma Krishnaswamy** — Senior AI Solutions Engineer, Weights & Biases
