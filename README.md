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

The active Phase A check is a paper-exact A1 run on CoreWeave H100s. It uses the
paper-scale rollout shape (`MAX_NEW_TOKENS=10000`, `ROLLOUTS_PER_QUERY=24`,
`QUERIES_PER_STEP=8`) on 2 nodes x 8 GPUs with `reward_mode=per_aspect_ia_f1`.

**W&B Run**: [rl-phase-a-A1-paper-2node-20260427T061638Z](https://wandb.ai/wandb-healthcare/bioreason-pro-custom/runs/s4snxgvy)

**Slurm Job**: `3451` (`bioreason-a1-paper-2node`)

**Status as of 2026-04-27 18:07 JST / 09:07 UTC**: running; step 5 training
metrics are complete, and step 5 validation is still in progress.

### A1 Paper-Exact Train Metrics

| Step | Reward | Nonzero Rate | `<final_answer>` Rate | Format Valid | Valid Rollouts | Ratio Mean |
|------|--------|--------------|-----------------------|--------------|----------------|------------|
| 1 | 0.1859 | 61.46% | 96.35% | 76.04% | 192 | 0.5705 |
| 2 | 0.1454 | 52.60% | 94.79% | 67.71% | 192 | 0.4666 |
| 3 | 0.1410 | 53.12% | 96.35% | 77.60% | 192 | 0.3566 |
| 4 | 0.2268 | 63.54% | 95.31% | 69.27% | 192 | 0.5276 |
| 5 | 0.2514 | 58.33% | 96.35% | 75.00% | 192 | 0.4315 |

### Current Interpretation

The paper-exact A1 run is not reproducing the earlier all-zero A1 failure. Train
rollouts are valid, reward is nonzero, `<final_answer>` generation is stable at
roughly 95%, and there are no observed rollout failures, degraded responses, or
noop fallbacks through step 5. This points away from `reward_mode=per_aspect_ia_f1`
as the direct cause of the earlier broken run.

The earlier A1 run (`z66vrc7a`) used lightweight/debug settings and showed
immediate collapse: reward = 0, final-answer tag rate = 0, format-valid rate = 0,
and unstable ratio metrics. The current evidence suggests that failure was more
likely caused by the non-paper rollout configuration and runtime instability than
by the A1 reward definition itself.

### Current Known Issue: Validation Bottleneck

Validation starts at step 5, but it currently runs as a rank-0 serial section over
200 validation proteins with `MAX_NEW_TOKENS=10000`. That makes validation much
slower than the distributed training steps and leaves the other ranks waiting. At
the last check, validation had started and the log was still advancing, but no
`validation_done` marker or validation metrics had been written yet.

Before using this validation cadence for routine experiments, the validation path
should be changed to use a smaller validation generation budget, fewer samples, or
parallelized validation. Otherwise, successful training steps can appear stuck for
hours at validation boundaries.

### Operational Notes

- Runtime files, model output, and caches should stay under `/mnt/data`; `/mnt/home`
  quota pressure previously caused repeated exit-code-122 crashes.
- The 2-node paper-exact shape is the current working configuration. The earlier
  1-node 8-GPU attempt used a larger local repeat count and timed out before
  producing a stable comparison.
- Tracking metadata now records Phase A fields such as `reward_mode`,
  `disease_loss_weight`, `lin_partial_credit_cap`, and `ablation_tag` so W&B and
  checkpoint metadata can be compared across A0/A1/A2/A3 runs.

<br>

## Contributors

- **Keisuke Kamata** — Senior Manager, AI Solutions Engineer, Weights & Biases
- **Uma Krishnaswamy** — Senior AI Solutions Engineer, Weights & Biases
