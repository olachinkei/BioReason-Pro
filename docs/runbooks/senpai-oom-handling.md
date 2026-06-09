# Senpai OOM Handling

Use this runbook when BioReason-Pro Senpai runs fail with CUDA OOM, vLLM KV
cache allocation errors, or DeepSpeed memory pressure.

## Preserve Paper Token Length

Do not reduce the rollout token length as the first OOM response. The paper
target is:

```bash
SENPAI_MAX_NEW_TOKENS=10000
```

The Senpai wrapper defaults to the paper value. Keep this value unless a human
researcher explicitly chooses a shorter-context ablation.

`max_new_tokens` and `vllm_max_model_len` are different:

- `max_new_tokens` is the maximum number of newly generated tokens per rollout.
- `vllm_max_model_len` is the total context budget for prompt plus generation
  that vLLM uses when planning KV cache capacity.

The default context budget is:

```bash
SENPAI_VLLM_MAX_MODEL_LEN=32768
```

This is intentionally larger than `SENPAI_MAX_NEW_TOKENS=10000` so prompts plus
paper-length generations fit without truncating the rollout target.

## First Response For vLLM OOM

The default active vLLM sequence count is:

```bash
SENPAI_VLLM_MAX_NUM_SEQS=4
```

If a run still fails during vLLM startup or rollout generation, retry with:

```bash
SENPAI_VLLM_MAX_NUM_SEQS=2
```

This reduces active vLLM concurrency while preserving `SENPAI_MAX_NEW_TOKENS`.
It will usually slow generation down, but it keeps the paper token length.

Example CKS/SUNK retry:

```bash
SENPAI_VLLM_MAX_NUM_SEQS=2 python k8s/launch.py \
  --tag <retry-tag> \
  --image <registry>/bioreason-pro-senpai:cuda12.6
```

Or through launcher extra args:

```bash
python k8s/launch.py \
  --tag <retry-tag> \
  --image <registry>/bioreason-pro-senpai:cuda12.6 \
  --extra_train_args "--vllm_max_num_seqs 2 --max_new_tokens 10000 --vllm_max_model_len 32768"
```

## If OOM Happens During Training

If the stack trace points to loss computation, backward, gradient reduction, or
optimizer step instead of vLLM startup/generation, reduce the DeepSpeed training
microbatch before changing token length:

```bash
SENPAI_OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU=2
SENPAI_GRADIENT_ACCUMULATION_STEPS=8
```

Keep:

```bash
SENPAI_MAX_NEW_TOKENS=10000
SENPAI_VLLM_MAX_MODEL_LEN=32768
```

## Escalation Order

Use this order:

1. Keep paper token length: `SENPAI_MAX_NEW_TOKENS=10000`.
2. Confirm the run is using `SENPAI_VLLM_MAX_NUM_SEQS=4`.
3. If vLLM still OOMs, retry with `SENPAI_VLLM_MAX_NUM_SEQS=2`.
4. If training/backward OOMs, lower
   `SENPAI_OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU` and compensate with
   `SENPAI_GRADIENT_ACCUMULATION_STEPS`.
5. Only reduce `SENPAI_MAX_NEW_TOKENS` for an explicitly approved ablation.

Record any OOM retry in the PR or run summary, including whether the failure was
vLLM generation-side or training-side.
