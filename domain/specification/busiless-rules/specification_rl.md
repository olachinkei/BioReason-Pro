# BioReason-Pro RL Implementation Specification

Based on *BioReason-Pro* (bioRxiv, 2026), especially Section 4.3.4, Appendix B.4.3, and Table S18. This note is intended to be a full implementation reference for the RL path.

## Migrated Context From `specification.md`

The following operational assumptions were previously documented in `specification.md` section 8.2 and below and are now maintained here in expanded form.

### Inputs and Strict Rules

- Canonical RL input is the `bioreason-pro-rl-paper` checkpoint.
- RL rollout / reward optimization uses the benchmark `train` split.
- Checkpoint selection and offline sanity-check use the full `validation` split of **200 proteins**.
- The `test` split is not used for RL training.
- Reward metrics, KL metrics, and training stability indicators must be logged via `wandb.log()`.
- Rollout traces must be captured with Weave.
- RL output checkpoints must be registered as W&B Artifacts.

### Execution Conditions

- RL uses the same benchmark definition and versioning rules as the rest of the project.
- Maximum wall time for training jobs is `12:00:00` unless a specific deviation is documented.
- Local output directories are scratch; the source of truth is the W&B Artifact / run record.
- The canonical paper-faithful rollout unit is:
  - `1 step = 8 proteins`
  - `1 protein = 24 rollouts`
  - `1 step total = 192 trajectories`
- The intended execution model in the paper is `DeepSpeed + vLLM (colocate)`.
- If the implementation falls back to more serial rollout generation for systems stability, that must be treated as a runtime deviation from the paper.

## 1. Algorithm Overview

Algorithm name: **DR-GRPO** (the paper's terminology).

It is a hybrid that combines four ingredients:

| Base method | Adopted element |
|---|---|
| GRPO (Shao et al. 2024) | Group-relative reward centering, no value model |
| GSPO (Zheng et al. 2025) | **Sequence-level** importance sampling correction rather than token-level correction |
| Dr. GRPO (Liu et al. 2025) | Length-bias mitigation and `B·G·Lmax` normalization |
| DAPO (Yu et al. 2025) | Clip-Higher strategy with asymmetric clipping (`epsilon_low != epsilon_high`) |

The KL penalty is added directly to the loss with `beta = 1e-4`.

## 2. Per-Step Processing Flow

```text
Step t:
  [ROLLOUT PHASE]
  1. Sample B/G = 8 queries from the training pool
  2. Use old_policy (frozen copy) to generate G = 24 responses per query
     -> total B = (B/G) * G = 192 (query, response) pairs
  3. Compute reward for each response -> r(x, y_i) in [0, 1]

  [ADVANTAGE PHASE]
  4. Compute group mean reward r̄(x) for each query        [Eq. 4.19]
  5. Compute the standard deviation over the full batch of 192 rewards
  6. Compute Â_i = (r(x,y_i) - r̄(x)) / (std_batch + eps) [Eq. 4.20]

  [SCORING PHASE]
  7. Compute sequence log-prob under current_policy for all 192 responses
  8. Compute sequence log-prob under old_policy for all 192 responses
     (or reuse rollout-time values if cached)
  9. Compute the sequence-level IS ratio s_i(theta)       [Eq. 4.22]

  [LOSS PHASE]
  10. Compute the clipped surrogate loss                  [Eq. 4.24]
  11. Add KL penalty
  12. Update theta with AdamW

  [POLICY UPDATE]
  13. Every steps_per_generation = 2 optimizer steps, copy
      current_policy weights into old_policy
```

Important interpretation:

- `steps_per_generation = 2` means one rollout batch is followed by two optimization steps.
- `num_iterations = 1` means the same rollout batch is not reused for PPO-style multiple epochs.

## 3. Detailed Mathematical Implementation

### 3.1 Rollout Generation [Eq. 4.17]

```python
# old_policy generates G=24 responses for each query
# inputs: B/G = 8 queries
responses = old_policy.generate(
    inputs=queries.repeat_interleave(G, dim=0),  # shape: [B, seq_len]
    max_new_tokens=L_max,       # 10000
    temperature=1.0,
    top_k=20,
    top_p=0.95,
    min_p=0,
    repetition_penalty=1.0,
    do_sample=True,
)
# responses shape: [B, response_len], where B = 192 trajectories total
```

### 3.2 Reward Computation [Eq. 4.18]

```python
def compute_reward(response_text: str, ground_truth_terms: set, ia_weights: dict) -> float:
    \"\"\"
    Return IA-weighted F1 in [0, 1].
    \"\"\"
    predicted_raw = extract_go_terms_from_final_answer(response_text)
    if predicted_raw is None:
        return 0.0

    predicted_terms = propagate_to_ancestors(predicted_raw)

    intersection = predicted_terms & ground_truth_terms
    pr_w = sum(ia_weights[f] for f in intersection) / (sum(ia_weights[f] for f in predicted_terms) + 1e-8)
    rc_w = sum(ia_weights[f] for f in intersection) / (sum(ia_weights[f] for f in ground_truth_terms) + 1e-8)

    if pr_w + rc_w == 0:
        return 0.0
    return 2 * pr_w * rc_w / (pr_w + rc_w)
```

Important implementation rules:

- Reward is a continuous scalar in `[0, 1]`.
- GO terms are propagated through both `is_a` and `part_of`, following CAFA5.
- If truncation prevents the final answer block from appearing, reward is `0`.

### 3.3 Advantage Estimation [Eq. 4.19, 4.20]

```python
rewards = torch.tensor([...])  # shape [B] = [192]

rewards_grouped = rewards.view(num_proteins, G)  # [8, 24]
group_means = rewards_grouped.mean(dim=1, keepdim=True)  # [8, 1]
group_means_expanded = group_means.expand_as(rewards_grouped).reshape(-1)  # [192]

batch_std = rewards.std() + eps_std
advantages = (rewards - group_means_expanded) / batch_std
```

Why batch-global standard deviation?

- In GO tasks, reward variance is often very low.
- Per-group normalization can collapse when all 24 rollouts for a protein receive nearly the same reward.
- The paper explicitly uses the standard deviation over the full batch of `192` rewards.

### 3.4 Sequence-Level Importance Ratio [Eq. 4.21, 4.22]

```python
def compute_sequence_log_prob(model, input_ids, response_ids):
    with torch.no_grad():
        logits = model(input_ids=torch.cat([input_ids, response_ids], dim=1)).logits
    log_probs = F.log_softmax(logits[:, input_ids.shape[1]-1:-1, :], dim=-1)
    token_log_probs = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum(dim=-1)

log_ratio = current_log_probs - old_log_probs
s_i = torch.exp(log_ratio)
s_i_capped = s_i.clamp(max=IS_cap)  # IS_cap = 2
```

Why sequence-level?

- Token-level ratios are numerically unstable for long sequences.
- Since the reward is sequence-level, the IS correction should also be sequence-level.

### 3.5 Clipped Surrogate Loss [Eq. 4.23, 4.24]

```python
eps_low = 7e-4
eps_high = 9e-4

surr1 = s_i * advantages
surr2 = s_i.clamp(1 - eps_low, 1 + eps_high) * advantages
clipped = torch.min(surr1, surr2)

L_max = 10000
loss_policy = -clipped.sum() / (B * G * L_max)
```

Why divide by `B·G·Lmax`?

- Dividing only by response count introduces a structural bias favoring certain response lengths.
- Dr. GRPO uses fixed-length normalization to reduce this bias.

### 3.6 KL Penalty

```python
kl_divergence = (current_log_probs - ref_log_probs).mean()
loss_kl = beta * kl_divergence  # beta = 1e-4

loss = loss_policy + loss_kl
```

Important distinction:

- `old_policy`: rollout-generation policy, updated every `steps_per_generation = 2`
- `ref_policy`: frozen SFT checkpoint used as the KL anchor

## 4. Full Hyperparameter List (Table S18)

### 4.1 Algorithm

| Parameter | Value | Notes |
|---|---|---|
| Algorithm | DR-GRPO | — |
| Group size G | 24 | responses per query |
| Steps per generation | 2 | two optimization steps per rollout batch |
| Num. iterations (inner) | 1 | no PPO-style multi-epoch reuse |
| KL penalty beta | 1e-4 | KL anchor to frozen ref policy |
| Clipping epsilon_low | 7e-4 | lower clip for negative advantages |
| Clipping epsilon_high | 9e-4 | higher clip for positive advantages |
| Reward scaling | Batch | use batch-global std |
| IS correction | Yes (sequence-level) | not token-level |
| IS cap | 2 | cap on importance ratio |

### 4.2 LoRA

| Parameter | Value |
|---|---|
| Rank r | 16 |
| Alpha alpha | 32 |
| Dropout | 0.05 |
| Disable model dropout | Yes |

### 4.3 Optimizer

| Parameter | Value |
|---|---|
| Optimizer | AdamW (`beta1=0.9`, `beta2=0.999`, `eps=1e-8`) |
| Peak learning rate | 3e-5 |
| LR schedule | Cosine decay |
| Warmup ratio | 0.03 |
| Weight decay | 0 |
| Gradient clipping | 1.0 |
| Gradient checkpointing | Yes |

### 4.4 Batching

| Parameter | Value | Notes |
|---|---|---|
| Per-device batch size | 6 | per GPU |
| Gradient accumulation steps | 4 | effective per-device batch = 24 |
| GPUs | 8 (2 nodes × 4) | H100 |
| Effective batch size B | 192 | `6 × 4 × 8` |
| Unique proteins per step (B/G) | 8 | `192 / 24` |

### 4.5 Rollout Sampling

| Parameter | Value |
|---|---|
| Temperature | 1.0 |
| Top-k | 20 |
| Top-p | 0.95 |
| Min-p | 0 |
| Repetition penalty | 1.0 |
| Max completion length Lmax | 10,000 tokens |
| Max prompt length | 512 text tokens |

### 4.6 Training

| Parameter | Value |
|---|---|
| Init checkpoint | SFT epoch 8 |
| Max RL steps | 1,200 |
| Total tokens consumed | ~619M |
| Precision | bf16 |
| Seed | 42 |
| Framework | DeepSpeed + vLLM (colocate) |
| Wall time | ~32 hours |

## 5. Reward Extraction Specification

### 5.1 Output Format

The model is expected to produce:

```text
<|REASONING|>
... free-form reasoning trace ...
<|/REASONING|>
<|FINAL_ANSWER|>
... structured final answer ...
GO:0005737
GO:0003674
...
<|/FINAL_ANSWER|>
```

### 5.2 Regex Extraction

```python
import re

FINAL_ANSWER_PATTERN = re.compile(
    r'<\\|FINAL_ANSWER\\|>(.*?)<\\|/FINAL_ANSWER\\|>',
    re.DOTALL
)
GO_TERM_PATTERN = re.compile(r'GO:\\d{7}')

def extract_go_terms_from_final_answer(text: str):
    match = FINAL_ANSWER_PATTERN.search(text)
    if match is None:
        return None
    final_answer_block = match.group(1)
    return set(GO_TERM_PATTERN.findall(final_answer_block))
```

Important design choices:

1. GO terms in the reasoning trace are ignored for reward.
2. If truncation removes the final answer block, reward is `0`.
3. Only valid `GO:\\d{7}` patterns are extracted.

## 6. Failure Patterns Reported In Appendix B.4.3

### 6.1 Too-Late SFT Initialization

| Phenomenon | If initialized from a checkpoint that is too mature, rollouts collapse toward near-identical outputs |
|---|---|
| Consequence | Within-group reward variance collapses and the learning signal vanishes |
| Mitigation | Initialize from the selected SFT checkpoint (epoch 8) rather than an overly late checkpoint |

### 6.2 Per-Group Normalization Collapse

| Phenomenon | Advantages collapse toward zero |
|---|---|
| Cause | Per-group standard deviation becomes near-zero |
| Mitigation | Normalize by batch-global reward standard deviation |

### 6.3 Runaway Generation

| Phenomenon | Responses become increasingly long and repetitive |
|---|---|
| Cause | Long outputs can dominate learning dynamics |
| Mitigation | Dr. GRPO normalization + GSPO sequence-level IS + Clip-Higher |

### 6.4 Token-Level IS Instability

| Phenomenon | Token-level IS ratios overflow or underflow |
|---|---|
| Cause | Very long sequence lengths |
| Mitigation | Use sequence-level log-prob differences and exponentiate only once |

### 6.5 Limited Value Of Longer RL Runs

| Phenomenon | On-policy reward may continue improving while downstream evaluation saturates |
|---|---|
| Interpretation | The signal can saturate under the current data distribution and reward design |
| Conclusion | Stop at 1,200 steps unless data diversity changes |

## 7. Pseudocode For The Training Loop

```python
policy = load_checkpoint(\"sft_epoch8\")
old_policy = copy.deepcopy(policy)
ref_policy = copy.deepcopy(policy)
ref_policy.requires_grad_(False)

optimizer = AdamW(policy.parameters(), lr=3e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

for step in range(1200):
    proteins = dataloader.sample(B // G)  # 8 proteins
    queries = [build_prompt(p) for p in proteins]

    with torch.no_grad():
        repeated_queries = repeat_queries(queries, G)  # [192]
        responses = old_policy.generate(
            repeated_queries,
            max_new_tokens=10000,
            temperature=1.0,
            top_k=20,
            top_p=0.95,
        )
        old_seq_logprobs = compute_sequence_log_prob(old_policy, queries, responses)
        ref_seq_logprobs = compute_sequence_log_prob(ref_policy, queries, responses)

    rewards = torch.tensor([
        compute_reward(resp, proteins[i // G].go_terms, ia_weights)
        for i, resp in enumerate(responses)
    ])

    group_means = rewards.view(B // G, G).mean(dim=1).repeat_interleave(G)
    batch_std = rewards.std() + 1e-8
    advantages = (rewards - group_means) / batch_std

    for inner_step in range(2):
        current_seq_logprobs = compute_sequence_log_prob(policy, queries, responses)

        s_i = torch.exp(current_seq_logprobs - old_seq_logprobs).clamp(max=2.0)

        surr1 = s_i * advantages
        surr2 = s_i.clamp(1 - 7e-4, 1 + 9e-4) * advantages
        loss_policy = -torch.min(surr1, surr2).sum() / (B * G * L_max)

        kl = (current_seq_logprobs - ref_seq_logprobs).mean()
        loss_kl = 1e-4 * kl

        loss = loss_policy + loss_kl
        loss.backward()

        if (inner_step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    old_policy.load_state_dict(policy.state_dict())
```

## 8. Paper-vs-Implementation Review Checklist

- [ ] Are rollouts generated as a batch of `B/G × G = 192` trajectories rather than by a mostly serial per-example loop?
- [ ] Is the denominator for advantage normalization the batch-global std over all 192 rewards?
- [ ] Is the IS ratio sequence-level rather than token-level?
- [ ] Is the IS cap equal to `2.0`?
- [ ] Is the loss normalized by `B·G·Lmax`?
- [ ] Are `epsilon_low` and `epsilon_high` asymmetric (`7e-4` vs `9e-4`)?
- [ ] Is reward extraction scoped to the final answer block only?
- [ ] Does truncation without a final answer block give reward `0`?
- [ ] Is `old_policy` refreshed every `steps_per_generation=2` steps?
- [ ] Is `ref_policy` frozen at SFT epoch 8?
- [ ] Is the LoRA dropout / model dropout setup aligned with the paper?
- [ ] Is the initialization checkpoint the selected SFT epoch 8 checkpoint rather than an arbitrary late SFT checkpoint?
