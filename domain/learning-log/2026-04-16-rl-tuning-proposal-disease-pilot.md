# 2026-04-16 RL Tuning Proposal — Disease-Protein Pilot

Status: **Proposal only. No code changes yet. Implementation happens on a new branch once signed off.**

## 0. Pilot framing (the one sentence this doc has to serve)

> Pilot study to test whether **fine-tuning BioReason-Pro on disease-related human proteins** improves GO function prediction — **without changing the model architecture**.

Every choice below is filtered against that sentence. If a lever requires adding/removing ESM3, adding a new encoder, changing projection layers, or otherwise deviating from the frozen-arch contract, it is out of scope.

## 1. What "win" means here

| Thing | Value |
|---|---|
| Baseline we must beat | `wandb-healthcare/bioreasoning-pro/bioreason-pro-rl:production` (paper RL checkpoint) |
| Headline metric | Weighted F<sub>max</sub> (CAFA-evaluator semantics), separately reported per aspect (BP / MF / CC) |
| Primary split | 200-protein disease dev split (temporal releases 225) |
| Final reported split | 400-protein disease holdout (temporal release 228) |
| Data budget | The ~2k disease-related reasoning examples in `disease-temporal-reasoning` |
| Algorithm | GRPO on top of the paper RL checkpoint (continuation) |
| Architecture | Frozen: Qwen3 + ESM3 + GO graph encoder + projection, as shipped |

The current run already logs `reward_mean`, `validation_reward_mean`, and `validation_reward_nonzero_rate`. We need to additionally evaluate true **weighted F<sub>max</sub> per aspect** on the dev set at checkpoint time — reward-mean is an RL proxy, not the research claim.

## 2. Where the current setup stands

From `train_protein_grpo.py` + `scripts/sh_train_protein_grpo.sh` as of today:

```336:336:scripts/sh_train_protein_grpo.sh
# NOTE: hyperparameters referenced below live in this file + train_protein_grpo.py
```

- **Reward**: single scalar `ia_weighted_f1` (IA-weighted F1 after GO-DAG ancestor propagation), implemented by `compute_weighted_f1` at `train_protein_grpo.py:1322`.
- **RL algorithm**: DR-GRPO–style — global reward std across all queries, per-query group-mean baseline, normalized advantages (`compute_group_advantages` / `compute_global_reward_std`).
- **Key hyperparameters** (today's defaults):
  - `learning_rate=3e-5`, cosine schedule, `warmup_ratio=0.03`
  - `kl_beta=1e-4`
  - `temperature=1.0`, `max_new_tokens=10000`
  - `queries_per_step=8`, `rollouts_per_query=24` (group size 24)
  - `reasoning_prompt_style=paper_native_tight`

### What's good about this baseline

- Reward is already CAFA-aligned (IA-weighted F1 + ancestor propagation), so optimization pressure is biologically meaningful (see [Clark & Radivojac, 2013](https://academic.oup.com/bioinformatics/article/29/13/i53/195366) and the [CAFA-evaluator paper](https://www.ccs.neu.edu/home/radivojac/papers/piovesan_bioinformadv_2024.pdf)).
- DR-GRPO group normalization avoids the well-known [length-bias problem in vanilla GRPO](https://www.emergentmind.com/topics/dr-grpo).
- `paper_native_tight` keeps the decoding contract close to the shipped checkpoint, which was the diagnosis conclusion in `2026-04-09-paper-native-continuation-implementation.md`.

### Where it likely leaves signal on the table

1. **Reward is near-binary for bad rollouts.** `compute_weighted_f1` returns `0.0` whenever predicted ∩ target = ∅ (`train_protein_grpo.py:1328-1329`). For proteins where the model's top-level aspect is wrong, every rollout in the group scores 0, producing a zero-variance group, zero advantage, and a wasted step. This is the classic "advantage collapse" / zero-variance group problem discussed in [DAPO (Dynamic Sampling)](https://arxiv.org/abs/2503.14476) and [EDGE-GRPO](https://arxiv.org/abs/2507.21848).
2. **Reward ignores GO semantic closeness.** `GO:0004672` (protein kinase activity) vs `GO:0016301` (kinase activity) get 0 credit if neither is in target after propagation, even though they are 1 edge apart in MF. [DeepGO-SE](https://www.nature.com/articles/s42256-024-00795-w) and the [GOSemSim Lin/Resnik family](https://bioconductor.statistik.uni-dortmund.de/packages/3.11/bioc/vignettes/GOSemSim/inst/doc/GOSemSim.html) show that IC-based semantic similarity is a better learning signal for hierarchical ontologies.
3. **No aspect-awareness in the reward.** BP, MF, CC have very different IA distributions and prediction difficulties in CAFA. Treating them as one flat bag lets the model satisfy the reward by over-predicting easy CC terms and under-predicting BP.
4. **Vanilla symmetric clip + no dynamic sampling.** We're vulnerable to the two most-cited GRPO failure modes: entropy collapse ([GSPO paper](https://arxiv.org/abs/2507.18071)) and gradient-dead zones on always-0 or always-correct groups ([DAPO](https://arxiv.org/abs/2503.14476)).
5. **No curriculum.** The disease subset is small (~2k) and heterogeneous in annotation density. [DeepGO](https://pmc.ncbi.nlm.nih.gov/articles/PMC5860606/) and the CAFA5 post-hoc analyses show that annotation-count and IA stratification matter a lot for small datasets.
6. **No disease-weighting inside the train loop.** The pilot is *specifically* about disease proteins, but the loss treats every train example uniformly. Examples linked to OMIM/DisGeNET genes should probably carry more weight than incidental train examples.

## 3. Proposal — three phases, cheapest first

Each phase produces a **single-variable ablation** against the paper RL checkpoint baseline. Each variable is defended by at least one peer-reviewed reference. Nothing touches model architecture.

---

### Phase A — Reward shaping within CAFA semantics (Week 1)

**Thesis:** The biggest untapped lever in a 2k-sample regime is denser, more biologically informative reward, not algorithm changes. Evidence: DeepGO-SE showed that IC-based hierarchical losses gave multi-point CAFA improvements over flat BCE on the same architecture ([Nature MI 2024](https://www.nature.com/articles/s42256-024-00795-w)).

**A.1 — Per-aspect weighted F1 decomposition**
- Split the current scalar into three: F1<sub>BP</sub>, F1<sub>MF</sub>, F1<sub>CC</sub>, each computed after ancestor propagation inside that aspect only, weighted by the aspect's IA mass.
- Final reward = mean of the three (so the scale stays ~[0,1] and the group std still makes sense).
- Matches CAFA's official per-aspect reporting. Reference: [Piovesan et al., CAFA-evaluator, Bioinformatics Advances 2024](https://www.ccs.neu.edu/home/radivojac/papers/piovesan_bioinformadv_2024.pdf).
- Expected effect: reduces the "all-0 group" rate because partial aspect credit is common even when the overall F1 is 0.

**Plain-English version:**
GO terms come in three flavors — *biological process* (what the protein participates in), *molecular function* (what it chemically does), and *cellular component* (where it lives in the cell). The baseline reward mashes all three into one F1 score, so a model that nails the function flavor but bombs the process flavor still looks "okay" because the two average out. A.1 breaks the score into one F1 per flavor and then averages those. Same overall scale, but now the model can't hide weakness in one aspect behind strength in another — and we get a partial score when the model gets *one* flavor right, which means fewer rollouts scoring exactly 0.

**A.2 — Hierarchical partial credit via Lin similarity**
- When `predicted ∩ target = ∅`, instead of returning 0, return a scaled [0, 0.3] reward equal to mean Lin similarity between each predicted term and its nearest target term within the same aspect.
- Cap the bonus so it can never beat a real F1 hit; it only resolves ties between bad groups so advantages are non-degenerate.
- Reference: [Lin/Resnik semantic similarity in GOSemSim](https://bioconductor.statistik.uni-dortmund.de/packages/3.11/bioc/vignettes/GOSemSim/inst/doc/GOSemSim.html); used in [DeepGO-SE](https://www.nature.com/articles/s42256-024-00795-w).
- Expected effect: kills most zero-variance groups without letting the model reward-hack by flooding shallow GO terms.

**Plain-English version:**
GO terms are organized in a tree — `kinase activity` is the parent of `protein kinase activity`. Today, if the model predicts `protein kinase activity` but the target is `kinase activity` (or vice versa), the reward is flat 0 — no credit for being one edge away. Every rollout in the group scores 0, GRPO's advantage goes to 0, and the training step wastes compute. A.2 says: only when there's no direct overlap, hand out a small consolation reward based on how "tree-close" the guesses are to the targets (same flavor only — no cross-flavor credit). The consolation is capped at 0.3, so it can never beat a real hit; it exists purely so near-miss groups still have variance and still produce a gradient. Note: strict Lin similarity needs precomputed information-content values we don't currently ship, so the implementation uses ancestor-set Jaccard — the same quantity in spirit, no extra data files required. We can upgrade to true Lin later without changing the flag surface.

**A.3 — Disease-weighted example weighting (not reward, loss weighting)**
- At loss time (not reward time), multiply each trajectory's loss by a scalar ∈ {1.0, 1.5} depending on whether the protein is in the OMIM/DisGeNET disease set. This is loss-side, not reward-side, so GRPO group normalization is untouched.
- Matches the pilot's framing: we *explicitly* want disease proteins to dominate gradient mass.
- Reference: standard importance-weighted ERM; consistent with how BioReason-Pro's own RL stage is described in [the paper](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1.full) as domain-conditional.

**Plain-English version:**
The pilot is supposed to be about disease proteins, but the training loop treats every protein equally — a random enzyme and a cancer-linked kinase get the same gradient weight. A.3 fixes that by multiplying the loss for disease proteins by 1.5× (and everything else stays at 1.0×) so disease examples push the weights harder per step. Critically, this is done on the **loss** side, after the advantage is computed, so GRPO's group-level reward-standardization math doesn't get corrupted — the reward scale stays meaningful across all examples. Caveat: the current `disease_temporal_hc_reasoning_v2` dataset doesn't surface a per-protein "is-disease" flag yet, so until that lands A3 effectively runs as a uniform 1.5× loss scale (still a useful ablation — does a hotter learning rate help at all?). Once the pipeline surfaces OMIM/DisGeNET membership, A3 becomes the real disease-vs-non-disease comparison with no other code changes needed.

**Ablation grid for Phase A (4 runs):**

| Run | Reward | Loss weighting |
|---|---|---|
| A0 (baseline) | current IA-F1 | uniform |
| A1 | per-aspect IA-F1 (A.1) | uniform |
| A2 | per-aspect + Lin partial credit (A.1 + A.2) | uniform |
| A3 | per-aspect + Lin + disease-weighted loss (A.1 + A.2 + A.3) | disease 1.5× |

Metric: weighted F<sub>max</sub> on 200-protein dev, per aspect. Move forward with whichever is best.

---

### Phase B — GRPO algorithmic fixes (Week 2)

**Thesis:** With the reward now non-degenerate (Phase A), we can pick up the cheap, well-evidenced GRPO tricks that the paper RL checkpoint predates.

**B.1 — DAPO Clip-Higher**
- Replace the symmetric PPO clip `[1-ε, 1+ε]` with asymmetric `[1-ε_low, 1+ε_high]` where `ε_high > ε_low` (their numbers: `0.28 / 0.20`).
- Empirically avoids entropy collapse during long RL continuation; ~3 point AIME gain reported with no other change.
- Reference: [DAPO §3.2](https://arxiv.org/abs/2503.14476).
- Zero-risk, 5-line change inside the GRPO loss.

**B.2 — DAPO Dynamic Sampling**
- Skip any query-group whose reward variance is 0 (all-correct or all-wrong). Draw a replacement query instead.
- Directly addresses the "wasted compute" problem that the current sparse reward causes.
- Reference: [DAPO §3.3](https://arxiv.org/abs/2503.14476).
- Fits our existing `queries_per_step=8, rollouts_per_query=24` loop with minimal plumbing.

**B.3 — Overlong soft length penalty**
- Reasoning traces in `paper_native_tight` can blow past useful length. Add a soft penalty that only kicks in once `completion_length > 0.8 * max_new_tokens`.
- Reference: [DAPO §3.4 "Overlong reward shaping"](https://arxiv.org/abs/2503.14476).
- Secondary upside: reduces rollout wall-clock for the worst-case 10k-token completions.

**B.4 (optional) — GSPO-style sequence-level ratio**
- If after B.1–B.3 we still see entropy collapse or divergence on a long RL run, swap the per-token importance ratio for the [GSPO sequence-level ratio](https://arxiv.org/abs/2507.18071). Qwen3 specifically was stabilized this way; we're on Qwen3.
- Only pull this trigger if we observe collapse. Not in the default grid.

**Ablation grid for Phase B (3 runs, starting from best-of-Phase-A):**

| Run | Clip | Dynamic sampling | Length penalty |
|---|---|---|---|
| B0 | symmetric (current) | off | off |
| B1 | asymmetric (0.20 / 0.28) | on | off |
| B2 | asymmetric | on | soft (0.8·max) |

---

### Phase C — Curriculum and schedule (Week 3, if budget remains)

**Thesis:** Small-data RL benefits from ordered exposure. These are lower-ceiling than A and B but cheap to add if the first two phases land.

**C.1 — IA-stratified curriculum**
- Bin training proteins by mean IA of their target GO set. Run 20% of steps on the low-IA bin (easier, shallower), 50% on mid, 30% on high-IA (hard, deep terms). Advance only after a warmup.
- References: [DeepGO's observations on GO depth](https://pmc.ncbi.nlm.nih.gov/articles/PMC5860606/), plus general curriculum-RLVR results such as [GDRO-style difficulty classifiers](https://arxiv.org/pdf/2601.19280) showing ~10% gains on reasoning benchmarks vs vanilla GRPO.

**C.2 — Annotation-density filter**
- Drop train proteins with ≤2 target GO terms for the first N steps. These are almost always noisy / under-annotated in CAFA5. Re-introduce later.
- Reference: DeepGO and [CAFA5 Kaggle write-ups](https://kaggle.curtischong.me/competitions/CAFA-5-Protein-Function-Prediction) consistently show low-annotation proteins hurt CAFA metrics.

**C.3 — LR schedule for RL-on-RL continuation**
- We are continuing RL *from* an already-RL'd checkpoint. Current `lr=3e-5` is appropriate for RL from SFT, but tends to be too hot for RL-on-RL. Try `1e-5` with longer warmup (0.1 instead of 0.03).
- Reference: standard practice in [ProRL](https://arxiv.org/pdf/2602.03190) and related prolonged-RL papers that explicitly discuss re-RL stability.

**Ablation grid for Phase C (2 runs, on top of best B):**

| Run | Curriculum | Annot. filter | LR |
|---|---|---|---|
| C0 | off | off | 3e-5 |
| C1 | IA-stratified | drop ≤2 annotations (first 30% of steps) | 1e-5, warmup 0.1 |

---

## 4. What I am explicitly **not** proposing

- No ESM3 unfreezing. Out of scope per pilot framing.
- No architecture additions (e.g., no new retrieval tool, no new graph encoder).
- No new dataset. Disease temporal split stays as-is.
- No change to `paper_native_tight` continuation contract. Prompt changes would invalidate the checkpoint-native decoding guarantee from `2026-04-09-paper-native-continuation-implementation.md`.
- No ART / serverless RL. Kei's analysis already ruled out vLLM-based ART rollouts for this architecture.

## 5. Reference index

**RL algorithm references**
- DAPO: [arXiv 2503.14476](https://arxiv.org/abs/2503.14476)
- Dr.GRPO: [summary + comparison](https://www.emergentmind.com/topics/dr-grpo)
- GSPO (Qwen3-aligned): [arXiv 2507.18071](https://arxiv.org/abs/2507.18071)
- EDGE-GRPO (entropy-driven advantage): [arXiv 2507.21848](https://arxiv.org/abs/2507.21848)

**Protein function & GO evaluation references**
- BioReason-Pro paper: [bioRxiv 2026.03.19.712954](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1.full)
- BioReason-Pro repo: [bowang-lab/BioReason-Pro](https://github.com/bowang-lab/BioReason-Pro)
- BioReason (DNA, NeurIPS '25): [bowang-lab/BioReason](https://github.com/bowang-lab/BioReason)
- CAFA-evaluator: [BioComputingUP/CAFA-evaluator](https://github.com/BioComputingUP/CAFA-evaluator), [Piovesan 2024](https://www.ccs.neu.edu/home/radivojac/papers/piovesan_bioinformadv_2024.pdf)
- DeepGO: [arXiv 1705.05919](https://arxiv.org/abs/1705.05919), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5860606/)
- DeepGO-SE (semantic entailment): [Nature MI 2024](https://www.nature.com/articles/s42256-024-00795-w)
- GO semantic similarity (GOSemSim Lin/Resnik): [docs](https://bioconductor.statistik.uni-dortmund.de/packages/3.11/bioc/vignettes/GOSemSim/inst/doc/GOSemSim.html)
- ProteinReasoner CoT: [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.07.21.665832v1.full-text)

## 6. Proposed execution plan (once approved)

1. Create branch `rl-disease-tuning` off current main.
2. Implement Phase A's four reward modes behind a single CLI flag (e.g. `--reward_mode {ia_f1, per_aspect, per_aspect_lin}` and `--disease_loss_weight`). Default stays as today's `ia_f1`, uniform weighting, so the branch is a strict superset of current behavior.
3. Add a proper `weighted_fmax_per_aspect` validation metric (not just reward-mean) computed on the 200-protein dev split at each checkpoint.
4. Run A0→A3 on 1 node × 8×H100. Each run ~1 epoch over the disease subset. Compare on dev weighted F<sub>max</sub>.
5. Gate on Phase A result: only proceed to Phase B if at least one Phase A variant beats A0 on dev by ≥1 weighted-F<sub>max</sub> point.
6. Phase B and C similarly gated.
7. Each Phase writes a short follow-up learning-log doc (`2026-04-XX-phase-A-results.md` etc.) with run IDs and numbers.

## 7. Open questions for Uma / Kei before implementation

- Should Phase A also include the original 4-component reward (format / structure / completion / GO-overlap) as an ablation, or is that officially retired? (Kei's latest main retired it; Uma's earlier `0.1/0.1/1.0/0.1` change is stale.)
- Do we already have Lin/Resnik IC precomputed from `go-basic.obo` somewhere, or do we need to compute it offline as part of the data pipeline?
- Is the 200-protein disease dev split already hooked up to a `compute_weighted_fmax_per_aspect` evaluation pipeline, or does that need to be built?
- Kei's note said "reward being 2 to 3" — that was under the earlier 4-component reward and is no longer applicable with `ia_f1` (bounded in [0,1]). Confirming so we don't accidentally re-introduce scaling fixes for a dead problem.

---

**Next action on approval:** create `rl-disease-tuning` branch, implement Phase A.1 + A.2 reward modes behind flags, add `weighted_fmax_per_aspect` dev eval, kick off A0→A3 grid.
