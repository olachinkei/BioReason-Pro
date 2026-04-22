---
theme: default
title: "LLMOps for Drug Discovery Systems: Governing Models, Agents, and Evaluation"
info: |
  Foundation models and AI agents are transforming drug discovery, but real-world deployment demands strong MLOps and LLMOps.
  This talk presents a practical improvement loop for BioReason-Pro, including tracing, evaluation, and domain fine-tuning.
class: text-center
highlighter: shiki
drawings:
  persist: false
transition: slide-left
mdc: true
fonts:
  sans: "Source Sans Pro"
  serif: "Source Sans Pro"
  mono: "Fira Code"
---

<style>
:root {
  --slidev-theme-primary: #58d3db;
  --slidev-theme-secondary: #fac13c;
}
.slidev-layout {
  background: #1a1d24;
  color: #ffffff;
}
h1, h2, h3 {
  color: #ffffff !important;
}
pre {
  background: #2a2f3a !important;
  color: #e6e6e6 !important;
  border-radius: 8px !important;
  border: 1px solid rgba(88, 211, 219, 0.2) !important;
}
p code, li code {
  background: rgba(88, 211, 219, 0.15) !important;
  color: #58d3db !important;
  padding: 0.2em 0.4em !important;
  border-radius: 4px !important;
  font-size: 0.9em !important;
}
</style>

# LLMOps for Drug Discovery Systems

<div style="color: #cccccc; padding-top: 0.75rem;">
Governing Models, Agents, and Evaluation
</div>

<div style="padding-top: 2rem; max-width: 42rem; margin: 0 auto; color: #cccccc; font-size: 0.95rem; text-align: left;">
Foundation models and AI agents are transforming drug discovery, but real-world deployment demands strong MLOps and LLMOps. Using Weights & Biases Models and Weave, we demonstrate a reproducible workflow for fine-tuning, agent tracing, evaluation, and RL that improves reliability, speeds iteration, and supports compliance.
</div>

<div style="padding-top: 2.5rem; color: #888888; font-size: 0.9rem;">
BioIT World • 22-minute talk
</div>

<!--
Set context quickly: this is about operating these systems in production-like science workflows.
-->

---
layout: center
---

# Talk Structure (22 min)

<div style="padding-top: 1.5rem; text-align: left; max-width: 44rem; margin: 0 auto; color: #cccccc; font-size: 1.02rem;">

1. **Case for a systematic improvement loop** (6 min)
2. **W&B tools and what they unlock** (5 min)
3. **Our BioReason-Pro plan: ablations + tuning strategies** (6 min)
4. **Results + live W&B demo** (5 min)

</div>

<div style="padding-top: 2rem; color: #58d3db;">
Slides for narrative + live UI for evidence.
</div>

<!--
Make expectations explicit and time-boxed.
-->

---
layout: center
---

# 1) Why We Need a Systematic Loop

<div style="padding-top: 2rem; text-align: left; max-width: 44rem; margin: 0 auto; color: #cccccc;">

- AI systems now generate hypotheses, candidates, and rankings.
- But many pipelines are still stitched together with scripts and notebooks.
- That breaks reproducibility, governance, and learning speed.

</div>

<div style="padding-top: 2rem; color: #ffffff;">
<span style="color: #58d3db;">Building models is no longer enough; operating them is the real bottleneck.</span>
</div>

<!--
This is the hook/problem bridge.
-->

---
layout: center
---

# What "Ad Hoc" Looks Like

<div style="padding-top: 1.5rem;">

```
model_v1 -> notebook_eval -> spreadsheet_metrics -> manual notes -> model_v2
   |             |                 |                     |
 unclear      non-repeatable    no lineage         no audit trail
```

</div>

<div style="padding-top: 1.75rem; text-align: left; max-width: 44rem; margin: 0 auto; color: #cccccc;">

- Which exact run produced this claim?
- Why did performance shift this week?
- Is this gain real, or split leakage / randomness?

</div>

<!--
Give concrete pain points they have probably experienced.
-->

---
layout: center
---

# Systematic Improvement Loop

<div style="padding-top: 1rem;">

```
┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
│ Instrument │-> │ Evaluate   │-> │ Improve    │-> │ Govern     │
│ and trace  │   │ consistently│  │ (SFT / RL) │   │ and deploy │
└────────────┘   └────────────┘   └────────────┘   └─────┬──────┘
                                                          │
                    ┌─────────────────────────────────────┘
                    ▼
                 Learn and iterate
```

</div>

<div style="padding-top: 1.5rem; color: #cccccc;">
The point is not one good run; the point is a reliable learning system.
</div>

<!--
This is the core thesis slide.
-->

---
layout: center
class: text-center
---

<div style="font-size: 4rem; font-weight: bold; color: #fac13c;">2</div>

# W&B Tools and How They Help

<div style="padding-top: 1rem; color: #cccccc;">
Models + Weave + Evaluations as the operating layer
</div>

---
layout: center
---

# W&B Models: Versioning and Lineage

<div style="padding-top: 1.25rem; text-align: left; max-width: 44rem; margin: 0 auto; color: #cccccc;">

- Register base models, tuned checkpoints, and aliases.
- Track datasets, configs, and artifacts per run.
- Promote with explicit criteria (not intuition).

</div>

<div style="padding-top: 1.5rem;">

```yaml
model: bioreason-pro
alias: pilot-disease-go-v3
parent: bioreason-pro-base
training_data: disease-human-go-v2
eval_gate: temporal_holdout_go_v1 >= target
```

</div>

<!--
Explain model governance and reproducible promotion.
-->

---
layout: center
---

# Weave: Traces for Agents and Model Calls

<div style="padding-top: 2rem; text-align: left; max-width: 42rem; margin: 0 auto; color: #cccccc;">

- Capture each call, intermediate step, latency, and output.
- Debug failed predictions with full context.
- Compare behavior across versions and prompt templates.

</div>

<div style="padding-top: 1.5rem;">

```python
@weave.op
def go_function_agent(sample):
    # trace input -> retrieval -> reasoning -> output
    return run_agent(sample)
```

</div>

<!--
Emphasize this is not observability theater: it accelerates debugging and trust.
-->

---
layout: center
---

# Evaluation: Measuring Real Improvement

<div style="padding-top: 1.5rem;">

```
same split + same metrics + same protocol -> fair comparison
```

</div>

<div style="padding-top: 1.5rem; text-align: left; max-width: 44rem; margin: 0 auto; color: #cccccc;">

- Evaluate by biologically meaningful metrics and slices.
- Track confidence intervals and temporal holdout behavior.
- Decide promotions with explicit eval gates.

</div>

<!--
Quality discussions become concrete when everyone is looking at the same eval protocol.
-->

---
layout: center
class: text-center
---

<div style="font-size: 4rem; font-weight: bold; color: #fac13c;">3</div>

# What We're Doing: Pilot Plan

<div style="padding-top: 1rem; color: #cccccc;">
BioReason-Pro pilot on disease-related human proteins for GO function prediction
</div>

<div style="padding-top: 1.25rem; text-align: left; max-width: 44rem; margin: 0 auto; color: #cccccc;">

- Objective: improve GO prediction quality.
- Constraint: **no architecture changes**.
- Focus: data strategy + tuning strategy + evaluation rigor.

</div>

---
layout: center
---

# Ablation Grid (Planned)

<div style="padding-top: 1rem;">

```text
Axes:
1) Data scope
   - generic proteins
   - + disease-related human proteins
   - + curated hard cases

2) Training strategy
   - baseline (no tune)
   - SFT
   - RL (GRPO-style variants)

3) Prompt / context strategy
   - base prompt
   - ontology-aware prompt
   - retrieval-augmented context
```

</div>

<div style="padding-top: 1rem; color: #cccccc;">
Run each cell with the same eval suite for apples-to-apples comparisons.
</div>

---
layout: center
---

# Tuning Strategies We Are Testing

<div style="padding-top: 1.25rem; display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
  <div style="padding: 1rem; background: #2a2f3a; border-radius: 0.5rem; text-align: left;">
    <div style="color: #58d3db; font-weight: bold;">Baseline</div>
    <div style="color: #cccccc; font-size: 0.9rem; padding-top: 0.4rem;">Current BioReason-Pro, no tuning.</div>
  </div>
  <div style="padding: 1rem; background: #2a2f3a; border-radius: 0.5rem; text-align: left;">
    <div style="color: #58d3db; font-weight: bold;">SFT</div>
    <div style="color: #cccccc; font-size: 0.9rem; padding-top: 0.4rem;">Supervised fine-tuning on curated disease-protein examples.</div>
  </div>
  <div style="padding: 1rem; background: #2a2f3a; border-radius: 0.5rem; text-align: left;">
    <div style="color: #58d3db; font-weight: bold;">RL Variant A</div>
    <div style="color: #cccccc; font-size: 0.9rem; padding-top: 0.4rem;">Reward prioritizes GO correctness + calibration.</div>
  </div>
  <div style="padding: 1rem; background: #2a2f3a; border-radius: 0.5rem; text-align: left;">
    <div style="color: #58d3db; font-weight: bold;">RL Variant B</div>
    <div style="color: #cccccc; font-size: 0.9rem; padding-top: 0.4rem;">Reward adds robustness penalties for known failure modes.</div>
  </div>
</div>

---
layout: center
---

# Execution Plan

<div style="padding-top: 1rem;">

```text
Week 1: freeze baselines + finalize eval split
Week 2: run SFT ablations
Week 3: run RL variants + reward sensitivity
Week 4: compare, select, and package candidate model
```

</div>

<div style="padding-top: 1.5rem; text-align: left; max-width: 44rem; margin: 0 auto; color: #cccccc;">

- Every run logged in W&B with linked artifacts.
- Promotion candidate requires passing predefined eval gates.
- Keep all decisions traceable for technical and compliance review.

</div>

---
layout: center
class: text-center
---

<div style="font-size: 4rem; font-weight: bold; color: #fac13c;">4</div>

# Results + Live Demo

<div style="padding-top: 1rem; color: #cccccc;">
Slides for summary, W&B UI for evidence
</div>

---
layout: center
---

# Results Snapshot (Replace with latest)

<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; padding-top: 1.5rem; text-align: center;">
  <div>
    <div style="font-size: 2rem; font-weight: bold; color: #58d3db;">TBD</div>
    <div style="font-size: 0.8rem; color: #cccccc;">delta Fmax</div>
  </div>
  <div>
    <div style="font-size: 2rem; font-weight: bold; color: #58d3db;">TBD</div>
    <div style="font-size: 0.8rem; color: #cccccc;">delta AUPRC</div>
  </div>
  <div>
    <div style="font-size: 2rem; font-weight: bold; color: #58d3db;">TBD</div>
    <div style="font-size: 0.8rem; color: #cccccc;">best strategy</div>
  </div>
  <div>
    <div style="font-size: 2rem; font-weight: bold; color: #58d3db;">0</div>
    <div style="font-size: 0.8rem; color: #cccccc;">arch changes</div>
  </div>
</div>

<div style="padding-top: 1.75rem; text-align: left; max-width: 44rem; margin: 0 auto; color: #cccccc;">

- Show confidence intervals on final version of this slide.
- Keep metric definitions visible in backup slides.

</div>

---
layout: center
---

# Live W&B Demo: What I Will Show

<div style="padding-top: 1rem; text-align: left; max-width: 44rem; margin: 0 auto; color: #cccccc;">

1. **Run comparison dashboard** (baseline vs SFT vs RL variants)
2. **Trace deep dive** for one success and one failure case
3. **Model registry lineage** from base model to candidate
4. **Eval report + promotion gate** decision

</div>

<div style="padding-top: 2rem; color: #58d3db;">
Goal: show that every claim in slides is backed by inspectable evidence.
</div>

---
layout: center
---

# Closing Message

<div style="padding-top: 1.5rem; text-align: left; max-width: 44rem; margin: 0 auto; color: #cccccc;">

- Drug discovery AI needs model innovation **and** operating discipline.
- A systematic loop turns one-off experiments into cumulative progress.
- W&B tooling helps teams trace, evaluate, tune, and govern at production pace.
- Our pilot asks a focused question and tests it with reproducible evidence.

</div>

<div style="padding-top: 2rem; color: #ffffff;">
<span style="color: #58d3db;">If it cannot be reproduced and explained, it cannot be trusted.</span>
</div>

---
layout: center
class: text-center
---

# Questions

<div style="padding-top: 1rem; color: #cccccc;">
LLMOps for Drug Discovery Systems: Governing Models, Agents, and Evaluation
</div>
