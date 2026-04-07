# Note: Do not modify this file
# Principles
- Goal:
    - BioReason-Pro Demo Design: Problem Setting, Verification Steps, and Implementation Procedure
    - Can additional training be established using only disease-related GO annotations after the November 2022 cutoff, oriented toward disease context?

0. Why We Are Doing This Demonstration
0.1 The Question This Demo Aims to Answer

"To what extent can AI 'understand' and predict protein function?"

Current computational biology tools suffer from a fundamental problem: they can predict but cannot explain. BLAST can transfer function based on sequence similarity, but it cannot articulate why a protein has that function. Deep learning models have improved accuracy but remain black boxes.
BioReason-Pro is the first model to address this question through "reasoning traces." This demo validates that capability in the next challenge domain explicitly identified by the paper's authors — disease-associated proteins — and demonstrates that the capability can be extended through additional training from a checkpoint.

0.2 Anticipated Tough Questions and Answers

Q1. "Isn't this just a reproduction experiment of the paper?"

A. The paper only evaluated on the CAFA5 benchmark (general-purpose proteins). This demo ventures into the "disease-associated proteins" domain, an untested area that the paper's authors themselves listed as future work. While using the same architecture, this is an extension validation — not a reproduction experiment — in that it validates capabilities in a disease context that the paper could not demonstrate.


Q2. "Why is additional training from a checkpoint necessary? Isn't the general-purpose model sufficient?"

A. BioReason-Pro RL was trained on a broad dataset of 133,492 proteins across 3,135 species, but the training data is predominantly general-purpose GO annotations. Reasoning patterns specific to disease-associated proteins — mechanisms by which mutations cause pathway dysfunction, functional context of rare disease proteins — are relatively underrepresented.
The significance of additional training is twofold. First, disease-associated proteins have few similar sequences (due to the nature of rare diseases), making this the domain where sequence-similarity-based methods perform worst. Second, verifying whether disease mechanism descriptions become incorporated into reasoning traces is itself a contribution to the paper's unresolved question: "Is BioReason-Pro truly reasoning, or performing sophisticated pattern mimicry?"


Q3. "Is there no overlap with BioReason-Pro's training data?"

A. BioReason-Pro's training data cutoff is November 2022. The fine-tuning data for this demo uses only disease-associated proteins that received experimental GO annotations in UniProt after 2023. Evaluation data is drawn from an even later period (2025 onward). Temporal holdout ensures data independence.


Q4. "Why not ProteinGym fitness prediction or virus-host interaction prediction?"

A. These were considered and rejected due to fundamental architectural incompatibility. BioReason-Pro is optimized for the input-output pattern "1 protein -> GO terms + reasoning trace," and its core component GO-GPT is specialized for autoregressive prediction of GO annotations.
ProteinGym fitness prediction requires "wild-type vs. mutant delta -> continuous value," which needs dual-sequence input and a continuous-value output head. Virus-host interaction prediction requires "2 protein pairs -> interaction presence/absence," which similarly requires architectural changes. In both cases, GO-GPT would serve no purpose, and the condition of "using BioReason-Pro" would be satisfied only nominally.
Disease-associated protein GO prediction is viable without any architectural changes, and all components — GO-GPT, ESM3, and RL — function as-is.


Q5. "Shouldn't DeepGOPlus be included as a comparison target?"

A. DeepGOPlus has legacy dependencies (Python 3.7 + TensorFlow + CUDA 10.1), and stable operation on modern GPU environments cannot be guaranteed. It is intentionally excluded because fair comparison conditions cannot be ensured. Instead, we use BLAST (sequence-similarity-based standard baseline) and ESM3 alone (standalone use of a protein foundation model) as comparison targets, creating a setup that clearly demonstrates the "limits of sequence similarity-based methods" and the "limits of foundation models without reasoning."


Q6. "What are the issues with synthesizing reasoning traces using GPT-5?"

A. BioReason-Pro itself was SFT-trained using GPT-5-synthesized reasoning traces, and this demo follows the same methodology. As the paper acknowledges, the limitation that "subtle reasoning errors may be introduced in synthesized reasoning traces" exists, but this is the same condition as the base model. After synthesis, consistency checks against disease-related biological facts are performed via manual review on a 10% sample basis.


Q7. "What can and cannot be concluded from this demo?"

A. What can be concluded: Whether additional training from a general-purpose checkpoint improves GO prediction accuracy (F_max) in the domain of disease-associated proteins. Whether disease-context descriptions increase in reasoning traces.
What cannot be concluded: A definitive answer to the fundamental question of whether BioReason-Pro is "truly reasoning or performing sophisticated pattern mimicry." The paper's authors themselves left this as an open question, and this demo shares that limitation.

# Approach
## Proceed using a layer model
- Confirm or define L4
- Create L3 (specification / ADR)
- Transcribe L2 (tests) from L3
- Write L1 (implementation) to satisfy L2
- files
    - L4: domain/foundation/philosophy.md
    - L3: domain/specification/business-rules/specipication.md
   
## Prohibitions
- Do not write code (L1) without a specification (L3)
- Do not write tests (L2) by reading code
- Do not derive test expected values from code
- Do not generate "implementations that just happen to work"

## Domain Design Rules
- Write business rules in Entities
- UseCases are for orchestration only
- Isolate external dependencies in Adapters/Repositories
- Unify domain terminology (Ubiquitous Language)

## ADR Rules
Create a file (.md) in the following cases:
Destination: domain/specification/adr
- Technology selection decisions
- Model/algorithm changes
- Domain definition changes
- Structural changes (e.g., flat -> segmented)

ADRs must include:
- Context
- Decision
- Rationale
- Consequences

## Dependency Rules
- UseCases depend on interfaces
- Repositories are injected via DI
- Tests use InMemory implementations

## Dependency Rules
- UseCases depend on interfaces
- Repositories are injected via DI
- Tests use InMemory implementations

## Task Management
- Managed via GitHub repository
- Tasks are tagged with Phase, order, and type (type granularity should allow independent work) to enable parallel execution
