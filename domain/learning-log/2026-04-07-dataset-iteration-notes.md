# 2026-04-07: Dataset Iteration Notes

This memo records lessons from dataset iterations that were excluded from the final specification.  
`domain/specification/busiless-rules/specification.md` does not include the comparison versions or candidate proposals described here.

## 1. Comparison Benchmark

- A `214 -> 221 -> 225 -> 228` version was also created once
- Measured: train `932 proteins / 1,898 unique labels`, validation `662 proteins`, test `969 proteins`
- This is not the final benchmark; it is treated as a comparison variant for examining archive choice and count sensitivity

## 2. Broader Filter

- A broader query using `cc_disease:*` as the main condition was also examined
- Live count was `5,093 proteins`
- The final benchmark adopts the high-confidence filter that includes `MIM/Orphanet`

## 3. Small-Data Decision

- The current benchmark is the small-data version
- However, with a time-independent benchmark and strict split rules, it was judged viable as a target for additional training validation
- `3,000 unique labels` was assessed as a comfort-level guideline, not a hard gate

## 4. Local Scratch and W&B

- Previously, local `data/artifacts/...` was written about somewhat strongly
- The current policy organizes local output as scratch, with W&B Artifact ref as the source of truth
- The final spec explicitly states "do not retain unnecessary local datasets"

## 5. NK / LK

- NK / LK was included as a supplementary analysis target in EDA at one point
- However, the final spec does not place them in required benchmark artifacts or go/no-go conditions
- If needed, they will be handled separately as future analysis artifacts
