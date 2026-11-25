# Aegis-Core roadmap (Phase 0 draft, 2026-11-25)

Status: PROPOSED by the AI assistant for the owner's review. No phase below is
approved until the owner records it in docs/DECISIONS.md. Labels as in
docs/THREAT_MODEL.md (VERIFIED, REPORTED, INFERRED, PENDING, BLOCKER).

Goal (REPORTED, owner): an open-source, transparent, locally runnable model
for cyber security and red-team work, which may later act as a bounded agent.
Scope limit (S1): defensive/analytical use and support for authorised testing;
no working exploits or malware. No claim of "compliant", "certified" or
"production ready" (G6); only "mapped to <framework> with evidence".

## Tracks

- Track A, Aegis-Nano (33.7M): methodology lab (data, SFT, evaluation, safety
  experiments). Not expected to hold reliable domain knowledge (S3, INFERRED).
- Track B, product model: an adapted open 7-8B base chosen in Phase 1 by a
  bake-off. Candidates (INFERRED, to verify): Olmo 3 7B, Qwen3-8B-Base;
  Foundation-Sec-8B as evaluation baseline only until its licence is checked.

## Phases and exit gates

Each phase ends only when its gate is met with recorded evidence.

| # | Phase | Exit gate (all written, with hashes/counts) | Status |
|---|-------|---------------------------------------------|--------|
| 0 | Governance foundation | ROADMAP, DECISIONS, updated THREAT_MODEL merged; owner answers D1-D6 | IN PROGRESS |
| 1 | Evaluation and base choice | Fixed eval suite (6 layers) runs end to end; bake-off report; licence of chosen base read | NOT STARTED |
| 2 | Data and licences | Each source audited per S2/DATA.md; train/eval split, near-dup and contamination check | NOT STARTED |
| 3 | Domain adaptation | LoRA/SFT first; CPT only if a gap is measured; results vs base | NOT STARTED |
| 4 | Instruction and safety tuning | S4(b) refusal set measured, failures recorded | NOT STARTED |
| 5 | Isolated capability tests | Isolated range only; written authorisation; no external targets | NOT STARTED |
| 6 | Local runtime | Documented local inference on stated hardware, measured | NOT STARTED |
| 7 | Bounded agent | S6 scope + policy gate in code outside the model + human approval + immutable log | NOT STARTED |
| 8 | Release and governance | G2-G5 passed; legal review done by a qualified person | NOT STARTED |

Time estimates are deliberately not given here; the earlier 19-34 week range
was INFERRED and is not evidence.

## Evaluation layers (Phase 1)

1. General regression. 2. Cyber knowledge benchmarks (chosen and
version-checked at that time). 3. Execution/CTF/range tasks (isolated).
4. Refusal set (S4b). 5. Privacy and contamination. 6. Confabulation.
Contamination warning (INFERRED): MITRE/CWE training text may overlap with
benchmark items built from the same sources; check before reporting scores.

## Compute

Local machine: specifications PENDING (owner). Kaggle free GPU is a
candidate for Track A and short Track B runs. Its quota and session limits are
NOT VERIFIED: Kaggle documentation pages could not be read by the tooling on
2026-11-25; a community post titled "floating GPU quota" mentions a 30-hour
baseline (REPORTED, truncated, undated). The owner can read the current
numbers in the Kaggle account page. Inference memory for 7-8B fp16 is about
14-16 GB (INFERRED), which is tight on one T4 (no native bf16); to be measured
in Phase 1.

## Training rule

No training run for Track B before the Phase 1 gate. Track A SFT code exists
as a separate, not yet merged patch (tests not yet run by the owner).
