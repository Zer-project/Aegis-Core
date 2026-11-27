# Decision log (ADR)

Rule: every entry is PROPOSED until the owner writes ACCEPTED with a date.
The assistant never records an acceptance on the owner's behalf.

## D1 Target language(s) - ACCEPTED 2026-11-25 (owner's answer in session)
English + Indonesian. Impact: data, eval sets, tokenizer. Risk (INFERRED): public
cyber benchmarks found so far (CTIBench, CyberMetric) are in English, so Indonesian
evaluation sets must be written by the project; see ROADMAP, Phase 1 inputs.

## D2 Red-team scope - ACCEPTED 2026-11-25 (owner's answer in session)
S1 as written (defensive/analytical use and support for authorised testing). Working
exploits and malware stay out of scope, and outside what the assistant will help build.

## D3 Base model for Track B - PROPOSED
Owner's stated preference (2026-11-25): Olmo 3 7B; Qwen3-8B-Base as comparison. The choice
is made by the Phase 1 bake-off on the fixed eval suite, not by preference. Facts read on
2026-11-28 from the Hugging Face cards (see ROADMAP): both Apache 2.0. Open question: D1
needs Indonesian; Olmo 3 is trained on Dolma 3 (not described as multilingual on the card
page read; INFERRED to be mostly English), Qwen3 states 119 languages. The bake-off must
include Indonesian prompts before D3 is accepted.

## D4 Compute plan - PROPOSED
Local hardware (PENDING: owner to state RAM/GPU/VRAM) plus Kaggle when
needed. Kaggle limits NOT VERIFIED (see ROADMAP). Record GPU hours per run
(R10).

## D5 Adaptation method - PROPOSED
LoRA/SFT first; continued pretraining only if a measured gap justifies it.

## D6 Agent autonomy - PROPOSED
Stages 0-3; no stage without written human approval. Policy gate lives in
code outside the model; model output returns to the model as data (R9, S6).
