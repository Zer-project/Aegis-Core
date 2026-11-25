# Decision log (ADR)

Rule: every entry is PROPOSED until the owner writes ACCEPTED with a date.
The assistant never records an acceptance on the owner's behalf.

## D1 Target language(s) - PROPOSED
Options: English only; English + Indonesian. Impact: data, eval sets,
tokenizer (Aegis-Nano has en+id). Owner decision needed.

## D2 Red-team scope - PROPOSED
Options: S1 as written (defensive/analytical + support for authorised
testing); or narrower. Wider scope (working exploits/malware) is outside what
the assistant will help build. Owner decision needed.

## D3 Base model for Track B - PROPOSED
Candidates: Olmo 3 7B, Qwen3-8B-Base (comparison). Decided by Phase 1
bake-off on the fixed eval suite, after reading each licence. Not decided.

## D4 Compute plan - PROPOSED
Local hardware (PENDING: owner to state RAM/GPU/VRAM) plus Kaggle when
needed. Kaggle limits NOT VERIFIED (see ROADMAP). Record GPU hours per run
(R10).

## D5 Adaptation method - PROPOSED
LoRA/SFT first; continued pretraining only if a measured gap justifies it.

## D6 Agent autonomy - PROPOSED
Stages 0-3; no stage without written human approval. Policy gate lives in
code outside the model; model output returns to the model as data (R9, S6).
