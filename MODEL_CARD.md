# Aegis-Nano: experimental small language models (v3, v4 and Model B)

**Status: research artifact.** Small models trained from scratch (v1 to v4 on
CPU, Model B on a Kaggle GPU). Their text is unreliable: it repeats itself and
states false things. They were not evaluated on any task benchmark. Not
suitable for any production or user-facing use.
Weights are **not published**; see "License".

## Model

- Decoder-only, pre-LayerNorm Transformer; GELU MLP (4x); learned position
  embeddings; tied input/output embeddings.
- 6 layers, 6 heads, embedding width 384, context 256 tokens, dropout 0.1 (v3) or 0.0 (v4),
  no bias terms.
- Tokenizer: byte-level BPE (Hugging Face `tokenizers`), vocabulary 16,384,
  trained on the training split only. Tokenizer sha256 `a31e3dcf2c889f7b44ced454e560676767fcba0e871fc9bb9c4c3cb017cf4153`.
- Parameters: 17,011,584.
- Source code: Apache-2.0, https://github.com/Zer-project/Aegis-Core

## Model B (GPU)

A larger model trained later on a Kaggle GPU (fp16) with the same tokenizer and
`scripts/train.py` 0.2.0. It has no separate name; it is "Model B" in
docs/EXPERIMENTS.md.

- 8 layers, 8 heads, embedding width 512, context 256 tokens, dropout 0.0, no
  bias terms, tied embeddings. Parameters: 33,694,208.
- Training data: the 9x corpus (`tokens_9x_clean`), 388,442,420 tokens
  (English 221,440,917, Indonesian 167,001,503), same sources and revisions.
  Documents with core online-gambling terms were removed from the training
  split (about 11.5% of the Indonesian documents; see DATA.md). The independent
  evaluation set was not filtered.
- 23,700 steps, batch 64, block 256 (388,300,800 tokens, about one pass by
  count; windows are sampled with replacement), AdamW, learning rate 1e-3 to
  1e-4 (cosine), warmup 500, weight decay 0.1, gradient clip 1.0. Two seeds,
  1337 and 2024. GPU training is not expected to be bit-reproducible.
- Checkpoint sha256: seed 1337
  `5ef282664c70fa6e971f250f78e0c970205350987a464740da50a590ad6c5c66`; seed 2024
  (the file written by Kaggle)
  `ac710d4df085ca8f970e978637888d0021858d6696e4a48481b0e79dd736a656`. The file
  used for the seed-2024 evaluations is a re-saved copy with identical weights
  (see docs/EXPERIMENTS.md).
- Independent evaluation set, window stride 4: loss 3.6627 (seed 1337; English
  3.5951, Indonesian 3.7581, perplexity 39.0) and 3.6496 (seed 2024; English
  3.5828, Indonesian 3.7438). A control with the same architecture trained on
  93.4M tokens for 5,700 steps (checkpoint sha256
  `59664abfaf487be68042902d5673e60e09d51926c06d8a709676ddb52ce105f5`) had loss
  3.968. Several things differ between the control and the 9x runs (tokens seen,
  steps, warmup, the gambling filter, the language mix), so the difference is
  not attributable to one factor.

## Training data

English (FineWeb-Edu `sample/10BT`) and Indonesian (FineWeb-2 `ind_Latn`) at
pinned revisions. Training split: 20,633 English and 26,623 Indonesian
documents, 46,651,978 BPE tokens in total. Licenses: ODC-By 1.0 and Common Crawl
Terms of Use; the dataset license does not clear the rights of every
underlying page. Provenance, holdout rule, dedup and limitations: see
[DATA.md](DATA.md). Only exact duplicates were removed.

## Training

CPU only, seed 1337 (v4 was repeated once with seed 2024: loss 4.6642, see
docs/EXPERIMENTS.md). 22,800 steps, batch 8, block 256
(46,694,400 tokens, about one epoch-equivalent; windows are sampled at random
with replacement, so not a single full pass; by calculation, roughly 63% of
tokens are seen at least once). AdamW (betas 0.9/0.95), weight
decay 0.1 on tensors with 2+ dimensions, learning rate 1e-3 to 1e-4 (cosine),
warmup 400, gradient clip 1.0. The learning rate was not tuned beyond a short-run comparison of 3e-4 and 1e-3
(see docs/EXPERIMENTS.md).
v3 checkpoint sha256 `5802ebe849b81ce31f1fb6fd623d1e25e199ebe323ca97d8444fcde760de3647`.
Environment: Python 3.14.4, torch 2.14.1+cpu.
v4 repeats v3 with `--dropout 0.0` and nothing else changed (about 17 h 43 min
on CPU). v4 checkpoint sha256
`ceec71d961dbd7c97bd55c34ab4c5156b52a21607329676a20d6379132a02522`.

## Evaluation

Metric: cross-entropy loss (nats) per BPE-16K token on non-overlapping
256-token windows. Perplexity depends on the tokenizer and is not comparable
with models that use a different one. At most two seeds per
configuration (v4 and Model B on 9x), no confidence intervals.

| Checkpoint | Steps | Loss, independent set | Loss, earlier holdout |
|---|---|---|---|
| v1 | 2,000 | 6.361 | 6.376 |
| v2 | 10,000 | 5.380 | 5.367 |
| v3 | 22,800 | 4.788 | 4.759 |
| v4 | 22,800 | 4.665 | not measured |

v3 on the independent set: perplexity 120.1 (English loss 4.740, perplexity
114.4; Indonesian loss 4.857, perplexity 128.6).
v4 on the independent set: perplexity 106.2 (English loss 4.622,
Indonesian loss 4.726). The independent set was
evaluated with window stride 4 (1,893,120 of 7,571,884 tokens). English
independence is at source-file level only; near-duplicates were not checked.
The earlier holdout shares source files with the training data.

No task-level benchmark, safety or memorization evaluation has been run.
Factuality was probed with a single prompt (the boiling point of water); this
is not a benchmark (see docs/EXPERIMENTS.md).

## Limitations and risks

- Generated text is unreliable. v3 and v4 text is not fluent (repetition,
  invented words, loops under greedy decoding). Model B text is locally fluent
  for a sentence or two, but it still loops under greedy decoding, changes
  topic, and contains specific names, dates and numbers that nothing supports.
  In a one-prompt factual check no model was shown to give the right answer.
  This rests on small samples read by an AI assistant, not on a rating.
- Confabulation: because Model B text sounds plausible it can mislead more
  than the rougher v4 text. Do not rely on any statement a model produces.
- Training data is web text that may contain personal information,
  copyrighted material and bias. Memorization of such data was not tested.
- No instruction tuning, safety tuning or output filtering.
- v4 has lower loss than v3, but a generation check showed no measurable
  reduction in repetition and no coherence gain (see docs/EXPERIMENTS.md).
- Model B on 9x has a much lower loss than v4 (3.66 against 4.67), but its
  repetition rate did not improve and greedy decoding still loops in every
  generation (see docs/EXPERIMENTS.md).
- A threat model draft is in docs/THREAT_MODEL.md.
- Provenance note: the `catatan` field in the token metadata files says
  "trial data, not final". It is stale; the shards contain the full sample
  described above. The files are left unchanged because their hashes are
  bound to the checkpoint.

## License

Code: Apache-2.0. Weights: **UNDECIDED**, and not distributed. This will be
decided after the dataset licenses, distribution terms and the implications
for derived weights are reviewed.
