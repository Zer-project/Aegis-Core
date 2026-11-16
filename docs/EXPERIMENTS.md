# Experiments log

Results from small CPU-only experiments on Aegis-Nano (17,011,584 parameters).
All numbers are cross-entropy loss (nats per BPE-16K token) on the independent
evaluation set described in [DATA.md](../DATA.md). Every row is a single training
run per seed; no confidence intervals. Checkpoints, logs and raw result files
are not distributed (the `data/` directory is git-ignored).

## Common setup

6 layers, 6 heads, width 384, context 256, vocabulary 16,384 (tokenizer sha256
`a31e3dcf2c889f7b44ced454e560676767fcba0e871fc9bb9c4c3cb017cf4153`), batch 8,
AdamW (betas 0.9/0.95), weight decay 0.1, gradient clip 1.0, seed 1337 unless
stated. Training data: 46,651,978 tokens (data meta sha256
`248d6cf8785d6161e0336a1f7c3bf108e64eaab98ba5a2cfa6e13ccd4d386fdf`).
Python 3.14.4, torch 2.14.1+cpu, 4 cores, no GPU. Training is deterministic on
this machine: a rerun of the 2,000-step baseline (seed 1337) produced a
bit-identical checkpoint.

## Stage C: learning rate and dropout (2,000 steps, warmup 100)

Evaluation: window stride 16 (473,344 tokens).

| Cell | Peak LR | Dropout | Seed 1337 | Seed 2024 | Mean |
|---|---|---|---|---|---|
| C1 (control) | 1e-3 | 0.1 | 6.3567 | 6.3536 | 6.3552 |
| C3 | 3e-4 | 0.1 | 6.4033 | 6.4073 | 6.4053 |
| C4 | 1e-3 | 0.0 | 6.3179 | 6.3110 | 6.3145 |

Pre-registered rule: a difference counts only if it exceeds 0.02 nat and has the
same sign for both seeds. Seed-to-seed spread within a cell was 0.003 to 0.007.

- Lower LR (C3 vs C1): +0.050 nat, worse in both seeds. Hypothesis rejected at
  this training length.
- No dropout (C4 vs C1): -0.041 nat, better in both seeds. Supported at this
  training length.

## Longer check: 6,000 steps (warmup 200, seed 1337, one seed)

Evaluation: window stride 16.

| Cell | Dropout | Loss | Perplexity |
|---|---|---|---|
| C1 | 0.1 | 5.7004 | 299.0 |
| C4 | 0.0 | 5.5585 | 259.4 |

Difference -0.142 nat. The 8-batch training-log evaluation was lower for dropout
0 at every checkpoint (steps 999 to 5999).

## v4: full run without dropout (22,800 steps)

Identical to the v3 baseline (warmup 400, peak LR 1e-3 to 1e-4 cosine, seed
1337) except `--dropout 0.0`. Evaluation: window stride 4 (1,893,120 tokens).

| Model | Dropout | English | Indonesian | All | Perplexity |
|---|---|---|---|---|---|
| v3 (baseline) | 0.1 | 4.7396 | 4.8568 | 4.7882 | 120.1 |
| v4 | 0.0 | 4.6220 | 4.7263 | 4.6653 | 106.2 |

Difference -0.123 nat, one seed. Checkpoint sha256: v3
`5802ebe849b81ce31f1fb6fd623d1e25e199ebe323ca97d8444fcde760de3647`, v4
`ceec71d961dbd7c97bd55c34ab4c5156b52a21607329676a20d6379132a02522`. v4 took
about 17 h 43 min on CPU. Numbers measured at different strides are not
comparable with each other.

## Generation check: v3 vs v4

`scripts/eval_generation.py`, 10 fixed prompts (5 English, 5 Indonesian),
128 new tokens, 3 seeds for sampling strategies. "rep4" is the share of
generations that contain a repeated 4-gram.

| Strategy | rep4 en (v3 / v4) | rep4 id (v3 / v4) |
|---|---|---|
| greedy | 1.00 / 1.00 | 1.00 / 1.00 |
| top-k 40 | 0.80 / 0.87 | 0.60 / 0.87 |
| top-k 40, repetition penalty 1.2 | 0.33 / 0.60 | 0.27 / 0.20 |
| top-p 0.9 | 0.73 / 0.87 | 0.80 / 0.87 |
| top-p 0.9, repetition penalty 1.2 | 0.27 / 0.40 | 0.20 / 0.07 |

Each cell is about 15 generations (5 for greedy). Differences are within noise.
Reading samples: v4 text is locally more fluent but still not coherent.
Greedy decoding loops in both models, and the penalty reduces repetition at the
cost of malformed tokens.

## B1: 2x corpus, same compute (22,800 steps)

Same recipe as v4 (dropout 0, LR 1e-3, warmup 400, seed 1337, 22,800 steps),
trained on `data/tokens_2x`: 93,324,995 tokens (en 49,464,996, id 43,859,999),
about 2.0x the v4 corpus, from the same source files with more row groups. One
training document that was byte-identical to an independent-eval document was
removed before tokenization with `scripts/dedup_against_eval.py` (en: 42,095 ->
42,094 docs). Exact-text overlap with the independent eval was 0 for id.
Near-duplicates were not checked.

Independent eval, stride 4, same 1,893,120 tokens:

| | v4 | B1 | Diff |
|---|---|---|---|
| en | 4.6220 | 4.6075 | -0.0145 |
| id | 4.7263 | 4.7093 | -0.0170 |
| all | 4.6653 | 4.6497 | -0.0156 |

The pre-set criterion (more than 0.02 nat lower than v4) was **not met**.
Checkpoint sha256: `2716ec521c7f5efc1d5db2e2cd395abc5605fecd16e4f4dacc0fb0a7eba33d14`.

The run was interrupted and resumed with `--resume` from step 11,000. In the
log, 37 steps were written twice; 34 are identical and the largest difference
is 4.8e-7. No uninterrupted run exists, so bit-identical final weights are not
shown. The `eval_loss` in the training log uses the `tokens_2x` holdout and is
not comparable with v4.

Generation check (same settings as v3/v4), rep4 en / id:

| Strategy | B1 | v4 |
|---|---|---|
| greedy | 1.00 / 1.00 | 1.00 / 1.00 |
| top-k 40 | 0.80 / 0.80 | 0.87 / 0.87 |
| top-k 40, repetition penalty 1.2 | 0.27 / 0.20 | 0.60 / 0.20 |
| top-p 0.9 | 0.80 / 0.87 | 0.87 / 0.87 |
| top-p 0.9, repetition penalty 1.2 | 0.27 / 0.00 | 0.40 / 0.07 |

Greedy decoding still loops in 100% of generations. About 15 generations per
cell; differences are within noise. B1 text was not read, and rep4 does not
measure coherence.

Limits: one seed. The compute budget is fixed, so B1 sees about 0.5 epoch of
its corpus; it does not test longer training on more data. Source files are the
same as the v4 corpus, so this does not test data diversity in general. No
claim about language ability is made.

## v4 second seed (22,800 steps, seed 2024)

Same recipe and data as v4 (`data/tokens_full`); only `--seed 2024` and
`--out-dir data/runs_v4_s2024` differ. Independent eval, stride 4, same
1,893,120 tokens:

| | v4 seed 1337 | v4 seed 2024 | Diff | B1 (seed 1337) |
|---|---|---|---|---|
| en | 4.6220 | 4.6152 | -0.0068 | 4.6075 |
| id | 4.7263 | 4.7332 | +0.0069 | 4.7093 |
| all | 4.6653 | 4.6642 | -0.0011 | 4.6497 |

Checkpoint sha256: `185799894e961869c28dfa831008efb6e2dd4f3ba17209f25f8fbaba327e3bef`.

The run was interrupted and resumed with `--resume` from step 14,000. 30 steps
were written twice in the log and all are identical. The cause of the
interruption is unknown (B1 was also interrupted). The training-log `eval_loss`
(8 batches) was 4.6788 here versus 4.6384 for seed 1337; it depends on the seed
and is not used for comparison.

Reading: the spread between the two v4 seeds on the total is 0.0011 nat, much
smaller than the B1 difference (-0.0156 and -0.0145 against the two seeds). On
this evidence the B1 gain is probably not seed noise, but two seeds are not a
statistical estimate, B1 has one seed, and the pre-set 0.02 criterion is still
not met. Per-language differences between seeds (about 0.007) have opposite signs.

## Qualitative generation check: v4 vs B1

One prompt per language, seed 1337, strategies greedy, top-k 40, and top-p 0.9
with repetition penalty 1.2, first 300 characters of each sample. The prompt
texts are not stored in the result files (`meta` lists only `en` and `id`). The
samples were read by the AI assistant, not by an independent human rater.

- Greedy loops in both models. en: sentence-level loops ("The city is ... of
  the city"). id: v4 collapses into lines of "-", B1 repeats one sentence.
- Top-k 40 repeats phrases within about 100 tokens in both models.
- The repetition penalty removes the loops but produces malformed tokens and
  invented names in both models.
- Short phrases are locally grammatical, but there is no coherence across
  sentences in either model.

Extra statistics over all greedy generations (n = 5 per cell):

| | v4 en | B1 en | v4 id | B1 id |
|---|---|---|---|---|
| Mostly punctuation | 0 | 0 | 1 | 1 |
| Mean position of first repeated 4-gram | 12.0 | 8.6 | 11.4 | 14.0 |

The differences point in opposite directions between languages and the cells
are tiny, so there is no signal. No evidence was found that B1 improves
generation over v4. The hypothesis that scale (17M parameters, 46-93M tokens)
is the main limit has not been tested. Spam or advertising drift in id output
was not checked systematically.

## Model B (GPU) generation evaluation: decoding strategies, en and id

Status: preliminary. One run per model. Source files: data/gen_eval/gen_eval_bgpu.json
(Model B, checkpoint ckpt_bgpu.pt, step 5700) and data/gen_eval/gen_eval_v4.json
(v4, step 22800). Both were produced by scripts/eval_generation.py (version
0.1.0) with the same tokenizer (data/tokenizer_cmp/v16384), the same 10 fixed
prompts (5 en, 5 id; identical prompt hash), seeds 1337+i for 3 seeds, and at
most 128 new tokens. Each sampling cell has 15 generations (5 prompts x 3
seeds, so not independent); each greedy cell has 5 (5 prompts, one seed).

Model B (identity as reported by the author, not read from the checkpoint):
33,694,208 parameters (8 layers, 8 heads, 512 embedding dim), trained for 5700
steps on a training set of about 93.4M tokens (tokens_2x), on Kaggle GPU (2x
Tesla T4). v4 has about 17M parameters and was evaluated at step 22800, so the
two runs differ in size and in steps; the training data and batch size of v4
are not recorded here.

Metrics: d2/d3/d4 are distinct-n ratios; rep4% is the share of generations
containing a repeated 4-gram; firstrep is the mean token position of the first
repeated 4-gram, averaged only over the generations that repeat (so it is not
comparable across cells with different rep4%); tokens is the mean generated
length. Repetition is measured on token IDs.

| strategy | lang | d2 | d3 | d4 | rep4% | firstrep | tokens |
|---|---|---|---|---|---|---|---|
| greedy | en | 0.0945 | 0.1 | 0.1056 | 1.0 | 12.2 | 128.0 |
| greedy | id | 0.0913 | 0.1016 | 0.1104 | 1.0 | 10.8 | 128.0 |
| greedy_rep1.2 | en | 0.6173 | 0.7032 | 0.7616 | 1.0 | 44.2 | 128.0 |
| greedy_rep1.2 | id | 0.6126 | 0.6667 | 0.6992 | 1.0 | 30.6 | 128.0 |
| t1.0_pure | en | 0.9499 | 0.9852 | 0.9957 | 0.3333 | 86.0 | 114.5333 |
| t1.0_pure | id | 0.9608 | 0.9798 | 0.9864 | 0.4 | 83.1667 | 123.4 |
| t0.8_k40 | en | 0.646 | 0.7582 | 0.8152 | 0.9333 | 42.5 | 116.1333 |
| t0.8_k40 | id | 0.7837 | 0.873 | 0.9189 | 0.8 | 58.75 | 120.2667 |
| t0.8_k40_rep1.2 | en | 0.9648 | 0.9942 | 0.9995 | 0.0667 | 110.0 | 119.8 |
| t0.8_k40_rep1.2 | id | 0.9674 | 0.9917 | 0.9978 | 0.1333 | 56.5 | 110.6667 |
| t0.8_p0.9 | en | 0.6602 | 0.755 | 0.8082 | 0.9333 | 42.2143 | 122.0 |
| t0.8_p0.9 | id | 0.7243 | 0.782 | 0.8064 | 0.8 | 54.5833 | 114.2667 |
| t0.8_p0.9_rep1.2 | en | 0.978 | 0.9958 | 0.9984 | 0.0667 | 77.0 | 111.8667 |
| t0.8_p0.9_rep1.2 | id | 0.9658 | 0.9871 | 0.9924 | 0.3333 | 89.2 | 107.4667 |

Comparison with v4 (rep4%, same protocol, from the two result files):

| Strategy | rep4 en (v4 / Model B) | rep4 id (v4 / Model B) |
|---|---|---|
| greedy | 1.00 / 1.00 | 1.00 / 1.00 |
| top-k 40 | 0.87 / 0.93 | 0.87 / 0.80 |
| top-k 40, repetition penalty 1.2 | 0.60 / 0.07 | 0.20 / 0.13 |
| top-p 0.9 | 0.87 / 0.93 | 0.87 / 0.80 |
| top-p 0.9, repetition penalty 1.2 | 0.40 / 0.07 | 0.07 / 0.33 |

Share of generations that stopped on EOS (penalized strategies), v4 / Model B:

| Strategy | en | id |
|---|---|---|
| top-k 40, repetition penalty 1.2 | 0.33 / 0.20 | 0.53 / 0.40 |
| top-p 0.9, repetition penalty 1.2 | 0.20 / 0.20 | 0.60 / 0.53 |

Findings:

- Without the repetition penalty, Model B and v4 are about the same: greedy is
  1.00 for both, and top-k 40 and top-p 0.9 differ by one generation of 15.
- With the penalty, English rep4 is lower for Model B in both strategies
  (0.60 to 0.07 and 0.40 to 0.07, i.e. 9 to 1 and 6 to 1 generations of 15).
  Indonesian is mixed: 0.20 to 0.13 under top-k 40, but 0.07 to 0.33 under
  top-p 0.9 (1 to 5 of 15).
- The English gap is the only large difference. Generations share 5 prompts, so
  the effective sample is smaller than 15 and significance is not established.
- Early stopping does not explain the English gap: Model B stopped on EOS no
  more often than v4, so its generations were not shorter.
- Under the penalty, distinct-n is about 0.99 or higher (d4 0.998 to 0.9995).
  That alone does not indicate coherence. The v4 vs B1 check above found that
  the penalty produces malformed tokens and invented names.
- Samples were read (7 generations, seed 1337: penalized top-k 40 and top-p 0.9
  on en prompts 0 and 1, penalized id prompt 0 for both strategies, and one
  unpenalized top-k 40 en). Without the penalty, the en sample collapses into a
  loop ("Vell-Vell-Vell-..."). With the penalty there are no loops, and the
  first sentence is usually fluent, but later text drifts between topics and
  contains contradictory or invented content (for example "dates back to the
  19th century" followed by "20,000 years after its destruction"; the names
  "Arrangel Röscherberg" and "Power Floor Stewart, NSW"; Europe, Japan and
  Rasulullah SAW in one id paragraph), and table or list fragments from the
  training data ("|Developmentary||Starting with Buddhism|"). This is a small
  hand-read sample, not a rating.
- Greedy firstrep (12.2 en, 10.8 id) is close to the v4 greedy values reported
  above (12.0 en, 11.4 id), so greedy looping looks unchanged. firstrep 110.0
  at rep4 0.0667 (top-k 40 with penalty, en) rests on a single generation.
- Pure sampling (t1.0_pure) gives rep4 0.33 en and 0.40 id with d4 about 0.99.
  It has no v4 counterpart in the comparison above.

What this does not show:

- Coherence or text quality of Model B. rep4% dropped under the penalty, but
  the samples read show no evidence that text quality improved.
- That Model B improves on v4 without a repetition penalty.
- That scale or training length is the cause. Model B has about twice the
  parameters of v4 but 5700 steps against 22800, and the data and batch size of
  v4 are not recorded here, so size, steps and data are all different.
- Variation beyond the fixed prompt set and the seeds used.

Next: add prompts and seeds, and report held-out eval loss for Model B and v4.

## What these results do not show

- Lower loss did not produce measurably less repetition or coherent text.
  No claim about language ability is made.
- Findings come from one seed at 6,000 steps, two seeds at 2,000 steps, and
  two seeds at 22,800 steps for v4 only (B1 has one). Only the LR values 3e-4 and 1e-3, and dropout 0 and 0.1, were tried.
- The model sees about one epoch-equivalent of data. Dropout 0 was not tested
  with repeated data.
- No task benchmark, factuality, safety or memorization evaluation has been run.

## Reproduce v4

```bash
python scripts/train.py --data-dir data/tokens_full --out-dir data/runs_v4 \
  --seed 1337 --batch-size 8 --block-size 256 --n-layer 6 --n-head 6 --n-embd 384 \
  --dropout 0.0 --max-steps 22800 --lr 1e-3 --min-lr 1e-4 --warmup 400 \
  --weight-decay 0.1 --grad-clip 1.0 --log-interval 10 --ckpt-interval 500 \
  --eval-interval 2000 --eval-batches 8
python scripts/evaluate_lm.py --ckpt v4=data/runs_v4/ckpt.pt \
  --eval-set indep=data/tokens_eval_indep:4 --out data/eval_results/results_v4_s4.json
```

## Reproduce B1

```bash
python scripts/sample_corpus.py --profile full2x --run
python scripts/dedup_against_eval.py \
  --train-files data/sample_2x/en_train.jsonl data/sample_2x/id_train.jsonl \
  --eval-files data/eval_indep/en_eval.jsonl data/eval_indep/id_eval.jsonl \
  --out-dir data/sample_2x_dd
python scripts/prepare_tokens.py --split train \
  --files data/sample_2x_dd/en_train.jsonl data/sample_2x_dd/id_train.jsonl \
  --tokenizer-dir data/tokenizer_cmp/v16384 --out-dir data/tokens_2x
python scripts/prepare_tokens.py --split eval \
  --files data/sample_2x/en_eval.jsonl data/sample_2x/id_eval.jsonl \
  --tokenizer-dir data/tokenizer_cmp/v16384 --out-dir data/tokens_2x
python scripts/train.py --data-dir data/tokens_2x --out-dir data/runs_b1 \
  --seed 1337 --batch-size 8 --block-size 256 --n-layer 6 --n-head 6 --n-embd 384 \
  --dropout 0.0 --max-steps 22800 --lr 1e-3 --min-lr 1e-4 --warmup 400 \
  --weight-decay 0.1 --grad-clip 1.0 --log-interval 10 --ckpt-interval 500 \
  --eval-interval 2000 --eval-batches 8
python scripts/evaluate_lm.py --ckpt b1=data/runs_b1/ckpt.pt \
  --eval-set indep=data/tokens_eval_indep:4 --out data/eval_results/results_b1_s4.json
```
