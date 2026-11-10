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

## What these results do not show

- Lower loss did not produce measurably less repetition or coherent text.
  No claim about language ability is made.
- Findings come from one seed at 6,000 and 22,800 steps, and two seeds at
  2,000 steps. Only the LR values 3e-4 and 1e-3, and dropout 0 and 0.1, were tried.
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
