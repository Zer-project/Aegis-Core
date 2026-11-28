# Kaggle runbook: base-model bake-off, step 1 (bits per byte)

Status: DRAFT, not yet run. Purpose: compare candidate base models (docs/ROADMAP.md, Phase 1)
on domain text, general English and general Indonesian with one tokenizer-independent metric.
Rules: private notebook and private dataset; never put tokens or passwords in a notebook or
chat; record GPU hours used from the Kaggle quota counter (THREAT_MODEL R10).

## 1. On the local machine (WSL)

```bash
python scripts/make_eval_bundle.py \
  --cyber data/cyber_dataset/aegis_sec_v2_eval.jsonl \
  --en data/eval_indep_f1/en_eval.jsonl --id data/eval_indep_f1/id_eval.jsonl \
  --n-docs 200 --out-dir data/kaggle_bundle_v1
cat data/kaggle_bundle_v1/manifest.json
```

Copy the folder somewhere Windows can see (for example /mnt/c/Users/<name>/Downloads) and
upload its 4 files as a Kaggle Dataset, visibility PRIVATE (Create, New Dataset).

## 2. In a Kaggle notebook (Accelerator: GPU T4 x2; Internet: on)

```python
!pip -q install -U "transformers>=4.57" accelerate bitsandbytes
!git clone -q https://github.com/zerforge/Aegis-Core.git && git -C Aegis-Core log -1 --oneline
```

One cell per model (each run is a separate process, so memory is freed in between):

```python
D = "/kaggle/input/<dataset-slug>"
!python Aegis-Core/scripts/eval_hf_bpb.py --model-id allenai/Olmo-3-1025-7B --revision a81bae42db3975be1671e27b9c9a56da1a9f980f \
  --corpus cyber={D}/cyber_eval.jsonl --corpus en={D}/en_eval.jsonl --corpus id={D}/id_eval.jsonl \
  --out /kaggle/working/bpb_olmo3_7b.json
```

```python
!python Aegis-Core/scripts/eval_hf_bpb.py --model-id Qwen/Qwen3-8B-Base --revision 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --corpus cyber={D}/cyber_eval.jsonl --corpus en={D}/en_eval.jsonl --corpus id={D}/id_eval.jsonl \
  --out /kaggle/working/bpb_qwen3_8b.json
```

If a run fails with a non-finite loss (fp16 overflow), retry once with `--dtype bf16`
(T4 has no native bf16; it may be slow or fail) or `--load-4bit`, and record which was used:
results from different precisions are not directly comparable.

Revisions above are the ones used on 2026-11-28 (see docs/EXPERIMENTS.md). The result JSON
records the resolved revision from scripts/eval_hf_bpb.py 0.1.1 onward.

Paste the printed summary (bits per byte and per token per corpus, time, peak memory) and the
`commit_hash` field of each JSON into the chat. Lower bits per byte is better.

## What this does not show

Bits per byte measures how well a model predicts text. It does not measure cyber knowledge,
instruction following, refusal or safety. The byte attribution is proportional (see the script
docstring). Step 2 (multiple-choice cyber benchmarks and a project-written Indonesian set)
needs its own licence check first.
