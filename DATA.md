# Data Sources and Licenses

Status (2026-11-03): the full sample (see "Full sample") and an independent
evaluation set (see "Independent evaluation") have been downloaded and used for
tokenizer and model training and for evaluation. Both were downloaded and used
before this file was updated, which contradicts the earlier rule that this file
is updated first. The small trial run (2026-10-25, see "Trial run") only tested
the sampling script and is not used for training or evaluation.

## Source datasets

| Source | Language | License | Config |
|---|---|---|---|
| HuggingFaceFW/fineweb-edu | English | ODC-By 1.0 | sample-10BT |
| HuggingFaceFW/fineweb-2 | Indonesian | ODC-By 1.0, subject to Common Crawl Terms of Use | ind_Latn |

Only a small slice of each source is used (streamed, not fully downloaded).
Revisions are pinned in the next section.

## Pinned revisions

Checked on 2026-10-25 through the Hugging Face API (metadata only, no data downloaded).

| Dataset | Config / path | Revision (commit) | Last modified (UTC) | License tag |
|---|---|---|---|---|
| HuggingFaceFW/fineweb-edu | sample-10BT | 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 | 2025-07-11 | odc-by |
| HuggingFaceFW/fineweb-2 | ind_Latn | af9c13333eb981300149d5ca60a8e9d659b276b9 | 2025-10-27 | odc-by |

The dataset cards list FineWeb-Edu v1.4.0 (2025-07-11) and FineWeb-2 v2.1.1
(2025-10-27) as their latest changelog entries. These dates match the
last-modified dates above. Loading code must pass these commit hashes as the
`revision` argument so every contributor gets identical data.

Server-side file sizes (metadata, not downloaded): `sample/10BT` has 14 files
(about 28.5 GB); `data/ind_Latn/train` has 32 files (about 152.2 GB).

## Rules

- Raw data is NOT stored in this repository. The `data/` folder is git-ignored.
- Attribution to the sources above is required by ODC-By and is kept in this file.
- Evaluation documents are kept separate from tokenizer and model training data.
- The FineWeb-2 `test` split is reserved for evaluation only. It is never used
  to train the tokenizer or any model. The dataset card says most languages
  include a small test split that should not be trained on. Its presence for
  `ind_Latn` was verified on 2026-10-25 (metadata only): `data/ind_Latn/test`
  has 1 file of 58,341,633 bytes. It was not read or downloaded.
- FineWeb-Edu has only a `train` split on its card. English evaluation
  documents are held out from `sample-10BT` by a deterministic rule on the
  document `id` (see "Holdout rule") and excluded from training.
- Deduplication: exact duplicate texts are removed inside the sample (see
  "Sampling method"). Results for the full sample will be documented here.

## Sampling method

The sample is built by `scripts/sample_corpus.py` (version 0.1.4 as of 2026-11-14;
the trial run used 0.1.1). It plans first and downloads only when run with `--run`.

- Plan mode reads only parquet footers (metadata) and prints the worst-case
  traffic. No document text is read.
- With `--run`, only the `text` and `id` column chunks of the selected row
  groups are downloaded, using HTTP Range requests on the pinned revisions.
- Traffic caps per language: 150 MB for the trial profile and 400 MB for the
  full profile. The effective budget is 90% of the cap. The script never raises
  a cap by itself and stops if the plan does not fit. Its byte counter counts
  requested response bytes only; it is an estimate, not a measure of network
  traffic.
- Targets: trial = about 100 training documents per language; full = 100 MB of
  training text per language (Indonesian:English 50:50).
- Exact duplicate texts are removed within the sample (64-bit BLAKE2b digest of
  the text). Near-duplicates are not removed.
- Output goes to `data/`, which is git-ignored. `MANIFEST.json` records the
  script version and SHA-256, the pinned revisions, the files and row groups
  used, statistics, and checksums. It contains no document text and no
  document ids.

## Holdout rule

About 1% of the sampled documents (both languages) are set aside for evaluation.
A document is an evaluation document when

    int.from_bytes(sha256(utf8(id)).digest()[:8], "big") % 100 == 0

The decision depends only on the document `id`. Evaluation documents are written
to separate `*_eval.jsonl` files, are not counted toward the training target,
and are never used to train the tokenizer or any model. The official FineWeb-2
`test` split is a separate, additional evaluation source and is not read by
`sample_corpus.py`. It is read by `scripts/sample_eval.py` (see "Independent
evaluation").

## Trial run (2026-10-25)

A trial run of version 0.1.1 tested the script on the real sources. It
downloaded one row group per language (100 training documents each; 0 English
and 2 Indonesian evaluation documents) with about 20 MB of counted traffic
(about 21 MB measured on the network interface). The trial data was downloaded
before this section was written. It is for testing only: the cheapest row
groups were chosen, so it is not representative, and it is not used for
training or evaluation.

## Full sample (2026-10-30)

Built by `scripts/sample_corpus.py` v0.1.3 with `--profile full` on the pinned
revisions. Numbers below are read from `data/sample/MANIFEST.json`.

| Language | Source files used | Train docs | Train bytes | Holdout docs | Holdout bytes |
|---|---|---|---|---|---|
| en | 3 of 14 (`002`, `007`, `011`) | 20,633 | 100,001,478 | 201 | 1,069,075 |
| id | 4 of 32 (`000`-`003`, `_00004`) | 26,623 | 100,002,975 | 284 | 1,013,808 |

No exact duplicates were found inside the sample. Near-duplicates were not
checked. The holdout documents come from the same source files as the training
documents, so that separation is at document level only. The traffic cap was
400 MB per language (effective 360 MB); counted requests were about 70 MB (en)
and 93 MB (id).

## Full sample 2x (2026-11-11)

Built by `scripts/sample_corpus.py` v0.1.4 with `--profile full2x` on the same
pinned revisions and the same source files as the full sample (en `002`, `007`,
`011`; id `000`-`003`, `_00004`), with more row groups. Output is
`data/sample_2x/` (not tracked by git; see `MANIFEST.json` there).

Training documents byte-identical to an independent-eval document were removed
with `scripts/dedup_against_eval.py` (exact text match only): 1 en document, 0 id
documents. The result is `data/sample_2x_dd/`, which was tokenized with the same
tokenizer (sha256 `a31e3dcf2c889f7b44ced454e560676767fcba0e871fc9bb9c4c3cb017cf4153`)
into `data/tokens_2x`.

| Language | Train docs | Train bytes | Train tokens | Holdout docs | Holdout tokens |
|---|---|---|---|---|---|
| en | 42,094 | 203,905,805 | 49,464,996 | 392 | 448,522 |
| id | 53,894 | 204,777,386 | 43,859,999 | 561 | 450,286 |

Near-duplicates were not checked. As before, holdout documents come from the same
source files as the training documents. This corpus has not been filtered for
spam or advertising. A keyword count (not verified by reading documents) found
that about 9% of its id documents (4,754 of 53,894) contain online-gambling terms
(slot, judi, togel, gacor, maxwin) at a density above 0.5%, and about 27% trip a
broader commercial-keyword rule. The broader rule also matches ordinary words
(harga, situs, daftar), so it is not an estimate of spam.

## Full sample 9x (2026-11-16)

Built by `scripts/sample_corpus.py` with `--profile full9x` on the same pinned
revisions and the same source files as the full sample (en `002`, `007`, `011`;
id `000`-`003`, `_00004`), with more row groups. The manifest
(`data/sample_9x/MANIFEST.json`, created 2026-11-16T16:41:33Z, not tracked by git)
records the script sha256
`3ce9292e64c090aa7b8dad669522d2b5be0fcd0b63f2c674b8f09b5873ecf1aa` and the
version label 0.1.4; the label was not bumped when the `full9x` profile was added.
The traffic cap was 900 MB per language (effective 810 MB); counted requests were
about 558 MB (en) and 408 MB (id).

| Language | Train docs | Train text bytes | Holdout docs | Duplicates skipped by the sampler |
|---|---|---|---|---|
| en | 188,393 | 900,001,317 | 1,796 | 149 |
| id | 242,586 | 900,004,805 | 2,382 | 0 |

Raw output hashes (sha256): `en_train.jsonl` `8e10d492d37e2ca62de9c50c260f18672cb55888501277057049d708f92301d0`,
`id_train.jsonl` `4cd74052a5ad3609c0f124217c751f25ae9219cf5c31f5053d8012ccf53c4ee3`.

### Filter F1 (training split only)

`scripts/clean_f1.py` (sha256 starts with `8a9b502237eb`) writes
`data/sample_9x_clean/`. A document is dropped if it has fewer than 50 words or
contains any of the terms slot, judi, togel, gacor, maxwin, "situs slot" or
"bandar togel" (case-insensitive, whole words). One matching word is enough; there
is no density threshold. Bad JSON lines would be skipped without being counted,
but the counts below add up to the input counts, so none were skipped.

| Language | Dropped: short only | Dropped: core term | Total dropped | Kept |
|---|---|---|---|---|
| en | 57 | 406 | 463 (0.25%) | 187,930 |
| id | 72 | 27,898 | 27,970 (11.5%) | 214,616 |

The word "slot" can be legitimate in English (for example "time slot"); how many of
the 406 en documents are not gambling was not checked. Hashes of the kept files:
`en_train.jsonl` `e7c50107322a6fbec1e8f6e0176ea7082a8e2fbaa9734d90172bfb9b6c0a4f8a`,
`id_train.jsonl` `98b73d353974f9ecb657f334163fa55312b2a90e773196454ef669d2d2cde829`.
The holdout and the independent evaluation set were not filtered
(`sample_9x_clean` holds only train files).

### Tokens

Tokenized with the same tokenizer (sha256
`a31e3dcf2c889f7b44ced454e560676767fcba0e871fc9bb9c4c3cb017cf4153`) into
`data/tokens_9x_clean`: en 221,440,917 tokens, id 167,001,503 tokens, total
388,442,420; no document was skipped. `meta_train.json` sha256:
`2f84aca435dfc7a12ffb90c4ca903a8558e2d2cc2d135415e7b96b74035aa5ee`.

### Separation from the independent evaluation set

- Source files: no overlap. The independent evaluation set uses en `003` and `010`
  and the official id test split.
- Exact text (sha256 of the text), filtered train against the independent
  evaluation set: 6 en documents and 0 id documents are identical. These 6 were
  not removed, and the model trained on them. Their effect on the loss is expected
  to be small but was not measured.
- Exact text, filtered train against this corpus's own holdout: 0.
- The earlier full sample (30 Oct, `data/sample`) has 0 exact-text matches with the
  independent evaluation set in both languages.
- Near-duplicates were not checked.
- The independent evaluation set was not filtered: 455 of its 3,821 id documents
  and 9 of its 3,999 en documents contain one of the core terms. A model trained
  on filtered data is therefore evaluated on some documents of a kind it never
  saw.

## Independent evaluation

Built by `scripts/sample_eval.py` v0.1.0 to avoid sharing source files with
training. Numbers are read from `data/eval_indep/MANIFEST.json`.

| Language | Source | Docs | Bytes |
|---|---|---|---|
| en | FineWeb-Edu files `003`, `010` (not used for training) | 3,999 | 17,893,135 |
| id | FineWeb-2 official `ind_Latn` test split (`000_00000.parquet`) | 3,821 | 14,343,144 |

One English document was dropped as an exact duplicate of a training document.
English independence holds at file level only: near-duplicates and shared
upstream pages were not checked. The Indonesian test split is the official
separate split.

## Provenance (from the dataset cards)

- Both datasets come from Common Crawl web pages. Use is also subject to
  Common Crawl's Terms of Use.
- FineWeb-Edu keeps pages from FineWeb that scored 3 or higher (scale 0 to 5)
  with an educational-quality classifier. The classifier was trained on
  annotations produced by Llama-3-70B-Instruct. `sample-10BT` is a random
  sample of about 10B GPT-2 tokens.
- FineWeb-2 identifies languages with GlotLID, deduplicates per language
  across all snapshots, and applies per-language filters.

## Privacy

- FineWeb-2 anonymizes email addresses and public IP addresses. Phone numbers
  are not anonymized. The card states that other personal information is very
  likely still present.
- Both cards refer to a removal / opt-out process for personal data and for
  webmasters. Common Crawl respects robots.txt at crawl time.

## Bias and known limitations

- Adult and toxic content was filtered only at URL level. A significant number
  of documents that could be considered toxic or harmful remain, and web biases
  may be reproduced.
- FineWeb-Edu: code content is likely not prevalent, and processing was not
  tailored to individual websites.
- FineWeb-2: the language classifier can confuse closely related languages, and
  some filters may remove certain content disproportionately (the card gives
  poetry as an example).
- These are web-crawled datasets. They may contain copyrighted text, personal
  information, and social biases. The dataset license does not clear the rights
  of every underlying page.
- This is not legal advice.

## Attribution

This project uses data from HuggingFaceFW/fineweb-edu and
HuggingFaceFW/fineweb-2, released under ODC-By 1.0 and derived from Common
Crawl. Use is also subject to Common Crawl's Terms of Use. This is not legal
advice.

## Citations

- Lozhkov, Ben Allal, von Werra, Wolf. FineWeb-Edu: the Finest Collection of
  Educational Content. 2024. doi:10.57967/hf/2497
- Penedo et al. FineWeb2: One Pipeline to Scale Them All. 2025. arXiv:2506.20920

## Cyber Security Data (Aegis-Sec v1)

Data sourced on 2026-11-22 for the initial Aegis-Sec fine-tuning run.

- **MITRE ATT&CK:** Enterprise STIX data (`enterprise-attack.json`) downloaded from `github.com/mitre/cti` (master branch). 2,228 documents processed (includes some revoked and deprecated objects). Warning: 92.8% of this dataset contains `(Citation: ...)` markers, which the model learned to mimic. Not filtered. License: MITRE grants a royalty-free license for research and commercial use, but requires copyright designation in copies.
- **CWE (Common Weakness Enumeration):** Version 5.0 downloaded from `cwe.mitre.org`. 983 weaknesses processed. License: Free for research, development, and commercial purposes under CWE Terms of Use.
- **CVE:** Download failed (404 Not Found). Not included in v1.

**Note on Licence Compliance:** As of this commit, the weights for Aegis-Sec are strictly experimental and not published. If shared, the MITRE ATT&CK copyright and license terms must accompany the distribution.
