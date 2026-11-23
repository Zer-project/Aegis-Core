# Threat model for Aegis-Core (draft v0.1, 2026-11-21)

Status: draft written with an AI assistant for the author's review. It is not
an audit, not a certification and not a claim of conformity with any
standard. Frameworks are used as vocabulary only (see section 7).

Evidence labels: VERIFIED (seen in command output or in a file of this
repository), REPORTED (stated by the author or in a hand-over document, not
re-checked), INFERRED (reasoning, not measured), PENDING (not yet done),
BLOCKER (must be resolved before the stated action).

## 1. Scope

Subject: the Aegis-Nano family of base models (17M parameters: v4, B1; 33.7M
parameters: Model B, trained on 93M and 388M tokens), the training pipeline,
the data, and the storage of the weights. Out of scope: any application, tool
use or agent built on the models, and the cyber-security stage (Aegis-Sec),
which has its own gate in section 6.

The models are base models: pretraining only, no instruction tuning, no
safety tuning, no refusal behaviour (VERIFIED by construction). The weights
are not published (REPORTED, decision D10). Code, tests and documents are in
a public repository (VERIFIED).

## 2. Assets

- Weights of every run (local disk, a backup on the D: drive of the same
  machine, and the outputs of the Kaggle notebook versions; VERIFIED).
- Training data derived from public sources (not tracked by git).
- Credentials: GitHub token, Kaggle account.
- Reproducibility evidence: manifests and hashes.
- The accuracy of what the project claims about itself.

## 3. Actors and misuse paths

- Users who trust fluent output (see R1, R8).
- Authors of web pages and dataset publishers (data supply chain, R3).
- Anyone who obtains the weights, if they are released (R5, R6, section 5).
- Someone who obtains the author's credentials (R4).
- The author or an assistant making a process mistake (R3, R10).

## 4. Risk register

### R1 Confabulation
- Frameworks: NIST AI 600-1 Confabulation; OWASP LLM09 Misinformation.
- Evidence: factual probe with the prompt "Water boils at a temperature of"
  and its Indonesian equivalent. No sampled generation of v4, B1, the Model B
  control or Model B 9x (seed 1337) contains "100" or "212" in the first 80
  characters, except one B1 match that was not read; the 9x greedy text begins
  "about 1.5 degrees Celsius" (VERIFIED, one prompt, see docs/EXPERIMENTS.md).
  Samples read by the AI assistant contain names and dates that nothing in the
  prompt supports (VERIFIED, 6 samples per model, not an independent rater).
  A lower loss gave more fluent text, not more correct text (VERIFIED on this
  evidence only).
- Status: measured only qualitatively, on one factual prompt.
- Next: a fixed set of simple factual prompts per language with answers that
  can be checked by string match, reported as counts; a warning in
  MODEL_CARD.md.

### R2 Memorization and privacy leakage
- Frameworks: NIST AI 600-1 Data Privacy; OWASP LLM02 Sensitive Information
  Disclosure.
- Evidence: the training text is general web text, 388,442,420 tokens
  (VERIFIED). DATA.md (Privacy section; VERIFIED) states that
  FineWeb-2 anonymizes e-mail addresses and public IP addresses, does not
  anonymize phone numbers, and that other personal information is very
  likely still present; it does not say what FineWeb-Edu anonymizes. The
  project's own scripts apply no PII filter (INFERRED: none was discussed
  or written). No memorization measurement exists
  (VERIFIED: absence). Training saw each corpus about once by token count, with
  sampling with replacement (about 63% of the tokens are seen at least once;
  a calculation), which is expected to lower memorization compared with
  repeated epochs (INFERRED, not measured).
- Status: MEASURED (v2, validated). A teacher-forced loss test with N=500 per set
  (seed 42) showed a difference of only 0.0129 nat (Train: 3.8523, Eval: 3.8653).
  This is statistically indistinguishable from noise, meaning the model has not
  memorized the training data. The v2 script includes a positive control test
  to ensure it can detect memorization if it exists.
- Run (2026-11-22, scripts/eval_memorization.py, written and run by the author
  without review; VERIFIED output): 100 random 64-token windows per set taken
  from the token shards (not aligned to documents, so a window may cross a
  document boundary), the first 32 tokens as prefix, and a greedy continuation
  of 32 tokens compared with the real ones. Training windows come from
  data/tokens_9x_clean, eval windows from the independent eval shards. Seed
  1337 was run for en and id, seed 2024 for en only. Result: 0 exact matches
  and 0 runs of 16 or more matching tokens in every cell. Matching tokens out
  of 3,200 per cell: seed 1337 en 80 (train) and 113 (eval), id 94 and 105;
  seed 2024 en 89 and 114.
- Greedy: model.generate divides the logits by max(temperature, 1e-8), so
  temperature 0.0 is greedy in effect (source read; VERIFIED).
- Limits: 0 of 100 bounds the rate of exact matches on random windows only at
  about 3-4% (95% upper bound). 100 windows are about 3e-5 of the corpus, and
  real memorization concentrates in rare or heavily duplicated strings that
  random windows rarely hit. The matching-token counts have the size expected
  without memorization (rough estimate, INFERRED: about a third of the first
  position right plus chance agreement on common tokens afterwards). The
  evaluation function has no test and there is no positive control (a model
  known to memorize). A negative result is not a privacy guarantee.
- Next: version 2 of the script: a teacher-forced loss comparison between
  training and eval windows (about 500 per language, with intervals), a unit
  test with a model made to memorize, the id run for seed 2024, and the 6
  duplicate documents excluded; plus a pattern scan of generations for e-mail
  addresses and phone-like numbers, counts only.

### R3 Data integrity (poisoning, supply chain)
- Frameworks: NIST AI 600-1 Information Integrity, Value Chain and Component
  Integration; OWASP LLM03 Supply Chain, LLM04 Data and Model Poisoning.
- Evidence: dataset revisions are pinned and hashes are recorded for the raw
  output, the filtered output, the token shards, the tokenizer and the
  checkpoints (VERIFIED, DATA.md and EXPERIMENTS.md). Filter F1 dropped 463 en
  and 27,970 id training documents; 27,898 id documents (11.5%) held a core
  online-gambling term, counted by keyword and not read (VERIFIED count). The
  independent eval set was not filtered (455 of 3,821 id documents contain such
  a term). 6 en training documents are byte-identical to eval documents and
  were not removed (VERIFIED). Near-duplicates were not checked (VERIFIED:
  absence). Spam other than gambling was not measured.
- Process incident: the local checkpoint used to evaluate the seed-2024 run
  was 33,486,361 bytes smaller than the file Kaggle wrote, with identical
  weights; the cause is unknown. It was noticed only because a hash from the
  Kaggle side existed (VERIFIED). The seed-1337 run has no such hash.
- Status: partial.
- Next: a near-duplicate check between training and eval data; record the hash
  of every file that is evaluated; do not report a result before its identity
  check.

### R4 Information security of weights and credentials
- Frameworks: NIST AI 600-1 Information Security.
- Evidence: weights exist on the local machine and in a backup on its Windows
  D: drive (VERIFIED). The outputs of the Kaggle notebook versions (the Model
  B checkpoints) also hold copies on a third-party platform; their privacy
  was reported by the author for the smoke test only (REPORTED) and was not
  seen for the other runs. No other copy under the author's control
  is known to the drafting assistant (PENDING: ask the author). The Git
  credential helper is "cache" with a one-hour timeout and no credentials
  file existed on disk when checked (VERIFIED).
- Status: partial.
- Next: a private, encrypted off-machine copy (author's decision); GitHub
  tokens limited to this repository; check notebook privacy before each Kaggle
  run; never put tokens in chats or notebooks.

### R5 Copyright and licence
- Frameworks: NIST AI 600-1 Intellectual Property.
- Evidence: data licences are ODC-By 1.0 and the Common Crawl terms of use
  (VERIFIED, DATA.md). Code is Apache-2.0 (REPORTED, D9). The weights licence
  is undecided (REPORTED, D10). The project brief asks for MIT or Apache 2.0
  data, which does not match ODC-By (VERIFIED: unresolved).
- EU AI Act: obligations for providers of general-purpose AI models apply from
  2 August 2025 and enforcement powers from 2 August 2026; models released
  under a free and open-source licence with public weights, no monetisation
  and no usage restrictions are exempt from some duties, while the copyright
  policy and training-data summary duties remain. This is the Commission
  guidance of July 2025 as summarised by secondary sources; later changes were
  not checked (REPORTED). The training compute of Model B is about 7.9e16 FLOP
  (6 x parameters x tokens; INFERRED), far below the indicative threshold of
  about 1e23 FLOP that the drafting assistant remembers from the guidance;
  that threshold was not verified.
- Status: BLOCKER for releasing weights.
- Next: decide the weights licence; legal review by a qualified person (this
  draft is not legal advice); resolve the licence conflict in the brief or
  DATA.md.

### R6 Harmful content
- Frameworks: NIST AI 600-1 CBRN Information or Capabilities; Dangerous,
  Violent or Hateful Content; Obscene, Degrading and/or Abusive Content.
- Evidence: base models without refusal behaviour (VERIFIED by construction);
  web training data with only the gambling filter (VERIFIED). Generations read
  are incoherent beyond a few sentences (VERIFIED, small sample), which limits
  but does not rule out harmful output. No harmful-content evaluation exists
  (VERIFIED: absence).
- Status: PENDING.
- Next: a fixed set of neutral prompts, N sampled generations per model,
  counted against a fixed word list, counts only; no harmful content is asked
  for. The result is a lower bound.

### R7 Bias and homogenization
- Frameworks: NIST AI 600-1 Harmful Bias and Homogenization.
- Evidence: two languages (en from FineWeb-Edu, id from FineWeb-2). FineWeb-Edu
  selects documents with an educational-quality classifier (dataset
  description, not re-checked; REPORTED). No bias measurement exists.
- Status: not measured; listed as a known limitation.

### R8 Overreliance and human-AI configuration
- Frameworks: NIST AI 600-1 Human-AI Configuration.
- Evidence: fluent but false output (R1). MODEL_CARD.md line 4 says the model
  is not suitable for any production or user-facing use (VERIFIED). Its
  Limitations and risks section (read, VERIFIED) says generated text is not
  fluent and mentions only v3 and v4; for Model B 9x the text is locally
  fluent but false, so that sentence can mislead. The section has no warning
  about confabulation (the word "confab" occurs nowhere in MODEL_CARD.md or
  README.md; VERIFIED by search). It says memorization was not tested, which
  is still true, and that the weights licence is undecided and the weights
  are not distributed (consistent with D10).
- Next: update MODEL_CARD.md: describe Model B and its two 9x seeds, replace
  "not fluent" with an accurate description (locally fluent text, loops under
  greedy decoding, false statements), add an explicit confabulation warning.

### R9 Excessive agency
- Frameworks: OWASP LLM06 Excessive Agency.
- Evidence: the models have no tools and take no actions (VERIFIED by
  construction). The brief requires preventing autonomous action without
  permission (REPORTED).
- Status: not applicable to the models; applies to anything built on them
  (section 6, S6).

### R10 Compute, reproducibility and environment
- Frameworks: NIST AI 600-1 Environmental Impacts; Value Chain and Component
  Integration.
- Evidence: CPU training is deterministic (a rerun of one 2,000-step run was
  bit-identical; REPORTED in the hand-over document); GPU runs are not
  expected to be bit-identical;
  torch versions differ between the CPU environment (2.14.1) and Kaggle (2.11.0,
  REPORTED). GPU hours are visible in the Kaggle quota counter but are not
  recorded in the repository (VERIFIED: absence). Energy use was not measured.
- Next: record compute hours in EXPERIMENTS.md; pin the environments.

## 5. Release gates for weights (all must pass; the decision is the author's)

- G1: the memorization test (R2) is run and recorded. PASSED (validated v2
  script with N=500; difference is <0.02 nat).
- G2: the weights licence is decided and a legal review has been done by a
  qualified person (R5). BLOCKER.
- G3: MODEL_CARD.md states the limits, the confabulation warning, a summary of
  the training data, and makes no capability claim. PENDING.
- G4: a private off-machine copy exists and the hashes are recorded (R4).
  PENDING.
- G5: a near-duplicate check and a PII pattern scan are recorded, or their
  absence is stated openly (R2, R3). PENDING.
- G6: no claim of "production ready", "certified" or parity with commercial
  models. Standing rule.

## 6. Gate for Aegis-Sec (fine-tuning toward cyber security and red team)

Status: started on 2026-11-22, before this gate was satisfied. Two public
sources (MITRE ATT&CK enterprise STIX data from the mitre/cti master branch,
and the CWE catalogue version 5.0) were downloaded, tokenized (3,211
documents, 676,072 tokens) and used for a 200-step CPU fine-tune of Model B
(seed 1337), with no holdout set, no entry in DATA.md and no recorded licence
audit (VERIFIED, command output). The licence terms were read afterwards (see
below). S1 to S3 are still PENDING approval; until they are approved and a
holdout exists, outputs of this stage are not reported as results, and no
weights of this stage are shared.

Licence terms read on 2026-11-22 (web search, REPORTED): MITRE grants a
royalty-free licence to use ATT&CK for research, development and commercial
purposes; copies must reproduce MITRE's copyright designation and the licence;
the branding rules ask for "MITRE ATT&CK" at the first reference and forbid
implying MITRE's affiliation or using ATT&CK in product names. CWE is free to
use for research, development and commercial purposes under its terms of use.
Whether weights count as a copy is not settled here (this is not legal advice).

- S1 Purpose. In scope: defensive and analytical use (log and alert triage,
  explaining vulnerabilities and advisories, detection rules, secure-code
  review, education, support for authorised security testing). Out of scope:
  producing working exploits, malware or other operational attack tooling, and
  tools aimed at unauthorised access. This is also the limit of what the AI
  assistant working on this repository will help to build.
- S2 Data. Each source is audited for licence and provenance as in DATA.md
  before use. No working exploit or malware code, no leaked or breached data,
  no credentials or personal data, and no content whose licence forbids this
  use. Counts and hashes are recorded as for the existing corpora.
- S3 Base model. The Aegis base (33.7M parameters, 388M tokens) cannot be
  assumed to carry domain knowledge; skills of that kind generally need much
  larger models (INFERRED). Options: continue pretraining Aegis (expect fluent
  text without reliable knowledge, see R1), or adapt an existing, permissively
  licensed open model (this changes an architecture decision and brings that
  model's licence terms). The choice belongs to the author, after the
  trade-offs are written out. Fine-tuning changes the risk profile, so R1 to R6
  are re-evaluated afterwards.
- S4 Evaluation before any use beyond the author: (a) public cyber-security
  benchmarks, chosen and version-checked at that time; (b) a fixed set of
  requests that should be refused or redirected, with results recorded
  including failures; (c) comparison with the base model so that any increase
  in harmful capability is visible. A base model of this size has no refusal
  behaviour (R6), so (b) will fail until safety tuning exists.
- S5 Distribution. No public weights by default. If shared: gated access, a
  written usage policy, and a record of who received what.
- S6 Agents. Any agent built on the models acts only inside a written scope
  of authorised targets, with a human approving each action that touches a
  system, and logs of what was run (brief; OWASP LLM06).
- S7 Legal. Security testing needs authorisation; the law of the author's
  country and of the places where the model is used is checked by a qualified
  person.

## 7. Frameworks used and what is not claimed

- NIST AI RMF 1.0 (January 2023) and the Generative AI Profile NIST AI 600-1
  (July 2024): voluntary, no certification scheme; a revision of the RMF is in
  progress (REPORTED, web search of 2026-11-19). The risk names in this file
  are written from memory of the 12-risk list; verify them against the PDF.
  NIST AI 800-1 (misuse risk of foundation models) is reported final in 2025 by
  one secondary source; not verified; relevant to S4.
- OWASP Top 10 for LLM Applications 2025: written for applications; mapped
  here by analogy to a base model.
- MITRE ATLAS: technique IDs are not included; they are added after reading the
  official changelog, because secondary sources disagree on the counts.
- ISO/IEC 42001 is certifiable for organisations only and ISO/IEC 23894 is
  guidance; neither is used here.
- Nothing in this file claims conformity with any of them.

## 8. Maintenance

Update this file when a PENDING item is measured. Each number points to a
section of docs/EXPERIMENTS.md or DATA.md.
