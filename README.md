# Remark

Flag risky UPI/NEFT/IMPS payments from their free-text narration with a small, calibrated,
CPU-servable classifier that improves from analyst labels.

Six labels: `BENIGN`, `GAMBLING_BETTING`, `CRYPTO_P2P`, `LOAN_APP`, `IMPERSONATION_SCAM`,
`MULE_TASK_JOB`. Everything is synthetic, runs on a laptop CPU, and needs no network after
`pip install`.

## What's in the repo

| Piece | Module / command |
|---|---|
| Deterministic synthetic data with template hold-out | `python -m remark.generate --seed 42 --out data/` |
| TF-IDF (word + char 2–5) + logistic regression baseline | `python -m remark.train_baseline` |
| Compact char-level transformer (HF `BertForSequenceClassification`, trained from scratch) | `python -m remark.finetune` |
| Temperature scaling + 90%-recall alert threshold (val only) | `remark/calibration.py`, `remark/evaluate.py` |
| Side-by-side report | `python -m remark.compare` → `reports/comparison.md` |
| Scoring / review / drift API | `uvicorn remark.api:app` |
| Latency benchmark | `python scripts/bench.py --n 1000` |
| Active-learning simulation | `python -m remark.active_loop` → `reports/active_learning.md` |
| Drifted batch | `python -m remark.generate --seed 7 --drift --out data/drift/` |

## Architecture

```
            generate.py ──► data/{train,val,test}.csv        data/drift/drift.csv
                                │
        ┌───────────────────────┴───────────────────────┐
 train_baseline.py                                  finetune.py
 (TF-IDF + LR)                                      (char BERT, 2×128)
        │  fit: train   early-stop/temperature/threshold: val   evaluate: test
        ▼                                                ▼
 models/baseline/{model.joblib, meta.json, reference.json}   models/finetuned/{…safetensors, vocab.json, meta.json, reference.json}
        │                                                │
        └──────────── Predictor interface (models.py) ───┘
                raw_logits() → softmax(logits / T) → risk = 1 − P(BENIGN), alert = risk ≥ threshold
                                │
                           api.py (FastAPI) ── store.py (SQLite labels) ── drift.py (PSI, OOV)
```

Design decisions:
- **One `Predictor` interface.** Each model only implements `raw_logits`. Temperature, threshold and
  the drift reference live in `meta.json` / `reference.json` beside the weights, so calibration, the
  API, the review queue and drift work identically for both models (`REMARK_MODEL=baseline|finetuned`).
  The baseline's logits are its log-probabilities, so temperature scaling applies to both.
- **Split that can't be passed by memorisation.** Templates and payee handles come in three pools:
  *train*, *val-only* (30% of val rows, unseen in train) and *test-holdout* (30% of test rows, unseen
  in train **and** val). The held-out mask is recomputable from the CSVs: test rows whose
  `template_id` never occurs in train/val. Narrations are globally de-duplicated.
- **Leakage boundary.** `test.csv` is only passed to `evaluate_on_test` and `calibration_on_test`
  (`remark/evaluate.py`). Weights fit on train; early stopping, temperature and threshold on val.
- **Realistic narrations.** Three UPI formats (`UPI/<RRN>/<handle>/<remark>`, `UPI-<NAME>-<handle>-<IFSC>-…`,
  `UPI/DR/…`), NEFT UTRs, IMPS P2A/MMT strings, remark-field truncation, abbreviations (`pymt`, `trf`,
  `rfnd`), Hinglish (`paise wapas`, `ghar baithe kamao`, `account block hoga kyc`), typos, and
  benign look-alikes (Dream11/My11Circle refunds, Zerodha/Groww/Upstox, NBFC EMIs, "kyc updated at
  branch"). 30% of risky payments go to ordinary personal handles, so the handle alone is no giveaway.
- **Normalisation** strips 5+ digit runs (RRN/UTR are unique noise) and splits on separators; the char
  transformer reads the *end* of long narrations, where the remark lives.
- **Privacy.** The API never logs narrations; label rows store only `txn_id`, label, analyst id, time.
  Analyst ids are restricted to `[A-Za-z0-9_.-]`; SQL uses bound parameters.

## Setup and commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # CPU torch wheel
cp .env.example .env                        # optional; defaults work

python -m remark.generate --seed 42 --out data/
python -m remark.train_baseline             # ~1 s
python -m remark.finetune                   # a few CPU minutes, logs wall-clock
python -m remark.active_loop                # ~10 s, both strategies × 3 seeds
python -m remark.generate --seed 7 --drift --out data/drift/

REMARK_MODEL=baseline uvicorn remark.api:app --port 8000 &
python scripts/bench.py --n 1000            # writes reports/bench_baseline.json
# stop it, then REMARK_MODEL=finetuned uvicorn … and bench again
python -m remark.compare                    # reports/comparison.md
```

API (all bodies JSON):
- `POST /v1/score` `{"narration", "channel": "UPI|NEFT|IMPS", "amount"}` →
  `{"label", "probabilities" (6), "risk_score", "model_version", "threshold", "alert"}`.
  Empty/whitespace or > 200-char narrations, unknown channels, negative amounts, unknown fields → 422.
- `POST /v1/score/batch` `{"items": [ …up to 256… ]}` → `{"results": [...]}`; 0 or 257 items → 422.
- `GET /health` → loaded model and version.
- `GET /v1/review-queue?k=20` → most uncertain (entropy) train rows, labels hidden, minus labelled ones.
- `POST /v1/labels` `{"txn_id", "label", "analyst_id"}` → 201; unknown label or txn_id → 422.
- `POST /v1/drift` `{"narrations": [...]}` → `{"psi", "oov_rate", "drift", "thresholds", …}`.

## Tests

```bash
pytest            # offline, no downloads; trains a baseline on generated data in a tmp dir
```
Covers generator determinism and the no-overlap / hold-out / balance rules, the API contract and every
422 case, ECE against a hand-computed example, temperature scaling lowering NLL, the 90%-recall
threshold chooser, a training step reducing loss on a tiny random transformer, predictor save/load,
the review queue end to end, label validation, and drift true/false. There are no `slow` tests: the
transformer needs no downloaded weights. The `slow` marker is registered and deselected by default.

## Results (seed 42, 2-vCPU Linux sandbox)

### Task 1: baseline

| | Test macro-F1 |
|---|---:|
| All 700 test rows | **0.961** |
| Held-out templates (210 rows, new phrasings + new handles) | **0.866** |
| In-distribution (490 rows) | 1.000 |

Per-class test recall: BENIGN 0.988, GAMBLING_BETTING 0.875, CRYPTO_P2P 0.950, LOAN_APP 0.975,
IMPERSONATION_SCAM 0.950, MULE_TASK_JOB 0.975.

The in-distribution score is perfect because synthetic seen templates are memorisable; the
**held-out number is the one to trust**.

`python scripts/bench.py --n 1000` (baseline, uvicorn single worker):

```
model=baseline (baseline-d2771d88) n=1000 p50=3.15ms p95=4.11ms p99=5.75ms
```

### Task 2: transformer vs baseline

Full table: [`reports/comparison.md`](reports/comparison.md) (generated by `python -m remark.compare`).

| | Baseline | Transformer |
|---|---:|---:|
| Test macro-F1 | **0.961** | 0.640 |
| Held-out-template macro-F1 | **0.866** | 0.348 |
| ECE before → after temperature (10 bins) | 0.167 → **0.021** | 0.118 → 0.062 |
| NLL before → after | 0.253 → 0.123 | 0.771 → 0.644 |
| Threshold chosen on val (≥ 90% risky recall) | 1.0000 | 0.1584 |
| Test recall / precision at that threshold | 0.730 / 1.000 | 0.860 / 0.557 |
| Alerts per 1,000 test transactions | 208.6 | 441.4 |
| p95 latency (bench.py, 1,000 requests) | 4.11 ms | 7.06 ms |
| Size on disk / parameters | 0.41 MB / sparse LR | 1.22 MB / 300,550 (all trainable) |
| CPU training wall-clock | 0.2 s | **171.5 s (2.9 min, under the 15-min limit)** |

**Checkpoint:** none. Hugging Face downloads were blocked in my environment, so I used the brief's
alternative: a compact transformer trained from scratch (char vocabulary, 2 layers × 128 hidden,
4 heads), early-stopped on val macro-F1 (best epoch 21 of 26). Training is seeded
(`torch.manual_seed(0)`).

**Honest verdict:** the baseline wins on every axis. With ~1,450 labelled rows, a transformer with
no pretraining cannot learn the vocabulary that char/word TF-IDF gets for free, and it generalises
worst to the held-out templates. Its probabilities are better calibrated *before* scaling, but the
baseline is better after. A second run with lr 2e-3 diverged and was discarded (see AI_USAGE.md).
Model risk management should deploy the baseline; a pretrained small encoder is the next thing to try.

**Thresholds don't transfer perfectly:** val is near-separable for the baseline, so the 90%-recall
point sits at ≈ 1.0 and test recall at that threshold is 0.73. The 90% target holds on val by
construction; held-out phrasings in test are harder.

### Task 3: active learning and drift

From `reports/active_learning.md` (60 seed labels, batch 30, 10 rounds, 3 seeds): to reach 95% of
the full-train baseline's macro-F1, **uncertainty sampling needed 140 labels on average (120–150)
vs 320 for random (300–330)**, about 56% fewer analyst labels. Uncertainty sampling clearly beat
random on this data, and its curve is ahead at every round from 90 labels on.

Drift (baseline reference): the seed-7 drifted batch (new "digital arrest"/disconnection scam family,
new `@superyes`/`@navipay` handles, IMPS-heavy mix) gives PSI 1.09 and OOV 0.41 → `drift: true`.
Held-back val gives PSI 0.00 and OOV 0.10 → `drift: false`. Thresholds: PSI > 0.2 or OOV > 0.25.

## Limitations and what I'd do next
- The synthetic val set is easier than test (validation macro-F1 ~0.99), so the val-chosen
  90%-recall threshold under-delivers on test's held-out templates. In production I'd pick the
  threshold on a recent, labelled, out-of-time slice and add a recall margin.
- Feed amount and channel into the models (currently narration-only, by design of the brief).
- Retrain from analyst labels on a schedule, gated on the comparison report; alert on drift.
- Pretrained small encoder (e.g. bert-tiny) with LoRA once a download is allowed, as a third column.
- Per-prediction reason tokens (top TF-IDF features) for analysts; Dockerfile and CI.
