# AI usage

## Tools
- **Claude (Opus 5.5) in claude.ai with a code workspace** — drove most of the code generation, ran
  the commands and tests in a Linux sandbox, and produced commits on feature branches.
- I owned the design decisions, reviewed every file, and checked numbers against generated reports.

## Key prompts (paraphrased)
1. "Understand the assignment thoroughly first" — produced a breakdown of requirements, hidden traps
   (class-balance arithmetic, hold-out mask, determinism, leakage) and a time plan before any code.
2. "Start from the repo then do the complete process" — scaffold → Task 1 (generator, baseline, API,
   bench) → Task 2 (calibration, threshold, transformer, compare) → Task 3 (review queue, labels,
   active learning, drift), committing small Conventional Commits per step.

## Where the AI was wrong, and how it was caught
| Issue | How it was caught | Fix |
|---|---|---|
| Validation set used only templates seen in training, so val was ~100% separable: temperature hit its lower bound (0.05) and the 90%-recall threshold was exactly 1.0. | Inspected `reports/baseline.json` calibration block after the first calibrated run. | Added a *val-only* template/handle pool (unseen in train) so val is an honest proxy for calibration and threshold choice. |
| A shared helper was named `test_metrics`; pytest would collect it as a test when imported. Same for `test_f1` in the AL loop. | Code review before writing tests. | Renamed to `evaluate_on_test` / `fit_and_score`. |
| First transformer config used `MAX_LEN=192`; ~60 s/epoch on 2 CPU threads would break the 15-minute budget. | Watched the epoch log. | Measured normalised narration length (p99 = 65, max = 72 chars) and set `MAX_LEN=96`: ~6 s/epoch. |
| A synthetic payee handle mimicked a real registered lender with a ".fake" suffix. | Review of `templates.py`. | Renamed to an invented brand. |
| OOV threshold of 0.15 would have flagged the normal test set as drift (test has 30% new templates, OOV 0.149). | Measured PSI/OOV on val, test and the drift batch before choosing thresholds. | Threshold 0.25 (val 0.10, test 0.15, drift batch 0.41). |
| A second transformer run (lr 2e-3, batch 16) diverged: loss stuck at ~1.80, val macro-F1 ~0.14. | Watched the epoch log. | Stopped it and kept the lr 1e-3 / batch 32 config that produced the reported run; artifacts and code match. |
| `starlette` deprecation warning for `HTTP_422_UNPROCESSABLE_ENTITY`. | pytest warnings summary. | Use the literal 422. |
| The planned Hugging Face checkpoint could not be downloaded in the sandbox (network policy). | `curl` to huggingface.co returned 403. | Used the brief's alternative: a compact transformer trained from scratch — fully offline, no setup script. |

## What I verified by hand
- Reports are produced by commands (`remark.train_baseline`, `remark.finetune`, `remark.compare`,
  `remark.active_loop`); none are hand-written. Re-running with seed 42 reproduces the metrics.
- The test split is only passed to `evaluate_on_test` / `calibration_on_test` (`remark/evaluate.py`);
  temperature, threshold and early stopping use val only.
