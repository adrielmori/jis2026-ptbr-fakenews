# Extended study - new analyses for the multi-corpus fake-news benchmark

This folder contains everything added for the extended version of the original benchmark paper:

* **Study 1 - leave-one-dataset-out (LODO) generalisation** of the 32 encoder x classifier pipelines over the 10 corpora,
  plus a volume-controlled (balanced) variant and the Friedman / Wilcoxon-Holm statistics.
* **Study 2 - calibration and selective classification** (triage for human review) for SVC-RBF and LogReg on
  Albertina-900M and BERTimbau-large, in-distribution and under LODO.
* **Error analysis** of the best LODO pipeline (high-confidence errors, UMAP of the embedding space).

Everything reuses the published embeddings, splits and selected hyper-parameters; nothing outside `extension/` is modified.

## Reproduce in one command

```bash
# from the repository root (Git Bash on Windows, or any POSIX shell)
git lfs pull                       # train embeddings / train split are Git LFS objects
bash extension/run_all.sh --clean  # creates extension/.venv, installs requirements.txt, runs all steps
```

Options: `--clean` removes previous outputs first; `--no-venv` uses `$PYTHON` (default `python`) instead of creating
`extension/.venv`. Classifiers run on the CPU (`config.yaml: n_jobs`, default 10 workers) except the SVC-RBF Gram
matrix, which is computed on the GPU when CUDA is available (`gpu_svc.py`, `config.yaml: gpu.svc_rbf`; the solution is
identical to the CPU SVC, see `DECISIONS.md` D-24). `requirements.txt` installs the CUDA 12.8 build of PyTorch; on a
machine without a GPU everything falls back to the CPU SVC automatically.
Wall-clock on the reference machine (i5-14400F, 10 cores, 32 GB, RTX 5060 Ti), from `logs/timings.csv`: integrity check
< 1 s, Table 4 reproduction 1.6 h with the GPU SVC (3.0 h on the CPU), full LODO 6-11 h depending on machine load,
balanced LODO 1.8-1.9 h, calibration 11-13 min, error analysis 1 min, tables and figures < 1 min; 11-15 h in total over
the two complete runs. The leave-one-dataset-out study is dominated by the 40 gradient-boosting fits (44 CPU-hours, up
to 2 h each on one core); the 40 SVC-RBF fits take about 30 min on the GPU. A second run from scratch reproduces the
first one up to two single predictions (see `DECISIONS.md` D-25).

Pipeline order (`run_all.sh`):

| Step | Script | Main outputs |
|---|---|---|
| 0 | `check_integrity.py` | `results/sanity/integrity_report.md`, `integrity_counts.csv`, `selected_hyperparameters.csv` |
| 0 | `reproduce_table4.py` | `results/sanity/table4_reproduction.{csv,md}`, `results/predictions/indist_refit_test.parquet` |
| 1 | `run_lodo.py` | `results/lodo/lodo_raw.csv`, `lodo_table_by_dataset.csv`, `lodo_table_by_pipeline.csv`, `lodo_vs_indist.csv`, `lodo_vs_indist_spearman.csv`, `lodo_balanced_raw.csv`, `lodo_balanced_summary.csv`, `lodo_balanced_by_dataset.csv`, `lodo_stats.csv` (+ `_friedman`, `_ranking`), `results/predictions/lodo*.parquet` |
| 2 | `run_calibration.py` | `results/calibration/calibration_metrics.csv`, `reliability_bins.csv`, `selective_curves.csv`, `selective_summary.csv`, `triage_summary.csv`, `results/predictions/calibration_predictions.parquet` |
| 3 | `run_errors.py` | `results/errors/high_confidence_errors.csv`, `high_confidence_error_rates.csv`, `umap_coordinates.csv`, `best_pipeline.json` |
| 4 | `make_tables.py` | `tables/*.tex` (booktabs, with captions) |
| 5 | `make_figures.py` | `figures/*.pdf` + `*.png` (300 dpi) |
| 6 | `make_baseline_figures.py` | `figures/baseline/`, `tables/baseline/`, `results/baseline/`: the published in-distribution results (classical + zero-shot LLMs) redrawn in English for the report; index with captions in `FIGURES.md` |

Logs with per-step timing: `logs/<script>.log`, `logs/timings.csv`.

## Layout

```
extension/
  config.yaml          all paths and parameters
  requirements.txt     pinned environment
  run_all.sh           one-command reproduction
  common.py            shared code (pipelines, loading, metrics, statistics)
  check_integrity.py   reproduce_table4.py   run_lodo.py   run_calibration.py   run_errors.py
  make_tables.py       make_figures.py       make_baseline_figures.py
  results/{sanity,lodo,calibration,errors,predictions,baseline}/
  tables/{,baseline/}  figures/{,baseline/}  logs/
  FIGURES.md           index of every figure and table (both sets) with captions
  REPO_MAP.md          map of the original repository
  DECISIONS.md         every technical decision and its rationale
  SUMMARY.md           main findings with numbers, limitations
```

## Protocols in brief

* **LODO**: for each corpus, train on the other nine (all splits) with the published hyper-parameters, scaler fitted on
  the training pool, test on the whole held-out corpus. Balanced variant: each training corpus sub-sampled to the
  smallest one (stratified by class), seeds 42-46.
* **Calibration**: raw / sigmoid (Platt) / isotonic, calibrators fitted on the validation split (in-distribution) or on
  a stratified 15 % slice of the training corpora (LODO); ECE (15 equal-width bins), adaptive ECE (equal-mass), MCE,
  Brier, log-loss, reliability diagrams.
* **Selective classification**: confidence `max(p, 1-p)`, thresholds 0.50-0.99, coverage / macro-F1 / accuracy on
  covered items, AURC, minimum threshold for macro-F1 >= 0.90 and >= 0.95 with bootstrap 95 % CI (1 000 resamples).

See `DECISIONS.md` for the exact choices and `SUMMARY.md` for the findings.
