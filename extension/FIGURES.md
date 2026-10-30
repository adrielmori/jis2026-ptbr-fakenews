# Figure and table index (all in English, PDF + PNG 300 dpi, LaTeX booktabs)

Two sets are produced by `run_all.sh`:

* **Baseline set** (`figures/baseline/`, `tables/baseline/`, `results/baseline/`): the *published* in-distribution results
  of the original paper (32 classical pipelines + 10 zero-shot LLMs, bootstrap intervals, McNemar/Pearson pairwise tests,
  Friedman + Wilcoxon-Holm, item-level agreement, UMAP of the published embeddings) redrawn from the read-only CSV/NPY
  files under `codes/results` by `make_baseline_figures.py`. The original figures in `codes/results/**/*.png` (Portuguese
  titles, raw model names, numeric labels) are left untouched; nothing outside `extension/` is modified.
* **Extension set** (`figures/`, `tables/`): the new studies (LODO, calibration/selective classification, error analysis)
  produced by `make_figures.py` and `make_tables.py`.

Conventions (both sets): encoders in fixed colour order BERTimbau-base (blue), BERTimbau-large (orange), Albertina-100M
(green), Albertina-900M (yellow), zero-shot LLMs purple; sequential scales use a single blue ramp; diverging scales use
blue - grey - red with the neutral at zero; legends sit below the axes; raw identifiers are replaced by readable names
(`SVC-RBF`, `Gemma-2 9B`, ...). Label convention: FAKE = `label_id` 0, REAL = 1, as in `codes/results/splits` (see
`DECISIONS.md` D-26 for the swap in the published per-class analysis outputs).

## Baseline figures (published results, in-distribution)

| File | Content |
|---|---|
| `fig_b01_corpus_overview` | (a) instances per corpus by label with FAKE share, BOATOSBR greyed as "no embeddings"; (b) text length per corpus (whitespace tokens, log scale). |
| `fig_b02_split_composition` | Stratified 70/15/15 split per corpus with absolute counts. |
| `fig_b03_table4_heatmaps` | Published Table 4: mean per-corpus macro-F1 by classifier x encoder, validation and test. |
| `fig_b04_per_corpus_heatmap` | Test macro-F1 of the 32 pipelines on each of the 10 corpora (rows grouped by encoder), column means on top. |
| `fig_b05_model_ranking_ci` | All 42 models ranked by test macro-F1 and accuracy with 95 % bootstrap CIs (2 000 resamples). |
| `fig_b06_class_f1_mcc` | (a) F1 of FAKE vs F1 of REAL for every model; (b) Matthews correlation coefficient. |
| `fig_b07_sample_difficulty` | (a) ECDF of the number of models (of 42) that classify each test item correctly; (b) per-corpus distribution of the fraction of correct models. |
| `fig_b08_zoo_per_corpus_heatmap` | Per-corpus test macro-F1 of all 42 models (zero-shot rows: COVID19BR not covered, undefined answers excluded). |
| `fig_b09_mcnemar_matrix` | 42 x 42 pairwise exact McNemar tests: net advantage of the row model, crosses = not significant after Holm over 861 pairs. |
| `fig_b10_correctness_correlation` | 42 x 42 Pearson correlation between per-item correctness vectors (which models fail on the same items). |
| `fig_b11_friedman_ranks_test` / `_val` | Average rank over the 10 corpora with the Friedman statistic and the Wilcoxon-Holm outcome vs the best pipeline. |
| `fig_b12_embedding_vs_classifier_spread` | Per corpus, macro-F1 range across classifiers (encoder fixed) vs across encoders (classifier fixed). |
| `fig_b13_encoder_rank_within_classifier` | Rank of each encoder inside each classifier (validation and test). |
| `fig_b14_confusion_matrices` | Row-normalised confusion matrices: SVC-RBF for the four encoders and the four best zero-shot LLMs. |
| `fig_b15_zero_shot_profile` | Zero-shot LLMs: macro-F1 with CI vs the best classical pipeline, predicted-FAKE share vs the true share, undefined answers. |
| `fig_b16_umap_test_encoders` | UMAP of the test embeddings for the four encoders, coloured by label and by corpus, with the 20 hardest items ringed. |
| `fig_b17_umap_splits_by_corpus` | UMAP of train / validation / test for each encoder, coloured by corpus. |

## Baseline tables

| File | Content |
|---|---|
| `tab_b01_corpus_statistics` | Size, label balance, token statistics and split sizes per corpus (BOATOSBR included, marked as not embedded). |
| `tab_b02_table4_test_val` | Published Table 4 (test / validation macro-F1 per classifier x encoder). |
| `tab_b03_model_zoo_test` | All 42 models: accuracy and macro-F1 with 95 % CI, per-class F1, MCC, undefined answers. |
| `tab_b04_friedman_holm_test` | Published in-distribution ranking, Friedman statistic, Wilcoxon-Holm vs the best pipeline. |
| `tab_b05_zero_shot_profile` | Zero-shot LLMs: macro-F1, accuracy, per-class recall, predicted-FAKE share, undefined answers. |
| `tab_b06_hard_examples` | The two test items per corpus misclassified by most classical pipelines, with the number of LLMs that also fail. |
| `tab_b07_mcnemar_summary` | Share of significant McNemar pairs by pair type (classical/classical, classical/LLM, LLM/LLM). |

Derived data behind the baseline figures: `results/baseline/{corpus_statistics, model_zoo_per_corpus,
pairwise_mcnemar_holm, correctness_correlation_matrix, zero_shot_profile, hard_examples, umap_test_coordinates}.csv`.

## Extension figures (new studies)

| File | Content |
|---|---|
| `fig_lodo_heatmap` | Mean leave-one-dataset-out (LODO) macro-F1 by classifier x encoder. |
| `fig_indist_vs_lodo_by_corpus` | In-distribution vs LODO macro-F1 per corpus (mean of 32 pipelines, best LODO pipeline marked, delta annotated). |
| `fig_lodo_boxplot_by_encoder` | Distribution of the 80 LODO scores per encoder. |
| `fig_lodo_delta_heatmap` | LODO minus in-distribution macro-F1 per pipeline x corpus (diverging scale). |
| `fig_lodo_balanced_vs_full` | Full vs balanced (300 per training corpus, 5 seeds) LODO per held-out corpus, with the in-distribution reference. |
| `fig_reliability_indist` / `_lodo` | Reliability diagrams of the four Study 2 pipelines, raw / sigmoid / isotonic, with ECE. |
| `fig_risk_coverage_indist` / `_lodo` | Risk-coverage curves and AURC. |
| `fig_coverage_f1_{scenario}_{pipeline}` | Coverage vs macro-F1 per corpus when thresholding the calibrated probability (0.90 / 0.95 targets dotted). |
| `fig_triage_coverage_90` / `_95` | Coverage reached at the minimum threshold that attains macro-F1 0.90 / 0.95, per corpus and scenario, with bootstrap CIs. |
| `fig_umap_best_encoder` | UMAP of all 41 873 embeddings of the best LODO encoder, coloured by corpus and by LODO hit/miss. |
| `fig_umap_small_multiples` | The same UMAP, one panel per corpus with its LODO error rate. |

## Extension tables

`tab_table4_reproduction`, `tab_corpora_lodo_sizes`, `tab_lodo_by_pipeline`, `tab_lodo_by_dataset`,
`tab_lodo_vs_indist_encoders`, `tab_lodo_vs_indist_pipelines`, `tab_lodo_balanced`, `tab_lodo_stats`,
`tab_lodo_posthoc_top`, `tab_calibration_metrics`, `tab_selective_aurc`, `tab_triage_summary`, `tab_triage_by_corpus`,
`tab_high_conf_errors` (captions inside each `.tex`).
