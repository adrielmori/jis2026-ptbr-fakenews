# Summary of the new results (extended study)

Run of 2026-10-29 on the reference machine (i5-14400F, 10 cores, 32 GB, RTX 5060 Ti), then reproduced from scratch with
`bash extension/run_all.sh --clean` on 2026-10-30 (`DECISIONS.md` D-25: the two runs differ in 1 of 1 339 936 LODO
predictions and 1 of 200 992 Table 4 predictions, both SVC-RBF items at the decision boundary; every number below is
unchanged at the reported precision). Every number is taken from the CSV files under `results/`. The published
in-distribution results (Table 4 of the original paper) were reproduced first: all 32 pipelines within 0.0023 macro-F1 of the
published values and 29-30 of 32 (depending on the run) reproduce the published test predictions item by item
(`results/sanity/`).

Setting: 10 corpora (41 873 instances; BOATOSBR has no embeddings and is excluded, see `DECISIONS.md` D-02), 4 encoders
x 8 classical classifiers = 32 pipelines with the published hyper-parameters, macro-F1 as the main metric.

## Main findings

### (i) Encoder ranking under leave-one-dataset-out (LODO)

1. **Cross-corpus generalisation collapses for every pipeline.** Averaged over the 32 pipelines, macro-F1 falls from
   0.672 in-distribution to 0.448 under LODO (mean delta -0.224); 207 of the 320 LODO fits score below 0.50. The best
   LODO pipeline, Albertina-100M + kNN, reaches 0.510 (median over hold-outs 0.484), i.e. 0.21 below its own
   in-distribution value (`results/lodo/lodo_table_by_pipeline.csv`, `lodo_vs_indist.csv`).

2. **The encoder ranking is reversed and the differences vanish.** In-distribution the order is Albertina-900M (0.685) >
   BERTimbau-large (0.678) > BERTimbau-base (0.665) > Albertina-100M (0.662); under LODO it is Albertina-100M (0.454) >
   BERTimbau-large (0.450) > BERTimbau-base (0.445) > Albertina-900M (0.444), a spread of 0.010 (Spearman rho between
   the two encoder rankings = -0.80, n = 4). The classifier matters more than the encoder: kNN (0.488) > gradient
   boosting (0.466) = decision tree (0.466) > LinearSVC (0.459) > LogReg (0.456) > random forest (0.448) > SVC-RBF
   (0.446) >> Gaussian NB (0.357). SVC-RBF, the in-distribution winner (0.811), has the largest drop (-0.365): the
   margin it learns is corpus-specific. The 32-pipeline rankings in-distribution and under LODO are essentially
   unrelated (rho = 0.14, p = 0.46).

3. **No pipeline is statistically distinguishable from the best.** Friedman over the 32 pipelines with the 10 hold-outs
   as units: chi2 = 31.7, p = 0.43, Kendall W = 0.10 (full LODO); the Wilcoxon tests of Albertina-100M + kNN against
   the other 31 give 11 raw p < 0.05 but none survives Holm (smallest adjusted p = 0.30). With n = 10 units the smallest
   attainable exact p is 0.00195, so Holm over 31 comparisons can never reject; the balanced variant gives a
   significant Friedman (p = 0.0015, W = 0.19) and still 0 of 31 after Holm (`lodo_stats*.csv`, D-10).

### (ii) Which corpora degrade most, and volume vs. domain

4. **Degradation is concentrated on the class-balanced, single-source corpora.** Mean macro-F1 over the 32 pipelines,
   in-distribution -> LODO: FAKETRUEBR 0.782 -> 0.314 (-0.47), FRECOGNA 0.794 -> 0.370 (-0.42), FNEWSSET 0.486 -> 0.253
   (-0.23, below the 0.50 of a constant classifier), COVID19BR 0.636 -> 0.404 (-0.23), FAKEBR 0.746 -> 0.523 (-0.22).
   The fact-checking corpora degrade least: FCN 0.853 -> 0.725 (-0.13), TRE300 0.683 -> 0.544 (-0.14), FACTCKBR 0.718 ->
   0.544 (-0.17), CENTRALFATOS 0.508 -> 0.451 (-0.06, but it is 99.7 % FAKE and already near 0.5 in-distribution).
   Best single LODO results per corpus: FCN 0.894, TRE300 0.704, FACTCKBR 0.674, FAKEBR 0.593, MUMINPT 0.553, FNEWSSET
   0.511, COVID19BR 0.506, CENTRALFATOS 0.502, FRECOGNA 0.455, FAKETRUEBR 0.426 (`lodo_table_by_dataset.csv`).

5. **A large part of the loss is a label-prior shift, not only a vocabulary shift.** The pooled training set is 65 %
   FAKE (54-71 % depending on the hold-out), and the models inherit it: over the 320 LODO fits the mean recall is 0.710
   for FAKE and 0.312 for REAL, and 271 of 320 fits recall FAKE better than REAL. For the best pipeline the REAL recall
   on the balanced corpora is 0.084 (FAKETRUEBR), 0.026 (FRECOGNA), 0.077 (FNEWSSET) and 0.131 (COVID19BR). The
   definition of "REAL" is also corpus-specific (see finding 10), so the minority class cannot be recovered by
   re-weighting alone.

6. **Volume does not buy generalisation; domain does.** The balanced LODO trains on 2 700 instances (300 per training
   corpus; 5 400 when TRE300 is held out) instead of 30-41 k, and scores *higher*: mean macro-F1 0.460 vs. 0.448
   (+0.012; improvement in 57 % of the 320 pipeline x hold-out cells; seed standard deviation 0.021). Rankings are
   stable (Albertina-100M + kNN first in both, 0.517 vs. 0.510), the class bias shrinks (recall FAKE/REAL 0.657/0.392),
   and the corpora that gain most are the balanced ones (FAKEBR +0.079, FAKETRUEBR +0.074) while FNEWSSET (-0.024),
   TRE300 (-0.040) and CENTRALFATOS (-0.055) lose. Nine extra corpora with ~10x more text add nothing that a
   different-domain test corpus can use (`lodo_balanced_summary.csv`, `fig_lodo_balanced_vs_full`).

### (iii) Calibration and coverage at macro-F1 >= 0.90 / 0.95

7. **In-distribution, Platt-scaled SVC-RBF is both the most accurate and the best calibrated pipeline, and it can
   automate ~94 % of the items at macro-F1 >= 0.95.** On the pooled test split (6 281 items) the raw SVC score is
   under-confident (ECE 0.18, mean confidence 0.75 vs. accuracy 0.93); sigmoid calibration brings ECE to 0.010
   (Albertina-900M) / 0.009 (BERTimbau-large) and Brier from 0.085 to 0.049. LogReg is already calibrated (raw ECE
   0.015-0.017). Selective classification with the sigmoid condition: SVC-RBF reaches macro-F1 >= 0.90 at full coverage
   (0.928 at tau = 0.50) and macro-F1 >= 0.95 at tau = 0.65-0.67 with coverage 0.945 [95 % CI 0.921-0.962]
   (Albertina-900M) and 0.943 [0.932-0.962] (BERTimbau-large), i.e. only 5-6 % of the items go to human review; LogReg
   needs to route 6-11 % for the 0.90 target and 29-36 % for 0.95. AURC: SVC 0.011 (E-AURC 0.008), LogReg 0.033-0.039.
   Per corpus the picture is uneven: the 0.90 target is reached in 25 of the 40 pipeline x corpus cells (sigmoid), never
   for FNEWSSET and only at coverage <= 0.19 for MUMINPT, whose minority class is too small for a 0.90 macro-F1
   (`calibration_metrics.csv`, `triage_summary.csv`, `selective_summary.csv`).

8. **Under LODO the calibrators do not transfer and no pipeline reaches either target on the pooled data.** The
   sigmoid/isotonic maps fitted on the training corpora make the models over-confident on the held-out one: ECE
   0.205-0.276, mean confidence 0.78-0.84 against accuracy 0.55-0.58, Brier 0.30-0.34, AURC 0.36-0.50 (optimal 0.10-0.12).
   The raw SVC score has the lowest LODO ECE (0.040-0.062) only because its under-confidence happens to match the lower
   accuracy; it is not a usable calibration. With all corpora pooled, macro-F1 >= 0.90 is reached by no
   pipeline/condition (the covered macro-F1 stays at 0.52-0.55 along the whole curve). Per corpus, the 0.90 target is
   reached in 6 of 40 cells (sigmoid) and 0.95 in 3; the only operationally useful case is FCN: Albertina-900M + SVC-RBF
   (sigmoid) covers 0.933 [0.904-0.973] at tau = 0.60 for 0.90 and 0.779 [0.707-0.822] at tau = 0.79 for 0.95; TRE300
   is marginal (coverage 0.10-0.15 with CI including 0) and the remaining "reached" cells are < 15 % coverage on the
   FAKE-dominated tails of CENTRALFATOS and FACTCKBR (`fig_triage_coverage_90/95`, `tab_triage_by_corpus.tex`).

### (iv) Profile of the high-confidence errors

9. **Confidence carries no information about correctness out of domain.** For the best LODO pipeline (Albertina-100M +
   kNN, 7 distance-weighted neighbours; probabilities are its Study-1 vote fractions because it is not one of the four
   calibrated pipelines, D-20), 43.2 % of the 41 873 items are predicted with confidence >= 0.90 (nearly all of them
   unanimous 7/7 votes) and the error rate among them is 0.385, identical to the overall error rate (0.385); 43 % of all
   errors are high-confidence errors. The same holds for the calibrated SVC under LODO: 31-42 % of the items have
   sigmoid confidence >= 0.90 and only 50-63 % of them are correct; on FAKETRUEBR, FNEWSSET and MUMINPT the accuracy of
   these "sure" predictions is 0.12, 0.03 and 0.15 (`high_confidence_error_rates.csv`).

10. **High-confidence errors are almost all REAL items called FAKE, and they reveal corpus-specific label semantics.**
    By class, REAL items have a LODO error rate of 0.815 and a high-confidence error rate of 0.939 (6 640 of 7 070
    confident REAL predictions are wrong) against 0.154 / 0.029 for FAKE. Of the 300 most confident errors (30 per
    corpus, 279 with confidence 1.0), 235 are REAL -> FAKE; the 65 FAKE -> REAL cases come from CENTRALFATOS (30),
    MUMINPT (17) and FACTCKBR (8). Reading them shows that the corpora do not share one notion of the label: in
    CENTRALFATOS the text is a fact-check *debunking* a rumour and the label FAKE refers to the rumour; in FCN and
    FACTCKBR "REAL" items are "#Verificamos: ... sao verdadeiras" articles; FRECOGNA is lemmatised text ("enganoso
    postagem parir o programar bolsar") that forms its own island in the UMAP (error rate 52 %); FNEWSSET "REAL" items
    are short opinion-like sentences and COVID19BR "REAL" items are tweets. Errors are not confined to one region of the
    embedding space; apart from FRECOGNA the corpora overlap in a single cloud and the errors are spread through it
    (`high_confidence_errors.csv`, `fig_umap_*`).

### (v) Implications for human-in-the-loop fact-checking

11. **A classical pipeline on frozen embeddings is a viable triage tool only inside the corpus it was tuned on.**
    In-distribution, Platt-scaled SVC-RBF can auto-label ~94 % of the stream at macro-F1 >= 0.95 and send ~6 % to
    reviewers; on an unseen source the same model needs to send essentially everything (no threshold reaches 0.90 on
    the pooled LODO data), its confidence cannot be used to prioritise the queue (AURC 0.36-0.50) and its errors are
    systematically biased towards calling genuine content fake, which is the costly direction for a fact-checking team.
    Deployment therefore requires (a) a calibration slice *from the target source* before any threshold is set, (b)
    monitoring of the predicted FAKE rate as a drift signal (under LODO the models predict 62-76 % FAKE regardless of the
    true rate), and (c) harmonised label definitions across sources before corpora are pooled; more pooled data of other
    sources does not help (finding 6).

12. **Compute note.** The SVC-RBF Gram matrix was computed on the GPU and fed to the same libsvm solver (D-24; verified
    identical to the CPU SVC); everything else ran on the CPU. Wall-clock of the first run: Table 4 reproduction 3.0 h
    (CPU SVC), full LODO 6.0 h (gradient boosting alone 44 CPU-hours, up to 117 min per fit; SVC-RBF 0.5 h in total on
    the GPU), balanced LODO 1.8 h, calibration 11 min, error analysis 1 min, tables and figures < 1 min. The clean
    verification run took 14.9 h in total (Table 4 1.6 h with the GPU SVC, full LODO 11.1 h on a machine that was
    throttled and shared for part of the night, balanced LODO 1.9 h, calibration 13 min) (`logs/timings.csv`).

13. **Published-results figure set and a label-naming inconsistency in the published analysis files.** All figures of the
    published in-distribution study (32 classical pipelines and the 10 zero-shot LLMs: bootstrap intervals, 861 pairwise
    McNemar tests with Holm correction, correctness correlations, Friedman/Wilcoxon-Holm, item difficulty, UMAPs) were
    redrawn in English under `figures/baseline/` (index in `FIGURES.md`). Doing so revealed that the published
    per-sample analysis outputs (`codes/results/analysis_outputs_models`) name the classes of the classical pipelines
    with the opposite convention to the split files (0 -> REAL instead of 0 -> FAKE), whereas the zero-shot rows use
    the correct names (D-26). Accuracy, macro-F1, MCC, Table 4 and all pairwise tests are unaffected; only the
    per-class metrics and confusion counts of the classical models in those files are swapped (e.g. the "FAKE" support
    of 2 197 items is in fact the REAL support). With the corrected convention, SVC-RBF reaches F1 0.95 on FAKE and 0.91
    on REAL, the best zero-shot LLM (Gemma-2 9B) 0.78 / 0.46, and every zero-shot model over-predicts FAKE or REAL by a
    wide margin (predicted-FAKE share 0.38-0.83 against a true share of 0.66); 714 of the 861 McNemar pairs are
    significant after Holm (316 of the 320 classical-vs-LLM pairs). The 20 hardest test items (two per corpus,
    misclassified by 24-32 of the 32 pipelines) are REAL items predicted FAKE in 18 cases, the same direction as the
    LODO label-prior bias of finding 5.

## Limitations and threats to validity

* **Ten hold-out units.** Friedman/Wilcoxon-Holm over 10 corpora cannot reject anything at alpha = 0.05 after Holm
  correction of 31 comparisons (D-10); rankings between pipelines whose LODO means differ by < 0.05 should be read as ties.
* **BOATOSBR is excluded** (no embeddings in the repository; D-02); the published tables also exclude it.
* **Fixed hyper-parameters.** The LODO models reuse the hyper-parameters selected in-distribution; re-tuning inside
  each LODO fold could change the classifier ranking (not the in-distribution/LODO gap, which is dominated by the label
  and domain shifts of findings 5 and 10).
* **Frozen mean-pooled embeddings, no fine-tuning.** The study characterises the published benchmark; fine-tuned
  encoders or text-level domain adaptation are out of scope.
* **Calibration slice under LODO** comes from the training corpora (15 %), which is the realistic "no target labels"
  setting but the pessimistic one; finding 11(a) is the untested alternative.
* **Macro-F1 on extremely imbalanced hold-outs** (CENTRALFATOS 34 REAL, MUMINPT 65 REAL) is driven by a handful of
  items; the per-corpus 0.90/0.95 targets are unattainable there for any classifier and are reported as "not reached".
* **kNN confidence** (finding 9) is a vote fraction, not a calibrated probability; the calibrated-SVC numbers in the
  same finding show that the conclusion does not depend on that choice.
* **Zero-shot baselines** cover 5 765 of the 6 281 test items (no COVID19BR items) with 14 undefined answers excluded,
  so classical-vs-LLM comparisons are made on the common items only (`n_common` in the McNemar files).
* **Balanced LODO uses 5 seeds** of a 300-per-corpus sub-sample; the seed standard deviation (0.021) is of the same order
  as the balanced-minus-full differences per corpus, so only the pooled +0.012 and the per-corpus gains above 0.04 are
  robust.
