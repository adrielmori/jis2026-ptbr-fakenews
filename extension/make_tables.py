"""Build the LaTeX (booktabs) tables of the extension from the CSV results.

Every table is written to extension/tables/<name>.tex as a complete table environment
with caption and label, ready to be \\input{} in the manuscript.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config, model_id, rpath, setup_logging, step_timer

SCRIPT = "make_tables"
ENC_SHORT = {"bertimbau_base": "BERTimbau-base", "bertimbau_large": "BERTimbau-large",
             "albertina_ptbr_100m": "Albertina-100M", "albertina_ptbr_900m": "Albertina-900M"}
CLF_SHORT = {"logreg": "LogReg", "linearsvc": "LinearSVC", "svc_rbf": "SVC-RBF", "random_forest": "RandomForest",
             "gradient_boosting": "GradBoost", "gaussian_nb": "GaussianNB", "knn": "kNN", "decision_tree": "DecisionTree"}
COND_SHORT = {"raw": "raw", "sigmoid": "sigmoid", "isotonic": "isotonic"}
SCEN_SHORT = {"indist": "in-dist.", "lodo": "LODO"}


def tex_escape(s: str) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def pipe_short(mid: str) -> str:
    e, c = mid.split(" + ")
    return f"{ENC_SHORT.get(e, e)} + {CLF_SHORT.get(c, c)}"


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}".replace(",", "\\,")
    return f"{v:.{nd}f}"


def booktabs(rows: list[list[str]], header: list[str], caption: str, label: str, col_format: str,
             note: str = "", size: str = "\\small", star: bool = False) -> str:
    env = "table*" if star else "table"
    lines = [f"\\begin{{{env}}}[t]", "\\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{col_format}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines.append(f"\\par\\vspace{{2pt}}\\footnotesize {note}")
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines) + "\n"


def write(out_dir: Path, name: str, content: str, log) -> None:
    (out_dir / f"{name}.tex").write_text(content, encoding="utf-8")
    log.info("wrote %s.tex", name)


def main() -> None:
    cfg = load_config()
    log = setup_logging(cfg, SCRIPT)
    res = rpath(cfg, "out_results")
    out = rpath(cfg, "out_tables")
    out.mkdir(parents=True, exist_ok=True)
    encs = cfg["encoders"]
    clfs = [c for c in ["svc_rbf", "knn", "linearsvc", "logreg", "random_forest", "gradient_boosting", "decision_tree", "gaussian_nb"]
            if c in cfg["classifiers"]]

    with step_timer(cfg, log, SCRIPT, "Study 1 tables"):
        raw = pd.read_csv(res / "lodo" / "lodo_raw.csv")
        t5 = pd.read_csv(res / "lodo" / "lodo_table_by_dataset.csv")
        vs = pd.read_csv(res / "lodo" / "lodo_vs_indist.csv")
        rho = pd.read_csv(res / "lodo" / "lodo_vs_indist_spearman.csv")
        bal = pd.read_csv(res / "lodo" / "lodo_balanced_by_dataset.csv")
        bsum = pd.read_csv(res / "lodo" / "lodo_balanced_summary.csv")
        plan = pd.read_csv(res / "lodo" / "lodo_balanced_sampling_plan.csv")
        fried = pd.read_csv(res / "lodo" / "lodo_stats_friedman.csv")
        rank = pd.read_csv(res / "lodo" / "lodo_stats_ranking.csv")
        post = pd.read_csv(res / "lodo" / "lodo_stats.csv")
        counts = pd.read_csv(res / "sanity" / "integrity_counts.csv").set_index("dataset_id")

        # --- corpora and LODO sizes
        rows = []
        for ds in cfg["datasets"]:
            c = counts.loc[ds]
            r = raw[raw.held_out_dataset == ds].iloc[0]
            nb = plan[(plan.held_out_dataset == ds)].groupby("seed").n_sampled.sum().iloc[0]
            rows.append([tex_escape(ds), fmt(int(c.n_embedded)), fmt(int(c.n_fake)), fmt(int(c.n_real)),
                         f"{100 * c.n_real / c.n_embedded:.1f}", fmt(int(r.n_train)), fmt(int(nb))])
        write(out, "tab_corpora_lodo_sizes", booktabs(
            rows, ["Corpus", "$n$", "FAKE", "REAL", "\\% REAL", "LODO $n_{\\text{train}}$", "Balanced $n_{\\text{train}}$"],
            "Corpora used in the leave-one-dataset-out (LODO) study. $n$ is the number of embedded instances "
            "(train+val+test of the published splits), all of which form the test set when the corpus is held out. "
            "LODO $n_{\\text{train}}$ is the size of the union of the other nine corpora; balanced $n_{\\text{train}}$ "
            "is the size after sub-sampling every training corpus to the smallest one (stratified by class).",
            "tab:corpora-lodo", "lrrrrrr"), log)

        # --- Table 4 layout (mean LODO macro-F1)
        m = raw.groupby(["classifier", "encoder"]).macro_f1.mean().unstack("encoder")[encs]
        pub = pd.read_csv(rpath(cfg, "published_table4")).set_index("classifier")[encs]
        rows = []
        for c in clfs:
            best_e = m.loc[c].idxmax()
            cells = []
            for e in encs:
                v = f"{m.loc[c, e]:.3f}"
                cells.append(f"\\textbf{{{v}}}" if e == best_e else v)
            cells.append(f"{m.loc[c].mean():.3f}")
            rows.append([CLF_SHORT[c]] + cells)
        rows.append(["\\midrule Mean"] + [f"{m[e].mean():.3f}" for e in encs] + [f"{m.values.mean():.3f}"])
        rows.append(["In-distribution mean (published)"] + [f"{pub[e].mean():.3f}" for e in encs] + [f"{pub.values.mean():.3f}"])
        write(out, "tab_lodo_by_pipeline", booktabs(
            rows, ["Classifier"] + [ENC_SHORT[e] for e in encs] + ["Mean"],
            "Leave-one-dataset-out macro-F1 averaged over the ten held-out corpora, for every encoder $\\times$ classifier "
            "pipeline (same layout as the in-distribution Table~4 of the original paper). Bold marks the best encoder per "
            "classifier. The last row repeats the published in-distribution column means for reference.",
            "tab:lodo-by-pipeline", "lrrrrr"), log)

        # --- Table 5 layout (per held-out corpus)
        ind = vs[vs.scope == "corpus_mean_over_pipelines"].set_index("dataset_id")
        rows = []
        for _, r in t5.sort_values("mean_macro_f1", ascending=False).iterrows():
            rows.append([tex_escape(r.held_out_dataset), fmt(int(r.n_test)), f"{ind.loc[r.held_out_dataset, 'indist_macro_f1']:.3f}",
                         f"{r.mean_macro_f1:.3f}", f"{r.best_macro_f1:.3f}", f"{r.range_macro_f1:.3f}",
                         tex_escape(pipe_short(r.best_pipeline))])
        write(out, "tab_lodo_by_dataset", booktabs(
            rows, ["Held-out corpus", "$n_{\\text{test}}$", "In-dist. mean", "LODO mean", "LODO best", "Range", "Best LODO pipeline"],
            "Robustness per held-out corpus under LODO (layout of Table~5 of the original paper): mean and best macro-F1 over "
            "the 32 pipelines, range (best $-$ worst) and the best pipeline. ``In-dist.\\ mean'' is the published mean "
            "in-distribution macro-F1 of the same 32 pipelines on the test split of that corpus.",
            "tab:lodo-by-dataset", "lrrrrrl", star=True), log)

        # --- encoders: in-dist vs LODO ranking
        enc = vs[vs.scope == "encoder_mean"].sort_values("indist_macro_f1", ascending=False)
        rows = [[ENC_SHORT[r.encoder], f"{r.indist_macro_f1:.3f}", f"{int(r.rank_indist)}", f"{r.lodo_macro_f1:.3f}",
                 f"{int(r.rank_lodo)}", f"{r.delta_lodo_minus_indist:+.3f}"] for _, r in enc.iterrows()]
        rp = rho.set_index("level")
        note = ("Spearman $\\rho$ between in-distribution and LODO rankings: "
                f"encoders $\\rho={rp.loc['4_encoders_mean', 'spearman_rho']:.2f}$; "
                f"32 pipelines $\\rho={rp.loc['32_pipelines_mean_over_corpora', 'spearman_rho']:.2f}$ "
                f"($p={rp.loc['32_pipelines_mean_over_corpora', 'p_value']:.2g}$); "
                f"10 corpora $\\rho={rp.loc['10_corpora_mean_over_pipelines', 'spearman_rho']:.2f}$ "
                f"($p={rp.loc['10_corpora_mean_over_pipelines', 'p_value']:.2g}$).")
        write(out, "tab_lodo_vs_indist_encoders", booktabs(
            rows, ["Encoder", "In-dist. F1", "Rank", "LODO F1", "Rank", "$\\Delta$"],
            "Encoder ranking in-distribution (published, mean over classifiers and corpora) versus LODO. "
            "$\\Delta$ = LODO $-$ in-distribution macro-F1.", "tab:lodo-vs-indist-encoders", "lrrrrr", note=note), log)

        pipes = vs[vs.scope == "pipeline_mean_over_corpora"].sort_values("indist_macro_f1", ascending=False).head(12)
        rows = [[tex_escape(pipe_short(r.model_id)), f"{r.indist_macro_f1:.3f}", f"{int(r.rank_indist)}",
                 f"{r.lodo_macro_f1:.3f}", f"{int(r.rank_lodo)}", f"{r.delta_lodo_minus_indist:+.3f}"] for _, r in pipes.iterrows()]
        write(out, "tab_lodo_vs_indist_pipelines", booktabs(
            rows, ["Pipeline", "In-dist. F1", "Rank", "LODO F1", "Rank", "$\\Delta$"],
            "The twelve best in-distribution pipelines and their LODO macro-F1 and rank (out of 32).",
            "tab:lodo-vs-indist-pipelines", "lrrrrr"), log)

        # --- balanced LODO
        rows = []
        for _, r in bal.sort_values("full_mean_macro_f1", ascending=False).iterrows():
            sd = bsum[bsum.held_out_dataset == r.held_out_dataset].macro_f1_std.mean()
            rows.append([tex_escape(r.held_out_dataset), fmt(int(r.n_train_full)), fmt(int(r.n_train_balanced)),
                         f"{r.full_mean_macro_f1:.3f}", f"{r.balanced_mean_macro_f1:.3f} $\\pm$ {sd:.3f}",
                         f"{r.delta_balanced_minus_full_mean:+.3f}", f"{r.full_best_macro_f1:.3f}",
                         f"{r.balanced_best_macro_f1:.3f}", f"{r.delta_balanced_minus_full_best:+.3f}"])
        write(out, "tab_lodo_balanced", booktabs(
            rows, ["Held-out corpus", "$n_{\\text{tr}}$ full", "$n_{\\text{tr}}$ bal.", "Mean full", "Mean balanced",
                   "$\\Delta$", "Best full", "Best bal.", "$\\Delta$"],
            "Full versus balanced LODO (every training corpus sub-sampled to the smallest one, five seeds). Means are over "
            "the 32 pipelines; $\\pm$ is the average standard deviation across seeds; $\\Delta$ = balanced $-$ full. "
            "A small $|\\Delta|$ indicates that the LODO degradation is driven by domain shift rather than by training volume.",
            "tab:lodo-balanced", "lrrrrrrrr", star=True), log)

        # --- Table 6 layout: Friedman + Holm post-hoc
        rows = []
        for sc in ("lodo_full", "lodo_balanced"):
            f = fried[fried.scenario == sc].iloc[0]
            best = rank[(rank.scenario == sc) & (rank["rank"] == 1)].iloc[0]
            p = post[post.scenario == sc]
            n_rej = int(p.reject_h0_holm.sum())
            rows.append([sc.replace("lodo_", "LODO ").replace("_", " "), f"{int(f.n_models)}", f"{int(f.n_units)}",
                         f"{f.statistic:.2f}", f"{f.p_value:.2e}", f"{f.kendall_w:.3f}", tex_escape(pipe_short(best.model_id)),
                         f"{best.mean_macro_f1:.3f}", f"{n_rej}/{len(p)}"])
        write(out, "tab_lodo_stats", booktabs(
            rows, ["Scenario", "Models", "Hold-outs", "$\\chi^2_F$", "$p$", "Kendall $W$", "Best pipeline", "Mean F1", "Holm rej."],
            "Friedman test over the 32 pipelines with the ten held-out corpora as statistical units, followed by two-sided "
            "Wilcoxon signed-rank tests of the best pipeline against each of the other 31 with Holm--Bonferroni correction "
            "($\\alpha=0.05$); ``Holm rej.'' is the number of comparisons remaining significant (layout of Table~6 of the original paper).",
            "tab:lodo-stats", "lrrrrrlrr", star=True), log)

        # top post-hoc rows for the full scenario
        p = post[post.scenario == "lodo_full"].head(10)
        rows = [[tex_escape(pipe_short(r.comparison_model)), f"{r.mean_comparison:.3f}", f"{r.mean_diff:+.3f}",
                 f"{r.raw_p_value:.4f}", f"{r.holm_adjusted_p_value:.4f}", "yes" if r.reject_h0_holm else "no"] for _, r in p.iterrows()]
        write(out, "tab_lodo_posthoc_top", booktabs(
            rows, ["Comparison pipeline", "Mean F1", "Diff.", "raw $p$", "Holm $p$", "Reject"],
            f"Ten smallest raw $p$-values of the Wilcoxon comparisons of the best LODO pipeline "
            f"({tex_escape(pipe_short(p.reference_model.iloc[0]))}) against the others (full LODO).",
            "tab:lodo-posthoc", "lrrrrl"), log)

    with step_timer(cfg, log, SCRIPT, "Study 2 tables"):
        cm = pd.read_csv(res / "calibration" / "calibration_metrics.csv")
        tri = pd.read_csv(res / "calibration" / "triage_summary.csv")
        sel = pd.read_csv(res / "calibration" / "selective_summary.csv")
        pipes = [model_id(e, c) for e, c in cfg["calibration"]["pipelines"]]
        conds = cfg["calibration"]["conditions"]

        rows = []
        for sc in ("indist", "lodo"):
            for p_ in pipes:
                for cd in conds:
                    r = cm[(cm.scenario == sc) & (cm.pipeline == p_) & (cm.condition == cd) & (cm.corpus == "ALL")].iloc[0]
                    rows.append([SCEN_SHORT[sc], tex_escape(pipe_short(p_)), cd, f"{r.ece:.3f}", f"{r.ece_adaptive:.3f}",
                                 f"{r.mce:.3f}", f"{r.brier:.3f}", f"{r.log_loss:.3f}", f"{r.accuracy:.3f}", f"{r.macro_f1:.3f}"])
        write(out, "tab_calibration_metrics", booktabs(
            rows, ["Scenario", "Pipeline", "Cond.", "ECE", "ECE$_{\\text{ad}}$", "MCE", "Brier", "Log-loss", "Acc.", "Macro-F1"],
            "Calibration quality on the evaluation items (all corpora pooled). ECE: expected calibration error with 15 "
            "equal-width bins on the confidence $\\max(p,1-p)$; ECE$_{\\text{ad}}$: 15 equal-mass bins; MCE: maximum "
            "calibration error. Raw: uncalibrated scores (predict\\_proba for LogReg, logistic squash of the SVC decision "
            "function); sigmoid (Platt) and isotonic calibrators are fitted on held-out calibration data "
            "(validation split in-distribution; a stratified 15\\% slice of the training corpora under LODO).",
            "tab:calibration-metrics", "lllrrrrrrr", star=True), log)

        rows = []
        for sc in ("indist", "lodo"):
            for p_ in pipes:
                for cd in conds:
                    cells = [SCEN_SHORT[sc], tex_escape(pipe_short(p_)), cd]
                    for tgt in cfg["calibration"]["f1_targets"]:
                        r = tri[(tri.scenario == sc) & (tri.pipeline == p_) & (tri.condition == cd) & (tri.corpus == "ALL")
                                & (np.isclose(tri.target_macro_f1, tgt))].iloc[0]
                        if r.status == "reached":
                            cells += [f"{r.threshold:.2f}", f"{r.coverage:.3f} [{r.coverage_ci_low:.3f}, {r.coverage_ci_high:.3f}]"]
                        else:
                            cells += ["--", "not reached"]
                    rows.append(cells)
        write(out, "tab_triage_summary", booktabs(
            rows, ["Scenario", "Pipeline", "Cond.", "$\\tau_{0.90}$", "Coverage [95\\% CI]", "$\\tau_{0.95}$", "Coverage [95\\% CI]"],
            "Selective classification (all corpora pooled): minimum confidence threshold $\\tau$ at which the macro-F1 of the "
            "automatically classified items reaches 0.90 and 0.95, with the corresponding coverage (fraction of items not "
            "routed to human review) and its bootstrap 95\\% confidence interval (1\\,000 resamples). A threshold is only "
            "accepted when it covers at least 30 items; ``not reached'' means no threshold $\\le 0.99$ satisfies the target.",
            "tab:triage-summary", "lllrlrl", star=True), log)

        # per-corpus coverage at F1 >= 0.90, sigmoid condition, both scenarios, 4 pipelines
        rows = []
        for ds in cfg["datasets"] + ["ALL"]:
            cells = [tex_escape(ds)]
            for sc in ("indist", "lodo"):
                for p_ in pipes:
                    r = tri[(tri.scenario == sc) & (tri.pipeline == p_) & (tri.condition == "sigmoid") & (tri.corpus == ds)
                            & (np.isclose(tri.target_macro_f1, 0.90))].iloc[0]
                    cells.append(f"{r.coverage:.2f}" if r.status == "reached" else "--")
            rows.append(cells)
        header = ["Corpus"] + [f"{SCEN_SHORT[sc]} {tex_escape(pipe_short(p_)).replace('BERTimbau-large', 'BT-L').replace('Albertina-900M', 'Alb-900M')}"
                               for sc in ("indist", "lodo") for p_ in pipes]
        write(out, "tab_triage_by_corpus", booktabs(
            rows, header,
            "Coverage at macro-F1 $\\ge 0.90$ per corpus (sigmoid calibration), in-distribution and LODO. ``--'': the target "
            "is not reached at any threshold $\\le 0.99$. BT-L = BERTimbau-large; Alb-900M = Albertina-900M.",
            "tab:triage-by-corpus", "l" + "r" * 8, size="\\scriptsize", star=True), log)

        rows = []
        for sc in ("indist", "lodo"):
            for p_ in pipes:
                cells = [SCEN_SHORT[sc], tex_escape(pipe_short(p_))]
                for cd in conds:
                    r = sel[(sel.scenario == sc) & (sel.pipeline == p_) & (sel.condition == cd) & (sel.corpus == "ALL")].iloc[0]
                    cells += [f"{r.aurc:.3f}", f"{r.e_aurc:.3f}"]
                r = sel[(sel.scenario == sc) & (sel.pipeline == p_) & (sel.condition == "raw") & (sel.corpus == "ALL")].iloc[0]
                cells.append(f"{r.full_coverage_risk:.3f}")
                rows.append(cells)
        write(out, "tab_selective_aurc", booktabs(
            rows, ["Scenario", "Pipeline", "AURC raw", "E-AURC", "AURC sig.", "E-AURC", "AURC iso.", "E-AURC", "Risk@100\\%"],
            "Area under the risk--coverage curve (AURC, lower is better) and its excess over the optimal curve (E-AURC) per "
            "condition, all corpora pooled; Risk@100\\% is the error rate at full coverage.",
            "tab:selective-aurc", "llrrrrrrr", star=True), log)

    with step_timer(cfg, log, SCRIPT, "error-analysis and sanity tables"):
        rates = pd.read_csv(res / "errors" / "high_confidence_error_rates.csv")
        info = json.loads((res / "errors" / "best_pipeline.json").read_text(encoding="utf-8"))
        rows = []
        for _, r in rates[rates.level == "corpus"].sort_values("error_rate").iterrows():
            rows.append([tex_escape(r.group), fmt(int(r.n_items)), f"{r.error_rate:.3f}", f"{r.high_conf_coverage:.3f}",
                         f"{r.high_conf_error_rate:.3f}" if not np.isnan(r.high_conf_error_rate) else "--",
                         f"{r.frac_errors_that_are_high_conf:.3f}" if not np.isnan(r.frac_errors_that_are_high_conf) else "--"])
        for _, r in rates[rates.level == "class"].iterrows():
            rows.append([f"\\textit{{class {r.group}}}", fmt(int(r.n_items)), f"{r.error_rate:.3f}", f"{r.high_conf_coverage:.3f}",
                         f"{r.high_conf_error_rate:.3f}", f"{r.frac_errors_that_are_high_conf:.3f}"])
        r = rates[rates.level == "all"].iloc[0]
        rows.append(["\\midrule All", fmt(int(r.n_items)), f"{r.error_rate:.3f}", f"{r.high_conf_coverage:.3f}",
                     f"{r.high_conf_error_rate:.3f}", f"{r.frac_errors_that_are_high_conf:.3f}"])
        write(out, "tab_high_conf_errors", booktabs(
            rows, ["Corpus / class", "$n$", "Error rate", "Cov.\\ $\\ge 0.90$", "Err.\\ rate $\\ge 0.90$", "Errors that are HC"],
            f"High-confidence (HC, confidence $\\ge 0.90$) errors of the best LODO pipeline "
            f"({tex_escape(pipe_short(info['best_pipeline']))}) per held-out corpus and per true class. "
            "``Cov.'' is the fraction of items predicted with confidence $\\ge 0.90$; ``Err.\\ rate'' is the error rate among "
            "those items; ``Errors that are HC'' is the fraction of all errors made with confidence $\\ge 0.90$.",
            "tab:high-conf-errors", "lrrrrr"), log)

        rep = pd.read_csv(res / "sanity" / "table4_reproduction.csv")
        agr = pd.read_csv(res / "sanity" / "table4_prediction_agreement.csv")
        rows = []
        for c in clfs:
            cells = [CLF_SHORT[c]]
            for e in encs:
                r = rep[(rep.classifier == c) & (rep.encoder == e)].iloc[0]
                cells.append(f"{r['diff']:+.4f}")
            a = agr[agr.classifier == c].prediction_agreement
            cells.append(f"{a.min():.4f}")
            rows.append(cells)
        write(out, "tab_table4_reproduction", booktabs(
            rows, ["Classifier"] + [ENC_SHORT[e] for e in encs] + ["Min.\\ agreement"],
            f"Sanity check: difference (reproduced $-$ published) of the mean per-corpus macro-F1 of Table~4 after refitting "
            f"the 32 selected pipelines from the stored embeddings and hyper-parameters (tolerance $\\pm{cfg['sanity']['table4_tolerance']}$; "
            f"max $|\\Delta|={rep['diff'].abs().max():.4f}$), and the minimum per-item agreement with the published test predictions.",
            "tab:table4-reproduction", "lrrrrr"), log)
    log.info("Tables written to %s", out)


if __name__ == "__main__":
    main()
