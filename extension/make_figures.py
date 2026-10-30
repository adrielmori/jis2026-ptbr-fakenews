"""Figures of the extension (PDF + PNG 300 dpi, journal-column friendly fonts).

Colour conventions (fixed order, colour-blind-safe, validated):
  encoders  : bertimbau_base blue, bertimbau_large orange, albertina_ptbr_100m aqua, albertina_ptbr_900m yellow
  conditions: raw blue, sigmoid orange, isotonic aqua
  scenarios : in-distribution blue, LODO orange
  sequential: single-hue blue ramp; diverging: blue <-> red with grey midpoint.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from common import load_config, model_id, rpath, setup_logging, step_timer  # noqa: E402

SCRIPT = "make_figures"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ENC_ORDER = ["bertimbau_base", "bertimbau_large", "albertina_ptbr_100m", "albertina_ptbr_900m"]
ENC_COLOR = dict(zip(ENC_ORDER, CAT[:4]))
ENC_SHORT = {"bertimbau_base": "BERTimbau-base", "bertimbau_large": "BERTimbau-large",
             "albertina_ptbr_100m": "Albertina-100M", "albertina_ptbr_900m": "Albertina-900M"}
CLF_ORDER = ["svc_rbf", "knn", "linearsvc", "logreg", "random_forest", "gradient_boosting", "decision_tree", "gaussian_nb"]
CLF_SHORT = {"logreg": "LogReg", "linearsvc": "LinearSVC", "svc_rbf": "SVC-RBF", "random_forest": "RandomForest",
             "gradient_boosting": "GradBoost", "gaussian_nb": "GaussianNB", "knn": "kNN", "decision_tree": "DecisionTree"}
COND_COLOR = {"raw": CAT[0], "sigmoid": CAT[1], "isotonic": CAT[2]}
COND_STYLE = {"raw": ":", "sigmoid": "-", "isotonic": "--"}
SCEN_COLOR = {"indist": CAT[0], "lodo": CAT[1]}
SCEN_LABEL = {"indist": "In-distribution", "lodo": "LODO"}
# ten corpora: eight validated categorical slots + two extra dark tones (documented limitation of 10-class scatter)
CORPUS_COLORS = CAT + ["#5c4a1f", "#4d5a66"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#2a78d6", "#0d366b"])
GREY = "#9a9995"
INK = "#0b0b0b"
INK2 = "#52514e"

SINGLE = 3.5   # inches, one journal column
DOUBLE = 7.2   # inches, two columns


def style(cfg):
    fs = cfg["figures"]["font_size"]
    plt.rcParams.update({
        "font.size": fs, "axes.titlesize": fs + 1, "axes.labelsize": fs, "xtick.labelsize": fs - 1,
        "ytick.labelsize": fs - 1, "legend.fontsize": fs - 1, "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"], "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "axes.axisbelow": True,
        "lines.linewidth": 1.4, "figure.dpi": 100, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def save(fig, out_dir, name, dpi, log):
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=dpi)
    plt.close(fig)
    log.info("saved %s.{pdf,png}", name)


def pipe_short(mid):
    e, c = mid.split(" + ")
    return f"{ENC_SHORT[e]} + {CLF_SHORT[c]}"


# ----------------------------------------------------------------------------
def fig_lodo_heatmap(raw, out_dir, dpi, log):
    m = raw.groupby(["classifier", "encoder"]).macro_f1.mean().unstack("encoder").loc[CLF_ORDER, ENC_ORDER]
    fig, ax = plt.subplots(figsize=(SINGLE, 3.2))
    im = ax.imshow(m.values, cmap=SEQ, vmin=0.4, vmax=0.9, aspect="auto")
    ax.set_xticks(range(len(ENC_ORDER))); ax.set_xticklabels([ENC_SHORT[e] for e in ENC_ORDER], rotation=30, ha="right")
    ax.set_yticks(range(len(CLF_ORDER))); ax.set_yticklabels([CLF_SHORT[c] for c in CLF_ORDER])
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color="white" if v > 0.7 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Mean LODO macro-F1 (10 hold-outs)")
    cb.outline.set_visible(False)
    ax.set_title("Leave-one-dataset-out macro-F1")
    save(fig, out_dir, "fig_lodo_heatmap", dpi, log)


def fig_indist_vs_lodo_by_corpus(vs, t5, out_dir, dpi, log):
    c = vs[vs.scope == "corpus_mean_over_pipelines"].sort_values("indist_macro_f1", ascending=False)
    best = t5.set_index("held_out_dataset").best_macro_f1
    x = np.arange(len(c)); w = 0.38
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.9))
    ax.bar(x - w / 2, c.indist_macro_f1, w, color=SCEN_COLOR["indist"], label="In-distribution (published, mean of 32 pipelines)",
           edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, c.lodo_macro_f1, w, color=SCEN_COLOR["lodo"], label="LODO (mean of 32 pipelines)",
           edgecolor="white", linewidth=0.5)
    ax.scatter(x + w / 2, best.loc[c.dataset_id].values, marker="_", s=120, color=INK, linewidths=1.2,
               label="LODO best pipeline", zorder=3)
    for xi, (a, b, bb) in enumerate(zip(c.indist_macro_f1, c.lodo_macro_f1, best.loc[c.dataset_id].values)):
        ax.text(xi + w / 2, max(b, bb) + 0.015, f"{b - a:+.2f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(c.dataset_id, rotation=30, ha="right")
    ax.set_ylim(0, 1.0); ax.set_ylabel("Macro-F1")
    ax.axhline(0.5, color=GREY, linewidth=0.6, linestyle=":")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=7)
    ax.set_title("In-distribution vs. leave-one-dataset-out macro-F1 per corpus (delta annotated)")
    fig.subplots_adjust(bottom=0.34)
    save(fig, out_dir, "fig_indist_vs_lodo_by_corpus", dpi, log)


def fig_lodo_boxplot_by_encoder(raw, out_dir, dpi, log):
    fig, ax = plt.subplots(figsize=(SINGLE, 2.8))
    data = [raw[raw.encoder == e].macro_f1.values for e in ENC_ORDER]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color=INK, linewidth=1.2), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    for patch, e in zip(bp["boxes"], ENC_ORDER):
        patch.set_facecolor(ENC_COLOR[e]); patch.set_alpha(0.55); patch.set_edgecolor(INK2)
    rng = np.random.default_rng(0)
    for i, (e, d) in enumerate(zip(ENC_ORDER, data)):
        ax.scatter(i + 1 + rng.uniform(-0.18, 0.18, len(d)), d, s=6, color=ENC_COLOR[e], edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_xticks(range(1, len(ENC_ORDER) + 1)); ax.set_xticklabels([ENC_SHORT[e] for e in ENC_ORDER], rotation=20, ha="right")
    ax.set_ylabel("LODO macro-F1 (8 classifiers x 10 hold-outs)")
    ax.set_title("LODO macro-F1 by encoder")
    save(fig, out_dir, "fig_lodo_boxplot_by_encoder", dpi, log)


def fig_lodo_balanced_vs_full(bal, out_dir, dpi, log):
    b = bal.sort_values("full_mean_macro_f1", ascending=False)
    x = np.arange(len(b)); w = 0.27
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.8))
    ax.bar(x - w, b.indist_mean_macro_f1, w, color=SCEN_COLOR["indist"], label="In-distribution (published)", edgecolor="white", linewidth=0.5)
    ax.bar(x, b.full_mean_macro_f1, w, color=SCEN_COLOR["lodo"], label="LODO, full training pool", edgecolor="white", linewidth=0.5)
    ax.bar(x + w, b.balanced_mean_macro_f1, w, color=CAT[2], label="LODO, balanced pool (mean of 5 seeds)", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(b.held_out_dataset, rotation=30, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("Mean macro-F1 (32 pipelines)")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=7)
    ax.set_title("Volume vs. domain: full and balanced LODO per held-out corpus")
    fig.subplots_adjust(bottom=0.34)
    save(fig, out_dir, "fig_lodo_balanced_vs_full", dpi, log)


def fig_lodo_delta_heatmap(vs, out_dir, dpi, log):
    long = vs[vs.scope == "pipeline_x_corpus"]
    long = long.assign(pipeline=long.model_id.map(pipe_short))
    order = long.groupby("model_id").lodo_macro_f1.mean().sort_values(ascending=False).index
    m = long.pivot(index="model_id", columns="dataset_id", values="delta_lodo_minus_indist").loc[order]
    fig, ax = plt.subplots(figsize=(DOUBLE, 6.0))
    cmap = LinearSegmentedColormap.from_list("div", ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#7a1f1f"])
    im = ax.imshow(m.values, cmap=cmap, vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(m.shape[1])); ax.set_xticklabels(m.columns, rotation=30, ha="right")
    ax.set_yticks(range(m.shape[0])); ax.set_yticklabels([pipe_short(i) for i in m.index], fontsize=6.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.outline.set_visible(False)
    cb.set_label("LODO - in-distribution macro-F1")
    ax.set_title("Per-pipeline, per-corpus change in macro-F1 under LODO (pipelines sorted by mean LODO macro-F1)")
    save(fig, out_dir, "fig_lodo_delta_heatmap", dpi, log)


# ----------------------------------------------------------------------------
def fig_reliability(rel, cm, pipes, scenario, out_dir, dpi, log):
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 6.2), sharex=True, sharey=True)
    for ax, p in zip(axes.ravel(), pipes):
        ax.plot([0.5, 1], [0.5, 1], color=GREY, linewidth=0.8, linestyle="--", label="perfect calibration")
        for cond in ("raw", "sigmoid", "isotonic"):
            r = rel[(rel.pipeline == p) & (rel.scenario == scenario) & (rel.condition == cond) & (rel.corpus == "ALL")]
            r = r[r["count"] > 0]
            e = cm[(cm.pipeline == p) & (cm.scenario == scenario) & (cm.condition == cond) & (cm.corpus == "ALL")].iloc[0]
            ax.plot(r.mean_confidence, r.accuracy, marker="o", markersize=3.5, color=COND_COLOR[cond],
                    linestyle=COND_STYLE[cond], label=f"{cond} (ECE={e.ece:.3f})")
        ax.set_title(pipe_short(p)); ax.set_xlim(0.5, 1); ax.set_ylim(0.3, 1.02)
        ax.legend(frameon=False, loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("Confidence (mean per bin)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy in bin")
    fig.suptitle(f"Reliability diagrams - {SCEN_LABEL[scenario]} (15 equal-width bins, all corpora)", y=1.0)
    save(fig, out_dir, f"fig_reliability_{scenario}", dpi, log)


def fig_coverage_f1(curves, pipe, scenario, datasets, out_dir, dpi, log):
    panels = datasets + ["ALL"]
    n = len(panels); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(DOUBLE, 1.9 * nrow), sharex=True, sharey=True)
    for ax, ds in zip(axes.ravel(), panels):
        for cond in ("raw", "sigmoid", "isotonic"):
            c = curves[(curves.pipeline == pipe) & (curves.scenario == scenario) & (curves.condition == cond) & (curves.corpus == ds)]
            c = c.sort_values("coverage")
            ax.plot(c.coverage, c.macro_f1, color=COND_COLOR[cond], linestyle=COND_STYLE[cond], label=cond)
        ax.axhline(0.90, color=GREY, linewidth=0.6, linestyle=":")
        ax.axhline(0.95, color=GREY, linewidth=0.6, linestyle=":")
        ax.set_title(ds, fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0.0, 1.02)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes.ravel()[max(n - ncol, 0):n]:
        ax.set_xlabel("Coverage")
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro-F1 (covered)")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Coverage vs. macro-F1 - {pipe_short(pipe)} - {SCEN_LABEL[scenario]} (dotted: 0.90 / 0.95)", y=1.0)
    fig.subplots_adjust(bottom=0.16)
    tag = pipe.replace(" + ", "__")
    save(fig, out_dir, f"fig_coverage_f1_{scenario}_{tag}", dpi, log)


def risk_coverage(y, p):
    conf = np.maximum(p, 1 - p); pred = (p >= 0.5).astype(int)
    order = np.argsort(-conf, kind="mergesort")
    err = (pred != y).astype(float)[order]
    k = np.arange(1, len(y) + 1)
    return k / len(y), np.cumsum(err) / k


def fig_risk_coverage(preds, sel, pipes, scenario, out_dir, dpi, log):
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 5.6), sharex=True, sharey=True)
    for ax, p in zip(axes.ravel(), pipes):
        for cond in ("raw", "sigmoid", "isotonic"):
            g = preds[(preds.pipeline == p) & (preds.scenario == scenario) & (preds.condition == cond)]
            cov, risk = risk_coverage(g.true_label_id.to_numpy(), g.prob_real.to_numpy())
            a = sel[(sel.pipeline == p) & (sel.scenario == scenario) & (sel.condition == cond) & (sel.corpus == "ALL")].iloc[0]
            step = max(1, len(cov) // 2000)
            ax.plot(cov[::step], risk[::step], color=COND_COLOR[cond], linestyle=COND_STYLE[cond], label=f"{cond} (AURC={a.aurc:.3f})")
        ax.set_title(pipe_short(p)); ax.set_xlim(0, 1); ax.set_ylim(0, None)
        ax.legend(frameon=False, loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("Coverage")
    for ax in axes[:, 0]:
        ax.set_ylabel("Selective risk (error rate)")
    fig.suptitle(f"Risk-coverage curves - {SCEN_LABEL[scenario]} (all corpora pooled)", y=1.0)
    save(fig, out_dir, f"fig_risk_coverage_{scenario}", dpi, log)


def fig_triage_coverage(tri, pipes, datasets, out_dir, dpi, log, target=0.90, cond="sigmoid"):
    panels = datasets + ["ALL"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.0), sharey=True)
    w = 0.2
    x = np.arange(len(panels))
    for ax, sc in zip(axes, ("indist", "lodo")):
        for k, p in enumerate(pipes):
            e, c = p.split(" + ")
            rows = []
            for ds in panels:
                r = tri[(tri.pipeline == p) & (tri.scenario == sc) & (tri.condition == cond) & (tri.corpus == ds)
                        & np.isclose(tri.target_macro_f1, target)].iloc[0]
                rows.append(r)
            cov = np.array([r.coverage for r in rows]); lo = np.array([r.coverage_ci_low for r in rows]); hi = np.array([r.coverage_ci_high for r in rows])
            reached = np.array([r.status == "reached" for r in rows])
            hatch = "////" if c == "logreg" else None
            ax.bar(x + (k - 1.5) * w, cov, w, color=ENC_COLOR[e], hatch=hatch, edgecolor="white", linewidth=0.4,
                   label=pipe_short(p))
            ax.errorbar(x + (k - 1.5) * w, cov, yerr=[np.clip(cov - lo, 0, None), np.clip(hi - cov, 0, None)], fmt="none",
                        ecolor=INK2, elinewidth=0.6, capsize=1.2)
            for xi, ok in zip(x + (k - 1.5) * w, reached):
                if not ok:
                    ax.text(xi, 0.02, "x", ha="center", va="bottom", fontsize=6, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels(panels, rotation=35, ha="right")
        ax.set_ylim(0, 1.05); ax.set_title(SCEN_LABEL[sc])
    axes[0].set_ylabel(f"Coverage at macro-F1 >= {target:.2f}")
    axes[1].legend(frameon=False, loc="upper right", fontsize=6.5)
    fig.suptitle(f"Coverage achievable at macro-F1 >= {target:.2f} ({cond} calibration; x = target not reached; bars: 95% bootstrap CI)", y=1.02)
    save(fig, out_dir, f"fig_triage_coverage_{int(target * 100)}", dpi, log)


# ----------------------------------------------------------------------------
def fig_umap(coords, info, datasets, out_dir, dpi, log):
    sub = coords.sample(frac=1.0, random_state=0)  # shuffle draw order
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.9))
    ax = axes[0]
    for k, ds in enumerate(datasets):
        g = sub[sub.corpus == ds]
        ax.scatter(g.umap_1, g.umap_2, s=1.2, color=CORPUS_COLORS[k], linewidths=0, alpha=0.7, label=f"{ds} (n={len(g):,})", rasterized=True)
    ax.set_title(f"{ENC_SHORT[info['encoder']]} space, by corpus", fontsize=8)
    ax.legend(frameon=False, markerscale=6, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3,
              handletextpad=0.2, columnspacing=0.8)
    ax = axes[1]
    is_err = (sub.lodo_outcome == "error").to_numpy()
    # one scatter in shuffled order so that neither outcome hides the other
    ax.scatter(sub.umap_1, sub.umap_2, s=1.0, c=np.where(is_err, CAT[7], "#b8b7ac"), linewidths=0, alpha=0.55, rasterized=True)
    ax.scatter([], [], s=1.0, color="#b8b7ac", label=f"LODO correct (n={(~is_err).sum():,})")
    ax.scatter([], [], s=1.0, color=CAT[7], label=f"LODO error (n={is_err.sum():,})")
    ax.set_title(f"LODO outcome, {pipe_short(info['best_pipeline'])}", fontsize=8)
    ax.legend(frameon=False, markerscale=6, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=1)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    fig.subplots_adjust(bottom=0.3, wspace=0.08)
    save(fig, out_dir, "fig_umap_best_encoder", dpi, log)

    # small multiples: one corpus highlighted per panel, errors in red
    fig, axes = plt.subplots(2, 5, figsize=(DOUBLE, 3.4))
    for ax, ds, col in zip(axes.ravel(), datasets, CORPUS_COLORS):
        ax.scatter(sub.umap_1, sub.umap_2, s=0.4, color="#e6e5e1", linewidths=0, rasterized=True)
        g = sub[sub.corpus == ds]
        ax.scatter(g.umap_1, g.umap_2, s=1.2, color=col, linewidths=0, alpha=0.8, rasterized=True)
        ge = g[g.lodo_outcome == "error"]
        ax.scatter(ge.umap_1, ge.umap_2, s=1.6, color=CAT[7], linewidths=0, alpha=0.9, rasterized=True)
        ax.set_title(f"{ds} (err. {len(ge) / max(len(g), 1):.0%})", fontsize=6.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.subplots_adjust(wspace=0.12)
    fig.suptitle(f"Each corpus in the {ENC_SHORT[info['encoder']]} UMAP (red: LODO errors of {pipe_short(info['best_pipeline'])})", y=1.0)
    save(fig, out_dir, "fig_umap_small_multiples", dpi, log)


# ----------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    log = setup_logging(cfg, SCRIPT)
    style(cfg)
    dpi = cfg["figures"]["dpi"]
    res = rpath(cfg, "out_results")
    out = rpath(cfg, "out_figures")
    out.mkdir(parents=True, exist_ok=True)
    datasets = cfg["datasets"]
    pipes = [model_id(e, c) for e, c in cfg["calibration"]["pipelines"]]

    with step_timer(cfg, log, SCRIPT, "Study 1 figures"):
        raw = pd.read_csv(res / "lodo" / "lodo_raw.csv")
        # keep only the categories present (smoke/reduced configs)
        CLF_ORDER[:] = [c for c in CLF_ORDER if c in set(raw.classifier)]
        ENC_ORDER[:] = [e for e in ENC_ORDER if e in set(raw.encoder)]

        vs = pd.read_csv(res / "lodo" / "lodo_vs_indist.csv")
        t5 = pd.read_csv(res / "lodo" / "lodo_table_by_dataset.csv")
        bal = pd.read_csv(res / "lodo" / "lodo_balanced_by_dataset.csv")
        fig_lodo_heatmap(raw, out, dpi, log)
        fig_indist_vs_lodo_by_corpus(vs, t5, out, dpi, log)
        fig_lodo_boxplot_by_encoder(raw, out, dpi, log)
        fig_lodo_balanced_vs_full(bal, out, dpi, log)
        fig_lodo_delta_heatmap(vs, out, dpi, log)

    with step_timer(cfg, log, SCRIPT, "Study 2 figures"):
        rel = pd.read_csv(res / "calibration" / "reliability_bins.csv")
        cm = pd.read_csv(res / "calibration" / "calibration_metrics.csv")
        curves = pd.read_csv(res / "calibration" / "selective_curves.csv")
        tri = pd.read_csv(res / "calibration" / "triage_summary.csv")
        sel = pd.read_csv(res / "calibration" / "selective_summary.csv")
        preds = pd.read_parquet(res / "predictions" / "calibration_predictions.parquet",
                                columns=["pipeline", "scenario", "condition", "true_label_id", "prob_real"])
        for sc in ("indist", "lodo"):
            fig_reliability(rel, cm, pipes, sc, out, dpi, log)
            fig_risk_coverage(preds, sel, pipes, sc, out, dpi, log)
            for p in pipes:
                fig_coverage_f1(curves, p, sc, datasets, out, dpi, log)
        fig_triage_coverage(tri, pipes, datasets, out, dpi, log, target=0.90)
        fig_triage_coverage(tri, pipes, datasets, out, dpi, log, target=0.95)

    with step_timer(cfg, log, SCRIPT, "error-analysis figures"):
        coords = pd.read_csv(res / "errors" / "umap_coordinates.csv")
        info = json.loads((res / "errors" / "best_pipeline.json").read_text(encoding="utf-8"))
        fig_umap(coords, info, datasets, out, dpi, log)
    log.info("Figures written to %s", out)


if __name__ == "__main__":
    main()
