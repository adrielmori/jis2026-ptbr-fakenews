"""English, report-quality figures and tables of the PUBLISHED (in-distribution) results.

Everything here is derived from the read-only CSV/NPY files under ``codes/results`` (classical pipelines, zero-shot
LLM baselines, bootstrap intervals, McNemar / Pearson pairwise tests, Friedman + Wilcoxon-Holm, per-sample
predictions) and from the unified corpus. Nothing in ``codes/`` is modified; the original Portuguese/raw figures stay
where they are. Outputs:

  figures/baseline/fig_b*.{pdf,png}   tables/baseline/tab_b*.tex   results/baseline/*.csv (derived data behind figures)

Conventions follow make_figures.py (fixed categorical order, single-hue sequential ramp, blue-grey-red diverging).
"""
from __future__ import annotations

import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from common import (LABEL_NAMES, holm_bonferroni, load_config, load_encoder, rpath, setup_logging,  # noqa: E402
                    step_timer)
from make_figures import (CAT, CLF_ORDER, CLF_SHORT, CORPUS_COLORS, DOUBLE, ENC_COLOR, ENC_ORDER, ENC_SHORT, GREY,  # noqa: E402
                          INK, INK2, SEQ, SINGLE, save, style)
from make_tables import booktabs, fmt, tex_escape  # noqa: E402

SCRIPT = "make_baseline_figures"
DIV = LinearSegmentedColormap.from_list("div_bgr", ["#0d366b", "#2a78d6", "#d9d8d3", "#e34948", "#7a1414"])
ZS_COLOR = "#6250d6"   # zero-shot family (categorical slot 7, not used by the encoders)
SPLIT_COLOR = {"train": CAT[0], "val": CAT[2], "test": CAT[1]}
LABEL_COLOR = {"FAKE": CAT[7], "REAL": CAT[0]}
ZS_SHORT = {"deepseek-r1:8b": "DeepSeek-R1 8B", "gemma2:9b": "Gemma-2 9B", "gemma3:12b": "Gemma-3 12B",
            "gpt-oss:20b": "GPT-OSS 20B", "llama3.1:8b": "Llama-3.1 8B", "llama3.2:3b": "Llama-3.2 3B",
            "phi3:14b": "Phi-3 14B", "qwen2:7b": "Qwen-2 7B", "qwen2.5:7b": "Qwen-2.5 7B", "qwen2.5:14b": "Qwen-2.5 14B"}


# ----------------------------------------------------------------------------- helpers
def parse_source(name: str) -> tuple[str, str, str]:
    """'classic_models::bertimbau_base__svc_rbf' -> ('classical', encoder, classifier);
    'zero_shot::gemma2:9b' -> ('zero_shot', model, '')."""
    fam, rest = name.split("::", 1)
    if fam == "classic_models":
        enc, clf = rest.split("__", 1)
        return "classical", enc, clf
    return "zero_shot", rest, ""


def short_name(name: str) -> str:
    fam, a, b = parse_source(name)
    if fam == "classical":
        return f"{ENC_SHORT[a]} + {CLF_SHORT[b]}"
    return ZS_SHORT.get(a, a)


def family_color(name: str) -> str:
    fam, a, _ = parse_source(name)
    return ENC_COLOR[a] if fam == "classical" else ZS_COLOR


def corpus_of(idx: pd.Series) -> pd.Series:
    return idx.str.extract(r"^([A-Z0-9]+)")[0]


def macro_f1_from(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = []
    for c in ("FAKE", "REAL"):
        tp = np.sum((y_true == c) & (y_pred == c)); fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def recompute_report(ps: pd.DataFrame, published: pd.DataFrame) -> pd.DataFrame:
    """Per-model metrics with the split label convention (FAKE = label_id 0). Accuracy, macro-F1 and MCC are identical to the
    published report; the per-class columns and the confusion counts of the classical pipelines are the corrected ones."""
    rows = []
    for name, g in ps[~ps.excluded_from_metrics].groupby("source_name"):
        yt, yp = g.y_true.to_numpy(), g.y_pred.to_numpy()
        tp = int(np.sum((yt == "FAKE") & (yp == "FAKE"))); tn = int(np.sum((yt == "REAL") & (yp == "REAL")))
        fp = int(np.sum((yt == "REAL") & (yp == "FAKE"))); fn = int(np.sum((yt == "FAKE") & (yp == "REAL")))
        pf = tp / max(tp + fp, 1); rf = tp / max(tp + fn, 1); f1f = 0 if pf + rf == 0 else 2 * pf * rf / (pf + rf)
        pr = tn / max(tn + fn, 1); rr = tn / max(tn + fp, 1); f1r = 0 if pr + rr == 0 else 2 * pr * rr / (pr + rr)
        den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = 0.0 if den == 0 else (tp * tn - fp * fn) / den
        rows.append({"source_name": name, "n": len(g), "support_fake": tp + fn, "support_real": tn + fp,
                     "accuracy": (tp + tn) / len(g), "macro_f1": (f1f + f1r) / 2, "fake_precision": pf, "fake_recall": rf, "fake_f1": f1f,
                     "real_precision": pr, "real_recall": rr, "real_f1": f1r, "mcc": mcc, "TP": tp, "TN": tn, "FP": fp, "FN": fn})
    rep = pd.DataFrame(rows)
    pub = published.set_index("source_name")
    rep["n_excluded_undefined"] = rep.source_name.map(pub.n_excluded_undefined).fillna(0).astype(int)
    chk = rep.set_index("source_name").join(pub[["macro_f1", "accuracy"]], rsuffix="_pub")
    assert np.allclose(chk.macro_f1, chk.macro_f1_pub, atol=1e-9) and np.allclose(chk.accuracy, chk.accuracy_pub, atol=1e-9)
    return rep


def legend_below(fig, handles, labels, ncol, bottom, fontsize=7):
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.02), fontsize=fontsize)
    fig.subplots_adjust(bottom=bottom)


# ----------------------------------------------------------------------------- data loading
def load_published(cfg) -> dict:
    root = cfg["_repo_root"]
    ao = root / "codes/results/analysis_outputs_models"
    st = rpath(cfg, "published_stats_dir")
    d = {}
    d["report_test"] = pd.read_csv(ao / "test/test_detailed_model_report.csv")
    d["report_all"] = pd.read_csv(ao / "combined_detailed_model_report_all_splits.csv")
    d["ci_test"] = pd.read_csv(ao / "test/test_bootstrap_confidence_intervals.csv")
    d["mcnemar"] = pd.read_csv(ao / "test/test_pairwise_mcnemar_results.csv")
    d["pearson"] = pd.read_csv(ao / "test/test_pairwise_pearson_correctness.csv")
    d["agreement"] = pd.read_csv(ao / "test/test_sample_level_agreement_distribution.csv")
    d["cdf"] = pd.read_csv(ao / "test/test_cumulative_distribution_tables.csv")
    ps = pd.read_csv(ao / "test/test_standardized_predictions_per_sample.csv", dtype=str,
                     usecols=["source_group", "source_name", "idx", "y_true_raw", "y_pred_raw", "y_true", "y_pred", "excluded_from_metrics"])
    ps["excluded_from_metrics"] = ps.excluded_from_metrics.str.lower().eq("true")
    # D-26: the published analysis script mapped label_id 0 -> REAL / 1 -> FAKE for the classical family, whereas the split
    # files (and the zero-shot rows) use 0 = FAKE / 1 = REAL. Rebuild the class names of the classical rows from the raw ids.
    cl = ps.source_group == "classic_models"
    ps.loc[cl, "y_true"] = ps.loc[cl, "y_true_raw"].astype(int).map(LABEL_NAMES)
    ps.loc[cl, "y_pred"] = ps.loc[cl, "y_pred_raw"].astype(int).map(LABEL_NAMES)
    d["per_sample"] = ps
    d["report_test"] = recompute_report(ps, d["report_test"])
    d["undefined"] = pd.read_csv(ao / "test/test_zero_shot_undefined_exclusions.csv")
    for split in ("test", "val"):
        d[f"t4_{split}"] = pd.read_csv(st / f"{split}_classifier_embedding_mean_macro_f1.csv").set_index("classifier")
        d[f"t4std_{split}"] = pd.read_csv(st / f"{split}_classifier_embedding_std_macro_f1.csv").set_index("classifier")
        d[f"rank_within_{split}"] = pd.read_csv(st / f"{split}_classifier_embedding_rank_within_classifier.csv").set_index("classifier")
        d[f"spread_{split}"] = pd.read_csv(st / f"{split}_embedding_vs_classifier_spread.csv")
        d[f"friedman_{split}"] = pd.read_csv(st / f"{split}_friedman_result.csv")
        d[f"ranking_{split}"] = pd.read_csv(st / f"{split}_model_average_ranking.csv")
        d[f"holm_{split}"] = pd.read_csv(st / f"{split}_wilcoxon_posthoc_holm.csv")
        d[f"by_dataset_{split}"] = pd.read_csv(st / f"{split}_macro_f1_by_dataset_model.csv").set_index("dataset_id")
    d["results"] = pd.read_csv(rpath(cfg, "published_results"))
    d["test_preds"] = pd.read_csv(rpath(cfg, "published_test_predictions"))
    d["unified"] = pd.read_parquet(rpath(cfg, "unified_dataset"), columns=["idx", "dataset_id", "label", "text_no_url"])
    d["splits"] = pd.concat([pd.read_csv(rpath(cfg, "splits_dir") / f"{s}_split.csv", usecols=["idx", "dataset_id", "label"]).assign(split=s)
                             for s in ("train", "val", "test")], ignore_index=True)
    return d


def model_order(report: pd.DataFrame) -> list[str]:
    """Zero-shot models first (by macro-F1), then classical pipelines grouped by encoder (ENC_ORDER) and CLF_ORDER."""
    zs = report[report.source_name.str.startswith("zero_shot")].sort_values("macro_f1", ascending=False).source_name.tolist()
    cl = [f"classic_models::{e}__{c}" for e in ENC_ORDER for c in CLF_ORDER]
    return zs + [m for m in cl if m in set(report.source_name)]


# ----------------------------------------------------------------------------- figures
def fig_corpus_overview(d, datasets, out_dir, dpi, log):
    u = d["unified"].copy()
    u["n_tokens"] = u.text_no_url.fillna("").astype(str).str.split().str.len()
    counts = u.groupby(["dataset_id", "label"]).size().unstack(fill_value=0)
    order = counts.sum(axis=1).sort_values(ascending=True).index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    y = np.arange(len(order))
    for lab, left in (("FAKE", None), ("REAL", counts.loc[order, "FAKE"].values)):
        vals = counts.loc[order, lab].values
        ax.barh(y, vals, left=left, color=LABEL_COLOR[lab], edgecolor="white", linewidth=0.5, height=0.7, label=lab)
    for yi, ds in zip(y, order):
        tot = counts.loc[ds].sum(); share = counts.loc[ds, "FAKE"] / tot
        txt = f"{tot:,}  ({share:.0%} fake)" + ("  - no embeddings, excluded" if ds not in datasets else "")
        ax.text(tot + 150, yi, txt, va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.set_xlim(0, counts.sum(axis=1).max() * 1.55); ax.set_xlabel("Instances")
    ax.set_title("(a) Corpus size and label balance")
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    ax = axes[1]
    data = [u.loc[u.dataset_id == ds, "n_tokens"].clip(lower=1).values for ds in order]
    bp = ax.boxplot(data, vert=False, widths=0.6, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.1), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    for patch, ds in zip(bp["boxes"], order):
        patch.set_facecolor("#cde2fb" if ds in datasets else "#e6e5e1"); patch.set_edgecolor(INK2)
    ax.set_xscale("log"); ax.set_yticks(range(1, len(order) + 1)); ax.set_yticklabels([])
    ax.set_xlabel("Whitespace tokens per text (log scale)")
    ax.set_title("(b) Text length (median, IQR, 1.5 IQR whiskers)")
    fig.subplots_adjust(wspace=0.08)
    save(fig, out_dir, "fig_b01_corpus_overview", dpi, log)
    stats = u.groupby("dataset_id").agg(n=("idx", "size"), n_fake=("label", lambda s: int((s == "FAKE").sum())),
                                        n_real=("label", lambda s: int((s == "REAL").sum())),
                                        tokens_mean=("n_tokens", "mean"), tokens_median=("n_tokens", "median"),
                                        tokens_p90=("n_tokens", lambda s: s.quantile(0.9)))
    return stats


def fig_split_composition(d, datasets, out_dir, dpi, log):
    s = d["splits"]
    s = s[s.dataset_id.isin(datasets)]
    ct = s.groupby(["dataset_id", "split"]).size().unstack(fill_value=0).loc[datasets, ["train", "val", "test"]]
    share = ct.div(ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.9))
    y = np.arange(len(datasets)); left = np.zeros(len(datasets))
    for sp in ("train", "val", "test"):
        ax.barh(y, share[sp].values, left=left, color=SPLIT_COLOR[sp], edgecolor="white", linewidth=0.5, height=0.7,
                label=f"{sp} ({ct[sp].sum():,})")
        for yi, (l0, v, n) in enumerate(zip(left, share[sp].values, ct[sp].values)):
            if v > 0.12:
                ax.text(l0 + v / 2, yi, f"{n:,}", ha="center", va="center", fontsize=6, color="white")
        left += share[sp].values
    ax.set_yticks(y); ax.set_yticklabels(datasets); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("Share of the corpus"); ax.set_title("Stratified 70/15/15 split per corpus")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3, fontsize=7)
    fig.subplots_adjust(bottom=0.3)
    save(fig, out_dir, "fig_b02_split_composition", dpi, log)
    return ct


def fig_table4_heatmaps(d, out_dir, dpi, log):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.3), sharey=True)
    for ax, split, title in zip(axes, ("val", "test"), ("(a) Validation split", "(b) Test split")):
        m = d[f"t4_{split}"].loc[CLF_ORDER, ENC_ORDER]
        im = ax.imshow(m.values, cmap=SEQ, vmin=0.35, vmax=0.85, aspect="auto")
        ax.set_xticks(range(4)); ax.set_xticklabels([ENC_SHORT[e] for e in ENC_ORDER], rotation=30, ha="right")
        ax.set_yticks(range(len(CLF_ORDER))); ax.set_yticklabels([CLF_SHORT[c] for c in CLF_ORDER]); ax.grid(False)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.values[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color="white" if v > 0.66 else INK)
        ax.set_title(title)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02); cb.outline.set_visible(False)
    cb.set_label("Mean per-corpus macro-F1 (10 corpora)")
    fig.suptitle("In-distribution macro-F1 by classifier and encoder (published Table 4)", y=1.0)
    save(fig, out_dir, "fig_b03_table4_heatmaps", dpi, log)


def fig_per_corpus_heatmap(d, datasets, out_dir, dpi, log):
    m = d["by_dataset_test"]
    cols = [f"{e} + {c}" for e in ENC_ORDER for c in CLF_ORDER]
    m = m.loc[datasets, cols].T
    fig, ax = plt.subplots(figsize=(DOUBLE, 6.2))
    im = ax.imshow(m.values, cmap=SEQ, vmin=0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels([CLF_SHORT[c.split(" + ")[1]] for c in cols], fontsize=6.5)
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color="white" if v > 0.7 else INK)
    for k, e in enumerate(ENC_ORDER):
        y0 = k * len(CLF_ORDER) - 0.5
        if k:
            ax.axhline(y0, color="white", linewidth=2)
        ax.text(-2.6, y0 + len(CLF_ORDER) / 2, ENC_SHORT[e], rotation=90, va="center", ha="center", fontsize=7,
                color=ENC_COLOR[e], fontweight="bold")
    col_mean = m.mean(axis=0)
    for j, v in enumerate(col_mean):
        ax.text(j, -0.9, f"{v:.2f}", ha="center", va="bottom", fontsize=6, color=INK2)
    ax.text(-0.5, -0.9, "mean", ha="right", va="bottom", fontsize=6, color=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.outline.set_visible(False); cb.set_label("Test macro-F1")
    ax.set_title("Per-corpus test macro-F1 of the 32 classical pipelines", pad=14)
    save(fig, out_dir, "fig_b04_per_corpus_heatmap", dpi, log)


def fig_model_ranking_ci(d, out_dir, dpi, log):
    ci = d["ci_test"]
    rep = d["report_test"].set_index("source_name")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 6.4), sharey=True, gridspec_kw={"width_ratios": [1, 1]})
    order = rep.sort_values("macro_f1").index.tolist()
    y = np.arange(len(order))
    for ax, metric, title in zip(axes, ("macro_f1", "accuracy"), ("(a) Macro-F1", "(b) Accuracy")):
        c = ci[ci.metric == metric].set_index("source_name").loc[order]
        for yi, name in zip(y, order):
            col = family_color(name)
            ax.plot([c.ci_low[name], c.ci_high[name]], [yi, yi], color=col, linewidth=1.3, solid_capstyle="round")
            ax.plot(c.point_estimate[name], yi, "o", color=col, markersize=3.2, markeredgecolor="white", markeredgewidth=0.4)
        ax.set_title(title); ax.set_xlim(0.35, 1.0)
        ax.axvline(0.5, color=GREY, linewidth=0.6, linestyle=":")
    axes[0].set_yticks(y); axes[0].set_yticklabels([short_name(n) for n in order], fontsize=6.3)
    axes[0].set_xlabel("Test macro-F1 (95 % bootstrap CI, 2 000 resamples)"); axes[1].set_xlabel("Test accuracy (95 % bootstrap CI)")
    handles = [Line2D([], [], color=ENC_COLOR[e], marker="o", linestyle="-", markersize=3, label=ENC_SHORT[e]) for e in ENC_ORDER]
    handles.append(Line2D([], [], color=ZS_COLOR, marker="o", linestyle="-", markersize=3, label="Zero-shot LLM"))
    legend_below(fig, handles, [h.get_label() for h in handles], ncol=5, bottom=0.08)
    fig.suptitle("All 42 evaluated models on the test split (32 classical pipelines, 10 zero-shot LLMs)", y=0.995)
    save(fig, out_dir, "fig_b05_model_ranking_ci", dpi, log)


def fig_class_f1_mcc(d, out_dir, dpi, log):
    rep = d["report_test"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for _, r in rep.iterrows():
        fam = parse_source(r.source_name)[0]
        ax.scatter(r.fake_f1, r.real_f1, s=22 if fam == "zero_shot" else 16, color=family_color(r.source_name),
                   marker="s" if fam == "zero_shot" else "o", edgecolor="white", linewidth=0.4, zorder=3)
    ax.plot([0, 1], [0, 1], color=GREY, linewidth=0.6, linestyle=":")
    zs = rep[rep.source_name.str.startswith("zero_shot")].sort_values("real_f1")
    offsets = [(5, -7), (5, 3), (-5, 3), (-5, -7)]
    for k, (_, r) in enumerate(zs.iterrows()):
        dx, dy = offsets[k % 4]
        ax.annotate(short_name(r.source_name), (r.fake_f1, r.real_f1), xytext=(dx, dy), textcoords="offset points", fontsize=5.5,
                    color=INK2, ha="left" if dx > 0 else "right",
                    arrowprops=dict(arrowstyle="-", color=GREY, linewidth=0.4, shrinkA=0, shrinkB=2))
    best = rep.sort_values("macro_f1", ascending=False).iloc[0]
    ax.annotate(short_name(best.source_name), (best.fake_f1, best.real_f1), xytext=(-4, 4), textcoords="offset points",
                fontsize=5.5, color=INK2, ha="right")
    ax.set_xlabel("F1 of the FAKE class"); ax.set_ylabel("F1 of the REAL class"); ax.set_xlim(0.2, 1); ax.set_ylim(0.2, 1)
    ax.set_title("(a) Per-class F1 on the test split")
    ax = axes[1]
    order = rep.sort_values("mcc").source_name.tolist()
    ax.barh(range(len(order)), rep.set_index("source_name").loc[order, "mcc"], color=[family_color(n) for n in order],
            edgecolor="white", linewidth=0.4, height=0.75)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([short_name(n) for n in order], fontsize=5.6)
    ax.set_xlabel("Matthews correlation coefficient"); ax.set_title("(b) MCC"); ax.axvline(0, color=INK2, linewidth=0.6)
    handles = [Line2D([], [], color=ENC_COLOR[e], marker="o", linestyle="", markersize=4, label=ENC_SHORT[e]) for e in ENC_ORDER]
    handles.append(Line2D([], [], color=ZS_COLOR, marker="s", linestyle="", markersize=4, label="Zero-shot LLM"))
    legend_below(fig, handles, [h.get_label() for h in handles], ncol=5, bottom=0.14)
    fig.subplots_adjust(wspace=0.55)
    save(fig, out_dir, "fig_b06_class_f1_mcc", dpi, log)


def fig_sample_difficulty(d, datasets, out_dir, dpi, log):
    ag = d["agreement"].copy(); ag["corpus"] = corpus_of(ag.idx)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    k = np.sort(ag.n_models_correct.values); cdf = np.arange(1, len(k) + 1) / len(k)
    ax.step(k, cdf, where="post", color=CAT[0], linewidth=1.5)
    ax.set_xlabel("Models correct per item (of 42)"); ax.set_ylabel("Empirical CDF")
    ax.set_title("(a) Agreement across the model zoo"); ax.set_xlim(0, 42); ax.set_ylim(0, 1.02)
    for q in (0.1, 0.5):
        kq = np.quantile(ag.n_models_correct, q)
        ax.axvline(kq, color=GREY, linewidth=0.6, linestyle=":")
        ax.text(kq - 0.5, 0.97, f"P{q * 100:.0f} = {kq:.0f}", ha="right", va="top", fontsize=6.5, color=INK2)
    ax = axes[1]
    order = ag.groupby("corpus").fraction_models_correct.median().loc[datasets].sort_values(ascending=False).index.tolist()
    data = [ag.loc[ag.corpus == c, "fraction_models_correct"].values for c in order]
    bp = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.1), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(CORPUS_COLORS[datasets.index(c)]); patch.set_alpha(0.55); patch.set_edgecolor(INK2)
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("Fraction of the 42 models correct"); ax.set_ylim(0, 1.02)
    ax.set_title("(b) Item difficulty by corpus")
    save(fig, out_dir, "fig_b07_sample_difficulty", dpi, log)


def zoo_per_corpus(d, datasets) -> pd.DataFrame:
    ps = d["per_sample"]; ps = ps[~ps.excluded_from_metrics].copy()
    ps["corpus"] = corpus_of(ps.idx)
    rows = []
    for (name, c), g in ps.groupby(["source_name", "corpus"]):
        rows.append({"source_name": name, "corpus": c, "n": len(g),
                     "macro_f1": macro_f1_from(g.y_true.to_numpy(), g.y_pred.to_numpy()),
                     "accuracy": float((g.y_true == g.y_pred).mean()),
                     "predicted_fake_rate": float((g.y_pred == "FAKE").mean()), "true_fake_rate": float((g.y_true == "FAKE").mean())})
    return pd.DataFrame(rows)


def fig_zoo_per_corpus_heatmap(zoo, order, datasets, out_dir, dpi, log):
    m = zoo.pivot(index="source_name", columns="corpus", values="macro_f1").loc[order, datasets]
    fig, ax = plt.subplots(figsize=(DOUBLE, 7.4))
    im = ax.imshow(m.values, cmap=SEQ, vmin=0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([short_name(n) for n in order], fontsize=6)
    for lab, name in zip(ax.get_yticklabels(), order):
        lab.set_color(family_color(name))
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=5.3, color=GREY)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.3, color="white" if v > 0.7 else INK)
    n_zs = sum(n.startswith("zero_shot") for n in order)
    ax.axhline(n_zs - 0.5, color="white", linewidth=2.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.outline.set_visible(False)
    cb.set_label("Test macro-F1 within the corpus")
    ax.set_title("Per-corpus test macro-F1 of all 42 models (zero-shot runs do not cover COVID19BR; undefined answers excluded)",
                 fontsize=8)
    save(fig, out_dir, "fig_b08_zoo_per_corpus_heatmap", dpi, log)


def fig_mcnemar_matrix(d, order, alpha, out_dir, dpi, log) -> pd.DataFrame:
    mc = d["mcnemar"].copy()
    adj, rej = holm_bonferroni(mc.mcnemar_pvalue_exact.to_numpy(), alpha)
    mc["holm_p"] = adj; mc["reject_holm"] = rej
    n = len(order); pos = {m: i for i, m in enumerate(order)}
    M = np.full((n, n), np.nan); S = np.zeros((n, n), dtype=bool)
    for _, r in mc.iterrows():
        if r.model_a not in pos or r.model_b not in pos:
            continue
        i, j = pos[r.model_a], pos[r.model_b]
        adv = (r.a_correct_b_wrong - r.a_wrong_b_correct) / r.n_common
        M[i, j] = adv; M[j, i] = -adv; S[i, j] = S[j, i] = bool(r.reject_holm)
    fig, ax = plt.subplots(figsize=(DOUBLE, 6.9))
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    im = ax.imshow(M, cmap=DIV, norm=norm, aspect="auto")
    ii, jj = np.where(~S & ~np.isnan(M))
    ax.scatter(jj, ii, marker="x", s=5, color=INK2, linewidths=0.4, label="not significant (Holm)")
    ax.set_xticks(range(n)); ax.set_xticklabels([short_name(m) for m in order], rotation=90, fontsize=5.2)
    ax.set_yticks(range(n)); ax.set_yticklabels([short_name(m) for m in order], fontsize=5.2)
    for lab, name in zip(ax.get_yticklabels(), order):
        lab.set_color(family_color(name))
    for lab, name in zip(ax.get_xticklabels(), order):
        lab.set_color(family_color(name))
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.outline.set_visible(False)
    cb.set_label("Net advantage of the row model: (row right & column wrong - row wrong & column right) / n")
    n_rej = int(rej.sum())
    ax.set_title(f"Pairwise McNemar tests on the test split: {n_rej} of {len(mc)} pairs significant after Holm (alpha = {alpha})")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.22), fontsize=6.5, markerscale=2)
    save(fig, out_dir, "fig_b09_mcnemar_matrix", dpi, log)
    return mc


def fig_correctness_correlation(d, order, out_dir, dpi, log):
    pe = d["pearson"]
    n = len(order); pos = {m: i for i, m in enumerate(order)}
    R = np.eye(n)
    for _, r in pe.iterrows():
        if r.model_a in pos and r.model_b in pos:
            R[pos[r.model_a], pos[r.model_b]] = R[pos[r.model_b], pos[r.model_a]] = r.pearson_r_correctness
    fig, ax = plt.subplots(figsize=(DOUBLE, 6.9))
    im = ax.imshow(R, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels([short_name(m) for m in order], rotation=90, fontsize=5.2)
    ax.set_yticks(range(n)); ax.set_yticklabels([short_name(m) for m in order], fontsize=5.2)
    for lab, name in zip(ax.get_yticklabels(), order):
        lab.set_color(family_color(name))
    for lab, name in zip(ax.get_xticklabels(), order):
        lab.set_color(family_color(name))
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.outline.set_visible(False)
    cb.set_label("Pearson r between per-item correctness vectors")
    ax.set_title("Which models fail on the same items? Correctness correlation on the test split")
    save(fig, out_dir, "fig_b10_correctness_correlation", dpi, log)
    return pd.DataFrame(R, index=order, columns=order)


def fig_friedman_ranks(d, split, out_dir, dpi, log):
    rk = d[f"ranking_{split}"].sort_values("avg_rank_complete_cases")
    hm = d[f"holm_{split}"].set_index("comparison_model")
    fr = d[f"friedman_{split}"].iloc[0]
    ref = hm.reference_model.iloc[0]
    fig, ax = plt.subplots(figsize=(SINGLE, 4.6))
    y = np.arange(len(rk))
    for yi, r in zip(y, rk.itertuples()):
        e, c = r.model_id.split(" + ")
        is_ref = r.model_id == ref
        sig = (not is_ref) and bool(hm.loc[r.model_id, "reject_h0_holm"])
        ax.plot([1, r.avg_rank_complete_cases], [yi, yi], color="#e6e5e1", linewidth=0.8, zorder=1)
        ax.plot(r.avg_rank_complete_cases, yi, "*" if is_ref else "o", color=ENC_COLOR[e], markersize=7 if is_ref else 4.2,
                markerfacecolor=ENC_COLOR[e] if (sig or is_ref) else "white", markeredgewidth=1.0, zorder=3)
        ax.text(r.avg_rank_complete_cases + 0.5, yi, f"{r.mean_macro_f1:.3f}", va="center", fontsize=5.5, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([f"{ENC_SHORT[m.split(' + ')[0]]} + {CLF_SHORT[m.split(' + ')[1]]}" for m in rk.model_id], fontsize=5.6)
    ax.invert_yaxis(); ax.set_xlim(0, 33); ax.set_xlabel("Average rank over the 10 corpora (1 = best)")
    ax.set_title(f"Friedman chi2 = {fr.statistic:.1f}, p = {fr.p_value:.1e}, Kendall W = {fr.kendall_w:.2f} ({split})", fontsize=7.5)
    n_rej = int(hm.reject_h0_holm.sum())
    handles = [Line2D([], [], color=INK2, marker="*", linestyle="", markerfacecolor=INK2, markersize=7,
                      label=f"reference: {ENC_SHORT[ref.split(' + ')[0]]} + {CLF_SHORT[ref.split(' + ')[1]]} (best mean macro-F1)"),
               Line2D([], [], color=INK2, marker="o", linestyle="", markerfacecolor=INK2, markersize=4,
                      label=f"significantly worse than the reference (Wilcoxon + Holm): {n_rej} of {len(hm)}"),
               Line2D([], [], color=INK2, marker="o", linestyle="", markerfacecolor="white", markersize=4,
                      label="not significant after Holm (n = 10 corpora: minimum exact p = 0.002)")]
    legend_below(fig, handles, [h.get_label() for h in handles], ncol=1, bottom=0.16, fontsize=6)
    save(fig, out_dir, f"fig_b11_friedman_ranks_{split}", dpi, log)


def fig_spread(d, datasets, out_dir, dpi, log):
    s = d["spread_test"].set_index("dataset_id").loc[datasets]
    x = np.arange(len(datasets)); w = 0.38
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.7))
    ax.bar(x - w / 2, s.mean_classifier_spread_within_embedding, w, color=CAT[0], edgecolor="white", linewidth=0.5,
           label="across classifiers, encoder fixed")
    ax.bar(x + w / 2, s.mean_embedding_spread_within_classifier, w, color=CAT[1], edgecolor="white", linewidth=0.5,
           label="across encoders, classifier fixed")
    ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=30, ha="right"); ax.set_ylabel("Macro-F1 range (max - min)")
    ax.set_title("What drives the in-distribution variance: classifier choice vs. encoder choice (test split)")
    legend_below(fig, *ax.get_legend_handles_labels(), ncol=2, bottom=0.36)
    save(fig, out_dir, "fig_b12_embedding_vs_classifier_spread", dpi, log)


def fig_rank_within_classifier(d, out_dir, dpi, log):
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE * 1.6, 2.8), sharey=True)
    for ax, split, title in zip(axes, ("val", "test"), ("(a) Validation", "(b) Test")):
        m = d[f"rank_within_{split}"].loc[CLF_ORDER, ENC_ORDER]
        im = ax.imshow(m.values, cmap=SEQ.reversed(), vmin=1, vmax=4, aspect="auto")
        ax.set_xticks(range(4)); ax.set_xticklabels([ENC_SHORT[e] for e in ENC_ORDER], rotation=30, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(CLF_ORDER))); ax.set_yticklabels([CLF_SHORT[c] for c in CLF_ORDER]); ax.grid(False)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.values[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v < 2.5 else INK)
        ax.set_title(title)
    cb = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.03, ticks=[1, 2, 3, 4]); cb.outline.set_visible(False)
    cb.set_label("Encoder rank within classifier (1 = best)")
    save(fig, out_dir, "fig_b13_encoder_rank_within_classifier", dpi, log)


def fig_confusion(d, out_dir, dpi, log):
    rep = d["report_test"].set_index("source_name")
    top_cl = [f"classic_models::{e}__svc_rbf" for e in ENC_ORDER]
    top_zs = rep[rep.index.str.startswith("zero_shot")].sort_values("macro_f1", ascending=False).index[:4].tolist()
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE, 4.0))
    for ax, name in zip(axes.ravel(), top_cl + top_zs):
        r = rep.loc[name]
        # rows: true FAKE / REAL; columns: predicted FAKE / REAL (TP = FAKE predicted FAKE)
        cm = np.array([[r.TP, r.FN], [r.FP, r.TN]], dtype=float)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap=SEQ, vmin=0, vmax=1); ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j]):,})", ha="center", va="center", fontsize=6.3,
                        color="white" if cmn[i, j] > 0.6 else INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["FAKE", "REAL"], fontsize=6.5); ax.set_yticks([0, 1]); ax.set_yticklabels(["FAKE", "REAL"], fontsize=6.5)
        ax.set_title(f"{short_name(name)}\nmacro-F1 {r.macro_f1:.3f}", fontsize=7, color=family_color(name))
    for ax in axes[1]:
        ax.set_xlabel("Predicted")
    for ax in axes[:, 0]:
        ax.set_ylabel("True")
    fig.suptitle("Row-normalised confusion matrices: SVC-RBF per encoder (top) and the four best zero-shot LLMs (bottom)", y=1.0)
    fig.subplots_adjust(hspace=0.55, wspace=0.35)
    save(fig, out_dir, "fig_b14_confusion_matrices", dpi, log)


def fig_zero_shot_profile(d, zoo, out_dir, dpi, log) -> pd.DataFrame:
    rep = d["report_test"]; rep = rep[rep.source_name.str.startswith("zero_shot")].copy()
    ci = d["ci_test"]; ci = ci[ci.metric == "macro_f1"].set_index("source_name")
    ps = d["per_sample"]; ps = ps[ps.source_group == "zero_shot"]
    rate = pd.DataFrame({"predicted_fake_rate": ps.groupby("source_name").y_pred.apply(lambda s: float((s == "FAKE").mean())),
                         "n_undefined": ps.groupby("source_name").excluded_from_metrics.sum().astype(int)})
    true_fake = float((ps.drop_duplicates("idx").y_true == "FAKE").mean())
    rep = rep.set_index("source_name").join(rate).sort_values("macro_f1")
    y = np.arange(len(rep))
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.9), sharey=True, gridspec_kw={"width_ratios": [1.3, 1, 0.7]})
    ax = axes[0]
    for yi, name in zip(y, rep.index):
        ax.plot([ci.ci_low[name], ci.ci_high[name]], [yi, yi], color=ZS_COLOR, linewidth=1.3)
        ax.plot(rep.macro_f1[name], yi, "o", color=ZS_COLOR, markersize=3.5, markeredgecolor="white", markeredgewidth=0.4)
    best_cl = d["report_test"][~d["report_test"].source_name.str.startswith("zero_shot")].macro_f1.max()
    ax.axvline(best_cl, color=CAT[1], linewidth=0.8, linestyle="--"); ax.text(best_cl - 0.01, len(rep) - 0.6, "best classical", ha="right", fontsize=6, color=CAT[1])
    ax.axvline(0.5, color=GREY, linewidth=0.6, linestyle=":")
    ax.set_yticks(y); ax.set_yticklabels([short_name(n) for n in rep.index]); ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("Macro-F1 (95 % bootstrap CI)"); ax.set_title("(a) Test macro-F1")
    ax = axes[1]
    ax.barh(y, rep.predicted_fake_rate, color=ZS_COLOR, edgecolor="white", linewidth=0.5, height=0.7)
    ax.axvline(true_fake, color=INK, linewidth=0.9, linestyle="--"); ax.text(true_fake + 0.01, -0.6, f"true FAKE share {true_fake:.2f}", fontsize=6, color=INK2)
    ax.set_xlim(0, 1); ax.set_xlabel("Share of items predicted FAKE"); ax.set_title("(b) Label bias")
    ax = axes[2]
    ax.barh(y, rep.n_undefined, color=GREY, edgecolor="white", linewidth=0.5, height=0.7)
    for yi, v in zip(y, rep.n_undefined):
        ax.text(v + 0.1, yi, f"{int(v)}", va="center", fontsize=6, color=INK2)
    ax.set_xlim(0, rep.n_undefined.max() * 1.6 + 1); ax.set_xlabel("Undefined answers (excluded)"); ax.set_title("(c) Refusals")
    fig.suptitle("Zero-shot LLM baselines on the test split (5 765 items: the zero-shot runs do not cover COVID19BR)", y=1.0)
    save(fig, out_dir, "fig_b15_zero_shot_profile", dpi, log)
    rep["true_fake_rate"] = true_fake
    return rep.reset_index()


def hard_examples(d, k_per_corpus=2) -> pd.DataFrame:
    """The k test items per corpus misclassified by most classical pipelines (as in the published analysis notebook)."""
    tp = d["test_preds"]
    tp = tp[tp.split.str.lower() == "test"].copy()
    tp["err"] = (tp.true_label_id != tp.pred_label_id).astype(int)
    g = tp.groupby(["dataset_id", "idx"]).agg(n_pipelines_wrong=("err", "sum"), true_label_id=("true_label_id", "first")).reset_index()
    g = g.sort_values(["dataset_id", "n_pipelines_wrong", "idx"], ascending=[True, False, True])
    top = g.groupby("dataset_id", group_keys=False).head(k_per_corpus).reset_index(drop=True)
    ps = d["per_sample"]; zs = ps[(ps.source_group == "zero_shot") & ps.idx.isin(top.idx)]
    zs_wrong = zs.assign(wrong=(zs.y_true != zs.y_pred) & ~zs.excluded_from_metrics).groupby("idx").wrong.sum()
    top["n_llms_wrong"] = top.idx.map(zs_wrong).fillna(0).astype(int)
    top["label"] = top.true_label_id.map(LABEL_NAMES)
    return top


def umap_embed(X, cfg, seed):
    import umap
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    u = cfg["errors"]["umap"]
    Z = PCA(n_components=min(u["pca_components"], X.shape[1]), random_state=seed).fit_transform(X)
    Z = normalize(Z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        red = umap.UMAP(n_components=2, n_neighbors=u["n_neighbors"], min_dist=u["min_dist"], metric=u["metric"],
                        spread=u["spread"], random_state=seed, transform_seed=seed)
        return red.fit_transform(Z)


def fig_umap_test(cfg, d, datasets, hard, out_dir, dpi, log) -> pd.DataFrame:
    seed = cfg["random_state"]
    coords = []
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE, 4.3))
    for k, enc in enumerate(ENC_ORDER):
        ed = load_encoder(cfg, enc)
        m = ed.mask_split("test"); meta = ed.meta[m].reset_index(drop=True); X = ed.X[m]
        Z = umap_embed(X, cfg, seed)
        df = pd.DataFrame({"encoder": enc, "idx": meta.idx.values, "corpus": meta.dataset_id.values,
                           "label": meta.label_id.map(LABEL_NAMES).values, "umap_1": Z[:, 0], "umap_2": Z[:, 1]})
        coords.append(df)
        rng = np.random.default_rng(seed); perm = rng.permutation(len(df))
        ax = axes[0, k]
        ax.scatter(df.umap_1.values[perm], df.umap_2.values[perm], s=1.2, c=df.label.map(LABEL_COLOR).values[perm], linewidths=0, alpha=0.6, rasterized=True)
        ax.set_title(ENC_SHORT[enc], fontsize=8)
        ax = axes[1, k]
        ax.scatter(df.umap_1.values[perm], df.umap_2.values[perm], s=1.2,
                   c=[CORPUS_COLORS[datasets.index(c)] for c in df.corpus.values[perm]], linewidths=0, alpha=0.6, rasterized=True)
        h = df[df.idx.isin(hard.idx)]
        for ax_ in (axes[0, k], axes[1, k]):
            ax_.scatter(h.umap_1, h.umap_2, s=16, facecolor="none", edgecolor=INK, linewidths=0.7, zorder=4)
        log.info("UMAP %s: %d test items", enc, len(df))
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[0, 0].set_ylabel("coloured by label"); axes[1, 0].set_ylabel("coloured by corpus")
    h1 = [Line2D([], [], marker="o", linestyle="", color=LABEL_COLOR[l], markersize=4, label=l) for l in ("FAKE", "REAL")]
    h1.append(Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=INK, markersize=5, label="hard example (2 per corpus)"))
    h2 = [Line2D([], [], marker="o", linestyle="", color=CORPUS_COLORS[i], markersize=4, label=c) for i, c in enumerate(datasets)]
    fig.legend(h1 + h2, [h.get_label() for h in h1 + h2], frameon=False, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.02), fontsize=6,
               handletextpad=0.3, columnspacing=0.9)
    fig.suptitle("UMAP of the test-split embeddings (6 281 items) for the four encoders", y=1.0)
    fig.subplots_adjust(bottom=0.16, wspace=0.06, hspace=0.12)
    save(fig, out_dir, "fig_b16_umap_test_encoders", dpi, log)
    return pd.concat(coords, ignore_index=True)


def fig_umap_splits(cfg, d, datasets, out_dir, dpi, log):
    seed = cfg["random_state"]
    fig, axes = plt.subplots(4, 3, figsize=(DOUBLE * 0.85, 7.6))
    for k, enc in enumerate(ENC_ORDER):
        ed = load_encoder(cfg, enc)
        for j, sp in enumerate(("train", "val", "test")):
            m = ed.mask_split(sp); meta = ed.meta[m].reset_index(drop=True); Z = umap_embed(ed.X[m], cfg, seed)
            rng = np.random.default_rng(seed); perm = rng.permutation(len(Z))
            ax = axes[k, j]
            ax.scatter(Z[perm, 0], Z[perm, 1], s=0.8, c=[CORPUS_COLORS[datasets.index(c)] for c in meta.dataset_id.values[perm]],
                       linewidths=0, alpha=0.55, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if k == 0:
                ax.set_title(f"{sp} (n = {len(Z):,})", fontsize=8)
            if j == 0:
                ax.set_ylabel(ENC_SHORT[enc], fontsize=8)
            log.info("UMAP %s/%s: %d items", enc, sp, len(Z))
    h = [Line2D([], [], marker="o", linestyle="", color=CORPUS_COLORS[i], markersize=4, label=c) for i, c in enumerate(datasets)]
    fig.legend(h, [x.get_label() for x in h], frameon=False, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01), fontsize=6.5)
    fig.suptitle("UMAP of each split, coloured by corpus: the corpus structure is identical across splits", y=0.995)
    fig.subplots_adjust(bottom=0.07, wspace=0.05, hspace=0.08)
    save(fig, out_dir, "fig_b17_umap_splits_by_corpus", dpi, log)


# ----------------------------------------------------------------------------- tables
def tables(d, stats, split_ct, zoo, zs_profile, hard, mc, tab_dir, datasets, alpha, log):
    from make_tables import write
    # corpus statistics
    rows = []
    for ds in datasets + [x for x in stats.index if x not in datasets]:
        s = stats.loc[ds]; sp = split_ct.loc[ds] if ds in split_ct.index else None
        rows.append([tex_escape(ds), fmt(int(s.n)), fmt(int(s.n_fake)), fmt(int(s.n_real)), f"{s.n_fake / s.n:.2f}",
                     f"{s.tokens_median:.0f}", f"{s.tokens_mean:.0f}",
                     fmt(int(sp.train)) if sp is not None else "--", fmt(int(sp.val)) if sp is not None else "--",
                     fmt(int(sp.test)) if sp is not None else "--"])
    write(tab_dir, "tab_b01_corpus_statistics", booktabs(
        rows, ["Corpus", "$n$", "FAKE", "REAL", "FAKE share", "Tokens (median)", "Tokens (mean)", "Train", "Val", "Test"],
        "Corpora of the unified Brazilian-Portuguese fake-news benchmark: size, label balance, text length (whitespace tokens of the "
        "URL-free text) and stratified 70/15/15 split. BOATOSBR is part of the unified dataset but has no embeddings and is not used.",
        "tab:b-corpus-statistics", "lrrrrrrrrr", star=True), log)
    # Table 4 (test and val)
    rows = []
    for c in CLF_ORDER:
        r = [CLF_SHORT[c]]
        for e in ENC_ORDER:
            r.append(f"{d['t4_test'].loc[c, e]:.3f} / {d['t4_val'].loc[c, e]:.3f}")
        rows.append(r)
    write(tab_dir, "tab_b02_table4_test_val", booktabs(
        rows, ["Classifier"] + [ENC_SHORT[e] for e in ENC_ORDER],
        "Published in-distribution results: mean per-corpus macro-F1 (test / validation) of each classifier--encoder pair "
        "(10 corpora, best hyper-parameters from 5-fold cross-validation on the training split).",
        "tab:b-table4", "lcccc", star=True), log)
    # model zoo with CIs
    rep = d["report_test"].set_index("source_name"); ci = d["ci_test"]
    cim = ci[ci.metric == "macro_f1"].set_index("source_name"); cia = ci[ci.metric == "accuracy"].set_index("source_name")
    rows = []
    for name in rep.sort_values("macro_f1", ascending=False).index:
        r = rep.loc[name]
        rows.append([tex_escape(short_name(name)), f"{r.accuracy:.3f} [{cia.ci_low[name]:.3f}, {cia.ci_high[name]:.3f}]",
                     f"{r.macro_f1:.3f} [{cim.ci_low[name]:.3f}, {cim.ci_high[name]:.3f}]", f"{r.fake_f1:.3f}", f"{r.real_f1:.3f}",
                     f"{r.mcc:.3f}", fmt(int(r.n_excluded_undefined))])
    write(tab_dir, "tab_b03_model_zoo_test", booktabs(
        rows, ["Model", "Accuracy [95\\% CI]", "Macro-F1 [95\\% CI]", "F1 FAKE", "F1 REAL", "MCC", "Undef."],
        "All 42 models on the test split (6\\,281 items; 5\\,765 for the zero-shot runs, which do not cover COVID19BR): 32 classical "
        "pipelines and 10 zero-shot LLMs. Bootstrap intervals use 2\\,000 resamples; per-class metrics follow the split convention "
        "(FAKE = label 0, see DECISIONS D-26); ``Undef.'' is the number of undefined zero-shot answers excluded from the metrics.",
        "tab:b-model-zoo", "lllrrrr", size="\\scriptsize", star=True), log)
    # Friedman + Holm
    fr = d["friedman_test"].iloc[0]; hm = d["holm_test"]
    rows = []
    for r in d["ranking_test"].sort_values("mean_macro_f1_rank").itertuples():
        h = hm[hm.comparison_model == r.model_id]
        if len(h):
            h = h.iloc[0]; cell = f"{h.raw_p_value:.4f} / {h.holm_adjusted_p_value:.3f}" + (" *" if h.reject_h0_holm else "")
        else:
            cell = "reference"
        e, c = r.model_id.split(" + ")
        rows.append([fmt(int(r.mean_macro_f1_rank)), f"{ENC_SHORT[e]} + {CLF_SHORT[c]}", f"{r.mean_macro_f1:.3f}", f"{r.std_macro_f1:.3f}",
                     f"{r.avg_rank_complete_cases:.1f}", cell])
    write(tab_dir, "tab_b04_friedman_holm_test", booktabs(
        rows, ["\\#", "Pipeline", "Mean macro-F1", "SD", "Avg. rank", "Wilcoxon $p$ raw / Holm"],
        f"Published in-distribution ranking over the 10 corpora (test split). Friedman $\\chi^2 = {fr.statistic:.1f}$, "
        f"$p = {fr.p_value:.1e}$, Kendall $W = {fr.kendall_w:.2f}$; post-hoc Wilcoxon signed-rank tests against the best pipeline "
        f"with Holm correction ($\\alpha = {alpha}$; * = rejected).",
        "tab:b-friedman", "rlrrrl", size="\\scriptsize", star=True), log)
    # zero-shot profile
    rows = [[tex_escape(short_name(r.source_name)), f"{r.macro_f1:.3f}", f"{r.accuracy:.3f}", f"{r.fake_recall:.3f}", f"{r.real_recall:.3f}",
             f"{r.predicted_fake_rate:.2f}", fmt(int(r.n_undefined))] for r in zs_profile.sort_values("macro_f1", ascending=False).itertuples()]
    write(tab_dir, "tab_b05_zero_shot_profile", booktabs(
        rows, ["LLM", "Macro-F1", "Accuracy", "Recall FAKE", "Recall REAL", "Pred. FAKE share", "Undefined"],
        f"Zero-shot LLM baselines on the test split. The true FAKE share is {zs_profile.true_fake_rate.iloc[0]:.2f}; "
        "undefined answers (refusals, non-label outputs) are excluded from the metrics.",
        "tab:b-zero-shot", "lrrrrrr"), log)
    # hard examples
    rows = [[tex_escape(r.dataset_id), tex_escape(r.idx), r.label, fmt(int(r.n_pipelines_wrong)), fmt(int(r.n_llms_wrong))] for r in hard.itertuples()]
    write(tab_dir, "tab_b06_hard_examples", booktabs(
        rows, ["Corpus", "Item", "Label", "Pipelines wrong (of 32)", "LLMs wrong (of 10)"],
        "The two test items per corpus misclassified by the largest number of classical pipelines (highlighted in the UMAP figure).",
        "tab:b-hard-examples", "llllr", size="\\scriptsize"), log)
    # McNemar summary by family
    mc = mc.copy()
    mc["fam_a"] = mc.model_a.str.split("::").str[0]; mc["fam_b"] = mc.model_b.str.split("::").str[0]
    mc["pair"] = np.where(mc.fam_a == mc.fam_b, np.where(mc.fam_a == "classic_models", "classical vs classical", "LLM vs LLM"), "classical vs LLM")
    g = mc.groupby("pair").agg(n_pairs=("reject_holm", "size"), n_significant=("reject_holm", "sum"),
                               median_discordant=("discordant", "median")).reset_index()
    rows = [[r.pair, fmt(int(r.n_pairs)), fmt(int(r.n_significant)), f"{r.n_significant / r.n_pairs:.0%}".replace("%", "\\%"), fmt(int(r.median_discordant))]
            for r in g.itertuples()]
    write(tab_dir, "tab_b07_mcnemar_summary", booktabs(
        rows, ["Pair type", "Pairs", "Significant (Holm)", "Share", "Median discordant items"],
        f"Pairwise exact McNemar tests between all 42 models on the test split, Holm-corrected over the {len(mc)} pairs ($\\alpha = {alpha}$).",
        "tab:b-mcnemar", "lrrrr"), log)


# ----------------------------------------------------------------------------- main
def main() -> None:
    cfg = load_config()
    log = setup_logging(cfg, SCRIPT)
    style(cfg)
    dpi = cfg["figures"]["dpi"]; alpha = cfg["stats"]["alpha"]
    datasets = list(cfg["datasets"])
    fig_dir = rpath(cfg, "out_figures") / "baseline"; tab_dir = rpath(cfg, "out_tables") / "baseline"
    res_dir = rpath(cfg, "out_results") / "baseline"
    for p in (fig_dir, tab_dir, res_dir):
        p.mkdir(parents=True, exist_ok=True)

    with step_timer(cfg, log, SCRIPT, "load published results"):
        d = load_published(cfg)
        order = model_order(d["report_test"])
        log.info("%d models in the published test report (%d zero-shot)", len(order), sum(o.startswith("zero_shot") for o in order))

    with step_timer(cfg, log, SCRIPT, "corpus and split figures"):
        stats = fig_corpus_overview(d, datasets, fig_dir, dpi, log)
        split_ct = fig_split_composition(d, datasets, fig_dir, dpi, log)
        stats.to_csv(res_dir / "corpus_statistics.csv")

    with step_timer(cfg, log, SCRIPT, "classical in-distribution figures"):
        fig_table4_heatmaps(d, fig_dir, dpi, log)
        fig_per_corpus_heatmap(d, datasets, fig_dir, dpi, log)
        fig_friedman_ranks(d, "test", fig_dir, dpi, log)
        fig_friedman_ranks(d, "val", fig_dir, dpi, log)
        fig_spread(d, datasets, fig_dir, dpi, log)
        fig_rank_within_classifier(d, fig_dir, dpi, log)

    with step_timer(cfg, log, SCRIPT, "model-zoo figures (classical + zero-shot)"):
        fig_model_ranking_ci(d, fig_dir, dpi, log)
        fig_class_f1_mcc(d, fig_dir, dpi, log)
        fig_sample_difficulty(d, datasets, fig_dir, dpi, log)
        zoo = zoo_per_corpus(d, datasets); zoo.to_csv(res_dir / "model_zoo_per_corpus.csv", index=False)
        fig_zoo_per_corpus_heatmap(zoo, order, datasets, fig_dir, dpi, log)
        mc = fig_mcnemar_matrix(d, order, alpha, fig_dir, dpi, log); mc.to_csv(res_dir / "pairwise_mcnemar_holm.csv", index=False)
        R = fig_correctness_correlation(d, order, fig_dir, dpi, log); R.to_csv(res_dir / "correctness_correlation_matrix.csv")
        fig_confusion(d, fig_dir, dpi, log)
        zs_profile = fig_zero_shot_profile(d, zoo, fig_dir, dpi, log); zs_profile.to_csv(res_dir / "zero_shot_profile.csv", index=False)

    with step_timer(cfg, log, SCRIPT, "hard examples and UMAP of the published embeddings"):
        hard = hard_examples(d); hard.to_csv(res_dir / "hard_examples.csv", index=False)
        coords = fig_umap_test(cfg, d, datasets, hard, fig_dir, dpi, log); coords.to_csv(res_dir / "umap_test_coordinates.csv", index=False)
        fig_umap_splits(cfg, d, datasets, fig_dir, dpi, log)

    with step_timer(cfg, log, SCRIPT, "baseline tables"):
        tables(d, stats, split_ct, zoo, zs_profile, hard, mc, tab_dir, datasets, alpha, log)
    log.info("Baseline figures written to %s, tables to %s", fig_dir, tab_dir)


if __name__ == "__main__":
    main()
