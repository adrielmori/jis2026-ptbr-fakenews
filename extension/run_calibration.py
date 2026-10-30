"""Study 2 - Calibration and selective classification (triage for human review).

Pipelines: {svc_rbf, logreg} x {albertina_ptbr_900m, bertimbau_large}.
Conditions: raw, sigmoid (Platt), isotonic - calibrators always fitted on held-out
calibration data, never on the evaluation data.
Scenarios:
  (a) indist : fit on the published train split, calibrate on the val split, evaluate on the test split.
  (b) lodo   : for each held-out corpus, a stratified 15% slice of the 9 training corpora is reserved
               for calibration, the model is fitted on the remaining 85%, evaluation on the held-out corpus.

Outputs: extension/results/calibration/*.csv, extension/results/predictions/calibration_predictions.parquet
"""
from __future__ import annotations

import time
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from gpu_svc import gpu_svc_enabled
from common import (run_tasks, dataset_sizes, rpath, binary_metrics, build_pipeline, load_best_params, load_config, load_encoder, model_id,
                    model_scores, rpath, set_thread_limits, setup_logging, step_timer)

SCRIPT = "run_calibration"
ALL = "ALL"


# ----------------------------------------------------------------------------
# Probability extraction
# ----------------------------------------------------------------------------
def condition_probabilities(pipe, X_cal, y_cal, X_eval) -> Dict[str, np.ndarray]:
    """P(REAL) on X_eval under the three conditions."""
    out = {}
    _, raw, _ = model_scores(pipe, X_eval)
    out["raw"] = raw
    for cond, method in (("sigmoid", "sigmoid"), ("isotonic", "isotonic")):
        cal = CalibratedClassifierCV(estimator=pipe, method=method, cv="prefit")
        cal.fit(X_cal, y_cal)
        proba = cal.predict_proba(X_eval)
        out[cond] = proba[:, list(cal.classes_).index(1)]
    return out


def indist_task(cfg, encoder, classifier, params, use_gpu=False):
    if not use_gpu:
        set_thread_limits(1)
    warnings.filterwarnings("ignore")
    data = load_encoder(cfg, encoder)
    tr, va, te = data.mask_split("train"), data.mask_split("val"), data.mask_split("test")
    t0 = time.time()
    pipe = build_pipeline(classifier, params, cfg["random_state"], rf_n_jobs=cfg["rf_n_jobs"], use_gpu=use_gpu)
    pipe.fit(data.X[tr], data.y[tr])
    probs = condition_probabilities(pipe, data.X[va], data.y[va], data.X[te])
    frames = []
    for cond, p in probs.items():
        frames.append(pd.DataFrame({
            "pipeline": model_id(encoder, classifier), "encoder": encoder, "classifier": classifier,
            "scenario": "indist", "condition": cond, "corpus": data.dataset[te], "held_out_dataset": "none",
            "idx": data.idx[te], "true_label_id": data.y[te], "prob_real": p.astype(np.float64),
            "n_train": int(tr.sum()), "n_calib": int(va.sum()), "fit_seconds": time.time() - t0,
        }))
    return pd.concat(frames, ignore_index=True)


def lodo_task(cfg, encoder, classifier, held_out, params, use_gpu=False):
    if not use_gpu:
        set_thread_limits(1)
    warnings.filterwarnings("ignore")
    data = load_encoder(cfg, encoder)
    pool = np.flatnonzero(np.isin(data.dataset, cfg["datasets"]) & (data.dataset != held_out))
    test_pos = np.flatnonzero(data.dataset == held_out)
    strat = pd.Series(data.dataset[pool]).astype(str) + "__" + pd.Series(data.y[pool]).astype(str)
    train_pos, cal_pos = train_test_split(pool, test_size=cfg["calibration"]["lodo_calibration_fraction"],
                                          random_state=cfg["random_state"], stratify=strat.to_numpy())
    train_pos, cal_pos = np.sort(train_pos), np.sort(cal_pos)
    t0 = time.time()
    pipe = build_pipeline(classifier, params, cfg["random_state"], rf_n_jobs=cfg["rf_n_jobs"], use_gpu=use_gpu)
    pipe.fit(data.X[train_pos], data.y[train_pos])
    probs = condition_probabilities(pipe, data.X[cal_pos], data.y[cal_pos], data.X[test_pos])
    frames = []
    for cond, p in probs.items():
        frames.append(pd.DataFrame({
            "pipeline": model_id(encoder, classifier), "encoder": encoder, "classifier": classifier,
            "scenario": "lodo", "condition": cond, "corpus": held_out, "held_out_dataset": held_out,
            "idx": data.idx[test_pos], "true_label_id": data.y[test_pos], "prob_real": p.astype(np.float64),
            "n_train": len(train_pos), "n_calib": len(cal_pos), "fit_seconds": time.time() - t0,
        }))
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------
# Calibration metrics
# ----------------------------------------------------------------------------
def calibration_metrics(y: np.ndarray, p_real: np.ndarray, n_bins: int) -> dict:
    y = np.asarray(y).astype(int)
    p_real = np.clip(np.asarray(p_real, dtype=float), 0.0, 1.0)
    pred = (p_real >= 0.5).astype(int)
    conf = np.maximum(p_real, 1.0 - p_real)
    correct = (pred == y).astype(float)
    n = len(y)
    # equal-width bins on confidence in [0.5, 1]
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    ece = mce = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.sum() / n * gap
            mce = max(mce, gap)
    # equal-mass (adaptive) bins
    order = np.argsort(conf, kind="mergesort")
    chunks = np.array_split(order, n_bins)
    ece_a = 0.0
    for ch in chunks:
        if len(ch):
            ece_a += len(ch) / n * abs(correct[ch].mean() - conf[ch].mean())
    eps = 1e-15
    pc = np.clip(p_real, eps, 1 - eps)
    logloss = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    out = {"n": n, "ece": float(ece), "ece_adaptive": float(ece_a), "mce": float(mce),
           "brier": float(np.mean((p_real - y) ** 2)), "log_loss": logloss,
           "mean_confidence": float(conf.mean()), "accuracy": float(correct.mean())}
    out.update({k: v for k, v in binary_metrics(y, pred).items() if k not in ("accuracy",)})
    return out


def reliability_bins(y, p_real, n_bins) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    p_real = np.clip(np.asarray(p_real, dtype=float), 0, 1)
    pred = (p_real >= 0.5).astype(int)
    conf = np.maximum(p_real, 1 - p_real)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        rows.append({"bin": k, "bin_lower": edges[k], "bin_upper": edges[k + 1], "count": int(m.sum()),
                     "mean_confidence": float(conf[m].mean()) if m.any() else np.nan,
                     "accuracy": float(correct[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Selective classification (vectorised over thresholds; supports bootstrap weights)
# ----------------------------------------------------------------------------
class SelectiveEvaluator:
    def __init__(self, y, p_real, thresholds):
        self.y = np.asarray(y).astype(int)
        self.p = np.asarray(p_real, dtype=float)
        self.pred = (self.p >= 0.5).astype(int)
        self.conf = np.maximum(self.p, 1 - self.p)
        self.t = np.asarray(thresholds)
        self.nt = len(self.t)
        # bin j <-> conf in [t_j, t_{j+1}); items with conf >= t_j have bin >= j
        self.bin = np.searchsorted(self.t, self.conf, side="right") - 1  # -1 -> below the first threshold
        self.cell = self.y * 2 + self.pred  # 0: y0p0, 1: y0p1, 2: y1p0, 3: y1p1
        self.n = len(self.y)

    def curves(self, weights=None) -> pd.DataFrame:
        """Metrics at every threshold. weights = per-item multiplicities (bootstrap) or None."""
        w = np.ones(self.n) if weights is None else weights
        keep = self.bin >= 0
        counts = np.zeros((4, self.nt))
        for c in range(4):
            m = keep & (self.cell == c)
            h = np.bincount(self.bin[m], weights=w[m], minlength=self.nt)
            counts[c] = h[::-1].cumsum()[::-1]  # covered at threshold j = bins >= j
        total = w.sum()
        tn, fp, fn, tp = counts  # for class REAL (1): tp = y1p1, fp = y0p1, fn = y1p0, tn = y0p0
        cov_n = counts.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = (tp + tn) / cov_n
            # F1 for class 1 (REAL) and class 0 (FAKE); labels absent in both truth and prediction are
            # excluded from the macro average (sklearn convention over the union of present labels).
            f1_real = 2 * tp / (2 * tp + fp + fn)
            f1_fake = 2 * tn / (2 * tn + fn + fp)
            present_real = (tp + fn + fp) > 0
            present_fake = (tn + fn + fp) > 0
            f1_real = np.where(present_real, np.nan_to_num(f1_real), np.nan)
            f1_fake = np.where(present_fake, np.nan_to_num(f1_fake), np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns (nothing covered)
                macro = np.nanmean(np.vstack([f1_real, f1_fake]), axis=0)
        macro[cov_n == 0] = np.nan
        acc[cov_n == 0] = np.nan
        return pd.DataFrame({"threshold": self.t, "coverage": cov_n / total, "n_covered": cov_n,
                             "macro_f1": macro, "accuracy": acc, "selective_risk": 1 - acc,
                             "fraction_human": 1 - cov_n / total})

    def aurc(self) -> dict:
        """Area under the risk-coverage curve (Geifman et al. 2019) and its excess over the optimal curve."""
        order = np.argsort(-self.conf, kind="mergesort")
        err = (self.pred != self.y).astype(float)[order]
        k = np.arange(1, self.n + 1)
        risk = np.cumsum(err) / k
        aurc = float(risk.mean())
        full_risk = float(err.mean())
        err_opt = np.sort(err)  # optimal ordering: all correct first
        aurc_opt = float((np.cumsum(err_opt) / k).mean())
        return {"aurc": aurc, "e_aurc": aurc - aurc_opt, "aurc_optimal": aurc_opt, "full_coverage_risk": full_risk}

    def triage(self, targets, n_boot, ci, rng, min_covered: int = 1) -> list:
        """Minimum threshold reaching each macro-F1 target. A threshold only counts when at least
        ``min_covered`` items are covered (a macro-F1 computed on a handful of items is noise)."""
        base = self.curves()
        rows = []
        boot_curves = []
        for _ in range(n_boot):
            idx = rng.integers(0, self.n, self.n)
            w = np.bincount(idx, minlength=self.n).astype(float)
            bc = self.curves(w)
            f1b = np.where(bc.n_covered.to_numpy() >= min_covered, bc.macro_f1.to_numpy(), np.nan)
            boot_curves.append(np.column_stack([f1b, bc.coverage.to_numpy()]))
        boot = np.stack(boot_curves)  # (B, nt, 2)
        lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
        for target in targets:
            ok = (base.macro_f1.to_numpy() >= target - 1e-12) & (base.n_covered.to_numpy() >= min_covered)
            if ok.any():
                j = int(np.argmax(ok))
                thr, cov, f1 = float(self.t[j]), float(base.coverage[j]), float(base.macro_f1[j])
                status = "reached"
            else:
                thr, cov, f1, status = np.nan, 0.0, np.nan, "not_reached"
            # bootstrap distribution of (min threshold, coverage at that threshold)
            b_ok = boot[:, :, 0] >= target - 1e-12  # (B, nt)
            reached = b_ok.any(axis=1)
            first = np.where(reached, np.argmax(b_ok, axis=1), -1)
            b_cov = np.where(reached, boot[np.arange(n_boot), np.clip(first, 0, None), 1], 0.0)
            b_thr = np.where(reached, self.t[np.clip(first, 0, None)], np.nan)
            rows.append({
                "target_macro_f1": target, "status": status, "threshold": thr, "coverage": cov,
                "macro_f1_at_threshold": f1, "fraction_human": 1 - cov if status == "reached" else 1.0,
                "coverage_ci_low": float(np.percentile(b_cov, lo)), "coverage_ci_high": float(np.percentile(b_cov, hi)),
                "threshold_ci_low": float(np.nanpercentile(b_thr, lo)) if reached.any() else np.nan,
                "threshold_ci_high": float(np.nanpercentile(b_thr, hi)) if reached.any() else np.nan,
                "bootstrap_fraction_reached": float(reached.mean()), "n_items": self.n, "n_bootstrap": n_boot,
                "min_covered": min_covered,
            })
        return rows


# ----------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    log = setup_logging(cfg, SCRIPT)
    ccfg = cfg["calibration"]
    out_dir = rpath(cfg, "out_results") / "calibration"
    pred_dir = rpath(cfg, "out_results") / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    bp = load_best_params(cfg)
    pipelines = [tuple(p) for p in ccfg["pipelines"]]
    thr = ccfg["thresholds"]
    thresholds = np.round(np.arange(thr["start"], thr["stop"] + 1e-9, thr["step"]), 2)

    gpu = gpu_svc_enabled(cfg)
    log.info("GPU Gram-matrix SVC: %s", gpu)

    def is_gpu(c):
        return gpu and c == "svc_rbf"

    with step_timer(cfg, log, SCRIPT, f"scenario (a) in-distribution: {len(pipelines)} fits (gpu={gpu})"):
        n_tr = int(pd.read_csv(rpath(cfg, "splits_dir") / "train_split.csv", usecols=["dataset_id"]).shape[0])
        g_items = [(cfg, e, c, bp[(e, c)], True) for e, c in pipelines if is_gpu(c)]
        ga, ca = run_tasks(indist_task, g_items,
                           [(cfg, e, c, bp[(e, c)], False) for e, c in pipelines if not is_gpu(c)],
                           min(cfg["n_jobs"], len(pipelines)), gpu_rows=[n_tr] * len(g_items))
    tasks = [(e, c, d) for e, c in pipelines for d in cfg["datasets"]]
    with step_timer(cfg, log, SCRIPT, f"scenario (b) LODO: {len(tasks)} fits (n_jobs={cfg['n_jobs']}, gpu={gpu})"):
        sizes = dataset_sizes(cfg)
        n_total = sum(sizes[d] for d in cfg["datasets"])
        frac = 1.0 - cfg["calibration"]["lodo_calibration_fraction"]
        g_items = [(cfg, e, c, d, bp[(e, c)], True) for e, c, d in tasks if is_gpu(c)]
        gb, cb = run_tasks(lodo_task, g_items,
                           [(cfg, e, c, d, bp[(e, c)], False) for e, c, d in tasks if not is_gpu(c)],
                           cfg["n_jobs"], gpu_rows=[int(frac * (n_total - sizes[it[3]])) + 1 for it in g_items])
    preds = pd.concat(ga + ca + gb + cb, ignore_index=True)
    preds["pred_label_id"] = (preds.prob_real >= 0.5).astype(int)
    preds["confidence"] = np.maximum(preds.prob_real, 1 - preds.prob_real)
    preds.to_parquet(pred_dir / "calibration_predictions.parquet", index=False)
    log.info("fit seconds:\n%s", preds.groupby(["pipeline", "scenario"]).fit_seconds.mean().round(1).to_string())

    with step_timer(cfg, log, SCRIPT, "calibration metrics + reliability bins"):
        rows, rel = [], []
        groups = list(preds.groupby(["pipeline", "scenario", "condition", "corpus"]))
        groups += [((p, s, c, ALL), g) for (p, s, c), g in preds.groupby(["pipeline", "scenario", "condition"])]
        for (p, s, c, corpus), g in groups:
            rec = {"pipeline": p, "scenario": s, "condition": c, "corpus": corpus}
            rec.update(calibration_metrics(g.true_label_id.to_numpy(), g.prob_real.to_numpy(), ccfg["n_bins"]))
            rows.append(rec)
            rb = reliability_bins(g.true_label_id.to_numpy(), g.prob_real.to_numpy(), ccfg["n_bins"])
            rb.insert(0, "corpus", corpus); rb.insert(0, "condition", c); rb.insert(0, "scenario", s); rb.insert(0, "pipeline", p)
            rel.append(rb)
        cm = pd.DataFrame(rows)
        cm.to_csv(out_dir / "calibration_metrics.csv", index=False)
        pd.concat(rel, ignore_index=True).to_csv(out_dir / "reliability_bins.csv", index=False)
        log.info("Aggregate calibration metrics:\n%s",
                 cm[cm.corpus == ALL][["pipeline", "scenario", "condition", "ece", "ece_adaptive", "mce", "brier",
                                       "log_loss", "accuracy", "macro_f1"]].round(4).to_string(index=False))

    with step_timer(cfg, log, SCRIPT, "selective classification curves, AURC and triage thresholds (bootstrap)"):
        curves, summ, triage = [], [], []
        rng = np.random.default_rng(cfg["random_state"])
        for (p, s, c, corpus), g in groups:
            ev = SelectiveEvaluator(g.true_label_id.to_numpy(), g.prob_real.to_numpy(), thresholds)
            cur = ev.curves()
            for pos, (k, v) in enumerate((("pipeline", p), ("scenario", s), ("condition", c), ("corpus", corpus))):
                cur.insert(pos, k, v)
            curves.append(cur)
            rec = {"pipeline": p, "scenario": s, "condition": c, "corpus": corpus, "n": ev.n}
            rec.update(ev.aurc())
            summ.append(rec)
            for r in ev.triage(ccfg["f1_targets"], ccfg["bootstrap_resamples"], ccfg["bootstrap_ci"],
                               np.random.default_rng(cfg["random_state"]), min_covered=ccfg.get("min_covered", 1)):
                r = {"pipeline": p, "scenario": s, "condition": c, "corpus": corpus, **r}
                triage.append(r)
        pd.concat(curves, ignore_index=True).to_csv(out_dir / "selective_curves.csv", index=False)
        pd.DataFrame(summ).to_csv(out_dir / "selective_summary.csv", index=False)
        tri = pd.DataFrame(triage)
        tri.to_csv(out_dir / "triage_summary.csv", index=False)
        log.info("Triage (aggregate):\n%s",
                 tri[tri.corpus == ALL][["pipeline", "scenario", "condition", "target_macro_f1", "status", "threshold",
                                         "coverage", "coverage_ci_low", "coverage_ci_high"]].round(3).to_string(index=False))
    log.info("Study 2 finished.")


if __name__ == "__main__":
    main()
