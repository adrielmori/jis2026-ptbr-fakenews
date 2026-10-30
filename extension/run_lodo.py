"""Study 1 - Leave-one-dataset-out (LODO) generalisation.

For each of the 32 encoder x classifier pipelines and each of the 10 corpora:
train on the other 9 corpora (train+val+test of those corpora) with the published
hyper-parameters, test on the entire held-out corpus. Variant: balanced training
(each training corpus sub-sampled to the size of the smallest one, stratified by class,
seeds 42-46).

Outputs: extension/results/lodo/*.csv, extension/results/predictions/lodo*.parquet
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from gpu_svc import gpu_svc_enabled
from common import (run_tasks, dataset_sizes, binary_metrics, build_pipeline, friedman_wilcoxon_holm, load_best_params, load_config,
                    load_encoder, model_id, model_scores, rpath, set_thread_limits, setup_logging,
                    step_timer, stratified_subsample)

SCRIPT = "run_lodo"
# Expensive classifiers first so that the parallel pool does not end with a long tail.
COST_ORDER = ["svc_rbf", "gradient_boosting", "random_forest", "knn", "logreg", "linearsvc",
              "decision_tree", "gaussian_nb"]


# ----------------------------------------------------------------------------
# Workers
# ----------------------------------------------------------------------------
def _fit_predict(cfg, data, classifier, params, train_pos, test_pos, use_gpu=False):
    t0 = time.time()
    pipe = build_pipeline(classifier, params, cfg["random_state"], rf_n_jobs=cfg["rf_n_jobs"], use_gpu=use_gpu)
    pipe.fit(data.X[train_pos], data.y[train_pos])
    pred, prob_real, kind = model_scores(pipe, data.X[test_pos])
    return pred.astype(int), prob_real.astype(np.float32), kind, time.time() - t0


def lodo_task(cfg: dict, encoder: str, classifier: str, held_out: str, params: dict, cache_dir: Path,
              use_gpu: bool = False):
    """Full LODO: one pipeline, one held-out corpus. ``use_gpu`` = Gram matrix on the GPU (svc_rbf only)."""
    if not use_gpu:
        set_thread_limits(1)
    warnings.filterwarnings("ignore")
    cache = cache_dir / f"full__{encoder}__{classifier}__{held_out}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    data = load_encoder(cfg, encoder)
    test_pos = np.flatnonzero(data.dataset == held_out)
    train_pos = np.flatnonzero(np.isin(data.dataset, cfg["datasets"]) & (data.dataset != held_out))
    pred, prob, kind, secs = _fit_predict(cfg, data, classifier, params, train_pos, test_pos, use_gpu=use_gpu)
    out = pd.DataFrame({
        "encoder": encoder, "classifier": classifier, "held_out_dataset": held_out,
        "idx": data.idx[test_pos], "split": data.split[test_pos], "true_label_id": data.y[test_pos],
        "pred_label_id": pred, "prob_real": prob, "score_kind": kind,
        "n_train": len(train_pos), "n_test": len(test_pos), "fit_seconds": secs,
        "backend": "gpu_gram_matrix" if use_gpu else "cpu",
    })
    out.to_parquet(cache, index=False)
    return out


def balanced_task(cfg: dict, encoder: str, held_out: str, seed: int, train_pos: np.ndarray,
                  params_by_clf: dict, cache_dir: Path):
    """Balanced LODO: one encoder, one held-out corpus, one seed, all 8 classifiers (same sub-sample)."""
    set_thread_limits(1)
    warnings.filterwarnings("ignore")
    cache = cache_dir / f"balanced__{encoder}__{held_out}__{seed}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    data = load_encoder(cfg, encoder)
    test_pos = np.flatnonzero(data.dataset == held_out)
    frames = []
    for classifier in cfg["classifiers"]:
        pred, prob, kind, secs = _fit_predict(cfg, data, classifier, params_by_clf[classifier], train_pos, test_pos)
        frames.append(pd.DataFrame({
            "encoder": encoder, "classifier": classifier, "held_out_dataset": held_out, "seed": seed,
            "idx": data.idx[test_pos], "true_label_id": data.y[test_pos], "pred_label_id": pred,
            "prob_real": prob, "score_kind": kind, "n_train": len(train_pos), "n_test": len(test_pos),
            "fit_seconds": secs,
        }))
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(cache, index=False)
    return out


# ----------------------------------------------------------------------------
# Aggregation helpers
# ----------------------------------------------------------------------------
def metrics_table(preds: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in preds.groupby(keys, sort=True):
        rec = dict(zip(keys, k))
        rec.update(binary_metrics(g.true_label_id, g.pred_label_id))
        rec["n_train"] = int(g.n_train.iloc[0])
        rec["n_test"] = int(len(g))
        rec["fit_seconds"] = float(g.fit_seconds.iloc[0])
        rec["score_kind"] = g.score_kind.iloc[0]
        rows.append(rec)
    return pd.DataFrame(rows)


def table_by_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Table 5 layout: per corpus, mean / std / best / worst macro-F1 over the 32 pipelines and the best pipeline."""
    rows = []
    for ds, g in raw.groupby("held_out_dataset"):
        g = g.sort_values("macro_f1", ascending=False)
        best = g.iloc[0]
        rows.append({
            "held_out_dataset": ds, "n_test": int(best.n_test), "n_pipelines": len(g),
            "mean_macro_f1": g.macro_f1.mean(), "std_macro_f1": g.macro_f1.std(ddof=0),
            "median_macro_f1": g.macro_f1.median(), "best_macro_f1": best.macro_f1,
            "worst_macro_f1": g.macro_f1.min(), "range_macro_f1": best.macro_f1 - g.macro_f1.min(),
            "best_pipeline": model_id(best.encoder, best.classifier),
            "best_accuracy": best.accuracy, "mean_accuracy": g.accuracy.mean(),
        })
    return pd.DataFrame(rows)


def table_by_pipeline(raw: pd.DataFrame, value: str = "macro_f1") -> pd.DataFrame:
    """Table 4 layout: classifier rows x encoder columns, mean over the 10 held-out corpora."""
    return raw.groupby(["classifier", "encoder"])[value].mean().unstack("encoder").sort_index()


def rank_desc(s: pd.Series) -> pd.Series:
    return s.rank(ascending=False, method="average")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--skip-balanced", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    log = setup_logging(cfg, SCRIPT)
    out_dir = rpath(cfg, "out_results") / "lodo"
    pred_dir = rpath(cfg, "out_results") / "predictions"
    cache_dir = out_dir / "_cache"
    for d in (out_dir, pred_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)
    bp = load_best_params(cfg)
    datasets = cfg["datasets"]
    n_jobs = cfg["n_jobs"]

    # ------------------------------------------------------------------ full LODO
    if not args.skip_full:
        clf_order = [c for c in COST_ORDER if c in cfg["classifiers"]]
        tasks = [(e, c, d) for c in clf_order for e in cfg["encoders"] for d in datasets]
        gpu = gpu_svc_enabled(cfg)
        gpu_items = [(cfg, e, c, d, bp[(e, c)], cache_dir, True) for e, c, d in tasks if gpu and c == "svc_rbf"]
        cpu_items = [(cfg, e, c, d, bp[(e, c)], cache_dir, False) for e, c, d in tasks if not (gpu and c == "svc_rbf")]
        log.info("GPU Gram-matrix SVC: %s (%d fits on GPU, %d on CPU)", gpu, len(gpu_items), len(cpu_items))
        sizes = dataset_sizes(cfg)
        n_total = sum(sizes[d] for d in datasets)
        gpu_rows = [n_total - sizes[it[3]] for it in gpu_items]
        with step_timer(cfg, log, SCRIPT, f"full LODO: {len(tasks)} fits (n_jobs={n_jobs}, gpu={gpu})"):
            g_frames, c_frames = run_tasks(lodo_task, gpu_items, cpu_items, n_jobs, gpu_rows=gpu_rows)
        preds = pd.concat(g_frames + c_frames, ignore_index=True)
        preds.to_parquet(pred_dir / "lodo_predictions.parquet", index=False)
        raw = metrics_table(preds, ["encoder", "classifier", "held_out_dataset"])
        raw["model_id"] = [model_id(e, c) for e, c in zip(raw.encoder, raw.classifier)]
        raw.to_csv(out_dir / "lodo_raw.csv", index=False)
        log.info("fit seconds by classifier (mean):\n%s", raw.groupby("classifier").fit_seconds.mean().round(1).to_string())
    else:
        raw = pd.read_csv(out_dir / "lodo_raw.csv")

    with step_timer(cfg, log, SCRIPT, "full LODO tables"):
        t5 = table_by_dataset(raw)
        t5.to_csv(out_dir / "lodo_table_by_dataset.csv", index=False)
        t4 = table_by_pipeline(raw)
        t4.to_csv(out_dir / "lodo_table_by_pipeline.csv")
        table_by_pipeline(raw, "accuracy").to_csv(out_dir / "lodo_table_by_pipeline_accuracy.csv")
        log.info("LODO Table-4 layout (mean macro-F1 over 10 holdouts):\n%s", t4.round(3).to_string())
        log.info("LODO Table-5 layout:\n%s", t5.round(3).to_string(index=False))

        # ---- in-distribution (published) vs LODO
        pub = pd.read_csv(rpath(cfg, "published_table5_source")).melt(id_vars="dataset_id", var_name="model_id",
                                                                        value_name="indist_macro_f1")
        long = raw[["model_id", "encoder", "classifier", "held_out_dataset", "macro_f1"]].rename(
            columns={"held_out_dataset": "dataset_id", "macro_f1": "lodo_macro_f1"})
        long = long.merge(pub, on=["dataset_id", "model_id"], how="left")
        long["delta_lodo_minus_indist"] = long.lodo_macro_f1 - long.indist_macro_f1
        long["rank_indist_within_corpus"] = long.groupby("dataset_id").indist_macro_f1.transform(rank_desc)
        long["rank_lodo_within_corpus"] = long.groupby("dataset_id").lodo_macro_f1.transform(rank_desc)
        long["scope"] = "pipeline_x_corpus"

        by_pipe = long.groupby(["model_id", "encoder", "classifier"], as_index=False)[["indist_macro_f1", "lodo_macro_f1"]].mean()
        by_pipe["delta_lodo_minus_indist"] = by_pipe.lodo_macro_f1 - by_pipe.indist_macro_f1
        by_pipe["rank_indist"] = rank_desc(by_pipe.indist_macro_f1)
        by_pipe["rank_lodo"] = rank_desc(by_pipe.lodo_macro_f1)
        by_pipe["scope"] = "pipeline_mean_over_corpora"
        rho_pipe, p_pipe = spearmanr(by_pipe.indist_macro_f1, by_pipe.lodo_macro_f1)

        by_enc = long.groupby("encoder", as_index=False)[["indist_macro_f1", "lodo_macro_f1"]].mean()
        by_enc["delta_lodo_minus_indist"] = by_enc.lodo_macro_f1 - by_enc.indist_macro_f1
        by_enc["rank_indist"] = rank_desc(by_enc.indist_macro_f1)
        by_enc["rank_lodo"] = rank_desc(by_enc.lodo_macro_f1)
        by_enc["scope"] = "encoder_mean"
        rho_enc, p_enc = spearmanr(by_enc.indist_macro_f1, by_enc.lodo_macro_f1)

        by_corpus = long.groupby("dataset_id", as_index=False)[["indist_macro_f1", "lodo_macro_f1"]].mean()
        by_corpus["delta_lodo_minus_indist"] = by_corpus.lodo_macro_f1 - by_corpus.indist_macro_f1
        by_corpus["rank_indist"] = rank_desc(by_corpus.indist_macro_f1)
        by_corpus["rank_lodo"] = rank_desc(by_corpus.lodo_macro_f1)
        by_corpus["scope"] = "corpus_mean_over_pipelines"
        rho_corpus, p_corpus = spearmanr(by_corpus.indist_macro_f1, by_corpus.lodo_macro_f1)

        # per-corpus rank correlation of the 32 pipelines
        per_corpus_rho = []
        for ds, g in long.groupby("dataset_id"):
            r, p = spearmanr(g.indist_macro_f1, g.lodo_macro_f1)
            per_corpus_rho.append({"level": f"pipelines_within_{ds}", "n": len(g), "spearman_rho": r, "p_value": p})
        rho_df = pd.DataFrame([
            {"level": "32_pipelines_mean_over_corpora", "n": len(by_pipe), "spearman_rho": rho_pipe, "p_value": p_pipe},
            {"level": "4_encoders_mean", "n": len(by_enc), "spearman_rho": rho_enc, "p_value": p_enc},
            {"level": "10_corpora_mean_over_pipelines", "n": len(by_corpus), "spearman_rho": rho_corpus, "p_value": p_corpus},
        ] + per_corpus_rho)
        rho_df.to_csv(out_dir / "lodo_vs_indist_spearman.csv", index=False)
        vs = pd.concat([by_pipe, by_enc, by_corpus, long], ignore_index=True)
        cols = ["scope", "model_id", "encoder", "classifier", "dataset_id", "indist_macro_f1", "lodo_macro_f1",
                "delta_lodo_minus_indist", "rank_indist", "rank_lodo", "rank_indist_within_corpus", "rank_lodo_within_corpus"]
        vs = vs.reindex(columns=cols)
        vs.to_csv(out_dir / "lodo_vs_indist.csv", index=False)
        log.info("Spearman correlations:\n%s", rho_df.round(4).to_string(index=False))
        log.info("Encoder means (in-dist vs LODO):\n%s", by_enc.round(4).to_string(index=False))

    # ------------------------------------------------------------------ balanced LODO
    if not args.skip_balanced:
        meta = load_encoder(cfg, cfg["encoders"][0]).meta  # same order for every encoder (verified)
        ds_arr = meta.dataset_id.to_numpy()
        y_arr = meta.label_id.to_numpy()
        plan_rows, tasks = [], []
        for held_out in datasets:
            train_ds = [d for d in datasets if d != held_out]
            sizes = {d: int((ds_arr == d).sum()) for d in train_ds}
            n_min = min(sizes.values())
            for seed in cfg["lodo"]["balanced_seeds"]:
                rng = np.random.default_rng(seed)
                pos = []
                for d in train_ds:  # fixed order -> deterministic
                    pd_ = np.flatnonzero(ds_arr == d)
                    sel = stratified_subsample(pd_, y_arr[pd_], n_min, rng, cfg["lodo"]["min_per_class"])
                    pos.append(sel)
                    plan_rows.append({"held_out_dataset": held_out, "seed": seed, "train_corpus": d,
                                      "n_available": sizes[d], "n_sampled": len(sel),
                                      "n_sampled_fake": int((y_arr[sel] == 0).sum()),
                                      "n_sampled_real": int((y_arr[sel] == 1).sum())})
                train_pos = np.sort(np.concatenate(pos))
                for e in cfg["encoders"]:
                    tasks.append((e, held_out, seed, train_pos))
        pd.DataFrame(plan_rows).to_csv(out_dir / "lodo_balanced_sampling_plan.csv", index=False)
        with step_timer(cfg, log, SCRIPT, f"balanced LODO: {len(tasks)} tasks x 8 classifiers (n_jobs={n_jobs})"):
            frames = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(balanced_task)(cfg, e, d, s, tp, {c: bp[(e, c)] for c in cfg["classifiers"]}, cache_dir)
                for e, d, s, tp in tasks)
        bpreds = pd.concat(frames, ignore_index=True)
        bpreds.to_parquet(pred_dir / "lodo_balanced_predictions.parquet", index=False)
        braw = metrics_table(bpreds, ["encoder", "classifier", "held_out_dataset", "seed"])
        braw["model_id"] = [model_id(e, c) for e, c in zip(braw.encoder, braw.classifier)]
        braw.to_csv(out_dir / "lodo_balanced_raw.csv", index=False)
    else:
        braw = pd.read_csv(out_dir / "lodo_balanced_raw.csv")

    with step_timer(cfg, log, SCRIPT, "balanced LODO summary"):
        agg = braw.groupby(["model_id", "encoder", "classifier", "held_out_dataset"]).agg(
            macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
            recall_fake_mean=("recall_fake", "mean"), recall_real_mean=("recall_real", "mean"),
            n_train=("n_train", "first"), n_test=("n_test", "first"), n_seeds=("seed", "nunique")).reset_index()
        full = raw[["model_id", "held_out_dataset", "macro_f1", "accuracy", "n_train"]].rename(
            columns={"macro_f1": "full_lodo_macro_f1", "accuracy": "full_lodo_accuracy", "n_train": "full_n_train"})
        agg = agg.merge(full, on=["model_id", "held_out_dataset"], how="left")
        agg["delta_balanced_minus_full"] = agg.macro_f1_mean - agg.full_lodo_macro_f1
        agg.to_csv(out_dir / "lodo_balanced_summary.csv", index=False)
        # corpus-level and pipeline-level digests
        corp = agg.groupby("held_out_dataset").agg(
            balanced_mean_macro_f1=("macro_f1_mean", "mean"), full_mean_macro_f1=("full_lodo_macro_f1", "mean"),
            balanced_best_macro_f1=("macro_f1_mean", "max"), full_best_macro_f1=("full_lodo_macro_f1", "max"),
            n_train_balanced=("n_train", "first"), n_train_full=("full_n_train", "first")).reset_index()
        corp["delta_balanced_minus_full_mean"] = corp.balanced_mean_macro_f1 - corp.full_mean_macro_f1
        corp["delta_balanced_minus_full_best"] = corp.balanced_best_macro_f1 - corp.full_best_macro_f1
        pub = pd.read_csv(rpath(cfg, "published_table5_source")).set_index("dataset_id").mean(axis=1).rename("indist_mean_macro_f1")
        corp = corp.merge(pub, left_on="held_out_dataset", right_index=True, how="left")
        corp.to_csv(out_dir / "lodo_balanced_by_dataset.csv", index=False)
        pipe = agg.groupby(["classifier", "encoder"]).macro_f1_mean.mean().unstack("encoder")
        pipe.to_csv(out_dir / "lodo_balanced_table_by_pipeline.csv")
        log.info("Balanced LODO by corpus:\n%s", corp.round(3).to_string(index=False))

    # ------------------------------------------------------------------ statistics
    with step_timer(cfg, log, SCRIPT, "Friedman / Wilcoxon / Holm"):
        alpha = cfg["stats"]["alpha"]
        piv_full = raw.pivot(index="held_out_dataset", columns="model_id", values="macro_f1")
        piv_bal = agg.pivot(index="held_out_dataset", columns="model_id", values="macro_f1_mean")
        st_full = friedman_wilcoxon_holm(piv_full, alpha, "lodo_full")
        st_bal = friedman_wilcoxon_holm(piv_bal, alpha, "lodo_balanced")
        pd.concat([st_full["friedman"], st_bal["friedman"]]).to_csv(out_dir / "lodo_stats_friedman.csv", index=False)
        pd.concat([st_full["ranking"], st_bal["ranking"]]).to_csv(out_dir / "lodo_stats_ranking.csv", index=False)
        pd.concat([st_full["posthoc"], st_bal["posthoc"]]).to_csv(out_dir / "lodo_stats.csv", index=False)
        piv_full.to_csv(out_dir / "lodo_macro_f1_by_dataset_model.csv")
        piv_bal.to_csv(out_dir / "lodo_balanced_macro_f1_by_dataset_model.csv")
        for st in (st_full, st_bal):
            log.info("\n%s", st["friedman"].to_string(index=False))
            log.info("best: %s ; rejected after Holm: %d / %d", st["ranking"].loc[0, "model_id"],
                     int(st["posthoc"].reject_h0_holm.sum()), len(st["posthoc"]))
    log.info("Study 1 finished.")


if __name__ == "__main__":
    main()
