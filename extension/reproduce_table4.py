"""Phase 0 - sanity check: reproduce Table 4 (mean per-dataset macro-F1, encoder x classifier)
by refitting the 32 selected pipelines on train+val (as in the original notebook) and
scoring the published test split. Tolerance: |diff| <= config.sanity.table4_tolerance.

Outputs (extension/results/sanity/):
  table4_reproduction.csv       long format: encoder, classifier, published, reproduced, diff, within_tolerance
  table4_reproduced_matrix.csv  same layout as the published CSV
  table4_prediction_agreement.csv  per-pipeline agreement with the published per-item predictions
  ../predictions/indist_refit_test.parquet  per-item predictions and scores of the refit models
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from gpu_svc import gpu_svc_enabled
from common import (run_tasks, binary_metrics, build_pipeline, load_best_params, load_config, load_encoder, macro_f1,
                    model_id, model_scores, rpath, set_thread_limits, setup_logging, step_timer)

SCRIPT = "reproduce_table4"


def fit_one(cfg: dict, encoder: str, classifier: str, params: dict, use_gpu: bool = False) -> pd.DataFrame:
    if not use_gpu:
        set_thread_limits(1)
    warnings.filterwarnings("ignore")
    data = load_encoder(cfg, encoder)
    tr = data.mask_split("train", "val")
    te = data.mask_split("test")
    t0 = time.time()
    pipe = build_pipeline(classifier, params, cfg["random_state"], rf_n_jobs=cfg["rf_n_jobs"], use_gpu=use_gpu)
    pipe.fit(data.X[tr], data.y[tr])
    pred, prob_real, kind = model_scores(pipe, data.X[te])
    out = pd.DataFrame({
        "encoder": encoder, "classifier": classifier, "idx": data.idx[te], "dataset_id": data.dataset[te],
        "true_label_id": data.y[te], "pred_label_id": pred.astype(int), "prob_real": prob_real.astype(float),
        "score_kind": kind, "fit_seconds": time.time() - t0, "backend": "gpu_gram_matrix" if use_gpu else "cpu",
    })
    return out


def main() -> None:
    cfg = load_config()
    log = setup_logging(cfg, SCRIPT)
    out_dir = rpath(cfg, "out_results") / "sanity"
    pred_dir = rpath(cfg, "out_results") / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    bp = load_best_params(cfg)
    jobs = [(e, c) for e in cfg["encoders"] for c in cfg["classifiers"]]

    gpu = gpu_svc_enabled(cfg)
    log.info("GPU Gram-matrix SVC: %s", gpu)
    with step_timer(cfg, log, SCRIPT, f"refit {len(jobs)} pipelines on train+val (n_jobs={cfg['n_jobs']}, gpu={gpu})"):
        n_trval = sum(pd.read_csv(rpath(cfg, "splits_dir") / f"{s}_split.csv", usecols=["dataset_id"]).shape[0]
                      for s in ("train", "val"))
        g_items = [(cfg, e, c, bp[(e, c)], True) for e, c in jobs if gpu and c == "svc_rbf"]
        g, c_ = run_tasks(fit_one, g_items,
                          [(cfg, e, c, bp[(e, c)], False) for e, c in jobs if not (gpu and c == "svc_rbf")],
                          cfg["n_jobs"], gpu_rows=[n_trval] * len(g_items))
    preds = pd.concat(g + c_, ignore_index=True)
    preds.to_parquet(pred_dir / "indist_refit_test.parquet", index=False)
    log.info("fit time per pipeline (s):\n%s",
             preds.groupby(["encoder", "classifier"]).fit_seconds.first().round(1).to_string())

    with step_timer(cfg, log, SCRIPT, "compare with published Table 4"):
        per_ds = (preds.groupby(["encoder", "classifier", "dataset_id"])
                  .apply(lambda g: macro_f1(g.true_label_id, g.pred_label_id)).rename("macro_f1").reset_index())
        per_ds = per_ds[per_ds.dataset_id.isin(cfg["datasets"])]
        repro = per_ds.groupby(["encoder", "classifier"]).macro_f1.mean().rename("reproduced").reset_index()
        pub = pd.read_csv(rpath(cfg, "published_table4")).melt(id_vars="classifier", var_name="encoder",
                                                                 value_name="published")
        cmp_ = pub.merge(repro, on=["encoder", "classifier"], how="outer")
        cmp_["diff"] = cmp_["reproduced"] - cmp_["published"]
        tol = cfg["sanity"]["table4_tolerance"]
        cmp_["within_tolerance"] = cmp_["diff"].abs() <= tol
        cmp_ = cmp_.sort_values(["classifier", "encoder"]).reset_index(drop=True)
        cmp_.to_csv(out_dir / "table4_reproduction.csv", index=False)
        repro.pivot(index="classifier", columns="encoder", values="reproduced").to_csv(out_dir / "table4_reproduced_matrix.csv")
        log.info("\n%s", cmp_.to_string(index=False))
        n_bad = int((~cmp_["within_tolerance"]).sum())
        log.info("max |diff| = %.5f ; pairs outside tolerance (%.3f): %d", cmp_["diff"].abs().max(), tol, n_bad)

        # Global test macro-F1 vs the published results table (test_f1_macro column)
        pub_res = pd.read_csv(rpath(cfg, "published_results"))[["embedding_model", "classifier", "test_f1_macro", "test_acc"]]
        glob = (preds.groupby(["encoder", "classifier"])
                .apply(lambda g: pd.Series(binary_metrics(g.true_label_id, g.pred_label_id))[["macro_f1", "accuracy"]])
                .reset_index().rename(columns={"macro_f1": "repro_test_f1_macro", "accuracy": "repro_test_acc"}))
        glob = glob.merge(pub_res, left_on=["encoder", "classifier"], right_on=["embedding_model", "classifier"])
        glob["diff_f1"] = glob["repro_test_f1_macro"] - glob["test_f1_macro"]

        # Per-item agreement with the published predictions
        pub_pred = pd.read_csv(rpath(cfg, "published_test_predictions"))[["embedding_model", "classifier", "idx", "pred_label_id"]]
        m = preds.merge(pub_pred, left_on=["encoder", "classifier", "idx"], right_on=["embedding_model", "classifier", "idx"],
                        suffixes=("", "_published"))
        agree = (m.assign(agree=lambda d: d.pred_label_id == d.pred_label_id_published)
                 .groupby(["encoder", "classifier"]).agree.mean().rename("prediction_agreement").reset_index())
        agree = agree.merge(glob[["encoder", "classifier", "repro_test_f1_macro", "test_f1_macro", "diff_f1"]],
                            on=["encoder", "classifier"])
        agree.to_csv(out_dir / "table4_prediction_agreement.csv", index=False)
        log.info("\n%s", agree.to_string(index=False))

    status = "PASS" if n_bad == 0 else "FAIL"
    summary = [f"# Table 4 reproduction: {status}", "",
               f"Tolerance |diff| <= {tol}; pairs outside tolerance: {n_bad} / {len(cmp_)}",
               f"max |diff| = {cmp_['diff'].abs().max():.5f}; mean |diff| = {cmp_['diff'].abs().mean():.5f}",
               f"mean per-item prediction agreement with published predictions: {agree.prediction_agreement.mean():.5f} "
               f"(min {agree.prediction_agreement.min():.5f})", "", cmp_.to_markdown(index=False)]
    (out_dir / "table4_reproduction.md").write_text("\n".join(summary), encoding="utf-8")
    log.info("Table 4 reproduction: %s", status)


if __name__ == "__main__":
    main()
