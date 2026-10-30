"""Error analysis for the best LODO pipeline (Study 1 ranking, full LODO).

Outputs (extension/results/errors/):
  best_pipeline.json                 which pipeline / which probability source was used
  high_confidence_errors.csv         30 most confident errors per corpus (id, corpus, label, prediction, confidence, text[:300])
  high_confidence_error_rates.csv    fraction of high-confidence (>= 0.90) errors per class and per corpus
  umap_coordinates.csv               2-D UMAP of the best encoder's embedding space + corpus + LODO hit/miss
"""
from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore", message="n_jobs value .* overridden")  # umap prints the venv path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from common import LABEL_NAMES, load_config, load_encoder, load_texts, model_id, rpath, setup_logging, step_timer

SCRIPT = "run_errors"


def main() -> None:
    cfg = load_config()
    log = setup_logging(cfg, SCRIPT)
    ecfg = cfg["errors"]
    res = rpath(cfg, "out_results")
    out_dir = res / "errors"
    out_dir.mkdir(parents=True, exist_ok=True)

    with step_timer(cfg, log, SCRIPT, "select best LODO pipeline and probability source"):
        ranking = pd.read_csv(res / "lodo" / "lodo_stats_ranking.csv")
        best = ranking[(ranking.scenario == "lodo_full") & (ranking["rank"] == 1)].iloc[0]
        best_id = best.model_id
        encoder, classifier = best_id.split(" + ")
        cal_pipes = {model_id(e, c) for e, c in cfg["calibration"]["pipelines"]}
        if best_id in cal_pipes:
            # Calibrated LODO probabilities from Study 2 scenario (b): sigmoid (Platt) fitted on a 15% slice of
            # the training corpora. Thresholds on calibrated probabilities are interpretable.
            cp = pd.read_parquet(res / "predictions" / "calibration_predictions.parquet")
            df = cp[(cp.pipeline == best_id) & (cp.scenario == "lodo") & (cp.condition == "sigmoid")].copy()
            df = df.rename(columns={"corpus": "held_out_dataset"})
            source = "study2_lodo_sigmoid_calibrated (model trained on 85% of the 9 training corpora, Platt on 15%)"
        else:
            lp = pd.read_parquet(res / "predictions" / "lodo_predictions.parquet")
            df = lp[(lp.encoder == encoder) & (lp.classifier == classifier)].copy()
            df["confidence"] = np.maximum(df.prob_real, 1 - df.prob_real)
            source = f"study1_full_lodo ({df.score_kind.iloc[0]})"
        df["correct"] = df.pred_label_id == df.true_label_id
        info = {"best_pipeline": best_id, "encoder": encoder, "classifier": classifier,
                "mean_lodo_macro_f1": float(best.mean_macro_f1), "probability_source": source,
                "n_items": int(len(df)), "accuracy": float(df.correct.mean())}
        (out_dir / "best_pipeline.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        log.info("best LODO pipeline: %s (mean macro-F1 %.4f); probabilities: %s", best_id, best.mean_macro_f1, source)

    with step_timer(cfg, log, SCRIPT, "high-confidence errors"):
        texts = load_texts(cfg)
        err = df[~df.correct].copy()
        err["text_300"] = err.idx.map(texts).astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, ecfg["text_chars"])
        err["label"] = err.true_label_id.map(LABEL_NAMES)
        err["prediction"] = err.pred_label_id.map(LABEL_NAMES)
        top = (err.sort_values(["held_out_dataset", "confidence"], ascending=[True, False])
               .groupby("held_out_dataset").head(ecfg["top_k_per_corpus"]))
        top = top[["idx", "held_out_dataset", "label", "prediction", "confidence", "prob_real", "text_300"]].rename(
            columns={"idx": "id", "held_out_dataset": "corpus"})
        top.insert(0, "pipeline", best_id)
        top.to_csv(out_dir / "high_confidence_errors.csv", index=False)

        thr = ecfg["high_confidence_threshold"]
        df["high_conf"] = df.confidence >= thr
        rows = []
        groups = [("corpus", ds, g) for ds, g in df.groupby("held_out_dataset")]
        groups += [("class", LABEL_NAMES[c], g) for c, g in df.groupby("true_label_id")]
        groups += [("corpus_x_class", f"{ds}|{LABEL_NAMES[c]}", g) for (ds, c), g in df.groupby(["held_out_dataset", "true_label_id"])]
        groups += [("all", "ALL", df)]
        for level, name, g in groups:
            n_err = int((~g.correct).sum())
            n_hc = int(g.high_conf.sum())
            n_hc_err = int((g.high_conf & ~g.correct).sum())
            rows.append({"level": level, "group": name, "n_items": len(g), "n_errors": n_err,
                         "error_rate": n_err / len(g), "n_high_conf": n_hc, "high_conf_coverage": n_hc / len(g),
                         "n_high_conf_errors": n_hc_err,
                         "frac_errors_that_are_high_conf": n_hc_err / n_err if n_err else np.nan,
                         "high_conf_error_rate": n_hc_err / n_hc if n_hc else np.nan,
                         "frac_items_high_conf_error": n_hc_err / len(g)})
        rates = pd.DataFrame(rows)
        rates.insert(0, "pipeline", best_id)
        rates["confidence_threshold"] = thr
        rates.to_csv(out_dir / "high_confidence_error_rates.csv", index=False)
        log.info("\n%s", rates[rates.level.isin(["corpus", "class", "all"])].round(4).to_string(index=False))

    with step_timer(cfg, log, SCRIPT, f"UMAP of {encoder} embedding space"):
        import umap  # noqa: E402 (slow import)
        data = load_encoder(cfg, encoder)
        X = normalize(data.X.astype(np.float32), norm="l2")
        u = ecfg["umap"]
        X = PCA(n_components=u["pca_components"], random_state=cfg["random_state"]).fit_transform(X)
        reducer = umap.UMAP(n_components=2, n_neighbors=u["n_neighbors"], min_dist=u["min_dist"], metric=u["metric"],
                            spread=u["spread"], random_state=cfg["random_state"], transform_seed=cfg["random_state"])
        emb = reducer.fit_transform(X)
        coords = pd.DataFrame({"idx": data.idx, "corpus": data.dataset, "label": [LABEL_NAMES[v] for v in data.y],
                               "umap_1": emb[:, 0], "umap_2": emb[:, 1]})
        coords = coords.merge(df[["idx", "correct", "confidence", "pred_label_id"]], on="idx", how="left")
        coords["lodo_outcome"] = np.where(coords.correct.isna(), "n/a", np.where(coords.correct.astype(bool), "correct", "error"))
        coords.to_csv(out_dir / "umap_coordinates.csv", index=False)
        log.info("UMAP coordinates saved for %d items (%d errors)", len(coords), int((coords.lodo_outcome == "error").sum()))
    log.info("Error analysis finished.")


if __name__ == "__main__":
    main()
