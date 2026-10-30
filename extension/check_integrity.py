"""Phase 0 - integrity check of the published artefacts.

Writes extension/results/sanity/integrity_report.md and integrity_counts.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import load_best_params, load_config, rpath, setup_logging, step_timer

SCRIPT = "check_integrity"


def main() -> None:
    cfg = load_config()
    log = setup_logging(cfg, SCRIPT)
    out_dir = rpath(cfg, "out_results") / "sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    issues: list[str] = []
    lines: list[str] = ["# Integrity report", ""]

    with step_timer(cfg, log, SCRIPT, "embeddings and metadata"):
        rows = []
        ref_idx = None
        for enc in cfg["encoders"]:
            d = rpath(cfg, "embeddings_dir") / enc
            for split in ("train", "val", "test"):
                X = np.load(d / f"{split}_embeddings.npy", mmap_mode="r")
                meta = pd.read_csv(d / f"{split}_metadata.csv")
                finite = bool(np.isfinite(np.asarray(X[: min(5000, X.shape[0])])).all())
                rows.append({"encoder": enc, "split": split, "n_rows": X.shape[0], "dim": X.shape[1],
                             "dtype": str(X.dtype), "n_meta": len(meta), "finite_head": finite})
                if X.shape[0] != len(meta):
                    issues.append(f"{enc}/{split}: {X.shape[0]} embeddings vs {len(meta)} metadata rows")
                if not finite:
                    issues.append(f"{enc}/{split}: non-finite values")
                key = (split, tuple(meta["idx"]))
                if ref_idx is None:
                    ref_idx = {}
                if split not in ref_idx:
                    ref_idx[split] = key[1]
                elif ref_idx[split] != key[1]:
                    issues.append(f"{enc}/{split}: instance order differs from the other encoders")
        shapes = pd.DataFrame(rows)
        shapes.to_csv(out_dir / "integrity_shapes.csv", index=False)
        lines += ["## Embedding matrices", "", shapes.to_markdown(index=False), ""]
        log.info("\n%s", shapes.to_string(index=False))

    with step_timer(cfg, log, SCRIPT, "corpus counts vs unified dataset and splits"):
        meta = pd.concat([pd.read_csv(rpath(cfg, "embeddings_dir") / cfg["encoders"][0] / f"{s}_metadata.csv")
                          for s in ("train", "val", "test")], ignore_index=True)
        uni = pd.read_parquet(rpath(cfg, "unified_dataset"), columns=["idx", "dataset_id", "label"])
        all_ds = sorted(set(meta.dataset_id) | set(uni.dataset_id))
        counts = meta.groupby(["dataset_id"]).size().rename("n_embedded").reindex(all_ds).to_frame()
        counts["n_unified"] = uni.groupby("dataset_id").size().reindex(all_ds)
        for s in ("train", "val", "test"):
            counts[f"n_{s}"] = meta[meta.split == s].groupby("dataset_id").size()
        counts["n_fake"] = meta[meta.label_id == 0].groupby("dataset_id").size()
        counts["n_real"] = meta[meta.label_id == 1].groupby("dataset_id").size()
        counts = counts.fillna(0).astype(int)
        counts["embedded"] = counts["n_embedded"] > 0
        counts["match_unified"] = counts["n_embedded"] == counts["n_unified"]
        counts.to_csv(out_dir / "integrity_counts.csv")
        lines += ["## Per-corpus counts (embeddings metadata vs unified dataset)", "",
                  counts.to_markdown(), ""]
        log.info("\n%s", counts.to_string())
        for ds, r in counts.iterrows():
            if ds in cfg["datasets"] and not r["match_unified"]:
                issues.append(f"{ds}: {r['n_embedded']} embedded vs {r['n_unified']} in unified dataset")
            if ds in cfg["datasets_not_embedded"]:
                issues.append(f"{ds}: present in the unified dataset ({r['n_unified']} rows) but never embedded "
                              "(not part of the published 10-corpus benchmark)")
        if not meta["idx"].is_unique:
            issues.append("duplicated idx in metadata")
        # split files
        for s in ("train", "val", "test"):
            sp = pd.read_csv(rpath(cfg, "splits_dir") / f"{s}_split.csv", usecols=["idx", "dataset_id", "label_id"])
            m = meta[meta.split == s]
            if len(sp) != len(m) or set(sp.idx) != set(m.idx):
                issues.append(f"split file {s}: {len(sp)} rows vs {len(m)} metadata rows / idx mismatch")
            else:
                mm = sp.merge(m, on="idx", suffixes=("_split", "_meta"))
                if (mm.label_id_split != mm.label_id_meta).any():
                    issues.append(f"split file {s}: label mismatch with metadata")
        lines += ["Split files (codes/results/splits) match the embedding metadata: "
                  + ("yes" if not any("split file" in i for i in issues) else "NO"), ""]

    with step_timer(cfg, log, SCRIPT, "hyper-parameters and published predictions"):
        bp = load_best_params(cfg)
        missing = [(e, c) for e in cfg["encoders"] for c in cfg["classifiers"] if (e, c) not in bp]
        lines += ["## Selected hyper-parameters", "", f"Pairs found: {len(bp)} / 32. Missing: {missing or 'none'}", ""]
        if missing:
            issues.append(f"missing best_params for {missing}")
        bp_df = pd.DataFrame([{"encoder": e, "classifier": c, "best_params": str(p)} for (e, c), p in bp.items()])
        bp_df.to_csv(out_dir / "selected_hyperparameters.csv", index=False)
        lines += [bp_df.to_markdown(index=False), ""]
        pred = pd.read_csv(rpath(cfg, "published_test_predictions"))
        n_models = pred.model.nunique()
        per_model = pred.groupby("model").size()
        lines += ["## Published per-item test predictions", "",
                  f"models: {n_models}; rows per model: min={per_model.min()} max={per_model.max()}", ""]
        if n_models != 32 or per_model.nunique() != 1:
            issues.append("published test predictions are incomplete")

    lines += ["## Divergences / notes", ""] + ([f"- {i}" for i in issues] or ["- none"])
    (out_dir / "integrity_report.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("Issues: %s", issues or "none")
    log.info("Report written to %s", out_dir / "integrity_report.md")


if __name__ == "__main__":
    main()
