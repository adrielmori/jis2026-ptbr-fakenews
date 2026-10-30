"""Shared utilities for the extended study of the multi-corpus fake-news benchmark.

Everything here mirrors the original training notebook (codes/01-ml-approach.ipynb):
same pipelines, same scaler placement, same random seeds, same metric definitions.
"""
from __future__ import annotations

import ast
import csv
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

EXT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXT_DIR.parent

LABEL_NAMES = {0: "FAKE", 1: "REAL"}


# ----------------------------------------------------------------------------
# Config / logging / timing
# ----------------------------------------------------------------------------
def load_config(path: Optional[Path] = None) -> dict:
    path = Path(path) if path else Path(os.environ.get("EXT_CONFIG", EXT_DIR / "config.yaml"))
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_repo_root"] = REPO_ROOT
    return cfg


def rpath(cfg: dict, key: str) -> Path:
    """Resolve a path from config.paths against the repository root."""
    return (cfg["_repo_root"] / cfg["paths"][key]).resolve()


class _RedactRepoRoot(logging.Filter):
    """Replace the absolute repository path by <repo> so logs carry no user or machine information."""

    def __init__(self):
        super().__init__()
        self.roots = sorted({str(REPO_ROOT), REPO_ROOT.as_posix()}, key=len, reverse=True)

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for r in self.roots:
            msg = msg.replace(r, "<repo>")
        record.msg, record.args = msg, ()
        return True


def setup_logging(cfg: dict, name: str) -> logging.Logger:
    log_dir = rpath(cfg, "out_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_dir / f"{name}.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    for h in (fh, sh):
        h.addFilter(_RedactRepoRoot())
        logger.addHandler(h)
    logger.propagate = False
    return logger


@contextmanager
def step_timer(cfg: dict, logger: logging.Logger, script: str, step: str):
    """Log the wall-clock time of a step and append it to logs/timings.csv."""
    t0 = time.time()
    logger.info("[START] %s", step)
    try:
        yield
    finally:
        elapsed = time.time() - t0
        logger.info("[DONE ] %s | %.1f s (%.2f min)", step, elapsed, elapsed / 60)
        path = rpath(cfg, "out_logs") / "timings.csv"
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["script", "step", "seconds", "finished_at"])
            w.writerow([script, step, f"{elapsed:.2f}", time.strftime("%Y-%m-%d %H:%M:%S")])


# ----------------------------------------------------------------------------
# Pipelines (verbatim from the original notebook) + saved hyper-parameters
# ----------------------------------------------------------------------------
def base_pipeline(classifier: str, random_state: int = 42, rf_n_jobs: int = -1, use_gpu: bool = False) -> Pipeline:
    if classifier == "logreg":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=3000, random_state=random_state))])
    if classifier == "linearsvc":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LinearSVC(random_state=random_state, max_iter=8000))])
    if classifier == "svc_rbf":
        if use_gpu:  # same C-SVC problem, Gram matrix computed on the GPU (see gpu_svc.py)
            from gpu_svc import GPUKernelSVC
            return Pipeline([("scaler", StandardScaler()), ("clf", GPUKernelSVC())])
        return Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf"))])
    if classifier == "random_forest":
        return Pipeline([("clf", RandomForestClassifier(random_state=random_state, n_jobs=rf_n_jobs))])
    if classifier == "gradient_boosting":
        return Pipeline([("clf", GradientBoostingClassifier(random_state=random_state))])
    if classifier == "gaussian_nb":
        return Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())])
    if classifier == "knn":
        return Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())])
    if classifier == "decision_tree":
        return Pipeline([("clf", DecisionTreeClassifier(random_state=random_state))])
    raise ValueError(f"Unknown classifier: {classifier}")


def build_pipeline(classifier: str, best_params: dict, random_state: int = 42, rf_n_jobs: int = -1,
                   use_gpu: bool = False) -> Pipeline:
    pipe = base_pipeline(classifier, random_state=random_state, rf_n_jobs=rf_n_jobs, use_gpu=use_gpu)
    pipe.set_params(**best_params)
    return pipe


def load_best_params(cfg: dict) -> Dict[Tuple[str, str], dict]:
    """Selected hyper-parameters of the 32 encoder x classifier pairs (from the published grid search)."""
    df = pd.read_csv(rpath(cfg, "published_results"))
    out = {}
    for _, r in df.iterrows():
        out[(r["embedding_model"], r["classifier"])] = ast.literal_eval(r["best_params"])
    return out


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
class EncoderData:
    """All embedded instances of one encoder (train+val+test concatenated, original order kept)."""

    def __init__(self, X: np.ndarray, meta: pd.DataFrame):
        self.X = X
        self.meta = meta.reset_index(drop=True)
        self.y = self.meta["label_id"].to_numpy().astype(int)
        self.dataset = self.meta["dataset_id"].to_numpy()
        self.split = self.meta["split"].to_numpy()
        self.idx = self.meta["idx"].to_numpy()

    def mask_split(self, *splits: str) -> np.ndarray:
        return np.isin(self.split, list(splits))

    def mask_dataset(self, *datasets: str) -> np.ndarray:
        return np.isin(self.dataset, list(datasets))


def load_encoder(cfg: dict, encoder: str) -> EncoderData:
    emb_dir = rpath(cfg, "embeddings_dir") / encoder
    Xs, metas = [], []
    for split in ("train", "val", "test"):
        X = np.load(emb_dir / f"{split}_embeddings.npy")
        meta = pd.read_csv(emb_dir / f"{split}_metadata.csv")
        if X.shape[0] != len(meta):
            raise ValueError(f"{encoder}/{split}: {X.shape[0]} embeddings vs {len(meta)} metadata rows")
        meta["split"] = split
        Xs.append(np.asarray(X, dtype=np.float32))
        metas.append(meta[["idx", "dataset_id", "label_id", "split"]])
    return EncoderData(np.vstack(Xs), pd.concat(metas, ignore_index=True))


def load_texts(cfg: dict) -> pd.Series:
    df = pd.read_parquet(rpath(cfg, "unified_dataset"), columns=["idx", "text_no_url"])
    return df.set_index("idx")["text_no_url"]


# ----------------------------------------------------------------------------
# Scores / metrics
# ----------------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def model_scores(model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return (pred, prob_real, kind). prob_real is P(label==1) when predict_proba exists,
    otherwise the logistic squash of the decision function (uncalibrated pseudo-probability)."""
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        prob_real = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
        return pred, prob_real, "predict_proba"
    d = model.decision_function(X)
    return pred, sigmoid(d), "sigmoid_of_decision_function"


def macro_f1(y_true, y_pred) -> float:
    """Macro-F1 over the union of labels present in y_true and y_pred (same as the published code)."""
    return float(f1_score(y_true, y_pred, average="macro"))


def binary_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": macro_f1(y_true, y_pred),
        "precision_fake": float(p[0]), "recall_fake": float(r[0]), "f1_fake": float(f[0]), "support_fake": int(s[0]),
        "precision_real": float(p[1]), "recall_real": float(r[1]), "f1_real": float(f[1]), "support_real": int(s[1]),
    }


# ----------------------------------------------------------------------------
# Stratified sub-sampling (balanced LODO)
# ----------------------------------------------------------------------------
def stratified_subsample(positions: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator,
                         min_per_class: int = 1) -> np.ndarray:
    """Sample n positions preserving class proportions (largest-remainder rounding),
    keeping at least `min_per_class` of every class that is present."""
    positions = np.asarray(positions)
    y = np.asarray(y)
    if n >= len(positions):
        return positions.copy()
    classes, counts = np.unique(y, return_counts=True)
    exact = counts / counts.sum() * n
    alloc = np.floor(exact).astype(int)
    alloc = np.maximum(alloc, np.minimum(min_per_class, counts))
    rem = n - alloc.sum()
    if rem > 0:
        order = np.argsort(-(exact - np.floor(exact)))
        for k in order:
            if rem == 0:
                break
            if alloc[k] < counts[k]:
                alloc[k] += 1
                rem -= 1
    while rem < 0:  # min_per_class pushed us above n; remove from the largest class
        k = int(np.argmax(alloc))
        alloc[k] -= 1
        rem += 1
    out = []
    for c, a in zip(classes, alloc):
        pos_c = positions[y == c]
        out.append(rng.choice(pos_c, size=int(min(a, len(pos_c))), replace=False))
    out = np.concatenate(out)
    rng.shuffle(out)
    return out


# ----------------------------------------------------------------------------
# Statistics (Friedman / Wilcoxon / Holm) - mirrors codes/04-statistic-result-analysis.ipynb
# ----------------------------------------------------------------------------
def holm_bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)
    order = np.argsort(p_values)
    adjusted = np.full(n, np.nan)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, p_values[i] * (n - rank)))
        adjusted[i] = running
    reject = np.zeros(n, dtype=bool)
    for rank, i in enumerate(order):
        if p_values[i] <= alpha / (n - rank):
            reject[i] = True
        else:
            break
    return adjusted, reject


def friedman_wilcoxon_holm(pivot: pd.DataFrame, alpha: float = 0.05, scenario: str = "") -> Dict[str, pd.DataFrame]:
    """pivot: rows = statistical units (datasets / holdouts), columns = models (pipelines)."""
    pivot = pivot.dropna(axis=0, how="any")
    n_units, n_models = pivot.shape
    stat, p = friedmanchisquare(*[pivot[c].to_numpy() for c in pivot.columns])
    friedman = pd.DataFrame([{
        "scenario": scenario, "n_models": n_models, "n_units": n_units, "alpha": alpha,
        "statistic": float(stat), "p_value": float(p),
        "kendall_w": float(stat / (n_units * (n_models - 1))),
        "significant": bool(p < alpha),
    }])
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    ranking = pd.DataFrame({
        "model_id": pivot.columns,
        "mean_macro_f1": pivot.mean(axis=0).values,
        "std_macro_f1": pivot.std(axis=0, ddof=0).values,
        "median_macro_f1": pivot.median(axis=0).values,
        "avg_rank": ranks.mean(axis=0).values,
        "n_units": n_units,
    }).sort_values(["mean_macro_f1", "avg_rank", "model_id"], ascending=[False, True, True],
                   kind="mergesort").reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.insert(0, "scenario", scenario)
    best = ranking.loc[0, "model_id"]
    rows = []
    for m in pivot.columns:
        if m == best:
            continue
        diff = pivot[best] - pivot[m]
        rec = {"scenario": scenario, "reference_model": best, "comparison_model": m, "n_paired": n_units,
               "mean_reference": float(pivot[best].mean()), "mean_comparison": float(pivot[m].mean()),
               "mean_diff": float(diff.mean()), "median_diff": float(diff.median()),
               "statistic": np.nan, "raw_p_value": np.nan, "holm_adjusted_p_value": np.nan,
               "reject_h0_holm": False, "alpha": alpha, "status": "ok", "note": ""}
        if not np.any(~np.isclose(diff.to_numpy(), 0.0, atol=1e-12)):
            rec.update(statistic=0.0, raw_p_value=1.0, status="all_zero_differences")
        else:
            s, pv = wilcoxon(pivot[best].to_numpy(), pivot[m].to_numpy(), alternative="two-sided",
                             zero_method="wilcox", method="auto")
            rec.update(statistic=float(s), raw_p_value=float(pv))
        rows.append(rec)
    posthoc = pd.DataFrame(rows)
    valid = posthoc["raw_p_value"].notna()
    adj, rej = holm_bonferroni(posthoc.loc[valid, "raw_p_value"].to_numpy(), alpha)
    posthoc.loc[valid, "holm_adjusted_p_value"] = adj
    posthoc.loc[valid, "reject_h0_holm"] = rej
    posthoc = posthoc.sort_values(["raw_p_value", "comparison_model"], kind="mergesort").reset_index(drop=True)
    return {"friedman": friedman, "ranking": ranking, "posthoc": posthoc}


def model_id(encoder: str, classifier: str) -> str:
    return f"{encoder} + {classifier}"


def run_tasks(fn, gpu_items: list, cpu_items: list, n_jobs: int, verbose: int = 5,
              gpu_rows: Optional[list] = None, per_worker_gb: float = 0.7, margin_gb: float = 1.5) -> Tuple[list, list]:
    """Run ``fn(*item)`` for every item. GPU items run sequentially in the main process (one Gram
    matrix of n^2 x 8 bytes, up to ~14 GB here) in a background thread while the CPU items run in a
    joblib (loky) pool with ``n_jobs`` workers. ``gpu_rows[i]`` = training rows of GPU item i: items
    whose Gram matrix would not fit next to the CPU pool are run *before* the pool starts (a few
    minutes of mostly idle CPU instead of swapping). Returns (gpu_results, cpu_results) in input order."""
    from concurrent.futures import ThreadPoolExecutor
    from joblib import Parallel, delayed
    log = logging.getLogger("run_tasks")

    first_idx: list = []
    if gpu_items and gpu_rows is not None:
        try:
            import psutil
            avail = psutil.virtual_memory().available / 2**30
        except Exception:  # pragma: no cover
            avail = float("inf")
        budget = avail - n_jobs * per_worker_gb - margin_gb
        first_idx = [i for i, n in enumerate(gpu_rows) if (n * n * 8) / 2**30 > budget]
        if first_idx:
            log.info("%d GPU fits need more than %.1f GB of host RAM: running them before the CPU pool starts",
                     len(first_idx), budget)
    results: dict = {}
    for i in sorted(first_idx, key=lambda i: -gpu_rows[i]):
        results[i] = fn(*gpu_items[i])
    rest = [i for i in range(len(gpu_items)) if i not in results]

    def _gpu_rest():
        return [(i, fn(*gpu_items[i])) for i in rest]

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_gpu_rest) if rest else None
        cpu_res = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(fn)(*it) for it in cpu_items) if cpu_items else []
        if fut is not None:
            results.update(dict(fut.result()))
    gpu_res = [results[i] for i in range(len(gpu_items))]
    return gpu_res, cpu_res


def dataset_sizes(cfg: dict) -> Dict[str, int]:
    """Number of embedded instances per corpus (from the published split metadata)."""
    meta = pd.concat(pd.read_csv(rpath(cfg, "splits_dir") / f"{s}_split.csv", usecols=["dataset_id"])
                     for s in ("train", "val", "test"))
    return meta.dataset_id.value_counts().to_dict()


def set_thread_limits(n: int = 1) -> None:
    """Limit BLAS threads inside worker processes to avoid oversubscription."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
