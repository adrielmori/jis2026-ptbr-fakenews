"""GPU-accelerated RBF SVC that is algorithmically identical to ``sklearn.svm.SVC(kernel="rbf")``.

The only expensive part of a C-SVC fit on high-dimensional embeddings is evaluating the RBF kernel
(1 536 multiply-adds per pair, re-evaluated by libsvm whenever its small cache misses). Here the full
Gram matrix is computed once on the GPU (PyTorch, float64) and handed to libsvm through
``SVC(kernel="precomputed")``; the optimisation problem, solver (SMO), tolerance, C and gamma are the
ones of the original pipeline, so the solution is the same up to floating-point round-off in the
kernel values. Prediction uses the support vectors only: ``dual_coef_ @ K(sv, x) + intercept_``,
which is exactly what libsvm computes; the kernel against the support vectors is also evaluated on
the GPU in chunks.

When CUDA (or torch) is unavailable the class degrades to the plain CPU ``SVC`` so that every script
still runs, only slower.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

_TORCH = None


def torch_cuda_available() -> bool:
    global _TORCH
    try:
        import torch  # noqa: WPS433 (optional dependency)
        _TORCH = torch
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch missing
        return False


def gpu_svc_enabled(cfg: dict) -> bool:
    """config.yaml: gpu.svc_rbf = auto | true | false."""
    mode = str(cfg.get("gpu", {}).get("svc_rbf", "auto")).lower()
    if mode in ("false", "0", "no", "off"):
        return False
    avail = torch_cuda_available()
    if mode in ("true", "1", "yes", "on") and not avail:
        raise RuntimeError("gpu.svc_rbf=true but no CUDA device / torch is available")
    return avail


def _resolve_gamma(gamma, X: np.ndarray) -> float:
    """Same rule as sklearn.svm.SVC (on the float64 training matrix)."""
    if isinstance(gamma, str):
        if gamma == "scale":
            var = X.var()
            return 1.0 / (X.shape[1] * var) if var != 0 else 1.0
        if gamma == "auto":
            return 1.0 / X.shape[1]
        raise ValueError(f"unknown gamma {gamma!r}")
    return float(gamma)


def rbf_kernel_gpu(A: np.ndarray, B: np.ndarray, gamma: float, device: str = "cuda",
                   row_chunk: int = 4096, out: np.ndarray | None = None) -> np.ndarray:
    """exp(-gamma * ||a - b||^2) for all pairs, computed in float64 on the GPU, returned on the host."""
    torch = _TORCH
    a_dev = torch.as_tensor(np.ascontiguousarray(A, dtype=np.float64), device=device)
    b_dev = torch.as_tensor(np.ascontiguousarray(B, dtype=np.float64), device=device)
    b_sq = (b_dev * b_dev).sum(1)
    if out is None:
        out = np.empty((A.shape[0], B.shape[0]), dtype=np.float64)
    for s in range(0, A.shape[0], row_chunk):
        a = a_dev[s:s + row_chunk]
        a_sq = (a * a).sum(1, keepdim=True)
        d2 = a_sq + b_sq[None, :] - 2.0 * (a @ b_dev.T)
        d2.clamp_(min=0.0)
        out[s:s + row_chunk] = torch.exp(-gamma * d2).cpu().numpy()
        del a_sq, d2
    del a_dev, b_dev, b_sq
    torch.cuda.empty_cache()
    return out


def _wait_for_host_memory(n_bytes: int, margin_bytes: int = 1 << 30, poll_s: float = 20.0, max_wait_s: float = 4 * 3600) -> None:
    """The Gram matrix lives in host RAM (n_train^2 x 8 bytes, up to ~14 GB here). When other workers
    occupy the memory, wait instead of swapping; the result is unaffected, only the start time."""
    try:
        import psutil
    except Exception:  # pragma: no cover
        return
    import logging, time
    need = n_bytes + margin_bytes
    t0 = time.time()
    while psutil.virtual_memory().available < need and time.time() - t0 < max_wait_s:
        logging.getLogger("gpu_svc").info("waiting for %.1f GB of free RAM (available %.1f GB)",
                                          need / 2**30, psutil.virtual_memory().available / 2**30)
        time.sleep(poll_s)


class GPUKernelSVC(ClassifierMixin, BaseEstimator):
    """Drop-in replacement for ``SVC(kernel="rbf", C=C, gamma=gamma)`` (binary or multiclass).

    Parameters mirror the sklearn names so that ``set_params(clf__C=..., clf__gamma=...)`` from the
    published ``best_params`` apply unchanged. ``cache_size`` only affects speed, never the solution.
    """

    def __init__(self, C: float = 1.0, gamma="scale", tol: float = 1e-3, shrinking: bool = True,
                 cache_size: float = 1000.0, class_weight=None, device: str = "cuda",
                 row_chunk: int = 4096):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.device = device
        self.row_chunk = row_chunk

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        X = np.ascontiguousarray(np.asarray(X), dtype=np.float64)
        y = np.asarray(y)
        self.gamma_ = _resolve_gamma(self.gamma, X)
        self.backend_ = "gpu_precomputed" if torch_cuda_available() else "cpu_libsvm"
        if self.backend_ == "cpu_libsvm":
            self.svc_ = SVC(kernel="rbf", C=self.C, gamma=self.gamma_, tol=self.tol, shrinking=self.shrinking,
                            cache_size=self.cache_size, class_weight=self.class_weight).fit(X, y)
            self.X_sv_ = None
        else:
            _wait_for_host_memory(X.shape[0] * X.shape[0] * 8)
            K = rbf_kernel_gpu(X, X, self.gamma_, self.device, self.row_chunk)
            self.svc_ = SVC(kernel="precomputed", C=self.C, tol=self.tol, shrinking=self.shrinking,
                            cache_size=self.cache_size, class_weight=self.class_weight).fit(K, y)
            del K
            self.X_sv_ = X[self.svc_.support_]
        self.classes_ = self.svc_.classes_
        self.support_ = self.svc_.support_
        self.n_support_ = self.svc_.n_support_
        self.intercept_ = self.svc_.intercept_
        self.n_features_in_ = X.shape[1]
        self.n_iter_ = getattr(self.svc_, "n_iter_", None)
        return self

    # ------------------------------------------------------------- predict
    def _kernel_to_sv(self, X) -> np.ndarray:
        X = np.ascontiguousarray(np.asarray(X), dtype=np.float64)
        return rbf_kernel_gpu(X, self.X_sv_, self.gamma_, self.device, self.row_chunk)

    def decision_function(self, X):
        check_is_fitted(self, "svc_")
        if self.backend_ == "cpu_libsvm":
            return self.svc_.decision_function(X)
        K = self._kernel_to_sv(X)
        if len(self.classes_) == 2:
            # sklearn exposes dual_coef_/intercept_ already in the convention of decision_function
            return K @ self.svc_.dual_coef_[0] + self.svc_.intercept_[0]
        # multiclass: let libsvm do the one-vs-one voting on a (n_test, n_train) matrix whose
        # non-support columns are never read
        full = np.zeros((K.shape[0], self.support_.max() + 1 if self.n_train_ is None else self.n_train_))
        full[:, self.support_] = K
        return self.svc_.decision_function(full)

    def predict(self, X):
        check_is_fitted(self, "svc_")
        if self.backend_ == "cpu_libsvm":
            return self.svc_.predict(X)
        if len(self.classes_) == 2:
            d = self.decision_function(X)
            return self.classes_[(d > 0).astype(int)]
        K = self._kernel_to_sv(X)
        full = np.zeros((K.shape[0], self.n_train_))
        full[:, self.support_] = K
        return self.svc_.predict(full)

    @property
    def n_train_(self):
        return None if self.X_sv_ is None else int(self.svc_.shape_fit_[0])

    @property
    def dual_coef_(self):
        return self.svc_.dual_coef_
