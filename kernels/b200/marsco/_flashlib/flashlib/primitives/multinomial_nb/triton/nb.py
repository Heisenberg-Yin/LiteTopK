"""Flash Multinomial Naive Bayes — H200-optimized BW-bound implementation.

Algorithm
---------
Multinomial NB on non-negative count features X.

Fit (closed form, sklearn / cuML conventions):
    feature_count[c, d] = sum_{i: y_i = c} X[i, d]
    class_count[c]      = N_c
    feature_log_prob[c, d] = log(feature_count[c, d] + alpha)
                             - log(sum_d feature_count[c, d] + alpha * D)
    class_log_prior[c]  = log(N_c / N)

Predict (joint log-likelihood, argmax over c):
    jll[n, c] = (X_test @ feature_log_prob.T)[n, c] + class_log_prior[c]
    pred[n]   = argmax_c jll[n, c]

Both stages are dominated by a single (N x D) x (C x ?) matmul: fit is
``one_hot.T @ X`` (computed via the shared :mod:`nb_core` Triton kernel),
predict is a single GEMM dispatched through :mod:`flashlib.linalg.gemm`
so the precision is selected by ``tol`` rather than an explicit dtype
branch.

Optimisations
-------------
1. Fit: shared ``_nb_count_kernel`` from :mod:`nb_core` --
   ``tl.dot(one_hot, X)`` per block, atomic-free. One pass over X.
2. Predict: single GEMM via :func:`flashlib.linalg.gemm.gemm`. ``tol``
   selects the variant (default ``1e-3`` -> bf16 with fp32 accumulator;
   ``None`` -> fp32 IEEE for sklearn-faithful argmax).
3. End-to-end is BW-bound (predict is one read of X_test, one read of
   FLP, one write of (N_test, C); arithmetic intensity ~D for large D).

Equivalence with sklearn / cuML
- sklearn smooths feature_count + alpha and the row sum + alpha*D.
- cuML's MultinomialNB matches sklearn's formulas.
- For ``predict_dtype="bf16"`` (legacy keyword) we set ``tol=1e-3``;
  the new universal knob is ``tol``.
"""
import torch

from flashlib.linalg.gemm import gemm as _flash_gemm
from flashlib.primitives.multinomial_nb.triton.nb_core import nb_count_features


def flash_multinomial_nb_fit(X: torch.Tensor, y: torch.Tensor, n_classes: int,
                             alpha: float = 1.0):
    """Fit Multinomial NB. Returns params dict for predict.

    Args:
        X: (N, D) on cuda. Non-negative counts (we don't check). Any float dtype.
        y: (N,) int labels in [0, n_classes).
        n_classes: C.
        alpha: Laplace smoothing (sklearn / cuML default = 1.0).

    Returns:
        dict with class_log_prior (C,) fp32, feature_log_prob (C, D) fp32,
        feature_count (C, D) fp32, class_count (C,) fp32, n_classes,
        n_features, alpha.
    """
    assert X.is_cuda and X.ndim == 2 and y.ndim == 1
    N, D = X.shape
    C = n_classes

    feature_count, class_count = nb_count_features(X, y, n_classes, binarize=None)

    smoothed_fc = feature_count + float(alpha)
    smoothed_cc = smoothed_fc.sum(dim=1, keepdim=True)
    feature_log_prob = torch.log(smoothed_fc) - torch.log(smoothed_cc)

    safe_count = class_count.clamp(min=1.0)
    class_log_prior = torch.log(safe_count) - torch.log(class_count.sum())

    return {
        "class_log_prior": class_log_prior.contiguous(),
        "feature_log_prob": feature_log_prob.contiguous(),
        "feature_count": feature_count,
        "class_count": class_count,
        "n_classes": C,
        "n_features": D,
        "alpha": float(alpha),
    }


def _resolve_tol(tol, predict_dtype):
    """Map the legacy ``predict_dtype`` kw onto the universal ``tol`` knob."""
    if predict_dtype is None:
        return tol
    if predict_dtype == "bf16":
        return 1e-3 if tol is None else tol
    if predict_dtype == "fp32":
        return None
    raise ValueError(f"Unknown predict_dtype: {predict_dtype!r}")


def flash_multinomial_nb_predict_log_proba_unnormalized(
    X_test: torch.Tensor,
    params: dict,
    predict_dtype=None,
    *,
    tol=None,
):
    """Compute joint log-likelihood: ``jll[n, c] = X @ FLP.T + class_log_prior``.

    Single GEMM via :func:`flashlib.linalg.gemm.gemm` with precision
    routed by ``tol``. The legacy ``predict_dtype`` keyword is mapped
    onto ``tol`` so existing call sites continue to work::

        predict_dtype="bf16" <-> tol=1e-3
        predict_dtype="fp32" <-> tol=None
    """
    assert X_test.is_cuda and X_test.ndim == 2
    feature_log_prob = params["feature_log_prob"]
    class_log_prior = params["class_log_prior"]

    if not X_test.is_contiguous():
        X_test = X_test.contiguous()

    eff_tol = _resolve_tol(tol, predict_dtype)

    # _flash_gemm dispatches by tol (bf16 / TF32 / IEEE). The global
    # allow_tf32 flag only matters when an internal helper falls through
    # to torch.matmul; we only flip it when the caller has opted into a
    # lossy budget.
    use_tf32 = eff_tol is not None and eff_tol > 0
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        jll = _flash_gemm(X_test, feature_log_prob.t(), tol=eff_tol)
    finally:
        if use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    if jll.dtype != torch.float32:
        jll = jll.to(torch.float32)
    jll = jll + class_log_prior.unsqueeze(0)
    return jll


def flash_multinomial_nb_predict(X_test: torch.Tensor, params: dict,
                                 return_log_proba: bool = False,
                                 predict_dtype=None,
                                 *,
                                 tol=None):
    """Predict labels (and optionally normalized log-probabilities)."""
    jll = flash_multinomial_nb_predict_log_proba_unnormalized(
        X_test, params, predict_dtype=predict_dtype, tol=tol,
    )
    labels = jll.argmax(dim=1)
    if return_log_proba:
        log_proba = jll - torch.logsumexp(jll, dim=1, keepdim=True)
        return labels, log_proba
    return labels


def flash_multinomial_nb(X_train, y_train, X_test, n_classes,
                         alpha: float = 1.0,
                         return_log_proba: bool = False,
                         predict_dtype=None,
                         *,
                         tol=None):
    """End-to-end fit + predict.

    ``tol`` (universal): routed through :func:`flashlib.linalg.gemm.gemm`.
        ``None`` -> fp32 IEEE; ``1e-3`` -> bf16 (~2-3x faster, ~1e-3 rel
        err on jll, safe for argmax).
    ``predict_dtype`` (legacy): explicit ``"bf16"`` / ``"fp32"`` override
        mapped onto ``tol``.
    """
    params = flash_multinomial_nb_fit(X_train, y_train, n_classes, alpha=alpha)
    return flash_multinomial_nb_predict(
        X_test, params, return_log_proba=return_log_proba,
        predict_dtype=predict_dtype, tol=tol,
    )
