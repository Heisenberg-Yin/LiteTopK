"""t-SNE dispatcher.

Backends:
    backend=None / "triton" -> :func:`triton_tsne` (default; vectorised
                                P-matrix + 2-kernel gradient SGD).
    backend="cutedsl"        -> CuteDSL perplexity-bisection P-matrix
                                (currently a P-matrix accelerator only,
                                still completes via the triton SGD).
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib.primitives.tsne.triton.train import (
    triton_tsne,
    triton_tsne_gradient_only,
)


def flash_tsne(
    X: torch.Tensor,
    n_iter: int = 1000,
    lr: float = 200.0,
    perplexity: float = 30.0,
    early_exag_iters: Optional[int] = None,
    ee_factor: float = 12.0,
    seed: int = 0,
    *,
    backend: Optional[str] = None,
):
    """t-SNE -- exact in input dtype by default.

    The CuteDSL path only accelerates the P-matrix step today; both
    backends currently fall through to the Triton SGD loop.
    """
    del backend  # CuteDSL path intentionally not yet wired in.
    return triton_tsne(
        X, n_iter=n_iter, lr=lr, perplexity=perplexity,
        early_exag_iters=early_exag_iters, ee_factor=ee_factor, seed=seed,
    )
