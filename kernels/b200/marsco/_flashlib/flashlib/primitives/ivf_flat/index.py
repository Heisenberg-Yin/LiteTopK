"""``IvfFlatIndex`` -- the in-memory container for a built IVF-Flat index.

Kept in its own module (no Triton import) so both the Triton builder
(:mod:`flashlib.primitives.ivf_flat.triton.build`) and the pure-torch
reference (:mod:`flashlib.primitives.ivf_flat.torch_fallback`) can
construct/consume it without an import cycle through
:mod:`flashlib.primitives.ivf_flat.impl`.

Layout
------
The database vectors are stored **cell-contiguous**: all vectors assigned
to inverted list ``c`` occupy the half-open row range
``[list_offsets[c], list_offsets[c + 1])`` of ``data``. This CSR-style
layout is the GPU-oriented index layout that lets the fused fine-scan
kernel stream a probed list with fully coalesced reads.

``ids[p]`` maps a stored row ``p`` back to the caller's original row id,
so search results are reported in the caller's index space.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class IvfFlatIndex:
    """A built IVF-Flat index.

    Attributes:
        centroids: ``(nlist, Dp)`` coarse-quantizer centroids (working dtype).
        data: ``(M, Dp)`` database vectors in cell-contiguous order.
        ids: ``(M,)`` int64 -- original row id for each stored row.
        list_offsets: ``(nlist + 1,)`` int64 CSR offsets into ``data``.
        metric: distance metric (only ``"l2"`` supported).
        D: original feature dimension as passed by the caller.
        Dp: padded working dimension (``max(D, 16)``); zero columns added
            for ``D < 16`` never affect squared-L2 distances.
        nlist: number of inverted lists / coarse centroids.
        nprobe: default number of lists to probe at search time.
        max_list_len: longest inverted list, recorded at build time so
            search can size the kernel's chunk loop without a D2H sync.
    """

    centroids: torch.Tensor
    data: torch.Tensor
    ids: torch.Tensor
    list_offsets: torch.Tensor
    metric: str
    D: int
    Dp: int
    nlist: int
    nprobe: int
    max_list_len: int = 0

    @property
    def M(self) -> int:
        return int(self.data.shape[0])

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    def list_lengths(self) -> torch.Tensor:
        """``(nlist,)`` int64 number of vectors in each inverted list."""
        return self.list_offsets[1:] - self.list_offsets[:-1]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"IvfFlatIndex(M={self.M}, D={self.D}, Dp={self.Dp}, "
            f"nlist={self.nlist}, nprobe={self.nprobe}, metric={self.metric!r}, "
            f"dtype={self.dtype}, device={self.device})"
        )


__all__ = ["IvfFlatIndex"]
