"""Estimate + Variant dataclasses returned by flashlib.info. Pure stdlib.

``Estimate`` is the unit of cost analysis. Each primitive's
``cost.estimate()`` returns one. Compound primitives (PCA, DBSCAN,
UMAP, ...) populate :attr:`Estimate.subops` by recursively calling
their child primitives' ``cost.estimate()`` -- the returned estimate
carries the full call-stack tree of sub-primitive estimates.

Print the tree with :meth:`Estimate.format_tree` or
:meth:`Estimate.print_tree`.

Derived performance fields
--------------------------

In addition to ``runtime_ms / flops / bytes_moved`` the dataclass
exposes computed properties for at-a-glance perf review:

* ``achieved_tflops``         -- ``flops / (runtime_ms * 1e9)``
* ``achieved_gbs``            -- ``bytes_moved / (runtime_ms * 1e6)``
* ``arithmetic_intensity``    -- ``flops / bytes_moved`` (FLOP / byte)
* ``utilization_pct``         -- achieved / theoretical-peak * 100 (when
  the device + dtype are known via :attr:`Estimate.device` /
  :attr:`Estimate.dtype`)

Confidence tiers (sorted from most reliable to least):

* ``"calibrated"`` -- the cost function consulted a measured throughput
  table for this op_class on this device (see
  :data:`flashlib.info.roofline._SUSTAINED_TFLOPS`); end-to-end
  predicted runtime should be within ±20 % of wall-clock.
* ``"measured"``   -- the cost function hard-codes a runtime
  expression fitted to benchmark data for a specific shape regime.
* ``"roofline"``   -- vendor-peak * default-efficiency-class; coarse but
  consistent across primitives.
* ``"heuristic"``  -- guess (typically for first-call CuteDSL kernels
  whose compile cost dominates).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from flashlib.info.roofline import (
    canonicalize_dtype,
    get_bandwidth_peak,
    get_compute_peak,
)


@dataclass
class Estimate:
    """Resource estimate for a workload on a target device.

    Attributes:
        op_name: identifier of the op this estimate is for (e.g., ``"pca"``,
            ``"eigh_qdwh"``, ``"cov_gemm"``). Used for the call-stack tree
            print.
        runtime_ms, flops, bytes_moved, memory_peak_gb: summary numbers.
        bound: ``'compute' | 'memory' | 'mixed' | 'latency'``.
        confidence: ``'calibrated' | 'measured' | 'roofline' | 'heuristic'`` --
            see module docstring.
        n_kernel_launches: GPU launches expected.
        suggested_config: kernel hyperparameters (tile size, num_warps, ...).
        subops: per-sub-op breakdown -- a list of child :class:`Estimate`
            instances composing this op. Each child carries its own ``subops``.
        notes: free-form human-readable reasoning.
        expected_residual: relative residual / error bound for the op's
            output. ``None`` when the op has no precision knob.
        precision_tier: coarse label (``'exact' | 'mixed' | 'fast' | 'loose'``).
        tol: the tolerance value used to pick the variant for this estimate.
        dtype: canonical dtype the op was estimated under (set by the
            dispatcher when known). Powers :attr:`utilization_pct`.
        device: device key (e.g. ``"H200"``). Set by the dispatcher when
            known. Powers :attr:`utilization_pct`.
    """
    op_name: str = ""
    runtime_ms: float = 0.0
    flops: float = 0.0
    bytes_moved: float = 0.0
    memory_peak_gb: float = 0.0
    bound: str = "memory"
    confidence: str = "roofline"
    n_kernel_launches: int = 0
    suggested_config: dict = field(default_factory=dict)
    subops: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    expected_residual: Optional[float] = None
    precision_tier: Optional[str] = None
    tol: Optional[float] = None
    dtype: Optional[str] = None
    device: Optional[str] = None

    # ─── Derived performance numbers ──────────────────────────────────────

    @property
    def achieved_tflops(self) -> float:
        """Effective achieved throughput in TFLOPS.

        ``flops / (runtime_ms * 1e9)``. Zero when ``runtime_ms`` is 0.
        """
        if self.runtime_ms <= 0:
            return 0.0
        return self.flops / (self.runtime_ms * 1e9)

    @property
    def achieved_gbs(self) -> float:
        """Effective achieved bandwidth in GB/s.

        ``bytes_moved / (runtime_ms * 1e6)``.
        """
        if self.runtime_ms <= 0:
            return 0.0
        return self.bytes_moved / (self.runtime_ms * 1e6)

    @property
    def arithmetic_intensity(self) -> float:
        """FLOP / byte ratio; >= the device's balance ratio means compute-bound."""
        if self.bytes_moved <= 0:
            return float("inf")
        return self.flops / self.bytes_moved

    @property
    def utilization_pct(self) -> Optional[float]:
        """Percent of peak (compute or bandwidth, whichever bound).

        Requires :attr:`device` and :attr:`dtype` to be set. Returns
        ``None`` otherwise.
        """
        if self.device is None or self.runtime_ms <= 0:
            return None
        dtype = canonicalize_dtype(self.dtype or "fp32")
        if self.bound == "memory":
            peak_gbs = get_bandwidth_peak(self.device) * 1000.0  # TB/s -> GB/s
            return 100.0 * self.achieved_gbs / max(peak_gbs, 1e-6)
        # compute-bound (or mixed/latency: still report against compute peak)
        peak_tf = get_compute_peak(dtype, "gemm", self.device)
        return 100.0 * self.achieved_tflops / max(peak_tf, 1e-6)

    # ─── Serialisation / display ──────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        out = {
            "op_name": self.op_name,
            "runtime_ms": self.runtime_ms,
            "flops": self.flops,
            "bytes_moved": self.bytes_moved,
            "memory_peak_gb": self.memory_peak_gb,
            "bound": self.bound,
            "confidence": self.confidence,
            "n_kernel_launches": self.n_kernel_launches,
            "suggested_config": self.suggested_config,
            "expected_residual": self.expected_residual,
            "precision_tier": self.precision_tier,
            "tol": self.tol,
            "dtype": self.dtype,
            "device": self.device,
            "achieved_tflops": self.achieved_tflops,
            "achieved_gbs": self.achieved_gbs,
            "arithmetic_intensity": self.arithmetic_intensity,
            "utilization_pct": self.utilization_pct,
            "notes": list(self.notes),
        }
        if self.subops:
            out["subops"] = [s.to_dict() for s in self.subops]
        return out

    def __repr__(self) -> str:
        prec = ""
        if self.expected_residual is not None:
            prec = f", residual~{self.expected_residual:.0e}"
        if self.tol is not None:
            prec += f", tol={self.tol:.0e}"
        name = self.op_name or "?"
        achieved = ""
        if self.bound == "compute" and self.achieved_tflops > 0:
            achieved = f", {self.achieved_tflops:.0f} TF"
        elif self.bound == "memory" and self.achieved_gbs > 0:
            achieved = f", {self.achieved_gbs:.0f} GB/s"
        util = ""
        u = self.utilization_pct
        if u is not None:
            util = f" ({u:.0f}% peak)"
        return (
            f"Estimate({name!r}: runtime={self.runtime_ms:.2f}ms, "
            f"peak={self.memory_peak_gb:.2f}GB, bound={self.bound!r}"
            f"{achieved}{util}{prec})"
        )

    def format_tree(self, prefix: str = "", is_last: bool = True,
                    is_root: bool = True, show_residual: bool = True,
                    show_achieved: bool = True) -> str:
        """Format this Estimate as a call-stack tree, one line per sub-op.

        Args:
            show_residual: include ``res~`` and ``[tol=...]`` decorations.
            show_achieved: include effective TFLOPS / GB/s column on the
                right of each row.
        """
        if is_root:
            connector = ""
            extension = ""
        else:
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "

        name = self.op_name or "?"
        tol_tag = f" [tol={self.tol:.0e}]" if self.tol is not None else ""
        residual_tag = ""
        if show_residual and self.expected_residual is not None:
            residual_tag = f"  res~{self.expected_residual:.0e}"
        achieved_tag = ""
        if show_achieved:
            if self.bound == "compute" and self.achieved_tflops > 0:
                achieved_tag = f"  {self.achieved_tflops:>5.0f} TF"
            elif self.bound == "memory" and self.achieved_gbs > 0:
                achieved_tag = f"  {self.achieved_gbs:>5.0f} GB/s"
        line = (
            f"{prefix}{connector}{(name + tol_tag):<32}  "
            f"{self.runtime_ms:>8.2f} ms  "
            f"{self.memory_peak_gb:>5.2f} GB  "
            f"{self.bound:<7}{achieved_tag}{residual_tag}"
        )
        out = [line]
        n = len(self.subops)
        for i, child in enumerate(self.subops):
            child_prefix = prefix + extension
            out.append(child.format_tree(
                prefix=child_prefix, is_last=(i == n - 1),
                is_root=False, show_residual=show_residual,
                show_achieved=show_achieved,
            ))
        return "\n".join(out)

    def print_tree(self, **kw) -> None:
        """Convenience: print the call-stack tree to stdout."""
        print(self.format_tree(**kw))

    def summary_line(self) -> str:
        """One-line agent-friendly summary of the estimate."""
        name = self.op_name or "?"
        head = f"{name:<28} {self.runtime_ms:>7.2f} ms  bound={self.bound:<7}"
        if self.bound == "compute" and self.achieved_tflops > 0:
            head += f" {self.achieved_tflops:>5.0f} TF"
        elif self.bound == "memory" and self.achieved_gbs > 0:
            head += f" {self.achieved_gbs:>5.0f} GB/s"
        u = self.utilization_pct
        if u is not None:
            head += f"  ({u:>3.0f}% peak)"
        if self.expected_residual is not None:
            head += f"  res~{self.expected_residual:.0e}"
        head += f"  [{self.confidence}]"
        return head


@dataclass
class Variant:
    """One implementation of an op family, with its own cost estimate."""
    name: str
    estimate: Estimate

    def __repr__(self) -> str:
        e = self.estimate
        prec = ""
        if e.expected_residual is not None:
            prec = f"  residual~{e.expected_residual:.0e}"
        return f"Variant({self.name!r}: {e.runtime_ms:.1f}ms{prec})"


def is_pareto_optimal(target: Estimate, others: list[Estimate]) -> bool:
    """True iff no other Estimate dominates ``target`` on (runtime, residual).

    A variant with ``expected_residual=None`` is treated as "no precision
    knob" and is always considered Pareto-optimal (no one beats it on a
    dimension it doesn't expose).
    """
    if target.expected_residual is None:
        return True
    for o in others:
        if o is target:
            continue
        if o.expected_residual is None:
            continue
        rt_better_or_eq = o.runtime_ms <= target.runtime_ms
        prec_better_or_eq = o.expected_residual <= target.expected_residual
        rt_strict = o.runtime_ms < target.runtime_ms
        prec_strict = o.expected_residual < target.expected_residual
        if rt_better_or_eq and prec_better_or_eq and (rt_strict or prec_strict):
            return False
    return True
