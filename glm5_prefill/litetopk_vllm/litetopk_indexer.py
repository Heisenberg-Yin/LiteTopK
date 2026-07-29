#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LiteTopK fused sparse top-k indexer for vLLM's DSA prefill path.

Replaces (per prefill chunk, when the context is long enough):
    logits = deep_gemm.fp8_fp4_mqa_logits(...)   # full [Q, S] materialized
    ops.top_k_per_row_prefill(logits, ...)       # reads it all back
with:
    exact hot-subset scoring (official kernel, [Q, SAMPLE]) -> threshold
    fused sparse scan (LiteTopK litetopk kernel, only candidates written)
    compact threshold-aware radix select -> top-k indices

Env knobs:
  VLLM_LITETOPK=1            enable
  VLLM_LITETOPK_MIN_S        min gathered context length to engage (default 262144)
  VLLM_LITETOPK_SAMPLE       sample prefix length (default 65536)
  VLLM_LITETOPK_CAP          candidate buffer width per row (default 131072)
  VLLM_LITETOPK_CHECK=1      also run the official path and log top-k recall
"""
import os

import torch

# Resolve the in-repository CUDA source by default; an installed integration
# may still point at a separate source tree with LITETOPK_DSA_DIR.
_REPO_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_DSA_DIR = os.environ.get(
    "LITETOPK_DSA_DIR",
    os.path.join(_REPO_DIR, "kernels", "b200", "dsa"))
_BUILD_DIR = os.environ.get(
    "VLLM_LITETOPK_BUILD",
    os.path.expanduser("~/.cache/litetopk_build"))

ENABLED = os.environ.get("VLLM_LITETOPK", "0") == "1"
MIN_S = int(os.environ.get("VLLM_LITETOPK_MIN_S", "262144"))
SAMPLE = int(os.environ.get("VLLM_LITETOPK_SAMPLE", "65536"))
CAP = int(os.environ.get("VLLM_LITETOPK_CAP", "131072"))
NB = int(os.environ.get("VLLM_LITETOPK_NB", "256"))
# The CUDA source fixes the numerical-FP16 Gate4 W256/L18 sparse-refresh
# contract. Refresh cadence is part of that supported ABI, not a runtime knob.
REFRESH = 64
CHECK = os.environ.get("VLLM_LITETOPK_CHECK", "0") == "1"
# Merged-chunk mode scans a whole prefill step in one call. MERGE_CAP bounds
# the per-row candidate buffer for these large-Q calls.
MERGE = os.environ.get("VLLM_LITETOPK_MERGE", "0") == "1"
# Strided threshold probe: sample across the global score range instead of a
# prefix. No seeds are emitted; the full scan uses it only to set o/inv/th.
STRIDED = os.environ.get("VLLM_LITETOPK_STRIDED", "0") == "1"
# Threshold probe: append PROBE extra sampled columns (from beyond the head
# window) to the sample, used ONLY for the bucket scale + histogram — never
# emitted as seeds (kernel emit_limit), so the scan still covers them exactly
# once and exactness is preserved.
PROBE = int(os.environ.get("VLLM_LITETOPK_PROBE", "0"))
# Strided probe size for STRIDED/auto mode.
SSAMPLE = int(os.environ.get("VLLM_LITETOPK_SSAMPLE", "16384"))
# Auto-escalation: switch large-Q chunks to the strided probe (sticky) when
# the deferred feedback sees the prefix threshold emitting > this many x K.
# In the exact-threshold regime the strided flip is performance-neutral, so
# this is purely a CAP-overflow guard (cap/K = 64): fire only on genuine
# blowups, not on ordinary looseness.
AUTO_XK = float(os.environ.get("VLLM_LITETOPK_AUTO_XK", "12.0"))
# Probe gathers whole 64-token pages (paged-attention compatible) instead of
# single strided columns; simulated equivalent, more coalesced.
PAGE_PROBE = os.environ.get("VLLM_LITETOPK_PAGE_PROBE", "1") == "1"
# Two-step strided sample: half the page budget is uniform and half is
# densified around hot anchors. The union remains a genuine row subset, so
# the kq=K threshold remains an exact bound.
# Seeds come only from the uniform half (arithmetic index map); the dense
# half stays in the scan workspace (re-scanned, no duplicate candidates)
# and its histogram contribution is subtracted to keep refresh exact.
TWOSTEP = os.environ.get("VLLM_LITETOPK_TWOSTEP", "0") == "1"
_AUTO = {"strided": False, "n": 0}
# Absolute forward headroom on the bucket scale (fraction of the sample span
# prepended ABOVE the sample max). Pair with a proportionally larger NB to
# keep bucket width unchanged (e.g. HEADROOM=1.0 + NB=512 == today's width).
HEADROOM = float(os.environ.get("VLLM_LITETOPK_HEADROOM", "0.0"))
MERGE_CAP = int(os.environ.get("VLLM_LITETOPK_MERGE_CAP", "32768"))
MEMSTATS = os.environ.get("VLLM_LITETOPK_MEMSTATS", "0") == "1"
# OVF_LOG: print the running max of sampled per-row candidate counts (from
# the existing deferred 1-in-8 probe; sync-free). Sizes MERGE_CAP.
OVF_LOG = os.environ.get("VLLM_LITETOPK_OVF_LOG", "0") == "1"
# HOT_PREFETCH runs part-0's hot selection on a side stream. Dependencies are
# event-precise: the carry-write event plus the workspace gather event.
HOT_PREFETCH = os.environ.get("VLLM_LITETOPK_HOT_PREFETCH", "0") == "1"
_HOT_STREAM = {}
PROBE_EVERY = int(os.environ.get("VLLM_LITETOPK_PROBE_EVERY", "8"))
# Probe-page compaction (strided mode): the workspace passed to the scan has
# the 256 probe pages REMOVED (they were already scored by the probe, which
# now emits them as seeds with original indices) — the scan covers S-16K
# columns and maps emitted indices back to original space in-kernel.
COMPACT = os.environ.get("VLLM_LITETOPK_COMPACT", "0") == "1"
# Prefix-mode prep subsampling affects threshold estimation only; emit still
# reads every element. It is disabled by default.
PREP_SUB = int(os.environ.get("VLLM_LITETOPK_PREP_SUB", "1"))
# Row-tile the sample GEMM + seed_prep pair to bound transient memory. Zero
# disables tiling.
PREP_TILE = int(os.environ.get("VLLM_LITETOPK_PREP_TILE", "2048"))
# Split big merged calls into QSPLIT-row scans: candidate/scratch buffers
# scale with rows-per-scan (8192 -> 2048 = 4x smaller), total scan work
# unchanged (same CTA count overall). Certificate stays valid at 2048 by
# FORCING num_kv_splits=1 (512 CTAs still cover 148 SMs; the 2368 floor
# only existed because of the wrapper's 4-wave auto-split heuristic).
QSPLIT = int(os.environ.get("VLLM_LITETOPK_QSPLIT", "0"))
# Enriched sampling uses positions from the previous part's top-k indices.
# Any selected position set still gives a valid exact-subset bound. Zero
# disables it; a positive value is the hot-column budget.
HOTSAMPLE = int(os.environ.get("VLLM_LITETOPK_HOTSAMPLE", "8192"))
# HOTONLY: the hot columns ARE the whole sample (no uniform probe). The
# subset bound stays provable for any chosen sample; drift discovery is
# carried by the scan->select->carry loop itself. Probe survives only as
# the cold-start fallback (hot_prev is None).
HOTONLY = os.environ.get("VLLM_LITETOPK_HOTONLY", "1") == "1"
# HOTLAST uses the last row's naturally deduplicated top-k as the sample.
HOTLAST = os.environ.get("VLLM_LITETOPK_HOTLAST", "0") == "1"
_HOT_CARRY = {}
# GATE4 builds write BUCKET-SPACE floats as cand_val (affine
# order-preserving); select must run with o'=0, inv'=1, th'=(th-o)*inv.
# Valid only for emit_lim==0 modes (prefix seeds would be x-space = mixed).

PREP_MARGIN = float(os.environ.get("VLLM_LITETOPK_PREP_MARGIN", "1.15"))
_COMPACT_IDX = {}  # (dev, S, pstp, npage) -> kept-position index tensor

_EXT = None
_FAILED = False
_AUX_CACHE = {}  # (device, head) -> (zeros[Qmax], full_head[Qmax]) int32


def _ks0_keh(Q, head, dev):
    """Cached zero-starts and sample-end tensors: torch.zeros/torch.full are a
    kernel launch each and measurably cost ~0.1-0.2ms/chunk on the hot path."""
    key = (str(dev), head)
    entry = _AUX_CACHE.get(key)
    if entry is None or entry[0].shape[0] < Q:
        qmax = max(Q, 1024)
        entry = (torch.zeros(qmax, dtype=torch.int32, device=dev),
                 torch.full((qmax,), head, dtype=torch.int32, device=dev))
        _AUX_CACHE[key] = entry
    return entry[0][:Q], entry[1][:Q]


def _ext():
    global _EXT, _FAILED
    if _EXT is None and not _FAILED:
        try:
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")
            from torch.utils.cpp_extension import load
            dg25 = os.environ.get("DEEPGEMM_DIR", "/opt/glm5_prefill_test/DeepGEMM")
            name = "litetopk_dsa_b200_v3_fp16_w256_l18"
            src = "dsa_litetopk.cu"
            bdir = f"{_BUILD_DIR}_v3_fp16_w256_l18"
            dg_inc = os.path.join(dg25, "deep_gemm/include")
            cutlass_inc = os.path.join(
                dg25, "third-party/cutlass/include")
            if os.path.isdir(dg_inc) and os.path.isdir(cutlass_inc):
                incs = [_DSA_DIR, dg_inc, cutlass_inc]
            else:
                # A pinned DeepGEMM wheel may bundle both DeepGEMM and CUTLASS
                # headers under its package include directory.
                import deep_gemm

                pkg_inc = os.path.join(
                    os.path.dirname(deep_gemm.__file__), "include")
                if not os.path.isfile(
                        os.path.join(pkg_inc, "cutlass/arch/barrier.h")):
                    raise RuntimeError(
                        "DeepGEMM/CUTLASS headers not found; set "
                        "DEEPGEMM_DIR to a DeepGEMM 2.5 checkout")
                incs = [_DSA_DIR, pkg_inc]
            cuda_flags = [
                "-O3", "-std=c++17", "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-gencode=arch=compute_100a,code=sm_100a",
            ]
            if os.environ.get("VLLM_LITETOPK_LINEINFO", "0") == "1":
                cuda_flags.append("-lineinfo")
                name, bdir = name + "_li", bdir + "_li"
            os.makedirs(bdir, exist_ok=True)
            _EXT = load(
                name=name,
                sources=[os.path.join(_DSA_DIR, src)],
                extra_include_paths=incs,
                extra_cuda_cflags=cuda_flags,
                build_directory=bdir,
                extra_ldflags=["-lcuda"],
                verbose=False,
            )
            print("[litetopk] using v3 fp16_w256_l18 kernel", flush=True)
        except Exception as e:  # noqa: BLE001
            _FAILED = True
            print(f"[litetopk] extension build failed, falling back: {e}", flush=True)
    return _EXT


def plan_sampling(Q, ke_min):
    """Sampling policy, shared by try_chunk AND the container's merged path
    (which must size the probe pre-gather before calling in). Returns
    (use_strided, probe_head). The policy uses a representative probe for
    long contexts; the overflow guard can also force strided sampling."""
    use_strided = plan_strided(Q)
    if not use_strided and Q >= 2048 and ke_min >= 262_144:
        use_strided = True
    head = SSAMPLE
    if use_strided and "VLLM_LITETOPK_SSAMPLE" not in os.environ \
            and ke_min >= 262_144:
        head = 65536
    return use_strided, head


def plan_strided(Q):
    """Single source of truth for the strided-vs-prefix decision, including
    the auto drift state and the every-33rd exploration flip. The vLLM
    container calls this BEFORE gathering (compacted vs full workspace) and
    passes the decision back via strided_plan= so the counter advances once."""
    use_strided = (STRIDED or _AUTO["strided"]) and Q >= 2048
    if use_strided and not STRIDED:
        _AUTO["n"] += 1
        if _AUTO["n"] % 33 == 0:
            use_strided = False
    return use_strided


_HINTS_VALIDATED = False
_PENDING_OVF = None  # (cuda event, pinned int32 tensor) deferred overflow probe
_PROBE_RES = None    # cached (pinned buffer, event): allocating these per
                     # arming blocked the CPU ~17ms inside cudaHostAlloc-class
                     # calls (nsys), starving the GPU stream for ~3.4ms/call
                     # at 256K/Q=512 when arming was throttled


def _deferred_overflow_poll():
    """Non-blocking check of the previous chunk's candidate-overflow probe."""
    global _PENDING_OVF
    if _PENDING_OVF is not None:
        ev, pinned, capv, kk, was_strided = _PENDING_OVF
        if ev.query():  # finished long ago; no sync
            mx = int(pinned[0])
            if mx > capv:
                print(f"[litetopk] WARNING: candidate overflow ({mx} > cap {capv}); "
                      f"recall may dip on that chunk — raise VLLM_LITETOPK_CAP", flush=True)
            if OVF_LOG and mx > _AUTO.get("mxmax", 0):
                # running-max telemetry (sync-free: pinned value already read);
                # a handful of prints per run, sizes the cap for memory work
                _AUTO["mxmax"] = mx
                print(f"[litetopk] cand max -> {mx} "
                      f"(mean {float(pinned[1]) / kk:.2f}xK)", flush=True)
            mean_xk = float(pinned[1]) / kk
            if not was_strided:  # honest prefix-mode reading
                if not _AUTO["strided"] and mean_xk > AUTO_XK:
                    _AUTO["strided"] = True
                    print(f"[litetopk] drift detected (cand {mean_xk:.1f}xK): "
                          f"large-Q chunks -> strided threshold probe", flush=True)
                elif _AUTO["strided"] and mean_xk < 0.8 * AUTO_XK:
                    _AUTO["strided"] = False  # drift regime ended
                    print(f"[litetopk] drift cleared (cand {mean_xk:.1f}xK): "
                          f"back to prefix sampling", flush=True)
            _PENDING_OVF = None


_ARANGE = {}
_MEM_SEEN = set()
_PREP_BUFS = {}  # (dev, NB) -> dict of caller-owned seed_prep buffers
_CAND_BUFS = {}  # dev -> numerical-FP16 candidate slab
_VOTE_BUF = {}      # dev -> persistent int32 vote-histogram slab (main stream)
_VOTE_BUF_HOT = {}  # dev -> separate slab for the HOT_PREFETCH side stream
                     # (must NOT alias the main-stream slab: concurrent
                     # zero_/scatter_add_ on two streams would race)


def _vote_hist(nv, dev, hot=False):
    """Reused int32 vote histogram over [0, nv). nv tracks the live prefix
    length S (up to max_model_len), so a fresh torch.zeros(nv,...) every
    try_chunk call means a shape that grows every step -- keep one
    persistent slab per device (two when the HOT_PREFETCH side stream is in
    play) and zero only the live [:nv] prefix each call."""
    cache = _VOTE_BUF_HOT if hot else _VOTE_BUF
    key = str(dev)
    b = cache.get(key)
    if b is None or b.numel() < nv:
        b = torch.empty(max(nv, 1024), dtype=torch.int32, device=dev)
        cache[key] = b
    buf = b[:nv]
    buf.zero_()
    return buf


def _cand_bufs(Q, cap, dev):
    key = str(dev)
    b = _CAND_BUFS.get(key)
    if b is None or b["cap"] != cap or b["q"] < Q:
        _CAND_BUFS[key] = None  # drop old slab BEFORE allocating the new one
        qm = max(Q, 1024)
        b = {"q": qm, "cap": cap,
             # float16 is opaque 16-bit storage in the packed ABI; CUDA
             # interprets its bits as uint16 rather than doing half arithmetic.
             "cv": torch.empty(qm, cap, dtype=torch.float16, device=dev),
             "ci": torch.empty(qm, cap, dtype=torch.int32, device=dev)}
        _CAND_BUFS[key] = b
    return b


def _prep_bufs(Q, nb, cap, dev):
    """Caller-owned seed_prep outputs. Reusing these kills the 0.1-0.5GB
    per-call alloc churn that forced the CUDA allocator into pathological
    behavior at small Q (256K: 4.5ms without an event-paced probe)."""
    key = (str(dev), nb)
    b = _PREP_BUFS.get(key)
    if b is None or b["q"] < Q:
        qm = max(Q, 1024)
        b = {"q": qm,
             "o": torch.empty(qm, device=dev),
             "inv": torch.empty(qm, device=dev),
             "th": torch.empty(qm, dtype=torch.int32, device=dev),
             "bc": torch.empty(qm, nb, dtype=torch.int32, device=dev),
             "cc": torch.empty(qm, dtype=torch.int32, device=dev)}
        _PREP_BUFS[key] = b
    return b


def _arange32(n, dev):
    key = str(dev)
    a = _ARANGE.get(key)
    if a is None or a.shape[0] < n:
        a = torch.arange(max(n, 8192), dtype=torch.int32, device=dev)
        _ARANGE[key] = a
    return a[:n]


def stash_carry(hot_key, idx, S):
    """Seed a layer's hot carry from the OFFICIAL path's topk output, called
    by the container on the LAST official chunk before MIN_S. The
    official->ours boundary is deterministic, so this one seed is all the
    first ours-chunk needs to run HOT (no cold start, no cold prefix). Stored
    compressed (voted hot columns, ~64KB/layer).

    The vote+topk selection and the store run on a per-device SIDE STREAM
    (async): seeding overlaps the model forward instead of stalling the
    official path. The consumer (try_chunk's carry read) waits on the stored
    event before touching the carry."""
    if not (HOTONLY and HOTSAMPLE > 0):
        return
    dev = idx.device
    nv = int(S)
    ss = _HOT_STREAM.get(str(dev))
    if ss is None:
        ss = torch.cuda.Stream(device=dev)
        _HOT_STREAM[str(dev)] = ss
    ss.wait_stream(torch.cuda.current_stream())  # see the just-written topk
    idx.record_stream(ss)                        # keep it alive for the read
    with torch.cuda.stream(ss):
        # separate slab (_vote_hist hot=True): the main stream reuses its own
        # slab, so a shared one would race the concurrent zero_/scatter_add_.
        votes = _vote_hist(nv, dev, hot=True)
        hpf = idx.reshape(-1).long().clamp_(0, nv - 1)
        votes.scatter_add_(0, hpf, torch.ones_like(hpf, dtype=torch.int32))
        hot = votes.topk(min(HOTSAMPLE, nv)).indices
        ev = torch.cuda.Event()
        ev.record()
    _HOT_CARRY[(str(dev), hot_key)] = (hot, nv, ev)


def try_merged(q, k, k_scale, weights, out_idx, topk, S, Qtot,
               probe_k=None, probe_scale=None, gather_event=None,
               strided_plan=None, pre_compacted=False, hot_key=None) -> bool:
    """One call for the whole prefill step (single-request causal):
    ks = 0, ke = S - Qtot + 1 + row. Gathers must already be done."""
    if not MERGE or Qtot < 2048:
        return False
    dev = q.device
    ks = _ks0_keh(Qtot, SAMPLE, dev)[0]
    ke = (S - Qtot + 1) + _arange32(Qtot, dev)
    if MEMSTATS and (Qtot, S) not in _MEM_SEEN:
        _MEM_SEEN.add((Qtot, S))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        ok = try_chunk(q, k, k_scale, weights, ks, ke, out_idx, topk,
                       num_reqs=1, ke_min_hint=S - Qtot + 1, cap=MERGE_CAP,
                       probe_k=probe_k, probe_scale=probe_scale,
                       gather_event=gather_event,
                       strided_plan=strided_plan, pre_compacted=pre_compacted,
                       hot_key=hot_key)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        print(f"[litetopk] merged Q={Qtot} S={S} ok={ok} "
              f"mem_overhead_peak={(peak - base) / 2**30:.2f} GiB", flush=True)
        return ok
    return try_chunk(q, k, k_scale, weights, ks, ke, out_idx, topk,
                     num_reqs=1, ke_min_hint=S - Qtot + 1, cap=MERGE_CAP,
                     probe_k=probe_k, probe_scale=probe_scale,
                     gather_event=gather_event,
                     strided_plan=strided_plan, pre_compacted=pre_compacted,
                     hot_key=hot_key)


def try_chunk(q, k, k_scale, weights, ks, ke, out_idx, topk,
              num_reqs=None, ke_min_hint=None, cap=None,
              probe_k=None, probe_scale=None, gather_event=None,
              strided_plan=None, pre_compacted=False, hot_prev=None,
              hot_key=None, hot_pre=None, _carry_io=True) -> bool:
    """Fill out_idx [Q, topk] with the per-row top-k indices; True on success.

    Falls back (returns False) whenever an assumption does not hold; the
    caller then runs the official dense-logits path.

    num_reqs / ke_min_hint: CPU-side hints from the chunk metadata. When
    provided (vLLM path), the GPU-tensor guards (`ks.max()`, `ke.min()`) are
    skipped entirely -- no `.item()` device syncs on the hot path. The first
    engaged chunk still validates the hints against the tensors once.
    """
    global _HINTS_VALIDATED, _PENDING_OVF
    try:
        if q.dim() != 3 or q.shape[1] != 32 or q.shape[2] != 128:
            return False
        Q = q.shape[0]
        # The exact-subset gate contract is certified for merged Q=8192
        # calls. Smaller calls use the official dense fallback.
        if Q < 8192:
            return False
        S = k.shape[0]
        if S < MIN_S and not pre_compacted:
            return False
        if _carry_io and HOTSAMPLE > 0 and hot_prev is None:
            # UNSPLIT direct call: read the per-layer carry here (split
            # calls get it via the QSPLIT wrapper). Same staleness guard.
            _kem2 = int(ke_min_hint) if ke_min_hint is not None \
                else S - q.shape[0] + 1
            _hc2 = _HOT_CARRY.get((str(q.device), hot_key))
            if _hc2 is not None and (_hc2[1] <= _kem2 or _hc2[1] == S):
                hot_prev = _hc2[0]
                if _hc2[2] is not None:
                    # async boundary seed (stash_carry side stream): order the
                    # main stream behind the seed's topk and keep it alive.
                    torch.cuda.current_stream().wait_event(_hc2[2])
                    hot_prev.record_stream(torch.cuda.current_stream())
        head = SAMPLE
        if num_reqs is not None and ke_min_hint is not None:
            if num_reqs != 1:
                return False  # multi-request chunk: ks offsets nonzero
            if not _HINTS_VALIDATED:  # one-time sanity sync, then trust the hints
                assert int(ks.max().item()) == 0, "ks!=0 despite num_reqs==1"
                real_ke_min = int(ke.min().item())
                assert real_ke_min == ke_min_hint, \
                    f"ke_min hint {ke_min_hint} != actual {real_ke_min}"
                _HINTS_VALIDATED = True
                print("[litetopk] CPU hints validated; sync-free path active", flush=True)
        else:
            if int(ks.max().item()) != 0:
                return False  # sliding-window starts not supported by the sample stage
        # There is intentionally no SAMPLE-sized strict-prefix guard here.
        # This code is hot-only: after ke_min is known below, the exact carry
        # indices are filtered to the range shared by every causal row and the
        # resulting (usually 8192-column) subset is validated against topk.
        _deferred_overflow_poll()
        ext = _ext()
        if ext is None:
            return False

        import deep_gemm  # DeepGEMM 2.5 (vLLM-pinned) for the sample scoring

        dev = q.device
        if not q.is_contiguous():
            q = q.contiguous()
        if weights.dtype != torch.float32:
            weights = weights.float()
        if not weights.is_contiguous():
            weights = weights.contiguous()
        ke_min = ke_min_hint if ke_min_hint is not None else int(ke.min().item())
        if ke_min <= topk:
            return False
        # The hot path uses positions carried from the previous chunk/layer.
        # Their Kth score is an exact subset bound; cold starts fall back to
        # the official path until a carry is available.
        use_strided = (HOTONLY and HOTSAMPLE > 0 and hot_prev is not None
                       and Q >= 2048)
        if use_strided:
            head = SSAMPLE
            if "VLLM_LITETOPK_SSAMPLE" not in os.environ \
                    and ke_min >= 262_144:
                head = 65536  # big representative probe for the drift band
            if probe_k is not None:
                head = probe_k.shape[0]  # container pre-gathered: its size wins
        elif "VLLM_LITETOPK_SAMPLE" not in os.environ:
            # Size-dependent exact-subset sample size.
            head = 131072 if 400_000 <= ke_min < 900_000 else 65536
        ks0, keh = _ks0_keh(Q, head, dev)
        stp = max((ke_min - topk) // head, 1) if use_strided else 1
        probe_extra = 0
        pstp_c = npage_c = 0
        if use_strided and stp > 1:
            npage_c = head // 64
            pstp_c = max(((ke_min - topk) // 64) // npage_c, 1)
        hot_here = (use_strided and stp > 1 and HOTONLY and HOTSAMPLE > 0
                    and hot_prev is not None)
        if hot_here:
            # HOT-ONLY sample: the prev part's vote-top columns are the whole
            # sample (no uniform probe, no dedup needed). Sample GEMM shrinks
            # 68K -> 4K columns; emit_lim stays 0 (hist/th only, scan covers
            # everything once). Subset bound provable as ever.
            if hot_pre is not None:
                # selection prefetched on the side stream (overlapped with
                # the previous layer's scan); consume with event ordering +
                # record_stream (cross-stream allocator safety).
                torch.cuda.current_stream().wait_event(hot_pre[2])
                hot_pre[0].record_stream(torch.cuda.current_stream())
                hot_pre[1].record_stream(torch.cuda.current_stream())
                _smp = (hot_pre[0], hot_pre[1])
                slog = None
                head = int(hot_pre[0].shape[0])
                ks_scan = ks0
            elif hot_prev.dim() == 1:
                # pre-voted carry (compressed store / official seeding):
                # the hot columns ARE these indices; no votes needed.
                # Filter out-of-range entries rather than clamping them:
                # clamping creates duplicates and can make the histogram
                # illegally tight.
                nv = int(ke_min)
                hot_idx = hot_prev[hot_prev < nv]
            elif HOTLAST:
                nv = int(ke_min)
                hot_idx = hot_prev[-1].long().clamp(0, nv - 1)  # unique by construction
            else:
                nv = int(ke_min)
                votes = _vote_hist(nv, dev)
                hpf = hot_prev.reshape(-1).long().clamp_(0, nv - 1)
                votes.scatter_add_(0, hpf, torch.ones_like(hpf, dtype=torch.int32))
                hot_idx = votes.topk(HOTSAMPLE).indices
            if hot_pre is None:
                _smp = (k.index_select(0, hot_idx), k_scale.index_select(0, hot_idx))
                slog = None
                head = int(hot_idx.shape[0])
                ks_scan = ks0  # scan the FULL range
        else:
            # This is a pure hot-start path. Without a carry there is no
            # certified sample, so defer to the official dense implementation.
            return False
        if head < topk:
            return False
        cap_eff = cap if cap is not None else CAP
        probe_group = probe_add_max = 0  # legacy paths never set these

        def _slog_rows(r0, r1):
            return deep_gemm.fp8_fp4_mqa_logits(
                (q[r0:r1], None), _smp, weights[r0:r1],
                *_ks0_keh(r1 - r0, head, dev), clean_logits=False)
        tile = PREP_TILE if (PREP_TILE > 0 and not TWOSTEP) else 0
        if slog is None and (tile == 0 or tile >= Q):
            slog, tile = _slog_rows(0, Q), 0
        # Fused v3 prep derives the bucket scale, histogram, and threshold.
        kq = topk
        hist_stride = 1
        if use_strided and stp > 1:
            # Exact subset bound: probe columns are a true subset of the row's
            # range, so the probe's topk-th value cannot be tighter than the
            # global topk-th. Sparse refresh pays down the loose initial gate.
            kq = topk
        elif PREP_SUB > 1 and head >= 32768:
            # prefix mode: subsampled threshold estimation
            hist_stride = PREP_SUB
            kq = max(int(topk / PREP_SUB * PREP_MARGIN), 128)
        k_scan, ksc_scan, ke_scan = k, k_scale, ke
        probe_group = probe_add_max = probe_stride_tok = 0
        if pre_compacted and pstp_c >= 2:
            # workspace was gathered COMPACTED by the container (probe
            # pages skipped at gather time): no index_select needed.
            g64 = (pstp_c - 1) * 64
            probe_add_max = npage_c * 64
            k_scan, ksc_scan = k, k_scale
            ke_scan = ke - probe_add_max
            probe_group = g64
            probe_stride_tok = pstp_c * 64
        elif use_strided and stp > 1 and COMPACT and pstp_c >= 2:
            # Remove the probe pages from the scan workspace: the probe
            # emits them as seeds (original indices via probe_stride_tok),
            # the scan covers S-16K columns, emitted indices map back
            # in-kernel. Saves the 1.6-3.1% probe-page rescan.
            g64 = (pstp_c - 1) * 64
            probe_add_max = npage_c * 64
            ckey = (str(dev), S, pstp_c, npage_c)
            kidx = _COMPACT_IDX.get(ckey)
            paged = (S % 64) == 0
            if kidx is None:
                if paged:
                    # page-level index: 64x smaller table, 8KB-contiguous
                    # segment copies instead of 128B row gathers
                    npg = S // 64
                    keep = torch.ones(npg, dtype=torch.bool, device=dev)
                    keep[torch.arange(npage_c, device=dev, dtype=torch.int64)
                         * pstp_c] = False
                    kidx = keep.nonzero(as_tuple=False).squeeze(1)
                else:
                    base = torch.arange(
                        npage_c * g64, device=dev, dtype=torch.int64)
                    part1 = base + 64 * (base // g64 + 1)
                    tail = torch.arange(
                        npage_c * pstp_c * 64, S, device=dev,
                        dtype=torch.int64)
                    kidx = torch.cat([part1, tail])
                _COMPACT_IDX[ckey] = kidx
            if paged:
                npg = S // 64
                k_scan = (
                    k.view(npg, 64, k.shape[1])
                    .index_select(0, kidx)
                    .view(-1, k.shape[1])
                )
                ksc_scan = (
                    k_scale.view(npg, 64)
                    .index_select(0, kidx)
                    .view(-1)
                )
            else:
                k_scan = k.index_select(0, kidx)
                ksc_scan = k_scale.index_select(0, kidx)
            ke_scan = ke - probe_add_max  # every probe page < ke_min <= ke
            probe_group = g64
            probe_stride_tok = pstp_c * 64
        # probe mode without compaction: no seeds (scan covers probes);
        # WITH compaction: seeds ON (probe_stride_tok maps their indices)
        emit_lim = head if (probe_stride_tok > 0
                            or not (use_strided and stp > 1)) else 0
        _b = _prep_bufs(Q, NB, cap_eff, dev)
        _cb = _cand_bufs(Q, cap_eff, dev)
        o, inv, th = _b["o"][:Q], _b["inv"][:Q], _b["th"][:Q]
        bcount = _b["bc"][:Q]
        cand_val = _cb["cv"][:Q]
        cand_idx = _cb["ci"][:Q]
        cand_cnt = _b["cc"][:Q]
        if tile:
            # row-tiled: slog lives only TILE rows at a time; the
            # freed tile is reused by the allocator next iteration
            for r0 in range(0, Q, tile):
                r1 = min(r0 + tile, Q)
                ext.seed_prep_litetopk_(
                    _slog_rows(r0, r1), NB, kq, cap_eff, emit_lim,
                    HEADROOM, probe_stride_tok, hist_stride,
                    o[r0:r1], inv[r0:r1], th[r0:r1], bcount[r0:r1],
                    cand_val[r0:r1], cand_idx[r0:r1], cand_cnt[r0:r1])
        else:
            ext.seed_prep_litetopk_(
                slog, NB, kq, cap_eff, emit_lim, HEADROOM,
                probe_stride_tok, hist_stride,
                o, inv, th, bcount, cand_val, cand_idx, cand_cnt)
        if use_strided and stp > 1 and PAGE_PROBE and TWOSTEP:
            # dense-half columns are re-scanned by the scan kernel:
            # remove their histogram contribution so refresh counts
            # each position exactly once (recall-safe: subtraction
            # can only loosen the refresh threshold... it removes a
            # genuine count, keeping totals exact, not loose).
            pb = ((slog[:, head:].neg() - o.view(-1, 1))
                  * inv.view(-1, 1)).floor_().clamp_(
                      0, NB - 1).to(torch.int64)
            bcount.scatter_add_(
                1, pb, torch.full(
                    pb.shape, -1, dtype=torch.int32, device=dev))
        if probe_extra:
            # Exact double-count removal: probe columns live in the scan
            # range and will be re-counted there; subtract their histogram
            # contribution so refresh sees each position exactly once.
            pb = ((slog[:, head:].neg() - o.view(-1, 1))
                  * inv.view(-1, 1)).floor_().clamp_(
                      0, NB - 1).to(torch.int64)
            bcount.scatter_add_(
                1, pb, torch.full(
                    pb.shape, -1, dtype=torch.int32, device=dev))
            probe_extra = 0
        # Probe mode (emit_lim=0): pass 3 is skipped. The production
        # zero-base scan initializes its CTA-local histogram directly,
        # so seed_prep also skips the dead global bcount zero write.
        if gather_event is not None:
            torch.cuda.current_stream().wait_event(gather_event)
        ke_scan_c = ke_scan.contiguous()
        cv, ci, cc = ext.mqa_logits_dsa_litetopk_ext(
            q, k_scan, ksc_scan, weights, ks_scan, ke_scan_c,
            o, inv, th, cand_val, cand_idx, cand_cnt, bcount, NB,
            topk + probe_extra, REFRESH, -1,
            probe_group, probe_add_max)
        # Overflow safety without a device sync: writes and selection clamp to
        # the cap, while a deferred probe reports attempted overflows.
        _AUTO["m"] = _AUTO.get("m", 0) + 1
        # Arm on the first call so helper cubins load during warmup.
        if _PENDING_OVF is None and ((_AUTO["m"] % PROBE_EVERY) == 0
                                     or _AUTO["m"] == 1):
            global _PROBE_RES
            if _PROBE_RES is None:
                _PROBE_RES = (torch.empty(2, dtype=torch.int32, pin_memory=True),
                              torch.cuda.Event())
            pinned, ev = _PROBE_RES
            pinned.copy_(torch.stack([cc.max(), cc.float().mean().int()]),
                         non_blocking=True)
            ev.record()
            _PENDING_OVF = (ev, pinned, cap_eff, topk, stp > 1)
        # Gate4 values are already in bucket space. The fixed in-place boundary
        # selector writes indices directly into the caller-owned output.
        ext.compact_topk_min_thr_inplace_idx_out_litetopk(
            cv, ci, cc, th, bcount, NB, topk, out_idx)
        idx = out_idx
        if CHECK:
            lg = deep_gemm.fp8_fp4_mqa_logits(
                (q, None), (k, k_scale), weights, ks.contiguous(),
                ke.contiguous(), clean_logits=True)
            ref = lg.topk(topk, dim=1).indices
            refs, _ = ref.sort(dim=1)
            p = torch.searchsorted(refs, idx.long().sort(dim=1).values)
            p = p.clamp(max=topk - 1)
            rec = (torch.gather(refs, 1, p) == idx.long().sort(dim=1).values).float().mean()
            print(f"[litetopk] chunk Q={Q} S={S} recall={100 * rec.item():.3f}%", flush=True)

        if _carry_io and HOTONLY and HOTSAMPLE > 0:
            # UNSPLIT calls own their carry here (split calls: the QSPLIT
            # wrapper does it around the part loop). Compressed store, same
            # semantics as the wrapper's.
            _nv = int(k.shape[0])
            _vt = _vote_hist(_nv, dev)
            _hf = idx.reshape(-1).long().clamp_(0, _nv - 1)
            _vt.scatter_add_(0, _hf, torch.ones_like(_hf, dtype=torch.int32))
            _HOT_CARRY[(str(dev), hot_key)] = (
                _vt.topk(min(HOTSAMPLE, _nv)).indices, _nv, None)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[litetopk] chunk fallback due to: {e}", flush=True)
        return False
