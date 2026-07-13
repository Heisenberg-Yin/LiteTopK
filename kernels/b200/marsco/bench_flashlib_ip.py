"""flashlib IP-mode matrix: k=128 with B200 autotune, k=1024 heuristic.

The METRIC_IP patch (see _flashlib/flashlib/primitives/knn/triton/insert.py)
ranks by -x.c so flashlib measures the same IP retrieval problem as the other
baselines. K and M are Triton constexprs: the K=1024 variant recompiles per
corpus size and each compile takes ~10 min (tl.sort over [BN,1024] u64).
"""
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "_flashlib"))
sys.path.insert(0, os.path.join(_HERE, "_flashlib", "nvidia_cutlass_dsl", "python_packages"))

import numpy as np
import torch

from bench_marsco_b200 import load_base_fp16, read_fvecs, DATA, cuda_time, recall
from flashlib.primitives.knn import flash_knn


def main():
    qv, d = read_fvecs(os.path.join(DATA, "query.fvecs"), max_rows=64)
    q = torch.from_numpy(qv.astype(np.float16)).cuda().contiguous()
    for m in (1000000, 2000000, 4000000, 5000000):
        base = load_base_fp16(m)
        for k, tune in ((128, True), (1024, False)):
            tag = "flashlib-IP-autotuned" if tune else "flashlib-IP"
            try:
                def fn():
                    return flash_knn(q, base, k, return_distances=False,
                                     metric="ip", autotune=tune)
                got = fn()
                torch.cuda.synchronize()
                ref = (q @ base.t()).topk(k, dim=-1).indices
                rec = recall(got, ref)
                t = cuda_time(fn, 2, 10 if k >= 1024 else 20)
                print(f"{tag} m={m} k={k}: {t:.4f} ms  recall={rec:.4f}", flush=True)
            except Exception:
                print(f"{tag} m={m} k={k}: FAILED", flush=True)
                traceback.print_exc()
        del base
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
