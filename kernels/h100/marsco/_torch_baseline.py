import os, sys, time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
import litetopk_fused as F

DATA = os.path.join(os.path.dirname(_HERE), "data", "marsco")


def read_fvecs(path, limit=None):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    a = a.reshape(-1, d + 1)
    if limit is not None:
        a = a[:limit]
    return np.ascontiguousarray(a[:, 1:]).view(np.float32)


def timed(fn, iters=3):
    fn(); torch.cuda.synchronize()
    ev0 = torch.cuda.Event(True); ev1 = torch.cuda.Event(True)
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record(); torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / iters


def main():
    dev = "cuda"
    base = read_fvecs(os.path.join(DATA, "base.fvecs"))
    query = read_fvecs(os.path.join(DATA, "query.fvecs"))
    M, D = base.shape
    c = torch.from_numpy(base).to(dev).half()
    N = int(os.environ.get("VN", "1000"))
    x = torch.from_numpy(query[:N]).to(dev).half()
    k = int(os.environ.get("VK", "1024"))
    print(f"corpus {base.shape}  query [{N},{D}]  k={k}")

    cf = c.float(); xf = x.float()

    # IP 路径：score = x @ c.t()，取最大的 k（内积越大越近）。无 csq/xsq elementwise。
    def torch_topk_ip():
        s = x @ c.t()
        return s.topk(k, largest=True).indices

    def torch_gemm_only():
        return (x @ c.t())
    scores = (x @ c.t())
    def torch_topk_only():
        return scores.topk(k, largest=True).indices

    def litetopk_dense():
        os.environ["FLASHTOPK_DENSE"] = "1"
        os.environ["FLASHTOPK_DENSE_CUDA_THR"] = "1"
        return F._fused(x, c, k, is_l2=False, use_tf32=False, return_distances=False, sorted=False, buf_mult=50)

    t_full = timed(torch_topk_ip)
    t_gemm = timed(torch_gemm_only)
    t_tk = timed(torch_topk_only)
    t_st = timed(litetopk_dense)
    print(f"torch full (gemm+topk): {t_full:.2f}ms")
    print(f"  torch gemm only     : {t_gemm:.2f}ms")
    print(f"  torch topk only     : {t_tk:.2f}ms")
    print(f"litetopk DENSE cuda_thr: {t_st:.2f}ms")
    print(f"speedup vs torch full : {t_full/t_st:.2f}x")


if __name__ == "__main__":
    main()
