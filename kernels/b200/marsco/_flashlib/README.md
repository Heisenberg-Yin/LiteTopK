# Vendored flashlib (patched)

`flashlib/` = flashlib 0.2.0 (PyPI) **with our METRIC_IP patch**
(`triton/insert.py`: constexpr METRIC_IP, rank by -x.c, drop corpus-norm;
`dispatch.py`/`_run` threading; `flash_knn(metric="ip")`) — used by
`../bench_flashlib_ip.py` for the IP-metric baseline. The tcgen05 probe
experiment is documented in comments inside `triton/insert.py` (reverted).

NOT vendored (stock PyPI, reinstall next to `flashlib/`):

```bash
pip download nvidia-cutlass-dsl==4.3.5 --python-version 3.12 --no-deps -d /tmp/dsl
# extract the wheel here so that nvidia_cutlass_dsl/ sits beside flashlib/
```

(The container's py3.12 has no pip module — that is why download+extract,
not pip install; see the marsco README.)
