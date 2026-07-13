"""Download MS MARCO v2.1 (Snowflake arctic-embed-l, 768-d) corpus shards via the
the HF endpoint and write a 5M-vector fvecs file for the top-k benchmark.

Network notes (internal env):
  * huggingface_hub's normal download path gets 302-redirected to the public
    HF/Xet CDN, which is unreachable here. Instead we hit the mirror's
    `resolve/main/...` URL directly with urllib; that 302s to the *mirror's* CDN
    (cas-bridge-...example.com), which is reachable, and we stream the whole body.
  * Each corpus shard is ~7.85 GB. We download one shard at a time, append its
    embeddings to the output fvecs, then delete the parquet to bound disk use.

fvecs format (matches existing data/base.fvecs): per row int32 dim then
dim float32 values; dim is 768.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "marsco"
EP = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
# arctic-embed-m-v1.5 is natively 768-d (matches existing base/query.fvecs and
# the kernel's D<=768 limit). The -l variant is 1024-d and is NOT used here.
REPO = os.environ.get("HF_REPO", "Snowflake/msmarco-v2.1-snowflake-arctic-embed-m-v1.5")
DIM = 768


def download_shard(shard: str, dst: Path, retries: int = 10) -> None:
    # Use huggingface-cli, which streams the full body in one connection at
    # ~20 MB/s. Its resume path issues HTTP Range requests that 302 to the public
    # (blocked) Xet CDN, so on any failure we delete the partial cache and retry
    # the whole download from scratch. Xet is disabled so it falls back to plain
    # HTTP through the mirror.
    if dst.exists():
        print(f"    {shard}: already present ({dst.stat().st_size} bytes)", flush=True)
        return
    local_dir = dst.parent / "_hf"
    cached = local_dir / shard  # huggingface-cli writes corpus/NN.parquet here
    env = dict(os.environ)
    env["HF_ENDPOINT"] = EP
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    for attempt in range(1, retries + 1):
        # Drop any stale .incomplete so the CLI starts fresh (no Range resume).
        cache_dl = local_dir / ".cache" / "huggingface" / "download"
        if cache_dl.exists():
            for p in cache_dl.rglob("*.incomplete"):
                p.unlink(missing_ok=True)
        t0 = time.time()
        proc = subprocess.run(
            ["huggingface-cli", "download", REPO, shard,
             "--repo-type", "dataset", "--local-dir", str(local_dir)],
            env=env, capture_output=True, text=True)
        # The CLI may exit 0 even on a mid-stream error; the only reliable signal
        # is whether it renamed the .incomplete cache to the final shard path.
        if cached.exists():
            cached.replace(dst)
            shutil.rmtree(local_dir, ignore_errors=True)
            print(f"    {shard}: download complete ({dst.stat().st_size} bytes, "
                  f"{time.time()-t0:.0f}s)", flush=True)
            return
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-2:]
        print(f"    [retry {attempt}/{retries}] {shard}: rc={proc.returncode} "
              f"{' | '.join(tail)[:200]}", flush=True)
        time.sleep(min(30, 5 * attempt))
    raise RuntimeError(f"failed to download {shard} after {retries} attempts")


def find_embedding_column(pf: pq.ParquetFile) -> str:
    # Prefer a column literally named "embedding"/"emb"; otherwise fall back to
    # the first column whose first value is a length-DIM numeric list.
    names = pf.schema_arrow.names
    for pref in ("embedding", "emb", "vector"):
        if pref in names:
            return pref
    batch = next(pf.iter_batches(batch_size=1))
    for name in names:
        v = batch.column(name)[0].as_py()
        if isinstance(v, (list, tuple)) and len(v) == DIM:
            return name
    raise RuntimeError(f"no length-{DIM} list column found; schema={names}")


def append_embeddings(parquet_path: Path, out_f, rows_needed: int, batch_size: int) -> int:
    pf = pq.ParquetFile(parquet_path)
    emb_col = find_embedding_column(pf)
    written = 0
    header = np.array([DIM], dtype=np.int32)
    for batch in pf.iter_batches(batch_size=batch_size, columns=[emb_col]):
        if rows_needed <= 0:
            break
        arr = np.asarray(batch.column(emb_col).to_pylist(), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != DIM:
            raise RuntimeError(
                f"embedding dim mismatch: got shape {arr.shape}, expected (*, {DIM}). "
                f"Wrong REPO? current REPO={REPO}")
        if arr.shape[0] > rows_needed:
            arr = arr[:rows_needed]
        # Interleave the int32 dim header in front of each row.
        n = arr.shape[0]
        rec = np.empty((n, DIM + 1), dtype=np.float32)
        rec[:, 0] = header.view(np.float32)[0]
        rec[:, 1:] = arr
        rec.tofile(out_f)
        written += n
        rows_needed -= n
        print(f"      converted +{n} (shard total {written})", flush=True)
    return written


def _is_valid_parquet(path: Path) -> bool:
    try:
        pf = pq.ParquetFile(path)
        _ = pf.metadata.num_rows
        return True
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=int(os.environ.get("TARGET_ROWS", 5_000_000)))
    ap.add_argument("--out", default=os.environ.get("OUT_FVECS",
                    str(DATA_DIR / "base_5m.fvecs")))
    ap.add_argument("--workdir", default=os.environ.get("WORKDIR",
                    str(DATA_DIR / "_shards")))
    ap.add_argument("--num-shards", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=50_000)
    ap.add_argument("--keep-parquet", action="store_true",
                    help="do not delete each parquet after conversion")
    args = ap.parse_args()

    out = Path(args.out)
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"target rows: {args.rows}  out: {out}  endpoint: {EP}", flush=True)
    remaining = args.rows
    total_written = 0
    with out.open("wb") as out_f:
        for i in range(args.num_shards):
            if remaining <= 0:
                break
            shard = f"corpus/{i:02d}.parquet"
            local = work / f"{i:02d}.parquet"
            if local.exists() and not _is_valid_parquet(local):
                # A previous run promoted a truncated file. Demote it to .part so
                # download_shard can resume instead of restarting from zero.
                print(f"[{i}] existing {local.name} is not a valid parquet; "
                      f"demoting to .part for resume", flush=True)
                local.replace(local.with_suffix(local.suffix + ".part"))
            print(f"[{i}] downloading {shard} ...", flush=True)
            if not local.exists():
                download_shard(shard, local)
            if not _is_valid_parquet(local):
                raise RuntimeError(f"{local} still invalid after download")
            print(f"[{i}] converting (need {remaining} more) ...", flush=True)
            w = append_embeddings(local, out_f, remaining, args.batch_size)
            total_written += w
            remaining -= w
            if not args.keep_parquet:
                local.unlink(missing_ok=True)
            print(f"[{i}] done. total_written={total_written} remaining={remaining}", flush=True)

    print(f"FINISHED. wrote {total_written} vectors to {out} "
          f"({out.stat().st_size} bytes)", flush=True)
    if total_written < args.rows:
        print(f"WARNING: only got {total_written} < {args.rows}; "
              f"need more than {args.num_shards} shards.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
