from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen

DEFAULT_SRC = "msmarco_v2.1_doc_segmented_00.parquet"
DEFAULT_DST = "msmarco_embeddings_1050000.parquet"
DEFAULT_ROWS = 1_050_000
DEFAULT_HF_REPO = "Snowflake/msmarco-v2.1-snowflake-arctic-embed-l"
DEFAULT_HF_FILE = "corpus/train-00000-of-00010.parquet"


def _download_with_url(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"downloading from url: {url}")
    with urlopen(url) as resp, tmp.open("wb") as f:
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else None
        copied = 0
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            copied += len(chunk)
            if total_bytes:
                pct = copied * 100.0 / total_bytes
                print(f"downloaded {copied}/{total_bytes} bytes ({pct:.1f}%)", flush=True)
            else:
                print(f"downloaded {copied} bytes", flush=True)
    tmp.replace(dst)


def _download_from_hf(repo_id: str, repo_file: str, dst: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "missing huggingface_hub; please `pip install huggingface_hub` or provide --url"
        ) from exc

    print(f"downloading from hf dataset: {repo_id}/{repo_file}")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=repo_file,
        repo_type="dataset",
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded, dst)


def ensure_source_file(src: Path, url: str | None, repo_id: str, repo_file: str) -> None:
    if src.exists():
        print(f"source exists, reuse: {src}")
        return
    if url:
        _download_with_url(url, src)
        return
    _download_from_hf(repo_id, repo_file, src)


def slice_parquet(src: Path, dst: Path, rows: int, batch_size: int) -> None:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(src)
    print("schema:", pf.schema_arrow.names)
    print("total rows in file:", pf.metadata.num_rows)

    writer = None
    written = 0
    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            remaining = rows - written
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)
            if writer is None:
                writer = pq.ParquetWriter(dst, batch.schema)
            writer.write_batch(batch)
            written += batch.num_rows
            print(f"written {written}/{rows}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    print("DONE. total written:", written)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="下载（可选）并截取 MSMARCO parquet 的前若干行。"
    )
    parser.add_argument("--src", default=os.environ.get("MSMARCO_SRC", DEFAULT_SRC))
    parser.add_argument("--dst", default=os.environ.get("MSMARCO_DST", DEFAULT_DST))
    parser.add_argument(
        "--rows",
        type=int,
        default=int(os.environ.get("MSMARCO_ROWS", DEFAULT_ROWS)),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("MSMARCO_BATCH_SIZE", 50_000)),
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("MSMARCO_SOURCE_URL"),
        help="可选：直接下载 parquet 的 URL；未提供时尝试从 Hugging Face 数据集下载。",
    )
    parser.add_argument(
        "--hf-repo",
        default=os.environ.get("MSMARCO_HF_REPO", DEFAULT_HF_REPO),
    )
    parser.add_argument(
        "--hf-file",
        default=os.environ.get("MSMARCO_HF_FILE", DEFAULT_HF_FILE),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    ensure_source_file(src, args.url, args.hf_repo, args.hf_file)
    dst.parent.mkdir(parents=True, exist_ok=True)
    slice_parquet(src, dst, args.rows, args.batch_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
