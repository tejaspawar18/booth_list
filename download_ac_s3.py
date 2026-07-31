"""
Download every booth PDF for one AC from the source S3 bucket to a local dir,
in parallel. Skips files already present locally (resumable).

Usage:
    python download_ac_s3.py --state-cd S13 --roll-id s13-2024-fir --ac 216 \
        --bucket electoral-roll-pdfs --workers 32
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

HERE = Path(__file__).resolve().parent


def list_keys(s3, bucket: str, prefix: str) -> list[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(o["Key"] for o in page.get("Contents", []))
    return keys


def download_one(s3, bucket: str, key: str, dest: Path) -> tuple[str, bool]:
    if dest.exists() and dest.stat().st_size > 0:
        return key, False
    tmp = dest.with_suffix(dest.suffix + ".part")
    s3.download_file(bucket, key, str(tmp))
    tmp.rename(dest)
    return key, True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-cd", required=True)
    ap.add_argument("--roll-id", required=True)
    ap.add_argument("--ac", required=True)
    ap.add_argument("--bucket", default="electoral-roll-pdfs")
    ap.add_argument("--out-dir", type=Path,
                     help="Default: booth_list_pdf/<state>/<roll>/<ac> (lowercased)")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    prefix = f"{args.state_cd}/{args.roll_id}/{args.ac}/"
    out_dir = args.out_dir or (HERE / "booth_list_pdf" / args.state_cd.lower()
                                / args.roll_id.lower() / args.ac)
    out_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")
    keys = [k for k in list_keys(s3, args.bucket, prefix) if k.endswith(".pdf")]
    print(f"{len(keys)} booth PDFs under s3://{args.bucket}/{prefix}")

    downloaded = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, s3, args.bucket, key, out_dir / Path(key).name): key
                for key in keys}
        for i, fut in enumerate(as_completed(futs), start=1):
            key, did_download = fut.result()
            downloaded += did_download
            skipped += not did_download
            if i % 25 == 0 or i == len(keys):
                print(f"[{i}/{len(keys)}] downloaded={downloaded} skipped(existing)={skipped}")

    print(f"done -> {out_dir} ({downloaded} downloaded, {skipped} already present)")


if __name__ == "__main__":
    main()
