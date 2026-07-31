"""
Convert a voters CSV (from extract_voters_tesseract.py / escalate_voters_gemini.py)
into one JSON file per booth and upload it to S3.

Key layout in the destination bucket:
    s3://<bucket>/<state_cd>/<roll_id>/<ac_no>/<booth_pdf stem>.json

Usage
-----
    python voters_csv_to_json.py --csv booth_list_csv/voters_216_hybrid.csv \
        --state-cd S13 --roll-id s13-2024-fir --ac 216 \
        --bucket electoral-voter-details --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import boto3

VOTER_FIELDS = ["serial_no", "epic_number", "name", "relation_type", "relation_name",
                "house_number", "age", "gender", "marker"]
PAGE_FIELDS = ["section_no", "section_name", "part_no", "page_no"]


def load_rows(csv_path: Path) -> dict[str, list[dict]]:
    by_booth: dict[str, list[dict]] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            by_booth.setdefault(row["booth_pdf"], []).append(row)
    return by_booth


def build_booth_json(booth_pdf: str, rows: list[dict], state_cd: str, roll_id: str,
                      ac_no: str) -> dict:
    voters = []
    for row in sorted(rows, key=lambda r: (int(r["page_no"]), int(r["serial_no"] or 0))):
        voter = {f: row.get(f, "") for f in VOTER_FIELDS}
        voter.update({f: row.get(f, "") for f in PAGE_FIELDS})
        provenance = "gemini" if "gemini" in (row.get("flags") or "") else "ocr"
        voter["source"] = provenance
        voters.append(voter)
    # part_no can repeat across different booth_pdf files (duplicate/re-scanned
    # uploads of the same electoral part), so it's surfaced at the booth level
    # for a dedup pass keyed on (ac_no, part_no) before this JSON is trusted.
    part_counts = Counter(r["part_no"] for r in rows if r.get("part_no"))
    part_no = part_counts.most_common(1)[0][0] if part_counts else ""
    return {
        "state_cd": state_cd,
        "roll_id": roll_id,
        "ac_no": ac_no,
        "booth_pdf": booth_pdf,
        "part_no": part_no,
        "voter_count": len(voters),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "voters": voters,
    }


def upload_one(s3, bucket: str, key: str, payload: dict) -> str:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    return key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--state-cd", required=True, help="e.g. S13")
    ap.add_argument("--roll-id", required=True, help="e.g. s13-2024-fir")
    ap.add_argument("--ac", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--out-dir", type=Path, help="Also write JSON files locally here")
    ap.add_argument("--dry-run", action="store_true", help="Build JSON, skip S3 upload")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    by_booth = load_rows(args.csv)
    print(f"{len(by_booth)} booths, {sum(len(v) for v in by_booth.values())} voter rows in {args.csv}")

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3") if not args.dry_run else None
    jobs = []
    seen_names: dict[str, str] = {}
    for booth_pdf, rows in by_booth.items():
        payload = build_booth_json(booth_pdf, rows, args.state_cd, args.roll_id, args.ac)
        # Named by part_no, not the booth PDF filename: two booth PDFs can be
        # duplicate/re-scanned uploads of the same electoral part, and keying
        # on part_no lets a later upload naturally supersede an earlier one
        # instead of both landing under different, undeduped keys.
        part_no = (payload["part_no"] or "").strip().replace("/", "_")
        name = part_no or Path(booth_pdf).stem
        if name in seen_names:
            print(f"WARNING: part_no {name!r} from {booth_pdf} collides with "
                  f"{seen_names[name]} - {booth_pdf} will overwrite it at the same key")
        seen_names[name] = booth_pdf
        key = f"{args.state_cd}/{args.roll_id}/{args.ac}/{name}.json"
        if args.out_dir:
            (args.out_dir / f"{name}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.dry_run:
            jobs.append((key, payload))

    if args.dry_run:
        print("dry-run: JSON built, nothing uploaded")
        return

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(upload_one, s3, args.bucket, key, payload): key for key, payload in jobs}
        for fut in as_completed(futs):
            key = fut.result()
            done += 1
            print(f"[{done}/{len(jobs)}] uploaded s3://{args.bucket}/{key}")

    print(f"done. {done} booth JSON files -> s3://{args.bucket}/{args.state_cd}/{args.roll_id}/{args.ac}/")


if __name__ == "__main__":
    main()
