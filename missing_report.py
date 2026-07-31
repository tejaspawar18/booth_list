#!/usr/bin/env python
"""Report missing booth PDFs per AC: compares the ECI portal's official part
list against what's actually landed in S3, so you know exactly how many and
which booth (part) numbers are still missing after a download pass.

Usage:
    .venv/bin/python missing_report.py --state Maharashtra --year 2024 \
        --roll-id S13-2024-FIR --env-file /home/ubuntu/.env
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import boto3

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from download_eci_rolls import EciClient, _match, load_env_files  # noqa: E402
from download_eci_rolls_fast import retry_call  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("missing-report")

S3_REGION = "ap-south-1"


def s3_present_parts(s3, bucket: str, prefix: str) -> set[int]:
    have: set[int] = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            name = obj["Key"].rsplit("/", 1)[-1]
            if name.startswith("booth_") and name.endswith(".pdf"):
                digits = name[len("booth_"):-len(".pdf")]
                if digits.isdigit():
                    have.add(int(digits))
    return have


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", required=True, help="State name or code, e.g. Maharashtra / S13")
    ap.add_argument("--year", required=True)
    ap.add_argument("--roll-id", required=True, help="Exact roll id, e.g. S13-2024-FIR")
    ap.add_argument("--s3-bucket", default="electoral-roll-pdfs")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output CSV (default /home/ubuntu/missing_booths_<state-cd>.csv)")
    ap.add_argument("--env-file", type=Path)
    args = ap.parse_args()

    load_env_files(args.env_file)
    s3 = boto3.client("s3", region_name=S3_REGION)
    cli = EciClient(pool_size=8)

    state = _match(cli.states(), args.state, "stateName", "stateCd")
    if not state:
        sys.exit(f"State not found: {args.state!r}")
    state_cd = state["stateCd"]

    rolls = cli.eroll_types(state_cd, args.year)
    roll = _match(rolls, args.roll_id, "id")
    if not roll:
        sys.exit(f"Roll not found: {args.roll_id!r}")
    roll_prefix = f"{state_cd}/{str(roll['id']).lower()}"

    out = args.out or Path(f"/home/ubuntu/missing_booths_{state_cd}.csv")

    districts = cli.districts(state_cd)
    log.info("%s | roll %s | %d districts", state_cd, roll["id"], len(districts))

    rows: list[dict] = []
    total_booths = 0
    total_missing = 0
    for di, d in enumerate(districts, 1):
        dcd = d["districtCd"]
        acs = retry_call(cli.acs, dcd, what=f"district {dcd} AC list")
        for ac in acs:
            ac_no = int(ac["asmblyNo"])
            parts = retry_call(cli.part_list, state_cd, dcd, ac_no, what=f"AC {ac_no} part list")
            all_parts = sorted(int(p["partNumber"]) for p in parts)
            if not all_parts:
                continue
            have = s3_present_parts(s3, args.s3_bucket, f"{roll_prefix}/{ac_no}/")
            missing = sorted(set(all_parts) - have)
            total_booths += len(all_parts)
            total_missing += len(missing)
            rows.append({
                "ac_no": ac_no,
                "ac_name": ac.get("asmblyName", ""),
                "district": d.get("districtValue", ""),
                "total_booths": len(all_parts),
                "missing_count": len(missing),
                "missing_parts": " ".join(map(str, missing)),
            })
            if missing:
                log.info("[%d/%d districts] AC %s (%s): %d/%d missing",
                         di, len(districts), ac_no, ac.get("asmblyName", ""),
                         len(missing), len(all_parts))

    rows.sort(key=lambda r: -r["missing_count"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ac_no", "ac_name", "district", "total_booths",
                                          "missing_count", "missing_parts"])
        w.writeheader()
        w.writerows(rows)

    acs_with_gaps = sum(1 for r in rows if r["missing_count"])
    log.info("Total: %d/%d booths missing across %d ACs (%d ACs have gaps)",
             total_missing, total_booths, len(rows), acs_with_gaps)
    log.info("Report saved to %s", out)


if __name__ == "__main__":
    main()
