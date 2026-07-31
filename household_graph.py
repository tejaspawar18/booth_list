"""
Build within-booth household graphs from voter JSON and report per-household
stats: member count, age spread, gender counts.

A household is a connected component of a graph whose nodes are voters and
whose edges link a voter to whichever other voter *in the same booth* has a
name matching this voter's declared relation_name (father/husband/mother) -
the standard way an electoral roll encodes family structure. Union-find over
name matches, not a general graph library: the graphs here are small (one
booth's ~900-1200 voters split into many tiny family clusters), and a name
link is symmetric enough that connected components are all we need.

Usage
-----
    python household_graph.py --state-cd S13 --roll-id s13-2024-fir --ac 216 \
        --bucket electoral-voter-details --out-dir booth_list_csv/households
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import boto3

HOUSEHOLD_COLUMNS = ["ac_no", "booth_pdf", "part_no", "household_id", "member_count",
                     "distinct_surname_count", "age_mean", "age_stddev",
                     "implausible_age_count", "male_count", "female_count",
                     "other_gender_count", "member_serials"]
BOOTH_SUMMARY_COLUMNS = ["ac_no", "part_no", "booth_pdf", "voter_count", "household_count",
                         "mean_household_size", "mean_age_stddev",
                         "total_male", "total_female", "total_other_gender"]


def normalize(name: str) -> str:
    return " ".join((name or "").strip().upper().split())


def first_last(name: str) -> tuple[str, str] | None:
    """(first token, last token) of a normalized name - the key a relation
    link is matched on. A voter's own name is typically 3 tokens (first
    name, the father's/husband's first name as a middle name, surname); the
    relation_name field recording that same person elsewhere usually drops
    the middle token ("AMBADAS SAKHARAM AHIRE" as himself vs "AMBADAS AHIRE"
    as someone's husband) - so first+last token is the reliable match, not
    the full string."""
    tokens = normalize(name).split()
    if len(tokens) < 2:
        return None
    return tokens[0], tokens[-1]


def edit_distance_leq(a: str, b: str, k: int) -> bool:
    """True if edit distance between a and b is <= k. Length-difference
    short-circuit first, since most non-matches fail that cheaply."""
    if abs(len(a) - len(b)) > k:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            row_min = min(row_min, cur[j])
        if row_min > k:
            return False  # this row can only get worse - bail out
        prev = cur
    return prev[-1] <= k


def build_households(voters: list[dict[str, Any]], fuzzy_edits: int = 1
                     ) -> tuple[list[list[dict[str, Any]]], int]:
    """Union-find over voters: link voter i to voter j when i's relation_name
    and j's name share the same (first token, last token). A voter with no
    match in this booth is its own household of one (common - the relative
    may be in a different booth, deceased, or simply not registered).

    A relation_name with no exact-key match falls back to a fuzzy pass:
    <=fuzzy_edits character edits on each of the first and last token,
    against this booth's other names. This is meant for OCR noise ("Sanju"
    read as "Sanjy"), not genuine name variation - it will not and should
    not link "Sanju" to "Sanjay" (a real nickname/full-name difference, 2
    edits on a 5-6 char token, and guessing that kind of link wrong merges
    two unrelated people's households). Takes the first tolerable candidate
    rather than the closest, to stay cheap; ambiguity between two equally
    plausible OCR-noise candidates in one booth is rare enough not to be
    worth the extra bookkeeping.

    Returns (households, fuzzy_link_count) - the count is for the run's
    summary line, so a surprisingly high figure is visible rather than
    silently changing the household shapes."""
    by_key: dict[tuple[str, str], list[int]] = {}
    for i, v in enumerate(voters):
        key = first_last(v.get("name", ""))
        if key:
            by_key.setdefault(key, []).append(i)

    parent = list(range(len(voters)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    unmatched: list[tuple[int, tuple[str, str]]] = []
    for i, v in enumerate(voters):
        key = first_last(v.get("relation_name", ""))
        if not key:
            continue
        candidates = by_key.get(key)
        if candidates:
            for j in candidates:
                if j != i:
                    union(i, j)
        else:
            unmatched.append((i, key))

    fuzzy_links = 0
    if unmatched and fuzzy_edits > 0:
        distinct_keys = list(by_key.keys())
        for i, key in unmatched:
            for ck in distinct_keys:
                if (edit_distance_leq(key[0], ck[0], fuzzy_edits)
                        and edit_distance_leq(key[1], ck[1], fuzzy_edits)):
                    for j in by_key[ck]:
                        if j != i:
                            union(i, j)
                    fuzzy_links += 1
                    break

    groups: dict[int, list[dict[str, Any]]] = {}
    for i, v in enumerate(voters):
        groups.setdefault(find(i), []).append(v)
    return list(groups.values()), fuzzy_links


def household_stats(members: list[dict[str, Any]]) -> dict[str, Any]:
    # Tesseract-only runs (no Gemini escalation) leave known OCR artifacts in
    # age uncorrected - a spurious extra digit ("56" -> "556") is the common
    # one (see extract_voters_tesseract.py's age_bad flag). One implausible
    # age would otherwise blow up a whole household's mean/stddev, so those
    # are excluded from the stats and counted separately rather than trusted.
    raw_ages = [int(m["age"]) for m in members if str(m.get("age", "")).strip().isdigit()]
    ages = [a for a in raw_ages if 18 <= a <= 100]
    genders = [m.get("gender", "") for m in members]
    # A well-formed household is mostly one surname; a high count here on a
    # multi-member household flags a false merge (two different families
    # linked through a common first+last name combination) rather than a
    # real family - worth checking member_serials for those by hand.
    surnames = {normalize(m.get("name", "")).split()[-1]
               for m in members if normalize(m.get("name", "")).split()}
    return {
        "member_count": len(members),
        "distinct_surname_count": len(surnames),
        "age_mean": round(statistics.mean(ages), 1) if ages else "",
        "age_stddev": round(statistics.stdev(ages), 2) if len(ages) > 1 else 0.0,
        "implausible_age_count": len(raw_ages) - len(ages),
        "male_count": sum(1 for g in genders if g == "Male"),
        "female_count": sum(1 for g in genders if g == "Female"),
        "other_gender_count": sum(1 for g in genders if g not in ("Male", "Female")),
        "member_serials": ";".join(str(m.get("serial_no", "")) for m in members),
    }


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-cd", required=True)
    ap.add_argument("--roll-id", required=True)
    ap.add_argument("--ac", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("booth_list_csv/households"))
    ap.add_argument("--fuzzy-edits", type=int, default=1,
                    help="Max character edits per token (first, last) for the OCR-noise "
                         "fallback when no exact name match is found; 0 disables it")
    args = ap.parse_args()

    s3 = boto3.client("s3")
    prefix = f"{args.state_cd}/{args.roll_id}/{args.ac}/"
    paginator = s3.get_paginator("list_objects_v2")
    keys = [o["Key"] for page in paginator.paginate(Bucket=args.bucket, Prefix=prefix)
            for o in page.get("Contents", []) if o["Key"].endswith(".json")]
    print(f"{len(keys)} booth JSON files under s3://{args.bucket}/{prefix}")

    household_rows: list[dict[str, Any]] = []
    booth_rows: list[dict[str, Any]] = []
    total_fuzzy_links = 0
    for key in sorted(keys):
        obj = s3.get_object(Bucket=args.bucket, Key=key)
        booth = json.loads(obj["Body"].read())
        voters = booth["voters"]
        households, fuzzy_links = build_households(voters, fuzzy_edits=args.fuzzy_edits)
        total_fuzzy_links += fuzzy_links

        booth_male = booth_female = booth_other = 0
        stddevs = []
        for hid, members in enumerate(households, start=1):
            stats = household_stats(members)
            household_rows.append({
                "ac_no": booth["ac_no"], "booth_pdf": booth["booth_pdf"],
                "part_no": booth["part_no"], "household_id": hid, **stats,
            })
            booth_male += stats["male_count"]
            booth_female += stats["female_count"]
            booth_other += stats["other_gender_count"]
            if isinstance(stats["age_stddev"], (int, float)):
                stddevs.append(stats["age_stddev"])

        booth_rows.append({
            "ac_no": booth["ac_no"], "part_no": booth["part_no"], "booth_pdf": booth["booth_pdf"],
            "voter_count": len(voters), "household_count": len(households),
            "mean_household_size": round(len(voters) / len(households), 2) if households else 0,
            "mean_age_stddev": round(statistics.mean(stddevs), 2) if stddevs else 0,
            "total_male": booth_male, "total_female": booth_female,
            "total_other_gender": booth_other,
        })

    write_csv(args.out_dir / f"households_{args.ac}.csv", HOUSEHOLD_COLUMNS, household_rows)
    write_csv(args.out_dir / f"booth_summary_{args.ac}.csv", BOOTH_SUMMARY_COLUMNS, booth_rows)

    total_voters = sum(b["voter_count"] for b in booth_rows)
    total_households = sum(b["household_count"] for b in booth_rows)
    multi = [h for h in household_rows if h["member_count"] > 1]
    mean_surnames_all = (statistics.mean(h["distinct_surname_count"] for h in household_rows)
                         if household_rows else 0)
    mean_surnames_multi = (statistics.mean(h["distinct_surname_count"] for h in multi)
                           if multi else 0)
    print(f"AC {args.ac}: {len(booth_rows)} booths, {total_voters} voters, "
          f"{total_households} households, mean household size "
          f"{total_voters/total_households:.2f}" if total_households else "no households found")
    print(f"mean distinct surnames per household: {mean_surnames_all:.2f} (all), "
          f"{mean_surnames_multi:.2f} (multi-member only, {len(multi)} households) - "
          f"closer to 1.0 on the multi-member figure means fewer false merges")
    print(f"fuzzy (OCR-noise, <={args.fuzzy_edits} edit) links used: {total_fuzzy_links} "
          f"of {total_voters} voters")
    print(f"wrote {args.out_dir}/households_{args.ac}.csv "
          f"and {args.out_dir}/booth_summary_{args.ac}.csv")


if __name__ == "__main__":
    main()
