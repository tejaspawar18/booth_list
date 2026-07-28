"""
Group an AC's booths into "localities", where one locality = one place (the same
village / address).  Voter count is NOT a constraint - a locality can be a single
booth or many booths; it is whatever booths belong to the same locality.

Method
------
1. Sort booths by part_no (ECI numbers booths geographically, so a place's booths
   sit together in a contiguous run).
2. Canonicalize village- and ward-name OCR variants within the AC using a
   conservative fuzzy match plus shared-prefix and booth-proximity guards.
   Panchayat names use the panchayat_clean column instead, standardized once by
   extract_booth_info_gemini.py (one Gemini call per AC, cached there); the
   local fuzzy clustering is only a fallback for booth-info CSVs predating
   that column.
3. Walk the sequence; a locality is a maximal run of consecutive booths that share
   the same canonical village AND (when printed) the same ward. In urban ACs the
   "village" is just the city name for every booth, so wards are what partition
   the city into places, and inside the city a polling-station building change
   also partitions a ward (one ward there spans many separate mohallas). In
   rural ACs wards are blank and villages partition. A booth with a blank
   village/ward attaches to the current run.
4. Merge same-place fragments separated by only a small booth-number gap, which
   handles town booths interspersed with a few other villages.
5. Club every rural blank-ward run that shares the same gram panchayat into a
   single locality — one panchayat, one locality. The panchayat outranks the
   village name, so all of a panchayat's villages club together however far
   apart their booths sit (panchayat_clean is Gemini-standardized, so a shared
   clean panchayat proves the same place; no booth-gap limit applies).
6. In rural ACs only: club every run of one ward into a single locality
   regardless of gap or building (a ward there is one place, e.g. a small town
   inside an otherwise rural AC). Then club same-village localities that share
   one polling-station building even when their (tola-style) ward names
   differ — the building is a harder physical-place signal than a locally
   printed mohalla name. Urban ACs skip both steps and instead rely on
   --max-voters/--min-voters to size their per-building ward fragments.
7. Name each locality after its village, or after its ward when the village/city
   is subdivided into several ward localities.

By default voter count is ignored: the output is pure one-place-one-locality,
for understanding what localities look like from the booth data. Optional
--max-voters can split an oversized place into consecutive chunks cutting only
at polling-station boundaries, and optional --min-voters can coalesce small
localities ONLY when they share the same gram panchayat; pass positive values
only when you intentionally want practical survey-size units.

Usage
-----
    PY=../../jup_venv/bin/python
    $PY build_localities.py --ac 324                        # pure places
    $PY build_localities.py --ac 324 --min-voters 2000 --max-voters 6000
    $PY build_localities.py --csv booth_list_csv/booth_info_gemini_324.csv --out localities/localities_324.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("localities")
HERE = Path(__file__).resolve().parent
DEFAULT_CANON_THRESHOLD = 0.68
DEFAULT_CANON_PREFIX = 3
DEFAULT_CANON_WINDOW = 20
DEFAULT_MERGE_GAP = 20
DEFAULT_MIN_VOTERS = 0
DEFAULT_MAX_VOTERS = 0
# same-building addresses differ only by room number, so cluster them far more
# strictly than village names ("प्रा0वि0 X" vs "प्रा0वि0 Y" are different schools)
ADDRESS_CANON_THRESHOLD = 0.88
ADDRESS_CANON_WINDOW = 20

LOC_COLUMNS = ["locality_id", "locality_name", "block", "tehsil", "panchayats",
               "villages", "wards", "num_booths", "part_range", "booth_parts",
               "male", "female", "total_voters", "counts_flagged"]
MAP_COLUMNS = ["part_no", "booth_pdf", "locality_id", "locality_name",
               "main_town_village", "ward", "panchayat", "ps_address",
               "block", "tehsil", "total",
               "canonical_village", "canonical_ward", "canonical_panchayat",
               "canonical_ps_address"]


_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def _norm(s: str) -> str:
    s = (s or "").translate(_DEV_DIGITS)
    return re.sub(r"\s+", " ", s.strip()).strip(" -–:।")


def _similar(a: str, b: str) -> bool:
    """True if two village names are the same place (exact or fuzzy, OCR-tolerant)."""
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return True  # unknown -> don't treat as a boundary on its own
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.8


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def _canonical_match(a: str, b: str, threshold: float, min_prefix: int) -> bool:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return False
    if a == b:
        return True
    if _common_prefix_len(a, b) < min_prefix:
        return False
    return SequenceMatcher(None, a, b).ratio() >= threshold


# room/wing designators inside a polling-station address: same building, so they
# must not affect address clustering and are never part of an area name
_ROOM_RE = re.compile(
    r"(?:क0?\s*[.-]?\s*नं0?|कक्ष\s*सं(?:ख्या)?|रूम\s*नं(?:बर)?|कमरा\s*नं(?:बर)?)"
    r"\.?\s*[-–]?\s*[0-9०-९]*"
    r"|(?:उत्तरी|दक्षिणी|पूर्वी|पश्चिमी|मध्य|बायां|दायां|बाया|दाया)\s*भाग"
    r"|भाग\s*[-–]?\s*[0-9०-९]+"
)


def _strip_rooms(addr: str) -> str:
    addr = _ROOM_RE.sub(" ", addr or "")
    addr = re.sub(r"\s*,(\s*,)+", ", ", addr)
    return _norm(re.sub(r"\s+", " ", addr)).strip(", ")


def _int_field(r: dict, key: str) -> int:
    digits = re.sub(r"\D", "", (r.get(key) or "").strip())
    return int(digits) if digits else 0


def load_booths(csv_path: Path) -> list[dict]:
    booths, seen_parts = [], set()
    with csv_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "main_town_village" not in reader.fieldnames:
            raise SystemExit(f"{csv_path.name} does not look like a booth-info CSV "
                             f"(missing main_town_village column)")
        for r in reader:
            part_no = _int_field(r, "part_no") or _int_field(r, "booth_pdf")
            if not part_no:
                log.warning("skipping row without a part number: %s",
                            {k: r.get(k, "") for k in ("booth_pdf", "main_town_village")})
                continue
            if part_no in seen_parts:
                log.warning("duplicate part_no %d; keeping the first occurrence", part_no)
                continue
            seen_parts.add(part_no)
            booths.append({
                "part_no": part_no,
                "booth_pdf": r.get("booth_pdf", ""),
                "total": _int_field(r, "total"),
                "male": _int_field(r, "male"),
                "female": _int_field(r, "female"),
                "village": r.get("main_town_village") or "",
                "ward": r.get("ward") or "",
                "ps_address": r.get("ps_address") or "",
                "ps_clean": _strip_rooms(r.get("ps_address") or ""),
                "panchayat": r.get("panchayat") or "",
                "panchayat_clean": r.get("panchayat_clean") or "",
                "block": r.get("block") or "",
                "tehsil": r.get("tehsil") or "",
                "counts_ok": r.get("counts_ok") or "",
            })
    booths.sort(key=lambda b: b["part_no"])
    blank = sum(1 for b in booths if not _norm(b["village"]))
    if blank:
        log.warning("%d/%d booths have a blank village; they attach to the "
                    "preceding locality", blank, len(booths))
    return booths


def canonicalize_field(
    booths: list[dict],
    field: str,
    canon_field: str,
    threshold: float = DEFAULT_CANON_THRESHOLD,
    min_prefix: int = DEFAULT_CANON_PREFIX,
    proximity_window: int = DEFAULT_CANON_WINDOW,
) -> dict[str, str]:
    """Map OCR variants of a place field to its most common spelling in this AC."""
    counts = Counter(_norm(b[field]) for b in booths if _norm(b[field]))
    parts_by_value: dict[str, list[int]] = {}
    for b in booths:
        value = _norm(b[field])
        if value:
            parts_by_value.setdefault(value, []).append(b["part_no"])
    clusters: list[list[str]] = []
    canon_for: dict[str, str] = {}

    for value, _count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        matched: list[str] | None = None
        for cluster in clusters:
            close_to_cluster = any(
                abs(pa - pb) <= proximity_window
                for member in cluster
                for pa in parts_by_value.get(member, [])
                for pb in parts_by_value.get(value, [])
            )
            if close_to_cluster and _canonical_match(cluster[0], value, threshold, min_prefix):
                matched = cluster
                break
        if matched is None:
            clusters.append([value])
            canon_for[value] = value
        else:
            matched.append(value)
            canon_for[value] = matched[0]

    for b in booths:
        value = _norm(b[field])
        b[canon_field] = canon_for.get(value, value)

    merged = [c for c in clusters if len(c) > 1]
    if merged:
        log.info("canonicalized %d %s OCR clusters", len(merged), field)
        for cluster in merged:
            log.info("  %s <= %s", cluster[0], " / ".join(cluster[1:]))
    return canon_for


# "लुचुई प्रथम / लुचुई द्वितीय / लुचुई तृतीय" are one place split across booths;
# word ordinals never distinguish localities (numeric "नं0 2" villages DO differ)
_ORDINAL_RE = re.compile(r"(?<!\S)(?:प्रथम|द्वितीय|द्वितीया|तृतीय|चतुर्थ|पंचम)(?!\S)")


def _strip_ordinal(s: str) -> str:
    return _norm(_ORDINAL_RE.sub(" ", s))


# trailing numeric designators ("सेमरा नं0 2", "महादेव झारखण्डी . 01",
# "... टुकड़ा नं 3", "X नम्बर 2", "Y no. 2") are booth bookkeeping, not part
# of a place's name. Built to generalize across ACs:
#  - chunk words: टुकड़ा/खंड/भाग/पार्ट/हिस्सा (optional, before the number word)
#  - number words: नं / न0 / नं0 / नम्बर / नंबर / सं / सं0 / स0 / संख्या / no / n0
#  - separators: dot / dash / colon / comma / nothing; digits spaced or glued
#  - bare "[.-] 01" tails without a number word
# _norm has already mapped Devanagari digits to ASCII by the time this runs.
_CHUNK_WORDS = r"(?:टुकड़ा|टुकडा|टुक0|टु0|खंड|खण्ड|भाग|पार्ट|हिस्सा)"
_NUM_TAIL_RE = re.compile(
    r"(?:"
    r"\s*[.\-–,:]?\s*" + _CHUNK_WORDS + r"?\s*"
    r"(?<![^\W\d_])(?:नं|न0|नम्बर|नंबर|संख्या|सं0|सं|स0|no|n0)"
    r"\s*\.?\s*[-–:]?\s*[\d\s]*\d"
    r"|\s*[.\-–,:]?\s*(?<![^\W\d_])" + _CHUNK_WORDS + r"\s*[-–:]?\s*\d[\d\s]*"
    r"|\s*[.\-–,:]?\s*(?<![^\W\d_])" + _CHUNK_WORDS +  # residue after number strip
    r"|\s*[.\-–,:]\s*\d[\d\s]*"
    r")\s*$",
    re.IGNORECASE,
)

# if stripping leaves only a generic word, the number WAS the identity
# ("वार्ड नं0 12", "सेक्टर 5") — keep the original name in that case
_GENERIC_TAIL_WORDS = {
    "वार्ड", "वॉर्ड", "मोहल्ला", "मुहल्ला", "सेक्टर", "ब्लॉक", "ब्लाक", "गली",
    "लेन", "फेज", "पॉकेट", "पाकेट", "कॉलोनी", "कालोनी", "टोला", "पुरवा",
    "टुकड़ा", "टुकडा", "खंड", "खण्ड", "भाग", "पार्ट", "हिस्सा", "नगर", "ग्राम",
    "ward", "sector", "block", "lane", "phase", "pocket", "colony",
}


def _strip_number_tail(s: str) -> str:
    out = _norm(s)
    prev = None
    while out != prev:
        prev = out
        candidate = _norm(_NUM_TAIL_RE.sub("", out))
        # never strip down to nothing, digits/punctuation, or a lone generic word
        if not candidate or not re.search(r"[^\W\d_]", candidate):
            break
        words = candidate.split()
        if len(words) == 1 and words[0].lower() in _GENERIC_TAIL_WORDS:
            break
        out = candidate
    return out


def _cv(b: dict) -> str:
    return _strip_ordinal(_norm(b.get("canonical_village") or b["village"]))


def _cw(b: dict) -> str:
    return _strip_ordinal(_norm(b.get("canonical_ward") or b.get("ward", "")))


def _ca(b: dict) -> str:
    return _norm(b.get("canonical_ps") or b.get("ps_clean", "") or b.get("ps_address", ""))


def _cp(b: dict) -> str:
    return _norm(b.get("canonical_panchayat") or b.get("panchayat", ""))


def _main_panchayat(loc: list[dict]) -> str:
    """The locality's single gram panchayat, or "" if blank or mixed.

    Compared by exact equality, not fuzzy similarity: panchayat_clean is Gemini-
    standardized, so the same panchayat always has the identical clean spelling
    while genuinely different panchayats keep distinct names. Fuzzy matching
    here would wrongly fuse look-alike names (नरौली vs बनौली score 0.8)."""
    vals = [_cp(b) for b in loc if _cp(b)]
    if not vals:
        return ""
    return vals[0] if all(v == vals[0] for v in vals[1:]) else ""


def _main_block(loc: list[dict]) -> str:
    """The locality's dominant development block, or "" if none printed.
    A gram panchayat name is only unique within its block, so the block
    disambiguates two same-named panchayats in one AC."""
    vals = [_norm(b["block"]) for b in loc if _norm(b["block"])]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


def group_by_locality(booths: list[dict], city: str = "") -> list[list[dict]]:
    """A locality = a maximal run of consecutive booths sharing the same village
    and, when printed, the same ward.

    Rural booths have a village and usually no ward, so villages partition the
    run. Urban booths all carry the city as their "village", so wards are what
    partition the city into places — and inside the city a polling-station
    building change also partitions, since one ward there spans many separate
    mohallas; split_oversized/coalesce_small then re-assemble ward-sized units
    from those building fragments. Outside the city a village keeps all its
    schools in one locality, and same-ward runs are clubbed at ward level
    regardless of building (see merge_same_ward_runs). Blank values never
    break a run on their own.
    """
    localities: list[list[dict]] = []
    cur: list[dict] = []
    for b in booths:
        if not cur:
            cur = [b]
            continue
        v, w = _cv(b), _cw(b)
        anchor_v = next((_cv(x) for x in cur if _cv(x)), "")
        anchor_w = next((_cw(x) for x in cur if _cw(x)), "")
        same_place = ((not v) or (not anchor_v) or _similar(anchor_v, v)) and \
                     ((not w) or (not anchor_w) or _similar(anchor_w, w))
        if same_place and city and v == city:
            a = _ca(b)
            anchor_a = next((_ca(x) for x in cur if _ca(x)), "")
            same_place = (not a) or (not anchor_a) or a == anchor_a
        if same_place:
            cur.append(b)
        else:
            localities.append(cur)
            cur = [b]
    if cur:
        localities.append(cur)
    return localities


def dominant_city(booths: list[dict]) -> str:
    """The AC's city: a single "village" covering most booths (urban AC), else ""."""
    counts = Counter(_cv(b) for b in booths if _cv(b))
    if not counts:
        return ""
    top, n = counts.most_common(1)[0]
    return top if n >= len(booths) * 0.5 else ""


def _place_key(loc: list[dict]) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    """(village, wards, addresses) identity of a single-village locality, else None."""
    villages = {_cv(b) for b in loc if _cv(b)}
    if len(villages) != 1:
        return None
    return (
        next(iter(villages)),
        tuple(sorted({_cw(b) for b in loc if _cw(b)})),
        tuple(sorted({_ca(b) for b in loc if _ca(b)})),
    )


def merge_nearby_same_place(
    localities: list[list[dict]],
    max_gap: int | None = DEFAULT_MERGE_GAP,
    max_voters: int | None = None,
) -> list[list[dict]]:
    if max_gap is None or max_gap < 0:
        return localities

    merged: list[list[dict]] = []
    last_by_place: dict[tuple, list[dict]] = {}
    merges = 0
    for loc in localities:
        place = _place_key(loc)
        prev = last_by_place.get(place) if place else None
        gap = loc[0]["part_no"] - prev[-1]["part_no"] - 1 if prev else None
        over_cap = max_voters and prev and sum(b["total"] for b in prev + loc) > max_voters
        if prev and gap is not None and gap <= max_gap and not over_cap:
            prev.extend(loc)
            prev.sort(key=lambda b: b["part_no"])
            merges += 1
        else:
            merged.append(loc)
            if place:
                last_by_place[place] = loc
    if merges:
        log.info("merged %d same-place locality fragments with gap <= %d", merges, max_gap)
    return merged


def merge_rural_same_panchayat_runs(localities: list[list[dict]]) -> list[list[dict]]:
    """Club every rural blank-ward locality that shares one gram panchayat into a
    single locality — one panchayat, one locality.

    The panchayat is the place unit and outranks main_town_village: all of a
    panchayat's villages belong to it. Because panchayat_clean is Gemini-
    standardized (OCR variants collapsed, genuinely different panchayats kept
    apart), a shared clean panchayat proves the same place, so there is no
    booth-number gap limit — same-panchayat runs merge however far apart they
    sit. Panchayats are matched by exact clean name, never fuzzily: look-alike
    but distinct panchayats (नरौली vs बनौली) must not fuse. The block is part of
    the key too: a gram panchayat name is only unique within its development
    block, so two same-named panchayats in different blocks (e.g. भगवानपुर in
    पाली vs पिपरौली) stay separate — block is matched fuzzily since it is not
    standardized and carries OCR noise like "पिपरौली (आं0)". Ward-bearing
    (urban) localities are never touched."""
    out: list[list[dict]] = []
    reps: list[list[dict]] = []  # localities that started a (block, panchayat) group
    merges = 0

    for loc in localities:
        pan = "" if _uniq_wards(loc) else _main_panchayat(loc)
        blk = _main_block(loc)
        prev = next((rep for rep in reps if pan and _main_panchayat(rep) == pan
                     and _similar(_main_block(rep), blk)), None) if pan else None
        if prev is not None:
            prev.extend(loc)
            prev.sort(key=lambda b: b["part_no"])
            merges += 1
        else:
            out.append(loc)
            if pan:
                reps.append(loc)

    if merges:
        log.info("clubbed %d rural same-panchayat locality runs", merges)
    return out


def merge_same_ward_runs(localities: list[list[dict]]) -> list[list[dict]]:
    """Club localities at ward level: a ward is one place, so localities with
    the same village/city AND the same single ward merge into one locality no
    matter how far apart their booth runs sit or which polling-station
    buildings they use. The village stays in the key so identically numbered
    wards of two different towns never merge."""
    out: list[list[dict]] = []
    first_by_key: list[tuple[tuple[str, str], list[dict]]] = []
    merges = 0

    for loc in localities:
        uw = _uniq_wards(loc)
        uv = _uniq_villages(loc)
        key = (uv[0] if len(uv) == 1 else "", uw[0]) if len(uw) == 1 else None
        def same_key(k: tuple[str, str]) -> bool:
            # a blank village only matches blank: never club "वार्ड 1" of two towns
            villages_match = (k[0] == key[0]) or bool(k[0] and key[0] and _similar(k[0], key[0]))
            return villages_match and _similar(k[1], key[1])

        prev = next((l for k, l in first_by_key if key and same_key(k)), None)
        if prev is not None:
            prev.extend(loc)
            prev.sort(key=lambda b: b["part_no"])
            merges += 1
        else:
            out.append(loc)
            if key:
                first_by_key.append((key, loc))

    if merges:
        log.info("clubbed %d same-ward locality runs", merges)
    return out


def merge_same_building_runs(localities: list[list[dict]]) -> list[list[dict]]:
    """Club same-village localities that share one polling-station building.

    Some rural towns print a distinct mohalla/tola name in the ward column for
    almost every booth, so group_by_locality splits them into many one-booth
    localities. But several such tolas are often served by the very same
    school building, and that shared building is a harder physical-place
    signal than the locally printed tola name — so it outranks the ward here,
    the same way panchayat outranks village. No gap limit: the same building
    text (already OCR-canonicalized) means the same place regardless of
    distance."""
    out: list[list[dict]] = []
    first_by_key: list[tuple[str, str, list[dict]]] = []
    merges = 0

    for loc in localities:
        uv = _uniq_villages(loc)
        addr = _single_address(loc)
        village = uv[0] if len(uv) == 1 else ""
        key = (village, addr) if village and addr else None
        prev = next((l for v, a, l in first_by_key
                     if key and _similar(v, key[0]) and a == key[1]), None)
        if prev is not None:
            prev.extend(loc)
            prev.sort(key=lambda b: b["part_no"])
            merges += 1
        else:
            out.append(loc)
            if key:
                first_by_key.append((key[0], key[1], loc))

    if merges:
        log.info("clubbed %d same-building locality runs", merges)
    return out


def _voters(loc: list[dict]) -> int:
    return sum(b["total"] for b in loc)


def _building_groups(loc: list[dict]) -> list[list[dict]]:
    """Consecutive booths in the same polling-station building (per ps_address).
    A blank address never starts a new building."""
    groups: list[list[dict]] = []
    for b in loc:
        if not groups:
            groups.append([b])
            continue
        a = _ca(b)
        ga = next((_ca(x) for x in groups[-1] if _ca(x)), "")
        if not a or not ga or a == ga:
            groups[-1].append(b)
        else:
            groups.append([b])
    return groups


def split_oversized(localities: list[list[dict]], max_voters: int | None) -> list[list[dict]]:
    """Split localities over max_voters into one locality per polling-station
    building — each building is its own place, so different buildings are never
    packed into one chunk. A single building over the cap is split booth-wise
    as a last resort."""
    if not max_voters:
        return localities
    out: list[list[dict]] = []
    splits = 0
    for loc in localities:
        if _voters(loc) <= max_voters:
            out.append(loc)
            continue
        chunks: list[list[dict]] = []
        for g in _building_groups(loc):
            if _voters(g) > max_voters and len(g) > 1:
                cur: list[dict] = []
                for b in g:
                    if cur and _voters(cur) + b["total"] > max_voters:
                        chunks.append(cur)
                        cur = []
                    cur.append(b)
                if cur:
                    chunks.append(cur)
            else:
                chunks.append(g)
        splits += len(chunks) - 1
        out.extend(chunks)
    if splits:
        log.info("split oversized localities into %d extra per-building localities "
                 "(max %d voters)", splits, max_voters)
    return out


def coalesce_small(localities: list[list[dict]], min_voters: int | None,
                   max_voters: int | None) -> list[list[dict]]:
    """Merge consecutive under-sized localities ONLY when the booth details
    prove they are one place: both sides must belong to the same single gram
    panchayat, which then names the merged locality. Adjacency in part numbers
    alone is never enough."""
    if not min_voters:
        return localities
    out: list[list[dict]] = []
    merges = 0

    def can_join(a: list[dict], b: list[dict]) -> bool:
        if max_voters and _voters(a) + _voters(b) > max_voters:
            return False
        # exact clean-panchayat match only: look-alike but distinct panchayats
        # (नरौली vs बनौली) must never coalesce
        pa, pb = _main_panchayat(a), _main_panchayat(b)
        return bool(pa and pb and pa == pb)

    for loc in localities:
        if (out and (_voters(out[-1]) < min_voters or _voters(loc) < min_voters)
                and can_join(out[-1], loc)):
            out[-1].extend(loc)
            merges += 1
        else:
            out.append(loc)
    if merges:
        log.info("coalesced %d under-sized localities into same-panchayat "
                 "neighbours (min %d voters)", merges, min_voters)
    return out


def _uniq_values(loc: list[dict], getter) -> list[str]:
    out: list[str] = []
    for b in loc:
        v = getter(b)
        if v and not any(_similar(v, o) for o in out):
            out.append(v)
    return out


def _uniq_villages(loc: list[dict]) -> list[str]:
    return _uniq_values(loc, _cv)


def _uniq_wards(loc: list[dict]) -> list[str]:
    return _uniq_values(loc, _cw)


# words that end the institution part of a polling-station address; the text
# after the last of these (or after the last comma) is the area/mohalla
_INSTITUTION_WORDS = [
    "स्कूल", "विद्यालय", "कॉलेज", "कालेज", "एकेडमी", "एकेडेमी", "अकादमी",
    "पालिटेक्निक", "पॉलिटेक्निक", "मदरसा", "मन्दिर", "मंदिर", "केन्द्र", "केंद्र",
    "भवन", "हाईस्कूल", "पू0मा0वि0", "पू०मा०वि०", "प्रा0वि0", "प्रा०वि०",
    "उ0मा0वि0", "मा0वि0", "ई0का0", "इ0का0", "वि0", "वि०",
]


def _clean_area(area: str, strip_words: list[str]) -> str:
    area = _strip_rooms(area)
    area = re.sub(r"[.।,]", " ", area)
    for w in strip_words:
        if w:
            area = re.sub(rf"(?<!\S){re.escape(w)}(?!\S)", " ", area)
    area = re.sub(r"\(\s*\)", " ", area)
    area = _norm(area).strip("-, ")
    if area.count("(") != area.count(")"):
        area = _norm(area.replace("(", " ").replace(")", " "))
    area = _strip_number_tail(area)
    # a name must contain letters, not just leftover digits/punctuation
    return area if re.search(r"[^\W\d_]", area) else ""


def _single_address(loc: list[dict]) -> str:
    addrs = {_ca(b) for b in loc if _ca(b)}
    return next(iter(addrs)) if len(addrs) == 1 else ""


def _address_area(loc: list[dict], strip_words: list[str]) -> str:
    """The area/mohalla named in the locality's (single) polling-station address."""
    addr = _single_address(loc)
    if not addr:
        return ""
    if "," in addr:
        area = _clean_area(addr.rsplit(",", 1)[1], strip_words)
        if area:
            return area
    cut = max((addr.rfind(w) + len(w)) for w in _INSTITUTION_WORDS if w in addr) \
        if any(w in addr for w in _INSTITUTION_WORDS) else -1
    if cut > 0:
        return _clean_area(addr[cut:], strip_words)
    return ""


def _institution_from_address(addr: str) -> str:
    if not addr:
        return ""
    if "," in addr:
        return _norm(addr.split(",", 1)[0])
    cut = max((addr.rfind(w) + len(w)) for w in _INSTITUTION_WORDS if w in addr) \
        if any(w in addr for w in _INSTITUTION_WORDS) else -1
    return _norm(addr[:cut]) if cut > 0 else ""


def _institution_place_from_address(addr: str) -> str:
    inst = _institution_from_address(addr)
    if not inst:
        return ""
    return _clean_area(inst, _INSTITUTION_WORDS)


def _address_institution(loc: list[dict]) -> str:
    """The school/institution part of the locality's (single) address."""
    return _institution_from_address(_single_address(loc))


def _address_area_qualifier(loc: list[dict]) -> str:
    return _address_area(loc, _uniq_villages(loc) + _uniq_wards(loc))


def _address_institution_summary(loc: list[dict]) -> str:
    institutions = _uniq_values(loc, lambda b: _institution_place_from_address(_ca(b)))
    if len(institutions) <= 2:
        return " / ".join(institutions)
    return f"{institutions[0]} आदि ({len(institutions)} संस्थान)"


def _single_ward(loc: list[dict]) -> str:
    uw = _uniq_wards(loc)
    return uw[0] if len(uw) == 1 else ""


def _append_qualifier(name: str, qualifier: str) -> str:
    qualifier = _norm(qualifier)
    parts = [_strip_number_tail(_norm(p)) for p in qualifier.split("/")]
    parts = [p for p in parts if p and not _similar(name, p)]
    # only a single place name qualifies; "A / B" lists and "X आदि" summaries
    # confuse more than they clarify
    if len(parts) != 1 or "आदि" in parts[0]:
        return name
    return f"{name} ({parts[0]})"


def _dedupe_names_with_qualifiers(names: list[str], localities: list[list[dict]]) -> list[str]:
    """Make locality names unique by adding the least specific useful qualifier.

    Ward is most local, panchayat is the next administrative clue, and polling-
    station institution is only needed when the administrative labels still
    collide. Numbering remains for true same-building booth splits.
    """
    out = list(names)
    # qualifiers are PLACE names only: ward, panchayat, or the area printed in
    # the polling-station address. Never the institution — "लाल बहादुर शास्त्री
    # पू0मा0" is a school, not a location. Names that stay duplicated after
    # these rungs simply stay duplicated; locality_id/booth_parts distinguish.
    ladders = [
        _single_ward,
        _main_panchayat,
        _address_area_qualifier,
    ]

    for qualifier_for in ladders:
        counts = Counter(out)
        duplicate_names = {n for n, c in counts.items() if c > 1}
        if not duplicate_names:
            return out
        # cluster the qualifier strings first so OCR variants of one area
        # ("जं0बेनी माधो नं0 2" / "नं 0 2") collapse instead of producing
        # visually identical "different" names; identical qualifiers then
        # stay duplicates and fall through to the next, more specific rung
        quals = [qualifier_for(localities[i]) if out[i] in duplicate_names else ""
                 for i in range(len(out))]
        reps: list[str] = []
        for i, q in enumerate(quals):
            if not q:
                continue
            rep = next((r for r in reps if _similar(r, q)), None)
            if rep is None:
                reps.append(q)
                rep = q
            quals[i] = rep
        # adopt a qualified name only where it actually reduces ambiguity;
        # a qualifier shared by the whole group is noise, not a distinction
        tentative = [_append_qualifier(out[i], quals[i]) if out[i] in duplicate_names
                     else out[i] for i in range(len(out))]
        tcounts = Counter(tentative)
        for i, name in enumerate(out):
            if (name in duplicate_names and tentative[i] != name
                    and tcounts[tentative[i]] < counts[name]):
                out[i] = tentative[i]

    # names identical even after all qualifiers are one building/place whose
    # booth runs sit too far apart to merge; keep the same name for both —
    # numbers or booth ranges in a name say nothing about the area, and the
    # locality_id / part_range columns already distinguish the rows
    return out


def name_localities(localities: list[list[dict]], city: str = "") -> list[str]:
    # how many localities each single village spans: a city/town subdivided by
    # wards spans many, and those localities are best named after their ward
    village_span = Counter()
    for loc in localities:
        uv = _uniq_villages(loc)
        if len(uv) == 1:
            village_span[uv[0]] += 1

    # a locality built from merge_same_building_runs can carry many distinct
    # tola/mohalla ward names (one building serving several printed tolas);
    # listing them all is unwieldy and no single one names the place, so those
    # localities get "village (वार्ड N)" instead — counted per village in
    # booth order so repeats across one village number consistently
    ward_seq: Counter = Counter()

    names = []
    for loc in localities:
        uv = _uniq_villages(loc)
        uw = _uniq_wards(loc)
        if not uv:
            names.append((uw[0] if uw else "") or _norm(loc[0]["panchayat"]) or "(unknown)")
        elif len(uv) == 1:
            if village_span[uv[0]] > 1 and uw:
                if len(uw) <= 3:
                    base = " / ".join(uw)
                    if uv[0] and uv[0] != city and not _similar(uv[0], base):
                        base = f"{uv[0]} - {base}"
                else:
                    n = ward_seq[uv[0]]
                    ward_seq[uv[0]] += 1
                    base = uv[0] if n == 0 else f"{uv[0]} (वार्ड {n})"
                names.append(base)
            else:
                names.append(uv[0])
        else:
            # multi-village localities exist only via a shared panchayat, which
            # is the one common name the booth details give this place
            p = _main_panchayat(loc)
            if p:
                names.append(p)
            elif len(uv) <= 3:
                names.append(" / ".join(uv))
            else:
                names.append(f"{uv[0]} आदि ({len(uv)} गाँव)")
    # booth bookkeeping numbers ("नं0 2", ". 01") are not part of a place name
    names = [_strip_number_tail(n) or n for n in names]
    # a name shared by several localities (a split ward/village) is better
    # replaced by the area printed in the polling-station address
    counts = Counter(names)
    for i, loc in enumerate(localities):
        if counts[names[i]] <= 1:
            continue
        if _single_ward(loc):
            continue
        area = _address_area(loc, _uniq_villages(loc))
        if area and not _similar(area, names[i]):
            names[i] = area
    return _dedupe_names_with_qualifiers(names, localities)


def main() -> None:
    ap = argparse.ArgumentParser(description="Group AC booths into pure place/locality runs")
    ap.add_argument("--ac", help="AC number; reads booth_list_csv/booth_info_gemini_<AC>.csv")
    ap.add_argument("--csv", type=Path, help="Booth-info CSV (overrides --ac)")
    ap.add_argument("--out", type=Path, help="Output localities CSV")
    ap.add_argument("--max-voters", type=int, default=DEFAULT_MAX_VOTERS, dest="max_voters",
                    help="Optional cap: split localities larger than this at polling-station "
                         "boundaries (default 0 = no cap)")
    ap.add_argument("--min-voters", type=int, default=DEFAULT_MIN_VOTERS, dest="min_voters",
                    help="Optional floor: coalesce localities smaller than this into a "
                         "same-panchayat neighbour (default 0 = no floor)")
    ap.add_argument("--canonical-threshold", type=float, default=DEFAULT_CANON_THRESHOLD,
                    help="Village OCR-variant clustering threshold")
    ap.add_argument("--canonical-prefix", type=int, default=DEFAULT_CANON_PREFIX,
                    help="Minimum shared leading characters for village OCR-variant clustering")
    ap.add_argument("--canonical-window", type=int, default=DEFAULT_CANON_WINDOW,
                    help="Only cluster village OCR variants with occurrences this many booths apart")
    ap.add_argument("--merge-gap", type=int, default=DEFAULT_MERGE_GAP,
                    help="Merge same-village fragments when separated by at most this many booth numbers")
    args = ap.parse_args()

    if not args.ac and not args.csv:
        ap.error("provide --ac or --csv")
    csv_path = args.csv or (HERE / "booth_list_csv" / f"booth_info_gemini_{args.ac}.csv")
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    ac = args.ac or re.sub(r"\D", "", csv_path.stem) or csv_path.stem
    out_csv = args.out or HERE / "localities" / f"localities_{ac}.csv"
    map_stem = out_csv.stem.replace("localities", "booth_locality_map")
    if map_stem == out_csv.stem:
        map_stem = out_csv.stem + "_booth_map"
    # default map CSV lives in localities_map/; an explicit --out keeps it alongside
    map_csv = (HERE / "localities_map" / (map_stem + ".csv")) if args.out is None \
        else out_csv.with_name(map_stem + ".csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    map_csv.parent.mkdir(parents=True, exist_ok=True)

    booths = load_booths(csv_path)
    if not booths:
        raise SystemExit(f"No usable booth rows in {csv_path}")
    total_v = sum(b["total"] for b in booths)
    log.info("%d booths, %d voters in AC %s", len(booths), total_v, ac)

    canonicalize_field(booths, "village", "canonical_village", args.canonical_threshold,
                       args.canonical_prefix, args.canonical_window)
    canonicalize_field(booths, "ward", "canonical_ward", args.canonical_threshold,
                       args.canonical_prefix, args.canonical_window)
    # addresses use a strict similarity but no prefix guard: OCR errors often hit
    # the first letters (जामिया/जमिया) and 0.88 on a long string is proof enough
    canonicalize_field(booths, "ps_clean", "canonical_ps", ADDRESS_CANON_THRESHOLD,
                       1, ADDRESS_CANON_WINDOW)
    # panchayat_clean is standardized once by extract_booth_info_gemini.py (one
    # Gemini call per AC, cached); use it directly when the booth-info CSV has
    # it, and fall back to local fuzzy clustering only for older CSVs that don't
    if any(b["panchayat_clean"] for b in booths):
        for b in booths:
            b["canonical_panchayat"] = _norm(b["panchayat_clean"]) or _norm(b["panchayat"])
        log.info("using panchayat_clean column from the booth-info CSV")
    else:
        canonicalize_field(booths, "panchayat", "canonical_panchayat", args.canonical_threshold,
                           args.canonical_prefix, args.canonical_window)
    city = dominant_city(booths)
    if city:
        log.info("urban AC: %s covers most booths; buildings partition its wards", city)
    else:
        log.info("rural AC: villages/panchayats partition, wards club regardless of gap")
    localities = group_by_locality(booths, city)
    localities = merge_nearby_same_place(localities, args.merge_gap, args.max_voters)
    localities = merge_rural_same_panchayat_runs(localities)
    if not city:
        localities = merge_same_ward_runs(localities)
        localities = merge_same_building_runs(localities)
    localities = split_oversized(localities, args.max_voters)
    localities = coalesce_small(localities, args.min_voters, args.max_voters)
    names = name_localities(localities, city)

    loc_rows, map_rows = [], []
    for idx, (loc, nm) in enumerate(zip(localities, names), 1):
        lid = f"AC{ac}-L{idx:03d}"
        parts = [b["part_no"] for b in loc]
        loc_rows.append({
            "locality_id": lid, "locality_name": nm,
            "block": Counter(_norm(b["block"]) for b in loc if _norm(b["block"])).most_common(1)[0][0] if any(_norm(b["block"]) for b in loc) else "",
            "tehsil": Counter(_norm(b["tehsil"]) for b in loc if _norm(b["tehsil"])).most_common(1)[0][0] if any(_norm(b["tehsil"]) for b in loc) else "",
            "panchayats": " / ".join(dict.fromkeys(_cp(b) for b in loc if _cp(b))),
            "villages": " / ".join(_uniq_villages(loc)),
            "wards": " / ".join(_uniq_wards(loc)),
            "num_booths": len(loc),
            "part_range": f"{parts[0]}-{parts[-1]}" if parts[0] != parts[-1] else str(parts[0]),
            "booth_parts": ",".join(map(str, parts)),
            "male": sum(b["male"] for b in loc), "female": sum(b["female"] for b in loc),
            "total_voters": sum(b["total"] for b in loc),
            "counts_flagged": sum(1 for b in loc if b["counts_ok"] != "Y"),
        })
        for b in loc:
            map_rows.append({"part_no": b["part_no"], "booth_pdf": b["booth_pdf"],
                             "locality_id": lid, "locality_name": nm,
                             "main_town_village": b["village"], "ward": b["ward"],
                             "panchayat": b["panchayat"], "ps_address": b["ps_address"],
                             "block": b["block"], "tehsil": b["tehsil"],
                             "total": b["total"],
                             "canonical_village": _cv(b), "canonical_ward": _cw(b),
                             "canonical_panchayat": _cp(b),
                             "canonical_ps_address": _ca(b)})
    map_rows.sort(key=lambda r: r["part_no"])

    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOC_COLUMNS); w.writeheader(); w.writerows(loc_rows)
    with map_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MAP_COLUMNS); w.writeheader(); w.writerows(map_rows)

    sizes = [r["total_voters"] for r in loc_rows]
    nb = [r["num_booths"] for r in loc_rows]
    log.info("=> %d localities  (voters: min %d, max %d, avg %d | booths/locality: min %d, max %d, avg %.1f)",
             len(loc_rows), min(sizes), max(sizes), total_v // len(loc_rows),
             min(nb), max(nb), sum(nb) / len(nb))
    biggest = max(loc_rows, key=lambda r: r["num_booths"])
    if biggest["num_booths"] > max(25, len(booths) // 5):
        log.warning("locality %s (%s) spans %d booths — the booth-info CSV may lack "
                    "distinguishing village/ward values; inspect it before using",
                    biggest["locality_id"], biggest["locality_name"], biggest["num_booths"])
    log.info("wrote %s  and  %s", out_csv.name, map_csv.name)


if __name__ == "__main__":
    main()
