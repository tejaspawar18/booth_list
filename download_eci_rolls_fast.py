"""
Fast, resumable booth-roll downloader for the generate-on-demand ECI rolls.

Why this exists (see download_eci_rolls.py for the protocol itself)
-------------------------------------------------------------------
Only the UP 2026 SIR roll is pre-generated on the CDN.  Every other roll tested
- Maharashtra 2024 (all three), the 2026 Baramati bye-election roll, and every
Andhra Pradesh roll (2024/2025) - 404s on the CDN for every booth, so the only
route is `generate-published-pdfs` (captcha) -> a fileId redeemed at
gateway-vpd `ext-printing-publish/get-published-file`.

That endpoint has a hard, measured limit: the gateway allows ~15s to serve a
fileId.  Booths whose PDF can't be produced in the window return
`504 upstream request timeout` (or cut the chunked transfer mid-stream), and the
fileId is then dead forever - every later fetch says "File Fetch Time Expire".
So a fetch is one shot, and a meaningful fraction of booths fail every time.

Measured on the Baramati/AC-223 bye-election roll (2026-07-22):

  * hit rate is ~17-33% per attempt and is *unaffected* by concurrency
    (1/6 at conc=1, 1/6 at conc=3, 2/6 at conc=8) - so concurrency is free
  * wall time is dominated by the fixed ~15s per fetch: 15.7s/booth serially
    vs 2.4s/booth at conc=8, a ~7x speedup for the same yield
  * a *large* generate batch is fine and amortises the captcha: batch=24 at
    conc=8 scored 8/24 in 57s, one captcha for the lot
  * booths that fail are not permanently doomed - a later round with *fresh*
    fileIds recovers more of them, and on a healthy roll the rounds **converge
    to full coverage**: Andhra S01-2025-FIR AC 1 went 19/24 -> 24/24 (100%) in
    ~2.5 min total. So run enough rounds rather than assuming a low ceiling.

**Pick the right roll - it matters more than any tuning.** Yield per round, same
code path, same 24 booths:

    S01-2025-FIR  (AP, SSR final)      19/24  79%   -> 100% after 3 rounds
    S13-2024-FIR  (MH, 2nd SSR final)   9/24  38%
    S13-2026-BY-* (MH, bye-election)    ~1/3  33%   plateaus, roll is degraded
    S01-2024-GEN-FIR (AP, Gen. Elec.)   4/24  17%
    S13-2024-GEN-FIR (MH, Gen. Elec.)   0/24   0%   effectively unavailable

General Election rolls (pdfGenType FC-EROLLGEN) are barely servable, and the
portal lists them *first* - so the naive "first FinalRoll" pick is the worst
one. `pick_roll` below prefers SIR > ordinary/SSR > General Election.

Design that follows from those facts
------------------------------------
  1. Big generate batches (one captcha each), fetched at high concurrency.
  2. One fetch per fileId, ever.  A 504/short read means "regenerate", not
     "retry this id".
  3. Rounds of fresh generates for whatever is still missing, stopped early
     when a round adds nothing (plateau) instead of grinding pointlessly.
  4. A per-AC manifest so runs are resumable and coverage accumulates across
     days.  This is the intended way to use it: re-run periodically.
  5. The CDN is still probed first, so CDN-backed rolls (UP) stay fast.

Usage
-----
    PY=../../jup_venv/bin/python

    # one AC, resumable
    $PY download_eci_rolls_fast.py --state Maharashtra --year 2026 \
        --district Pune --ac 201

    # a whole district, more rounds
    $PY download_eci_rolls_fast.py --state Maharashtra --year 2024 \
        --district Akola --all-acs --rounds 6

    # pick a specific roll when a year publishes several
    $PY download_eci_rolls_fast.py --state Maharashtra --year 2024 \
        --roll-id S13-2024-FIR --district Akola --ac 30

    # coverage so far, no downloading
    $PY download_eci_rolls_fast.py --state Maharashtra --year 2026 \
        --district Pune --ac 201 --report
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

import download_eci_rolls as base
from download_eci_rolls import (
    EciClient, VPD_GATEWAY, cdn_path, download, generate_with_captcha,
    load_env_files, parse_parts, _match,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eci-fast")

FETCH_URL = f"{VPD_GATEWAY}/api/v1/ext-printing-publish/get-published-file"
FETCH_HEADERS = {"CurrentRole": "citizen", "PLATFORM-TYPE": "web"}
MIN_PDF_BYTES = 100_000
FETCH_SESSION: requests.Session | None = None


class Manifest:
    """Per-AC record of what has been fetched and what keeps failing.

    Coverage on these rolls accumulates over repeated runs, so the manifest is
    the difference between "re-run and make progress" and "re-run and redo the
    same 30%". `attempts` also lets us report which booths look truly stuck."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = {"done": [], "attempts": {}}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
                self.data.setdefault("done", [])
                self.data.setdefault("attempts", {})
            except (ValueError, OSError):
                log.warning("unreadable manifest %s; starting fresh", path)

    @property
    def done(self) -> set[int]:
        return {int(p) for p in self.data["done"]}

    def attempts(self, part: int) -> int:
        return int(self.data["attempts"].get(str(part), 0))

    def mark_done(self, part: int) -> None:
        if part not in self.done:
            self.data["done"].append(part)

    def mark_attempt(self, part: int) -> None:
        self.data["attempts"][str(part)] = self.attempts(part) + 1

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["done"] = sorted(self.done)
        tmp = self.path.with_suffix(".json.part")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(self.path)


def fetch_session(pool: int) -> requests.Session:
    """Session dedicated to fileId redemption, with retries *disabled*.

    The shared EciClient session mounts an adapter with max_retries=2, which is
    right for the CDN but actively harmful here: a fileId is single-use, so a
    transport-level retry of a timed-out fetch can never succeed and just costs
    another full timeout. Paired with the short read timeout below (a good fetch
    completes in ~15s; the gateway itself gives up at 15s), this bounds the cost
    of a hung request at ~45s instead of ~240s."""
    s = requests.Session()
    s.headers.update(base._HEADERS)
    s.mount("https://", requests.adapters.HTTPAdapter(
        pool_connections=pool, pool_maxsize=pool, max_retries=0))
    return s


def fetch_file(cli: EciClient, file_id: str, out: Path) -> bool:
    """Redeem one fileId. Single attempt by contract — see module docstring."""
    try:
        r = FETCH_SESSION.get(FETCH_URL, params={"fileId": file_id},
                              headers=FETCH_HEADERS, timeout=(10, 45))
        if r.status_code != 200:
            return False
        body = r.content
    except requests.RequestException:
        # includes ChunkedEncodingError: the gateway cut the transfer mid-stream
        return False
    if len(body) < 200:
        return False
    try:
        payload = r.json().get("payload") or ""
    except ValueError:
        return False
    if not payload:
        return False
    import base64
    raw = base64.b64decode(payload)
    if raw[:4] != b"%PDF" or len(raw) < MIN_PDF_BYTES:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    tmp.write_bytes(raw)
    tmp.replace(out)
    return True


def retry_call(fn, *a, tries: int = 5, what: str = "request"):
    """Retry a reference-data call through transient network failures.

    These calls (ac_languages, part_list) are cheap and idempotent, unlike a
    fileId fetch. A district run is hours long, so a single timed-out metadata
    request must not be allowed to end it — that once cost 10 unprocessed ACs."""
    for attempt in range(1, tries + 1):
        try:
            return fn(*a)
        except requests.RequestException as exc:
            if attempt == tries:
                raise
            wait = min(30.0, 2 ** attempt)
            log.warning("%s failed (%s); retry %d/%d in %.0fs",
                        what, type(exc).__name__, attempt, tries, wait)
            time.sleep(wait)


def pick_roll(rolls: list[dict], want: str) -> dict:
    """Choose the roll most likely to actually yield PDFs.

    Order matters a lot and the portal's own ordering is the wrong one. Measured
    on Akola AC 30, same 24 booths, same code path: the General Election roll
    `S13-2024-GEN-FIR` returned **0/24**, while the SSR roll `S13-2024-FIR`
    returned **9/24 (38%)**. GE rolls (pdfGenType FC-EROLLGEN) appear not to be
    retained in a servable form. Since GE rolls are typically listed first, the
    obvious "first FinalRoll" pick is the worst possible one — hence this
    explicit preference: SIR (CDN-backed) > ordinary/SSR > General Election."""
    def score(r: dict) -> tuple:
        ref = str(r.get("rollTypeRefId") or "")
        is_want = r.get("rollType") == want
        is_sir = ref.startswith("SIR-")
        is_ge = str(r.get("pdfGenType") or "").startswith("FC-") or "GEN-" in str(r.get("id"))
        # higher is better
        return (is_want, is_sir, not is_ge, r.get("revisionNo") or 0)

    best = max(rolls, key=score)
    ge = str(best.get("pdfGenType") or "").startswith("FC-")
    if ge:
        log.warning("only a General Election roll is available (%s); these often "
                    "yield no PDFs at all — check --list-rolls for alternatives",
                    best["id"])
    return best


def _chunks(seq: list[int], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def generate_batches(cli: EciClient, roll: dict, ac: int, dcd: str | None,
                     batches: list[list[int]], lang: str, out_dir: Path,
                     env_file: Path | None, captcha_tries: int,
                     out_q: "queue.Queue") -> None:
    """Producer: solve captcha + generate fileIds for each batch, one ahead of
    the fetcher. Generation costs a captcha solve (2-15s with rejections) that
    would otherwise be dead time between batches."""
    for chunk in batches:
        # The captcha service intermittently returns a payload with no `data`
        # (base64 of None -> TypeError) and can time out. Losing the whole batch
        # to that wastes a round, so give it a couple of fresh attempts here.
        for attempt in range(1, 4):
            try:
                gen = generate_with_captcha(cli, roll, ac, dcd, chunk, lang,
                                            out_dir, env_file, captcha_tries)
                payload = gen.get("payload") or []
                out_q.put((chunk, payload, gen.get("refId") == "CDN", None))
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 3:
                    out_q.put((chunk, [], False, exc))
                else:
                    log.warning("AC %s: generate attempt %d failed (%s); retrying",
                                ac, attempt, type(exc).__name__)
                    time.sleep(3 * attempt)
    out_q.put(None)


def run_round(cli: EciClient, roll: dict, ac: int, dcd: str | None,
              pending: list[int], lang: str, jobs: dict[int, Path],
              args, man: Manifest) -> set[int]:
    """One round: generate fresh fileIds for `pending` and fetch them all.
    Returns the parts still missing."""
    batches = list(_chunks(pending, max(1, args.batch)))
    q: "queue.Queue" = queue.Queue(maxsize=1)
    prod = threading.Thread(target=generate_batches, daemon=True,
                            args=(cli, roll, ac, dcd, batches, lang, jobs[pending[0]].parent,
                                  args.env_file, args.captcha_tries, q))
    prod.start()

    still = set(pending)
    while True:
        item = q.get()
        if item is None:
            break
        chunk, payload, via_cdn, exc = item
        if exc is not None:
            log.error("AC %s: generate failed for %d booth(s): %s", ac, len(chunk), exc)
            continue
        pairs = [(p, it) for p, it in zip(chunk, payload) if it]
        if not pairs:
            log.warning("AC %s: portal generated nothing for %d booth(s)", ac, len(chunk))
            continue

        def _one(pi: tuple[int, str]) -> tuple[int, bool]:
            part, item_id = pi
            got = (download(cli.s, item_id, jobs[part], attempts=1) if via_cdn
                   else fetch_file(cli, item_id, jobs[part]))
            return part, got

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max(1, min(args.workers, len(pairs)))) as ex:
            results = list(ex.map(_one, pairs))
        won = 0
        for part, got in results:
            man.mark_attempt(part)
            if got:
                man.mark_done(part)
                still.discard(part)
                won += 1
        man.save()
        log.info("AC %s: batch %d booths -> %d saved (%.1fs/booth)",
                 ac, len(pairs), won, (time.time() - t0) / max(1, len(pairs)))
    return still


def process_ac(cli: EciClient, roll: dict, ac: int, dcd: str | None, state_cd: str,
               base_out: Path, args, part_filter: set[int] | None) -> tuple[int, int]:
    lang = args.lang or next(iter(
        retry_call(cli.ac_languages, state_cd, dcd, ac,
                   what=f"AC {ac} languages") or {"ENG": ""}), "ENG")
    parts = [int(p["partNumber"]) for p in
             retry_call(cli.part_list, state_cd, dcd, ac, what=f"AC {ac} part list")]
    parts = sorted(set(parts) & part_filter) if part_filter else sorted(set(parts))
    if not parts:
        log.warning("AC %s: no booths listed", ac)
        return 0, 0

    ac_dir = base_out / str(ac)
    man = Manifest(ac_dir / "_manifest.json")
    jobs = {p: ac_dir / f"booth_{p:04d}.pdf" for p in parts}

    # trust the filesystem over the manifest: a file on disk is done
    for p in parts:
        if jobs[p].exists() and jobs[p].stat().st_size > MIN_PDF_BYTES:
            man.mark_done(p)
    pending = [p for p in parts if p not in man.done]
    log.info("AC %s: %d booths, %d already have PDFs, %d to fetch (lang %s)",
             ac, len(parts), len(parts) - len(pending), len(pending), lang)
    if args.report:
        stuck = [p for p in pending if man.attempts(p) >= 3]
        log.info("AC %s: coverage %d/%d (%.0f%%); %d booth(s) failed 3+ times",
                 ac, len(man.done & set(parts)), len(parts),
                 100 * len(man.done & set(parts)) / len(parts), len(stuck))
        return len(man.done & set(parts)), len(pending)
    if not pending:
        return len(parts), 0

    # CDN first: on a pre-generated roll (UP SIR) this finishes the whole AC.
    if not args.no_cdn:
        probe = {p: (cdn_path(roll, ac, p, lang), jobs[p]) for p in pending[:3]}
        missed = base.download_batch(cli.s, probe, args.workers)
        for p in probe:
            man.mark_attempt(p)
            if p not in missed:
                man.mark_done(p)
        if len(missed) < len(probe):
            log.info("AC %s: roll is on the CDN — downloading all booths directly", ac)
            rest = {p: (cdn_path(roll, ac, p, lang), jobs[p])
                    for p in pending if p not in probe}
            still = base.download_batch(cli.s, rest, args.workers)
            for p in rest:
                man.mark_attempt(p)
                if p not in still:
                    man.mark_done(p)
            man.save()
            pending = sorted(still | (missed & set(probe)))
        else:
            pending = [p for p in pending if p not in man.done]
        man.save()

    # generate-on-demand rounds
    t0 = time.time()
    for rnd in range(1, args.rounds + 1):
        if not pending:
            break
        # try least-attempted booths first: they have the best odds and it keeps
        # a stuck booth from soaking up every round
        pending.sort(key=lambda p: (man.attempts(p), p))
        before = len(pending)
        pending = sorted(run_round(cli, roll, ac, dcd, pending, lang, jobs, args, man))
        gained = before - len(pending)
        log.info("AC %s: round %d/%d +%d booth(s), %d still missing (%.0fs elapsed)",
                 ac, rnd, args.rounds, gained, len(pending), time.time() - t0)
        if gained == 0 and rnd >= args.min_rounds:
            # A no-progress round is the normal way a *successful* AC ends, so
            # only editorialise when the coverage actually looks bad.
            pct = 100 * len(man.done & set(parts)) / len(parts)
            if pct < 80:
                log.info("AC %s: round added nothing at %.0f%% — this roll is likely "
                         "degraded (General Election rolls especially); check "
                         "--list-rolls for a better one.", ac, pct)
            else:
                log.info("AC %s: round added nothing — done at %.0f%%", ac, pct)
            break

    have = len(man.done & set(parts))
    log.info("AC %s: coverage %d/%d (%.0f%%) after %.0fs",
             ac, have, len(parts), 100 * have / len(parts), time.time() - t0)
    return have, len(pending)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fast resumable downloader for generate-on-demand ECI booth rolls")
    ap.add_argument("--state", required=True, help="State name or code (e.g. Maharashtra / S13)")
    ap.add_argument("--year", required=True)
    ap.add_argument("--roll", default="final", choices=["final", "draft"])
    ap.add_argument("--roll-id", help="Exact roll id when a year publishes several "
                                      "(see --list-rolls), e.g. S13-2024-FIR")
    ap.add_argument("--list-rolls", action="store_true", help="Show published rolls and exit")
    ap.add_argument("--district", help="District name or code")
    ap.add_argument("--ac", nargs="+", type=int)
    ap.add_argument("--all-acs", action="store_true", help="Every AC in --district")
    ap.add_argument("--lang")
    ap.add_argument("--parts", help="Booth subset, e.g. '1-50'")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "booth_list_pdf")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel fetches (default 8). Hit rate is unaffected by "
                         "concurrency, so this is close to a free speedup.")
    ap.add_argument("--batch", type=int, default=24,
                    help="Booths per captcha/generate call (default 24). Bigger "
                         "batches amortise the captcha without hurting the hit rate.")
    ap.add_argument("--rounds", type=int, default=8,
                    help="Max rounds of fresh generates per AC (default 8). Rounds are "
                         "cheap (~25s) and a healthy roll converges to 100%% within a "
                         "few of them, so this is worth leaving high.")
    ap.add_argument("--min-rounds", type=int, default=3,
                    help="Rounds to run before the no-progress early stop (default 3)")
    ap.add_argument("--captcha-tries", type=int, default=8)
    ap.add_argument("--no-cdn", action="store_true", help="Skip the CDN probe entirely")
    ap.add_argument("--report", action="store_true", help="Print coverage, download nothing")
    ap.add_argument("--env-file", type=Path)
    args = ap.parse_args()

    load_env_files(args.env_file)
    global FETCH_SESSION
    FETCH_SESSION = fetch_session(max(4, args.workers))
    cli = EciClient(pool_size=max(4, args.workers))

    state = _match(cli.states(), args.state, "stateName", "stateCd")
    if not state:
        sys.exit(f"State not found: {args.state!r}")
    state_cd = state["stateCd"]

    rolls = cli.eroll_types(state_cd, args.year)
    if not rolls:
        sys.exit(f"No published rolls for {state_cd} {args.year}")
    if args.list_rolls:
        # Probe each roll on the CDN. This is the single most useful thing to
        # know before starting: a CDN-backed roll downloads at ~1 booth/s with
        # full coverage, anything else crawls at ~33% yield. Empirically only
        # SIR-FinalRoll / SIR-DraftRoll are pre-generated (UP, Gujarat, ...),
        # so this doubles as a way to poll for a state's SIR roll going live.
        probe_ac, probe_dcd = None, None
        if args.ac:
            probe_ac = args.ac[0]
            if args.district:
                dd = _match(cli.districts(state_cd), args.district,
                            "districtValue", "districtCd", "districtNo")
                probe_dcd = dd and dd["districtCd"]
        if probe_ac is None:
            dd = cli.districts(state_cd)[0]
            probe_dcd = dd["districtCd"]
            probe_ac = int(cli.acs(probe_dcd)[0]["asmblyNo"])
        langs = list(cli.ac_languages(state_cd, probe_dcd, probe_ac) or {}) or ["ENG"]
        print(f"(CDN probed on AC {probe_ac}, languages {','.join(langs)})\n")
        for r in rolls:
            speed = "-"
            if r.get("rollTypeRefId"):
                # 403 from the edge is a cached throttle (~10 min), NOT absence —
                # reporting it as SLOW would wrongly write off a CDN-backed roll.
                speed, throttled = "SLOW (generate)", False
                for lang in langs[:2]:
                    for part in (1, 2):
                        url = f"{base.CDN_ROOT}/{cdn_path(r, probe_ac, part, lang)}"
                        try:
                            code = cli.s.get(url, timeout=45, stream=True).status_code
                        except requests.RequestException:
                            continue
                        if code == 200:
                            speed = f"FAST (CDN, {lang})"
                            break
                        if code in (403, 429):
                            throttled = True
                    if speed.startswith("FAST"):
                        break
                if throttled and not speed.startswith("FAST"):
                    speed = "UNKNOWN (throttled)"
            print(f"{r['id']:<22} {str(r.get('rollTypeRefId') or '-'):<16} "
                  f"rev{r.get('revisionNo')} {speed:<18} {r.get('displayName')}")
            for by in (r.get("byElecAcList") or []):
                print(f"{'':<22}   bye-election AC {by['acNo']} {by.get('acName')}")
        return

    want = "FinalRoll" if args.roll == "final" else "DraftRoll"
    if args.roll_id:
        roll = _match(rolls, args.roll_id, "id")
        if not roll:
            sys.exit(f"Roll id not found: {args.roll_id!r} "
                     f"(available: {', '.join(r['id'] for r in rolls)})")
    else:
        roll = pick_roll(rolls, want)
    log.info("State %s | roll %s (%s)", state_cd, roll["id"], roll.get("displayName"))

    roll_slug = str(roll["id"]).lower()
    base_out = args.out / state_cd.lower() / roll_slug

    district = None
    if args.district:
        district = _match(cli.districts(state_cd), args.district,
                          "districtValue", "districtCd", "districtNo")
        if not district:
            sys.exit(f"District not found: {args.district!r}")

    ac_numbers = list(args.ac or [])
    ac_district: dict[int, str] = {}
    if args.all_acs:
        if not district:
            sys.exit("--all-acs requires --district")
        for a in cli.acs(district["districtCd"]):
            ac_numbers.append(int(a["asmblyNo"]))
            ac_district[int(a["asmblyNo"])] = a["districtCd"]
    if not ac_numbers:
        sys.exit("Nothing to do: pass --ac N... or --all-acs (with --district).")
    if district:
        for a in ac_numbers:
            ac_district.setdefault(a, district["districtCd"])

    part_filter = parse_parts(args.parts)
    total_have = total_left = 0
    failed_acs: list[int] = []
    for ac in sorted(set(ac_numbers)):
        # Never let one AC take down a multi-hour district run; progress is on
        # disk and in the manifests, so the next run resumes what this skips.
        try:
            have, left = process_ac(cli, roll, ac, ac_district.get(ac), state_cd,
                                    base_out, args, part_filter)
        except Exception as exc:  # noqa: BLE001
            log.error("AC %s failed (%s: %s) — continuing with the next AC; "
                      "re-run to retry it", ac, type(exc).__name__, exc)
            failed_acs.append(ac)
            continue
        total_have += have
        total_left += left
    if failed_acs:
        log.warning("ACs that errored out entirely: %s",
                    ", ".join(map(str, failed_acs)))
    log.info("Total: %d booth PDFs on disk, %d still missing. Output under %s",
             total_have, total_left, base_out)
    if total_left:
        log.info("Re-run the same command later to pick up more — these rolls "
                 "release booths gradually across runs.")


if __name__ == "__main__":
    main()
