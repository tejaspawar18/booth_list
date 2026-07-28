"""
Extract voter rows from the elector pages of ECI booth-roll PDFs using Gemini vision.

The elector pages (page 3 onward; the last pages are summary tables) print a
3 x 10 grid of voter cards, each holding:

    serial number, EPIC number, Name, Father's/Husband's/Mother's Name,
    House Number, Age, Gender

This script renders each such page to a JPEG, sends it to Gemini and asks for a
strict JSON array of voter objects, then appends them to one CSV per AC.

Credentials
-----------
Same as extract_booth_info_gemini.py -- put one of these in survey/.env,
survey/booth_list/.env, or your shell env:

    GEMINI_API_KEY=...                                # AI Studio key
    GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json   # Vertex AI (GCP credits)

Usage
-----
    PY=../../jup_venv/bin/python
    # one booth, first elector page only (cheap smoke test)
    $PY extract_voters_gemini.py \
        --pdf-dir booth_list_pdf/s13/2024-finalroll/228 --parts 1 --max-pages 1 --debug

    # whole AC folder
    $PY extract_voters_gemini.py --pdf-dir booth_list_pdf/s13/2024-finalroll/228 --workers 6

    # explicit page window (1-based, inclusive)
    $PY extract_voters_gemini.py --pdf-dir ... --first-page 3 --last-page 27

Resume
------
Pages already recorded in the usage CSV (<out>.usage.csv) are skipped, so an
interrupted run can simply be re-run. Use --refresh to redo everything.

Output
------
    booth_list_csv/voters_<AC>.csv        one row per voter
    booth_list_csv/voters_<AC>.usage.csv  one row per page: tokens + approx cost
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import logging
import os
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pdfplumber

from extract_booth_info_gemini import (
    API_URL,
    DEFAULT_CACHED_PRICE_PER_M,
    DEFAULT_INPUT_PRICE_PER_M,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_PRICE_PER_M,
    VertexAuth,
    _vertex_schema,
    api_request,
    estimate_cost_from_usage,
    extract_usage,
    load_env_files,
    parse_gemini_response,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extract-voters")

HERE = Path(__file__).resolve().parent
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Fields the model reads off each voter card.
VOTER_FIELDS = ["serial_no", "epic_number", "name", "relation_type", "relation_name",
                "house_number", "age", "gender", "marker"]
# Page-level context the model reads off the page header.
PAGE_FIELDS = ["section_no", "section_name", "part_no"]
COLUMNS = ["booth_pdf", "page_no", "ac_no"] + PAGE_FIELDS + VOTER_FIELDS
USAGE_COLUMNS = ["booth_pdf", "page_no", "voters", "input_tokens", "output_tokens",
                 "thought_tokens", "cached_tokens", "total_tokens", "approx_cost_usd"]

PROMPT = """\
This is one elector page of an Election Commission of India voter roll (part/booth
roll). It shows a grid of voter cards, normally 3 columns x 10 rows.

Read EVERY card on the page, in reading order (left to right, then top to bottom),
and return one JSON object per card.

Per card:
- serial_no: the number in the box at the top-left of the card.
- epic_number: the voter ID printed at the top-right of the card (e.g. TQT6600134).
  Copy it character by character; do not guess or auto-correct it.
- name: the value after "Name :".
- relation_type: exactly one of "Father", "Husband", "Mother", "Other" - whichever
  relation label the card prints ("Father's Name", "Husband's Name", ...).
- relation_name: the value of that relation line.
- house_number: the value after "House Number :" ("" when blank).
- age: integer age.
- gender: "Male", "Female" or "Third Gender" as printed.
- marker: any status letter/symbol printed inside the serial-number box next to the
  number (e.g. "#", "E", "S", "R", "M", "Q"); "" when there is none.

Also read the page header: section_no and section_name from "Section No and Name",
and part_no from "Part No.".

Rules:
- Transcribe text exactly as printed; do not translate or normalize spellings.
- Convert Devanagari digits to ASCII in numeric fields.
- Unreadable or absent value -> empty string (or 0 for age).
- If the page is a cover/summary page with no voter cards, return an empty voters list.
- Do not invent cards: return exactly as many objects as there are cards printed.
"""


def response_schema() -> dict[str, Any]:
    voter = {
        "type": "object",
        "properties": {
            "serial_no": {"type": "integer"},
            "epic_number": {"type": "string"},
            "name": {"type": "string"},
            "relation_type": {"type": "string",
                              "enum": ["Father", "Husband", "Mother", "Other"]},
            "relation_name": {"type": "string"},
            "house_number": {"type": "string"},
            "age": {"type": "integer"},
            "gender": {"type": "string"},
            "marker": {"type": "string"},
        },
        "required": VOTER_FIELDS,
    }
    return {
        "type": "object",
        "properties": {
            "section_no": {"type": "string"},
            "section_name": {"type": "string"},
            "part_no": {"type": "integer"},
            "voters": {"type": "array", "items": voter},
        },
        "required": ["section_no", "section_name", "part_no", "voters"],
    }


def thinking_config(model: str, vertex: bool) -> dict[str, Any]:
    """Gemini 3 takes thinking_level; 2.5 only understands thinkingBudget and
    rejects the newer field outright (HTTP 400). Both are set to the cheapest
    setting - this task is transcription, not reasoning."""
    if model.startswith("gemini-3") or model in {"gemini-flash-latest",
                                                 "gemini-flash-lite-latest"}:
        return {"thinkingLevel": "MINIMAL"} if vertex else {"thinking_level": "minimal"}
    return {"thinkingBudget": 0} if vertex else {"thinking_budget": 0}


class GeminiApiBackend:
    """Gemini API (AI Studio key) via the Interactions endpoint."""

    name = "gemini-api"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def url(self, model: str) -> str:
        return API_URL

    def headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "x-goog-api-key": self._api_key}

    def payload(self, model: str, image_b64: str) -> dict[str, Any]:
        return {
            "model": model,
            "input": [
                {"type": "text", "text": PROMPT},
                {"type": "image", "data": image_b64, "mime_type": "image/jpeg"},
            ],
            "response_format": {"type": "text", "mime_type": "application/json",
                                "schema": response_schema()},
            "generation_config": thinking_config(model, vertex=False),
        }


class VertexBackend:
    """Vertex AI generateContent, authenticated with a service-account key."""

    name = "vertex"

    def __init__(self, auth: VertexAuth, location: str = "global") -> None:
        self._auth = auth
        self._location = location

    def url(self, model: str) -> str:
        host = ("aiplatform.googleapis.com" if self._location == "global"
                else f"{self._location}-aiplatform.googleapis.com")
        return (f"https://{host}/v1/projects/{self._auth.project_id}/locations/"
                f"{self._location}/publishers/google/models/{model}:generateContent")

    def headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json",
                "Authorization": f"Bearer {self._auth.token()}"}

    def payload(self, model: str, image_b64: str) -> dict[str, Any]:
        return {
            "contents": [{"role": "user", "parts": [
                {"text": PROMPT},
                {"inlineData": {"mimeType": "image/jpeg", "data": image_b64}},
            ]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": _vertex_schema(response_schema()),
                "thinkingConfig": thinking_config(model, vertex=True),
            },
        }


def _vertex_preflight(backend: VertexBackend, model: str) -> str | None:
    """Free countTokens call to verify IAM/API access; returns the error or None."""
    url = backend.url(model).replace(":generateContent", ":countTokens")
    try:
        api_request(url, backend.headers(),
                    {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
                    timeout=30, retries=1)
        return None
    except (RuntimeError, SystemExit) as exc:
        return str(exc)


def make_backend(args: argparse.Namespace) -> GeminiApiBackend | VertexBackend:
    load_env_files(args.env_file)
    sa_key = args.sa_key or (Path(p) if (p := os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")) else None)
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if args.backend in {"auto", "vertex"} and sa_key and sa_key.exists():
        auth = VertexAuth(sa_key)
        vertex = VertexBackend(auth, args.location)
        err = _vertex_preflight(vertex, args.model)
        if err is None:
            log.info("backend: Vertex AI (project %s, location %s)", auth.project_id, args.location)
            return vertex
        hint = (f"Vertex AI not usable with this key yet: {err}\n"
                f"Enable aiplatform.googleapis.com and grant roles/aiplatform.user "
                f"in project {auth.project_id}.")
        if args.backend == "vertex" or not api_key:
            raise SystemExit(hint)
        log.warning("%s\nFalling back to the Gemini API key for this run.", hint)
    elif args.backend == "vertex":
        raise SystemExit("--backend vertex needs a service-account key: pass --sa-key "
                         "or set GOOGLE_APPLICATION_CREDENTIALS")

    if not api_key:
        raise SystemExit("No credentials: set GEMINI_API_KEY/GOOGLE_API_KEY or "
                         "GOOGLE_APPLICATION_CREDENTIALS (Vertex) in env or .env")
    log.info("backend: Gemini API (AI Studio key)")
    return GeminiApiBackend(api_key)


def page_count(pdf_path: Path) -> int:
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


def render_page_jpeg(pdf_path: Path, page_no: int, dpi: int, quality: int,
                     max_dim: int) -> str:
    """page_no is 1-based."""
    with pdfplumber.open(pdf_path) as pdf:
        img = pdf.pages[page_no - 1].to_image(resolution=dpi).original.convert("RGB")
    try:
        if max_dim and max(img.size) > max_dim:
            scale = max_dim / max(img.size)
            img = img.resize((round(img.width * scale), round(img.height * scale)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    finally:
        img.close()


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip(" :।")


def _to_int(value: Any) -> int | str:
    if value is None or value == "":
        return ""
    digits = re.sub(r"\D", "", str(value).translate(DEV_DIGITS))
    return int(digits) if digits else ""


def _clean_epic(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9/]", "", str(value or "")).upper()


def normalize_rows(raw: dict[str, Any], pdf_path: Path, page_no: int,
                   ac_no: str) -> list[dict[str, Any]]:
    part_no = int(re.sub(r"\D", "", pdf_path.stem).lstrip("0") or "0")
    model_part = _to_int(raw.get("part_no"))
    if isinstance(model_part, int) and model_part and model_part != part_no:
        log.warning("%s p%d: printed part number %d != filename part number %d",
                    pdf_path.name, page_no, model_part, part_no)

    base = {
        "booth_pdf": pdf_path.name,
        "page_no": page_no,
        "ac_no": _to_int(ac_no),
        "section_no": _clean(raw.get("section_no")),
        "section_name": _clean(raw.get("section_name")),
        "part_no": part_no,
    }
    rows: list[dict[str, Any]] = []
    for voter in raw.get("voters") or []:
        if not isinstance(voter, dict):
            continue
        row = dict(base)
        row["serial_no"] = _to_int(voter.get("serial_no"))
        row["epic_number"] = _clean_epic(voter.get("epic_number"))
        row["name"] = _clean(voter.get("name"))
        row["relation_type"] = _clean(voter.get("relation_type"))
        row["relation_name"] = _clean(voter.get("relation_name"))
        row["house_number"] = _clean(voter.get("house_number"))
        row["age"] = _to_int(voter.get("age"))
        row["gender"] = _clean(voter.get("gender"))
        row["marker"] = _clean(voter.get("marker"))
        if not (row["name"] or row["epic_number"]):
            continue
        rows.append(row)
    return rows


def process_page(pdf_path: Path, page_no: int, ac_no: str, backend: Any,
                 args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_b64 = render_page_jpeg(pdf_path, page_no, args.dpi, args.jpeg_quality,
                                 args.max_dim)
    response = api_request(backend.url(args.model), backend.headers(),
                           backend.payload(args.model, image_b64),
                           timeout=args.timeout, retries=args.retries)
    raw = parse_gemini_response(response)
    rows = normalize_rows(raw, pdf_path, page_no, ac_no)
    usage = extract_usage(response)
    cost = estimate_cost_from_usage(usage, args.input_price_per_m,
                                    args.output_price_per_m, args.cached_price_per_m)
    usage_row = {"booth_pdf": pdf_path.name, "page_no": page_no, "voters": len(rows),
                 **usage, "approx_cost_usd": f"{cost:.6f}"}
    if args.debug:
        log.info("%s p%d raw=%s", pdf_path.name, page_no,
                 json.dumps(raw, ensure_ascii=False)[:4000])
        log.info("%s p%d usage=%s", pdf_path.name, page_no, json.dumps(usage_row))
    return rows, usage_row


def check_serials(rows: list[dict[str, Any]]) -> None:
    """Warn about gaps/duplicates in each booth's serial numbers."""
    by_booth: dict[str, list[int]] = {}
    for r in rows:
        if isinstance(r.get("serial_no"), int):
            by_booth.setdefault(str(r["booth_pdf"]), []).append(r["serial_no"])
        elif str(r.get("serial_no") or "").isdigit():
            by_booth.setdefault(str(r["booth_pdf"]), []).append(int(r["serial_no"]))
    for booth, serials in sorted(by_booth.items()):
        seen = sorted(serials)
        dupes = {s for s in seen if seen.count(s) > 1}
        gaps = sorted(set(range(seen[0], seen[-1] + 1)) - set(seen))
        if dupes:
            log.warning("%s: %d duplicate serial numbers (e.g. %s)", booth, len(dupes),
                        ", ".join(map(str, sorted(dupes)[:10])))
        if gaps:
            log.warning("%s: %d missing serial numbers (e.g. %s)", booth, len(gaps),
                        ", ".join(map(str, gaps[:10])))


def read_csv_rows(path: Path, columns: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as f:
        return [{c: r.get(c, "") for c in columns} for r in csv.DictReader(f)]


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    booth = int(re.sub(r"\D", "", str(row.get("booth_pdf", ""))) or 0)
    page = int(str(row.get("page_no") or 0))
    serial = int(str(row.get("serial_no") or 0) or 0)
    return booth, page, serial


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract voter rows from booth-roll PDFs using Gemini vision")
    ap.add_argument("--pdf-dir", type=Path, help="Directory of booth_*.pdf files")
    ap.add_argument("--ac", help="AC number; used for --out naming and the ac_no column "
                                 "(default: the --pdf-dir folder name)")
    ap.add_argument("--out", type=Path, help="Output CSV (default booth_list_csv/voters_<AC>.csv)")
    ap.add_argument("--env-file", type=Path, help="Optional extra .env path")
    ap.add_argument("--backend", choices=["auto", "gemini", "vertex"], default="auto")
    ap.add_argument("--sa-key", type=Path,
                    help="Vertex service-account JSON key (default: GOOGLE_APPLICATION_CREDENTIALS)")
    ap.add_argument("--location", default="global", help="Vertex location (default global)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--first-page", type=int, default=3,
                    help="First elector page, 1-based (default 3)")
    ap.add_argument("--last-page", type=int, default=0,
                    help="Last elector page, 1-based; 0 = total pages - --skip-last")
    ap.add_argument("--skip-last", type=int, default=2,
                    help="Trailing summary pages to skip when --last-page is 0 (default 2)")
    ap.add_argument("--max-pages", type=int, help="Process at most N elector pages per booth")
    ap.add_argument("--parts", help="Comma-separated booth part numbers, e.g. 1,2,7")
    ap.add_argument("--limit", type=int, help="Process only the first N booth PDFs")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--max-dim", type=int, default=2400,
                    help="Downscale the page image to this longest side (0 = off)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--input-price-per-m", type=float, default=DEFAULT_INPUT_PRICE_PER_M)
    ap.add_argument("--output-price-per-m", type=float, default=DEFAULT_OUTPUT_PRICE_PER_M)
    ap.add_argument("--cached-price-per-m", type=float, default=DEFAULT_CACHED_PRICE_PER_M)
    ap.add_argument("--refresh", action="store_true", help="Ignore existing output and redo all pages")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not args.pdf_dir and not args.ac:
        ap.error("provide --pdf-dir (or --ac for booth_list_pdf/<AC>)")
    pdf_dir = args.pdf_dir or (HERE / "booth_list_pdf" / str(args.ac))
    if not pdf_dir.is_dir():
        raise SystemExit(f"PDF dir not found: {pdf_dir}")
    ac_no = args.ac or pdf_dir.name
    out_csv = args.out or HERE / "booth_list_csv" / f"voters_{ac_no}.csv"
    usage_csv = out_csv.with_suffix(".usage.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("booth_*.pdf"))
    if not pdfs:
        raise SystemExit(f"No booth_*.pdf files in {pdf_dir}")
    if args.parts:
        wanted = {int(x) for x in re.findall(r"\d+", args.parts)}
        pdfs = [p for p in pdfs if int(re.sub(r"\D", "", p.stem) or 0) in wanted]
        missing = wanted - {int(re.sub(r"\D", "", p.stem) or 0) for p in pdfs}
        if missing:
            log.warning("no PDF found for parts: %s", ",".join(map(str, sorted(missing))))
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        raise SystemExit("no booth PDFs selected")

    backend = make_backend(args)

    rows = [] if args.refresh else read_csv_rows(out_csv, COLUMNS)
    usage_rows = [] if args.refresh else read_csv_rows(usage_csv, USAGE_COLUMNS)
    done = {(u["booth_pdf"], str(u["page_no"])) for u in usage_rows}
    if done:
        log.info("resuming; %d pages already in %s", len(done), out_csv.name)

    tasks: list[tuple[Path, int]] = []
    for pdf in pdfs:
        total = page_count(pdf)
        last = args.last_page or (total - args.skip_last)
        last = min(last, total)
        pages = list(range(args.first_page, last + 1))
        if args.max_pages:
            pages = pages[: args.max_pages]
        if not pages:
            log.warning("%s: no elector pages in range (total_pages=%d)", pdf.name, total)
        tasks.extend((pdf, p) for p in pages if (pdf.name, str(p)) not in done)

    if not tasks:
        log.info("nothing to do; %s already covers the selected pages", out_csv)
        return

    log.info("processing %d pages across %d booths (model=%s, workers=%d)",
             len(tasks), len(pdfs), args.model, max(1, min(args.workers, len(tasks))))

    lock = threading.Lock()
    failed: list[str] = []
    completed = 0

    def record(rows_new: list[dict[str, Any]], usage_row: dict[str, Any]) -> None:
        nonlocal completed
        with lock:
            completed += 1
            rows.extend(rows_new)
            usage_rows.append(usage_row)
            log.info("[%d/%d] %s p%s -> %d voters", completed, len(tasks),
                     usage_row["booth_pdf"], usage_row["page_no"], len(rows_new))
            rows.sort(key=_sort_key)
            usage_rows.sort(key=_sort_key)
            write_csv(out_csv, COLUMNS, rows)
            write_csv(usage_csv, USAGE_COLUMNS, usage_rows)

    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, len(tasks)))) as ex:
        futs = {ex.submit(process_page, pdf, page, ac_no, backend, args): (pdf, page)
                for pdf, page in tasks}
        for fut in as_completed(futs):
            pdf, page = futs[fut]
            try:
                record(*fut.result())
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{pdf.name} p{page}")
                log.error("%s p%d failed: %s", pdf.name, page, exc)

    check_serials(rows)
    cost = 0.0
    for u in usage_rows:
        try:
            cost += float(u.get("approx_cost_usd") or 0)
        except ValueError:
            pass
    log.info("Done. %d voters from %d pages -> %s", len(rows), len(usage_rows), out_csv)
    log.info("Approx Gemini cost: $%.4f (%.5f per page)", cost,
             cost / max(1, len(usage_rows)))
    if failed:
        log.warning("%d pages FAILED (re-run to retry): %s", len(failed), ", ".join(failed))


if __name__ == "__main__":
    main()
