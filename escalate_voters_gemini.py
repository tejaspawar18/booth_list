"""
Hybrid voter extraction: free Tesseract everywhere, paid Gemini only where it pays.

extract_voters_tesseract.py reads a whole AC for nothing but gets ~93% of names
exactly right; extract_voters_gemini.py gets ~100% but costs ~$0.0073 a page.
This script keeps the OCR pass and re-reads only the cards a rule selects,
cropping each card from its PDF (the OCR CSV records card_box/dpi for exactly
this) and sending them to Gemini in **batches of many card images per request**.

Batching is what makes it cheap: the prompt and schema are sent once per batch
rather than once per card, and a card crop is a fraction of a page image.

Batching: one montage, not N crops
----------------------------------
Gemini charges a floor of image tokens *per image*, so N card crops sent as N
images cost ~1,120 input tokens each, while the same N cards tiled into one
image cost ~70. Measured on 48 cards: 53,652 input tokens as separate images
($0.040) vs 3,452 as a montage ($0.012). --pack montage (the default) tiles the
batch into a 3-column sheet, stamps each card with [i], and asks for one object
per printed index; --pack single keeps the naive form. With montage, a card
costs about the same as it would inside a full page ($0.00029), so cost scales
linearly with the escalated fraction.

Selecting what to escalate (--rule)
-----------------------------------
Measured end-to-end on AC 228 (5 booths, 126 pages, 3,470 cards with a
full-Gemini reference), single-model OCR baseline:

    pipeline                 cards sent   name     mean    cost     extra time
    OCR only                       0%     90.5%   96.5%   $0        -
    hybrid conf<85               4.9%     91.9%   97.0%   $0.05     +57 s
    hybrid conf<88              16.9%     94.2%   97.6%   $0.16     +94 s
    full Gemini                  100%    ~100%    99.8%   $1.01     -

**Tesseract's confidence is a weak error predictor**: below 80 it means
something (81% of those cards are wrong) but 80-85 and 85-88 both sit near 11%,
so a threshold cut mostly buys cards that were already right. With the two-model
OCR ensemble the `disagree` flag is a much better selector - 18.6% of cards for
57.6% of the errors, vs 14.4% for conf<85 - but it makes the OCR pass ~1.6x
slower. Pick by which resource is scarcer: CPU time or API budget.

A confidence collapse also detects the wrong-language case: English OCR on
Devanagari rolls scores ~42, so every card escalates. That is the right
behaviour, but it costs full-Gemini money - run those rolls with --lang hin
(or straight to extract_voters_gemini.py) instead.

    --rule conf<85              confidence threshold only
    --rule disagree             cards the OCR ensemble read two ways (default)
    --rule conf<88,disagree     union of both
    --rule flags:epic_bad       any flag substring works

Usage
-----
    PY=../../jup_venv/bin/python
    # fast path: single-model OCR, then Gemini on the low-confidence cards
    $PY extract_voters_tesseract.py --pdf-dir <dir> --ac 228 --no-ensemble \
        --workers 8 --out booth_list_csv/voters_228_ocr.csv
    $PY escalate_voters_gemini.py --csv booth_list_csv/voters_228_ocr.csv \
        --pdf-dir <dir> --rule "conf<85" --batch-size 30 --workers 8
    # what would it cost? (no API calls)
    $PY escalate_voters_gemini.py --csv ... --pdf-dir ... --rule "conf<88" --dry-run

Output
------
    <csv stem>_hybrid.csv         the OCR rows with escalated cards replaced
    <csv stem>_hybrid.usage.csv   one row per batch: cards, tokens, cost
Escalated rows carry `gemini` in flags, so the provenance of every field stays
visible, and `ocr_*` columns keep what the OCR had said.
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
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pdfplumber
from PIL import Image

from extract_booth_info_gemini import (
    DEFAULT_MODEL,
    VertexAuth,
    _vertex_schema,
    api_request,
    estimate_cost_from_usage,
    extract_usage,
    load_env_files,
    make_backend,
    parse_gemini_response,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("escalate-voters")

HERE = Path(__file__).resolve().parent
# gemini-3-flash-preview paid tier; 2.5-flash measured worse AND pricier per page,
# 2.5-flash-lite lands at OCR-level accuracy, so neither is worth escalating to.
INPUT_PRICE_PER_M = 0.50
OUTPUT_PRICE_PER_M = 3.00
CACHED_PRICE_PER_M = 0.05

VOTER_FIELDS = ["serial_no", "epic_number", "name", "relation_type", "relation_name",
                "house_number", "age", "gender", "marker"]
USAGE_COLUMNS = ["batch", "cards", "input_tokens", "output_tokens", "thought_tokens",
                 "cached_tokens", "total_tokens", "approx_cost_usd", "seconds"]

PROMPT = """\
This image is a sheet of voter cards cut out of Election Commission of India
electoral rolls and tiled together. Each card is stamped with an index in square
brackets, like [1], [2], ... above it. Read EVERY card and return one object per
card.

Per card:
- i: the index printed above that card, as an integer.
- serial_no: the number in the box at the top-left of the card.
- epic_number: the voter ID at the top-right (e.g. TQT6600134). Copy it character
  by character; do not guess or auto-correct it.
- name: the value after "Name :".
- relation_type: exactly one of "Father", "Husband", "Mother", "Other".
- relation_name: the value of that relation line.
- house_number: the value after "House Number :" ("" when blank).
- age: integer age.
- gender: "Male", "Female" or "Third Gender" as printed.
- marker: any status letter/symbol inside the serial-number box (e.g. "#", "E",
  "S", "R", "M", "Q"); "" when there is none.

Transcribe exactly as printed; do not translate or normalize spellings. Convert
Devanagari digits to ASCII. Unreadable value -> empty string (0 for age).
Return exactly one object per card on the sheet, no more.
"""


def response_schema(n_cards: int) -> dict[str, Any]:
    card = {
        "type": "object",
        "properties": {
            "i": {"type": "integer"},
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
        "required": ["i"] + VOTER_FIELDS,
    }
    return {"type": "object", "properties": {"cards": {"type": "array", "items": card}},
            "required": ["cards"]}


# ------------------------------------------------------------------ selection

def parse_rule(rule: str):
    """Build a predicate over one CSV row from a small rule language."""
    terms = [t.strip() for t in rule.split(",") if t.strip()]
    if not terms:
        raise SystemExit("--rule is empty")

    def predicate(row: dict[str, str]) -> bool:
        for term in terms:
            if m := re.fullmatch(r"conf\s*<\s*([\d.]+)", term):
                try:
                    if float(row.get("ocr_conf") or 0) < float(m.group(1)):
                        return True
                except ValueError:
                    return True
            elif term == "disagree":
                if "arbitrated" in (row.get("flags") or ""):
                    return True
            elif term.startswith("flags:"):
                if term[6:] in (row.get("flags") or ""):
                    return True
            elif term == "all":
                return True
            else:
                raise SystemExit(f"unknown rule term: {term!r} "
                                 "(use conf<N, disagree, flags:<substring>, all)")
        return False
    return predicate


# --------------------------------------------------------------- card cropping

def crop_cards(pdf_path: Path, rows: list[dict[str, str]], pad: int,
               jpeg_quality: int) -> list[tuple[dict[str, str], np.ndarray]]:
    """Render each page once and cut out the requested cards."""
    out: list[tuple[dict[str, str], np.ndarray]] = []
    by_page: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in rows:
        by_page.setdefault((int(row["page_no"]), int(row.get("dpi") or 200)), []).append(row)
    with pdfplumber.open(pdf_path) as pdf:
        for (page_no, dpi), page_rows in sorted(by_page.items()):
            image = pdf.pages[page_no - 1].to_image(resolution=dpi).original.convert("RGB")
            try:
                array = np.array(image)
            finally:
                image.close()
            height, width = array.shape[:2]
            for row in page_rows:
                try:
                    x, y, w, h = (int(v) for v in str(row["card_box"]).split(","))
                except (ValueError, KeyError):
                    log.warning("%s p%s serial %s: no usable card_box - skipped",
                                pdf_path.name, row["page_no"], row["serial_no"])
                    continue
                crop = array[max(0, y - pad): min(height, y + h + pad),
                             max(0, x - pad): min(width, x + w + pad)]
                if crop.size == 0:
                    continue
                out.append((row, crop))
    return out


def build_montage(crops: list[np.ndarray], columns: int, label_height: int = 26,
                  gap: int = 6) -> Image.Image:
    """Tile card crops into one image, each stamped with its 1-based index.

    Gemini charges a floor of image tokens per image, so N separate crops cost
    ~1,120 input tokens each while N cards inside one page-sized image cost ~63.
    Montaging recovers most of that: the model reads one image and returns one
    object per printed index, which also makes the card->row mapping explicit
    instead of relying on argument order.
    """
    from PIL import ImageDraw

    cell_w = max(c.shape[1] for c in crops)
    cell_h = max(c.shape[0] for c in crops)
    rows = -(-len(crops) // columns)
    width = columns * cell_w + (columns + 1) * gap
    height = rows * (cell_h + label_height) + (rows + 1) * gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, crop in enumerate(crops):
        r, c = divmod(index, columns)
        x = gap + c * (cell_w + gap)
        y = gap + r * (cell_h + label_height + gap)
        draw.text((x + 4, y + 4), f"[{index + 1}]", fill="black")
        sheet.paste(Image.fromarray(crop), (x, y + label_height))
    return sheet


def encode_jpeg(image: Image.Image, quality: int) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------------ gemini I/O

def batch_payload(backend: Any, model: str, images: list[str],
                  n_cards: int | None = None) -> dict[str, Any]:
    schema = response_schema(n_cards or len(images))
    parts: list[dict[str, Any]] = [{"text": PROMPT}]
    for image in images:
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": image}})
    if backend.name == "vertex":
        return {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": _vertex_schema(schema),
                "thinkingConfig": {"thinkingLevel": "MINIMAL"},
                "maxOutputTokens": 8192,
            },
        }
    inputs: list[dict[str, Any]] = [{"type": "text", "text": PROMPT}]
    for image in images:
        inputs.append({"type": "image", "data": image, "mime_type": "image/jpeg"})
    return {
        "model": model,
        "input": inputs,
        "response_format": {"type": "text", "mime_type": "application/json",
                            "schema": schema},
        "generation_config": {"thinking_level": "minimal"},
    }


def run_batch(backend: Any, model: str, batch: list[tuple[dict[str, str], np.ndarray]],
              index: int, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.time()
    crops = [crop for _row, crop in batch]
    if args.pack == "montage":
        images = [encode_jpeg(build_montage(crops, args.columns), args.jpeg_quality)]
    else:
        images = [encode_jpeg(Image.fromarray(c), args.jpeg_quality) for c in crops]
    response = api_request(backend.url(model), backend.headers(),
                           batch_payload(backend, model, images, len(batch)),
                           timeout=args.timeout, retries=args.retries)
    parsed = parse_gemini_response(response)
    cards = parsed.get("cards") or []
    usage = extract_usage(response)
    cost = estimate_cost_from_usage(usage, INPUT_PRICE_PER_M, OUTPUT_PRICE_PER_M,
                                    CACHED_PRICE_PER_M)
    if len(cards) != len(batch):
        log.warning("batch %d: asked for %d cards, got %d - matching by index",
                    index, len(batch), len(cards))
    usage_row = {"batch": index, "cards": len(batch), **usage,
                 "approx_cost_usd": round(cost, 6), "seconds": round(time.time() - started, 1)}
    return cards, usage_row


def apply_cards(batch: list[tuple[dict[str, str], np.ndarray]], cards: list[dict[str, Any]],
                keep_serial: bool) -> int:
    """Write Gemini's reading over the OCR row, keeping the OCR value in ocr_*."""
    by_index = {}
    for position, card in enumerate(cards, start=1):
        try:
            key = int(card.get("i") or position)
        except (TypeError, ValueError):
            key = position
        by_index[key] = card
    applied = 0
    for position, (row, _image) in enumerate(batch, start=1):
        card = by_index.get(position)
        if card is None:
            row["flags"] = ";".join(f for f in [row.get("flags", ""), "gemini_missing"] if f)
            continue
        for field in VOTER_FIELDS:
            if field == "serial_no" and keep_serial:
                continue        # the OCR serial is fitted to the page sequence
            value = card.get(field, "")
            row[f"ocr_{field}"] = row.get(field, "")
            row[field] = "" if value is None else str(value)
        row["flags"] = ";".join(f for f in [row.get("flags", ""), "gemini"] if f)
        applied += 1
    return applied


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True,
                    help="Output CSV from extract_voters_tesseract.py")
    ap.add_argument("--pdf-dir", type=Path, required=True, help="Where the booth PDFs live")
    ap.add_argument("--out", type=Path, help="Default: <csv stem>_hybrid.csv")
    ap.add_argument("--rule", default="disagree",
                    help="conf<N | disagree | flags:<substring> | all, comma-separated (OR)")
    ap.add_argument("--batch-size", type=int, default=30,
                    help="Cards per Gemini request (default 30, i.e. a page's worth)")
    ap.add_argument("--pack", default="montage", choices=["montage", "single"],
                    help="montage: tile the batch into ONE image (far cheaper - image "
                         "tokens have a per-image floor); single: one image per card")
    ap.add_argument("--columns", type=int, default=3,
                    help="Montage columns (3 mirrors the roll's own layout)")
    ap.add_argument("--workers", type=int, default=4, help="Concurrent requests")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="auto", choices=["auto", "gemini", "vertex"])
    ap.add_argument("--sa-key", type=Path)
    ap.add_argument("--location", default="global")
    ap.add_argument("--env-file", type=Path)
    ap.add_argument("--pad", type=int, default=4, help="Pixels of padding around a card crop")
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--keep-ocr-serial", action="store_true", default=True,
                    help="Keep the OCR serial_no (it is fitted to the page sequence)")
    ap.add_argument("--limit", type=int, help="Escalate at most N cards (costed dry run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be escalated and the projected cost, call nothing")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"{args.csv} not found - run extract_voters_tesseract.py first")
    with args.csv.open(encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if "card_box" not in columns:
        raise SystemExit(f"{args.csv} has no card_box column - re-run "
                         "extract_voters_tesseract.py (it records card geometry now)")

    predicate = parse_rule(args.rule)
    selected = [row for row in rows if predicate(row)]
    if args.limit:
        selected = selected[: args.limit]
    log.info("%d of %d cards selected by rule %r (%.1f%%)", len(selected), len(rows),
             args.rule, 100.0 * len(selected) / max(1, len(rows)))
    if not selected:
        log.info("nothing to escalate")
        return

    n_batches = -(-len(selected) // args.batch_size)
    if args.dry_run:
        # ~260 input tokens per card crop + ~330 for the prompt, ~55 output tokens
        # per card: measured on this AC, good to ~10%
        est = (n_batches * 330 + len(selected) * 260) / 1e6 * INPUT_PRICE_PER_M \
            + len(selected) * 55 / 1e6 * OUTPUT_PRICE_PER_M
        log.info("dry run: %d cards in %d batches of %d -> ~$%.4f (~$%.5f/card)",
                 len(selected), n_batches, args.batch_size, est, est / len(selected))
        return

    by_pdf: dict[str, list[dict[str, str]]] = {}
    for row in selected:
        by_pdf.setdefault(row["booth_pdf"], []).append(row)

    log.info("cropping %d cards from %d booth PDFs", len(selected), len(by_pdf))
    crops: list[tuple[dict[str, str], np.ndarray]] = []
    for pdf_name, pdf_rows in sorted(by_pdf.items()):
        pdf_path = args.pdf_dir / pdf_name
        if not pdf_path.exists():
            log.error("%s not found in %s - %d cards skipped", pdf_name, args.pdf_dir,
                      len(pdf_rows))
            continue
        crops.extend(crop_cards(pdf_path, pdf_rows, args.pad, args.jpeg_quality))
    if not crops:
        raise SystemExit("no card crops produced")

    backend = make_backend(args)
    batches = [crops[i:i + args.batch_size] for i in range(0, len(crops), args.batch_size)]
    log.info("escalating %d cards in %d batches of <=%d to %s (%s), %d workers",
             len(crops), len(batches), args.batch_size, args.model, backend.name,
             args.workers)

    usage_rows: list[dict[str, Any]] = []
    applied = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_batch, backend, args.model, batch, i, args): (i, batch)
                   for i, batch in enumerate(batches, start=1)}
        for future in as_completed(futures):
            index, batch = futures[future]
            try:
                cards, usage_row = future.result()
            except Exception as exc:  # noqa: BLE001 - a failed batch keeps its OCR values
                log.error("batch %d failed: %s", index, exc)
                for row, _image in batch:
                    row["flags"] = ";".join(f for f in [row.get("flags", ""),
                                                        "gemini_failed"] if f)
                continue
            applied += apply_cards(batch, cards, args.keep_ocr_serial)
            usage_rows.append(usage_row)
            done = len(usage_rows)
            log.info("[%d/%d batches] %d cards, $%.4f so far",
                     done, len(batches), applied,
                     sum(u["approx_cost_usd"] for u in usage_rows))

    out_csv = args.out or args.csv.with_name(args.csv.stem + "_hybrid.csv")
    out_columns = columns + [f"ocr_{f}" for f in VOTER_FIELDS]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in out_columns})
    usage_csv = out_csv.with_suffix(".usage.csv")
    with usage_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=USAGE_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(usage_rows, key=lambda u: u["batch"]))

    cost = sum(u["approx_cost_usd"] for u in usage_rows)
    tokens_in = sum(u["input_tokens"] for u in usage_rows)
    tokens_out = sum(u["output_tokens"] for u in usage_rows)
    elapsed = time.time() - started
    log.info("escalated %d/%d cards in %.0fs -> %s", applied, len(rows), elapsed, out_csv)
    log.info("cost $%.4f (%d in / %d out tokens) = $%.5f per escalated card, "
             "$%.5f per card in the AC", cost, tokens_in, tokens_out,
             cost / max(1, applied), cost / max(1, len(rows)))


if __name__ == "__main__":
    main()
