"""
Extract voter rows from the elector pages of ECI booth-roll PDFs with Tesseract OCR.

This is the local/offline alternative to extract_voters_gemini.py: same CSV
columns, no API calls, no cost. Elector pages print a 3 x 10 grid of voter cards:

    serial number, EPIC number, Name, Father's/Husband's/Mother's Name,
    House Number, Age, Gender

Rather than OCR-ing a whole page as one block (which interleaves the three
columns), the page image is segmented with OpenCV: the printed rules are
recovered by morphology, each card is a rectangle in that grid, and inside a
card the serial box and the photo box are rectangles too. Each region is then
OCR-ed on its own with a page-segmentation mode suited to it.

Requirements
------------
The tesseract binary must be on PATH, or point --tesseract / the TESSERACT_CMD
env var at it. There is no Homebrew here, so it was installed with conda:

    conda create -n tesseract-ocr -c conda-forge tesseract
    # binary: ~/anaconda3/envs/tesseract-ocr/bin/tesseract  (auto-detected below)

Python side: pytesseract, opencv-python, pdfplumber (all in jup_venv).

Engines
-------
The default engine reads every card twice - with the installed integer LSTM
model and with the tessdata_best float model in ./tessdata_best - and arbitrates
the two readings (see arbitrate()). Fetch the second model once with

    curl -sL -o tessdata_best/eng.traineddata \
      https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/eng.traineddata

When it is absent the script logs that line and runs single-model; --no-ensemble
forces single-model.

--engine tesseract (default) or easyocr. Both are free and local. Measured on
180 cards of AC 228 against the Gemini extraction of the same cards, exact-match
mean over the eight per-voter fields:

    tesseract  96.2%   16 s for 6 pages (6 CPU workers)
    easyocr    64.3%   85 s for 6 pages (Apple MPS)

so tesseract stays the default. Part of easyocr's gap is this harness rather
than the model - the card parsing is tuned to tesseract's line grouping, and the
one-glyph marker crop defeats its detector (0.6%) - but it is not close enough
to be worth re-tuning. PaddleOCR (PP-OCRv5), Surya and docTR are the other free
engines worth trying on a GPU box; neither is installed here. Add one by
implementing text()/block() and registering it in ENGINES.

Scaling: CPU parallel, GPU segmentation
---------------------------------------
Work is chunked as (booth PDF, run of pages) and spread over a process pool, so
each worker opens a PDF once and renders its pages sequentially - pdfium is not
thread-safe, but it is fine across processes. --workers defaults to the CPU
count; throughput is very close to linear in it.

**Tesseract has no CUDA path** (this build reports no OpenCL either), so with
the default engine OCR runs on CPU cores - on a GPU EC2 instance the speed-up
comes from that instance's larger vCPU count. What does move to the GPU is the
OpenCV segmentation stage (threshold, morphology, dilate, upscale) via cv2.cuda,
and the easyocr engine's torch models:

    --device auto   use the GPU when cv2 has a CUDA device, else CPU (default)
    --device gpu    require a CUDA device; fail loudly if there is none
    --device cpu    force the CPU path

Note that the PyPI opencv-python wheel is built without CUDA: --device gpu needs
an OpenCV built with CUDA support (as on the DL AMIs). auto quietly falls back,
logging which path it took. Expect a modest end-to-end win - segmentation is
~15% of page time here, OCR dominates.

Robustness
----------
- resumes by default: pages already in the output CSV are skipped (--refresh redoes)
- per-page retry (--retries) and a per-OCR-call timeout (--ocr-timeout), so one
  wedged tesseract call cannot hang the run
- output is flushed every --flush-every pages, and on Ctrl-C, so a killed run
  keeps its work; a page that fails every attempt is logged, counted, and the
  run exits non-zero with the page list to re-run
- refuses to overwrite a CSV that is not this script's output (the Gemini
  extractor's default path is one character away) unless --force
- warns when a page yields an unexpected number of cards (the grid is 3 x 10)

Accuracy
--------
Measured against a Gemini extraction of the same 3,470 voter cards (AC 228,
booths 1/2/4/7/21, 126 pages), exact match, single model -> two-model ensemble:

    relation_type 99.9% -> 99.9%   gender        98.2% -> 99.7%
    age           98.8% -> 98.8%   name          90.5% -> 92.8%
    marker        98.7% -> 98.7%   relation_name 94.7% -> 96.9%
    epic_number   95.7% -> 95.7%   house_number  94.1% -> 94.1%
    -> mean 96.3% -> 97.1% exact (~98% of names are within one edit)

The ensemble costs ~1.6x wall clock (432 s vs 295 s for the 126 pages, 6 workers).
section_name is the weak field at 63%: it is read once per page from the header
strip, so one bad read taints all 30 rows on that page.

Structured fields come out clean; what is left is character noise in the free
text ("Anmed" for "Ahmed", "Magbool" for "Maqbool"). Use extract_voters_gemini.py
where the names themselves have to be right.

Things that were tried and made it *worse*, so they are deliberately absent:

- corpus vocabulary voting (snap a rare name token to a frequent one within one
  edit). Broke 2-4x more rows than it fixed at every threshold - Indian names
  differ legitimately by one letter (AJIJ/AJIT, SHAKIR/SHAKIL, NAJIR/NASIR).
- rendering above 200 dpi, upscaling the card crop, sharpening, Otsu
  binarisation, psm 4: name accuracy 89.2% at the current settings vs 86.2%
  (300 dpi), 84.6% (400 dpi), 83.3% (300 dpi + 2x), 80.4% (sharpen), 55% (Otsu).
- disabling tesseract's dictionary (load_system_dawg/load_freq_dawg=0): no change.

Two layout facts are used to repair the fields tesseract reads worst, and both
repairs are recorded in the flags column so rows stay auditable:

- serials on a page are consecutive, so they are fitted to an arithmetic
  sequence; a read that contradicts the fit is flagged serial_mismatch.
- a booth uses only a handful of EPIC prefixes, so a prefix within 2 edits of
  one that is common on the page is rewritten to it (epic_prefix_fixed). The
  digits cannot be voted on, so an EPIC of the wrong shape is only flagged
  (epic_bad / epic_length) -- check those against the PDF.
- an EPIC is always <letters><digits> nationwide, so a digit-shaped glyph in
  the letter zone or a letter-shaped glyph in the digit zone is rewritten to
  its twin (epic_glyph_fixed) before the checks above run. I<->1 and O<->0
  are near-total confusions on some fonts (measured 557 and 237 char errors
  on one AC's ground truth); Z/S/B/G/Y<->2/5/8/6/7 are the same shape family.

Other flags: age_bad (outside 18-120), name_missing, epic_missing.

Usage
-----
    PY=../../jup_venv/bin/python
    $PY extract_voters_tesseract.py \
        --pdf-dir booth_list_pdf/s13/2024-finalroll/228 --ac 228 --parts 1 --max-pages 1 --debug
    $PY extract_voters_tesseract.py --pdf-dir booth_list_pdf/s13/2024-finalroll/228 --ac 228
    $PY extract_voters_tesseract.py --pdf-dir ... --workers 32 --device gpu   # GPU instance
    $PY extract_voters_tesseract.py --pdf-dir booth_list_pdf/s24/... --lang hin   # Hindi rolls

Output
------
    booth_list_csv/voters_<AC>_ocr.csv        one row per voter
    booth_list_csv/voters_<AC>_ocr.pages.csv  one row per page: cards, mean OCR confidence
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import statistics
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pdfplumber
import pytesseract
from PIL import Image

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extract-voters-ocr")

HERE = Path(__file__).resolve().parent
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
ALNUM_WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CARDS_PER_PAGE = 30          # the elector grid is 3 columns x 10 rows

COLUMNS = ["booth_pdf", "page_no", "ac_no", "section_no", "section_name", "part_no",
           "serial_no", "epic_number", "name", "relation_type", "relation_name",
           "house_number", "age", "gender", "marker", "flags", "ocr_conf",
           # card bbox at --dpi, so a later pass (escalation to a VLM) can crop
           # exactly this card without re-deriving the segmentation
           "card_box", "dpi"]
PAGE_COLUMNS = ["booth_pdf", "page_no", "cards", "mean_conf", "section_no", "section_name"]

# Candidate locations for a conda/brew/system tesseract, tried in order.
TESSERACT_CANDIDATES = [
    Path.home() / "anaconda3/envs/tesseract-ocr/bin/tesseract",
    Path.home() / "miniconda3/envs/tesseract-ocr/bin/tesseract",
    Path("/opt/homebrew/bin/tesseract"),
    Path("/usr/local/bin/tesseract"),
    Path("/usr/bin/tesseract"),
]

# Label lines inside a card, English rolls and Devanagari rolls.
# the label separator is a printed ":" that tesseract also reads as = ; . +
SEP = r"\s*[:;.+=]*\s*"
# "Father's Name :", but also the bare "Others :" that some rolls print
RE_RELATION = re.compile(
    # N\w{1,3}e rather than Name: "Narne"/"Nane" are routine tesseract misreads
    r"^\s*(?:(Father|Husband|Mother|Others?)\s*['’`]?\s*s?\s*(?:N\w{1,3}e" + SEP + r"|[:;.+=]+\s*)"
    r"|(पिता|पति|माता|अन्य)\s*(?:का|की)?\s*नाम" + SEP + r")(.*)$",
    re.IGNORECASE)
RE_NAME = re.compile(r"^\s*(?:Name|नाम)" + SEP + r"(.*)$", re.IGNORECASE)
RE_HOUSE = re.compile(r"^\s*(?:House\s*(?:Number|No\.?)|मकान\s*(?:संख्या|नंबर|नं))"
                      + SEP + r"(.*)$", re.IGNORECASE)
RE_AGE_GENDER = re.compile(
    # Devanagari digits belong in the age class: Hindi rolls print mixed forms
    # like "आयु : ३0", and without ०-९ the line fails to match, so the age is
    # lost *and* the whole line lands in whichever field preceded it.
    r"(?:Age|आयु)" + SEP + r"([0-9०-९OoIl|/S]{1,3})"
    r"(?:.*?(?:Gender|लिंग)" + SEP + r"([A-Za-zऀ-ॿ ]+))?",
    re.IGNORECASE)
RELATION_CANON = {"father": "Father", "husband": "Husband", "mother": "Mother",
                  "other": "Other", "others": "Other",
                  "पिता": "Father", "पति": "Husband", "माता": "Mother", "अन्य": "Other"}
GENDER_CANON = {"male": "Male", "female": "Female", "third gender": "Third Gender",
                "पुरुष": "Male", "महिला": "Female", "अन्य": "Third Gender"}
# OCR confusions seen in this font when a field should be digits.
DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1",
                           "|": "1", "/": "7", "S": "5", "B": "8", "Z": "2"})
EPIC_RE = re.compile(r"^[A-Z]{2,4}[0-9]{6,9}$")
# Natural letters/digits boundary, independent of the page's modal digit count -
# used to recover the prefix when a digit has been dropped (see repair_epics).
EPIC_NATURAL_SPLIT = re.compile(r"^([A-Z]+)([0-9]+)$")
# tesseract renders a capital I as one of these inside a word
I_CONFUSIONS = re.compile(r"[|!\]\[]")


def resolve_tesseract(explicit: str | None) -> str:
    for cand in [explicit, os.environ.get("TESSERACT_CMD")]:
        if cand and Path(cand).exists():
            return str(cand)
    from shutil import which
    if found := which("tesseract"):
        return found
    for cand in TESSERACT_CANDIDATES:
        if cand.exists():
            return str(cand)
    raise SystemExit(
        "tesseract binary not found. Install it (e.g. `conda create -n tesseract-ocr "
        "-c conda-forge tesseract`) and pass --tesseract /path/to/tesseract.")


# ------------------------------------------------------------------- device/GPU

def cuda_devices() -> int:
    """CUDA devices visible to this OpenCV build (0 for the PyPI wheel)."""
    try:
        return int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:  # noqa: BLE001 - no cv2.cuda module at all
        return 0


def torch_gpu() -> str:
    """CUDA/MPS as seen by torch - only the easyocr engine can use it."""
    try:
        import torch
    except ImportError:
        return ""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return ""


def resolve_device(preference: str, engine_name: str) -> tuple[str, str]:
    """Returns (device, what-was-found). 'gpu' explicit fails when nothing here
    can use one, so a GPU instance never silently runs the slow path."""
    cv_gpu = cuda_devices()
    engine_gpu = torch_gpu() if ENGINES[engine_name].gpu_capable else ""
    note = ", ".join([f"opencv-cuda={cv_gpu}"]
                     + ([f"torch={engine_gpu or 'cpu'}"] if ENGINES[engine_name].gpu_capable
                        else ["tesseract=cpu-only"]))
    if preference == "cpu":
        return "cpu", note
    if cv_gpu or engine_gpu:
        return "gpu", note
    if preference == "gpu":
        raise SystemExit(
            f"--device gpu requested but nothing here can use one ({note}).\n"
            f"  cv2 {cv2.__version__}: the PyPI opencv-python wheel is built without CUDA;\n"
            "  install an OpenCV built with CUDA (e.g. the DL AMI's) for GPU segmentation,\n"
            "  and use --engine easyocr for GPU OCR - tesseract itself is CPU-only.\n"
            "  Or run with --device cpu / auto.")
    return "cpu", note


class ImageOps:
    """Segmentation primitives, on GPU when one is usable and CPU otherwise.

    Only the pixel work moves: connected components stay on the host, and the
    OCR calls are Tesseract's own CPU threads. Both paths must return identical
    arrays - the GPU path is a drop-in, never a different algorithm.
    """

    def __init__(self, device: str) -> None:
        self.device = device

    # -- CPU/GPU pairs ------------------------------------------------------
    def threshold_otsu_inv(self, gray: np.ndarray) -> np.ndarray:
        # cv2.cuda.threshold has no OTSU flag, so the level is computed on the
        # host (cheap: a 256-bin histogram) and applied on the device.
        if self.device == "gpu":
            level = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
            g = cv2.cuda_GpuMat()
            g.upload(gray)
            out = cv2.cuda.threshold(g, level, 255, cv2.THRESH_BINARY_INV)[1]
            return out.download()
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    def open_rect(self, binary: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
        if self.device == "gpu":
            g = cv2.cuda_GpuMat()
            g.upload(binary)
            op = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, cv2.CV_8UC1, kernel)
            return op.apply(g).download()
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    def dilate3(self, binary: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        if self.device == "gpu":
            g = cv2.cuda_GpuMat()
            g.upload(binary)
            op = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC1, kernel)
            return op.apply(g).download()
        return cv2.dilate(binary, kernel)

    def resize(self, img: np.ndarray, scale: int) -> np.ndarray:
        if scale <= 1:
            return img
        if self.device == "gpu":
            g = cv2.cuda_GpuMat()
            g.upload(img)
            out = cv2.cuda.resize(g, (0, 0), fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)
            return out.download()
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------- OCR engines

class TesseractEngine:
    """Tesseract 5 via pytesseract. CPU only - there is no CUDA build."""

    name = "tesseract"
    gpu_capable = False

    def __init__(self, opts: dict[str, Any]) -> None:
        pytesseract.pytesseract.tesseract_cmd = opts["tesseract"]
        self.lang = opts["lang"]
        self.timeout = opts["ocr_timeout"]
        self.best_tessdata = opts.get("best_tessdata", "")

    def text(self, img: np.ndarray | None, psm: int, whitelist: str = "",
             scale: int = 1, lang: str | None = None) -> str:
        if img is None or img.size == 0:
            return ""
        img = ops().resize(img, scale)
        config = f"--psm {psm}"
        if whitelist:
            config += f" -c tessedit_char_whitelist={whitelist}"
        return pytesseract.image_to_string(Image.fromarray(img), lang=lang or self.lang,
                                           config=config, timeout=self.timeout).strip()

    def block(self, img: np.ndarray | None, psm: int = 6,
              tessdata: str = "") -> tuple[list[str], float]:
        """Lines of a multi-line region plus the mean word confidence.

        tessdata selects a model directory: "" is the installed (integer LSTM)
        model, and the tessdata_best float model is passed explicitly by the
        ensemble. The two disagree on different characters, which is the whole
        point - see arbitrate()."""
        if img is None or img.size == 0:
            return [], 0.0
        config = f"--psm {psm}" + (f" --tessdata-dir {tessdata}" if tessdata else "")
        data = pytesseract.image_to_data(Image.fromarray(img), lang=self.lang,
                                         config=config, timeout=self.timeout,
                                         output_type=pytesseract.Output.DICT)
        lines: dict[tuple[int, int, int], list[str]] = {}
        confs: list[float] = []
        for i, word in enumerate(data["text"]):
            if not word.strip():
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
            if (conf := float(data["conf"][i])) >= 0:
                confs.append(conf)
        return [" ".join(v) for _k, v in sorted(lines.items())], (
            statistics.mean(confs) if confs else 0.0)


class EasyOCREngine:
    """EasyOCR (CRAFT detector + CRNN recogniser), torch - uses CUDA or Apple
    MPS when present, CPU otherwise. Heavier per page than tesseract but it is
    the accuracy comparison worth having, and it is the one that a GPU instance
    actually accelerates."""

    name = "easyocr"
    gpu_capable = True
    LANG_MAP = {"eng": "en", "hin": "hi", "mar": "mr", "ben": "bn", "tam": "ta",
                "tel": "te", "kan": "kn", "guj": "gu", "urd": "ur"}

    def __init__(self, opts: dict[str, Any]) -> None:
        import easyocr  # imported lazily: torch costs ~2 s to load
        langs = [self.LANG_MAP.get(part, part)
                 for part in re.split(r"[+,]", opts["lang"]) if part]
        # easyocr picks cuda -> mps -> cpu itself when gpu=True
        self.reader = easyocr.Reader(langs or ["en"], gpu=opts["device"] != "cpu",
                                     verbose=False)
        self.batch_size = opts.get("batch_size", 16)
        self.device = str(getattr(self.reader, "device", "cpu"))

    def _read(self, img: np.ndarray, allowlist: str = "") -> list[tuple[Any, str, float]]:
        kwargs: dict[str, Any] = {"batch_size": self.batch_size, "paragraph": False}
        if allowlist:
            kwargs["allowlist"] = allowlist
        return self.reader.readtext(img, **kwargs)

    def text(self, img: np.ndarray | None, psm: int, whitelist: str = "",
             scale: int = 1, lang: str | None = None) -> str:
        if img is None or img.size == 0:
            return ""
        img = ops().resize(img, scale)
        found = self._read(img, whitelist)
        if not found:  # small crops often defeat the detector; recognise directly
            h, w = img.shape[:2]
            try:
                found = self.reader.recognize(img, horizontal_list=[[0, w, 0, h]],
                                              free_list=[],
                                              allowlist=whitelist or None)
            except Exception:  # noqa: BLE001 - treat as an empty read
                return ""
        return " ".join(t for _b, t, _c in sorted(found, key=lambda r: r[0][0][0])).strip()

    def block(self, img: np.ndarray | None, psm: int = 6,
              tessdata: str = "") -> tuple[list[str], float]:
        if img is None or img.size == 0:
            return [], 0.0
        found = self._read(img)
        if not found:
            return [], 0.0
        # group boxes into printed lines by vertical overlap, then read each L->R
        boxes = [(min(p[1] for p in b), max(p[1] for p in b), min(p[0] for p in b), t, c)
                 for b, t, c in found]
        boxes.sort(key=lambda r: r[0])
        lines: list[list[tuple[float, str]]] = []
        line_bottom = -1.0
        for top, bottom, left, text, _conf in boxes:
            if not lines or top > line_bottom - 0.4 * (bottom - top):
                lines.append([])
                line_bottom = bottom
            else:
                line_bottom = max(line_bottom, bottom)
            lines[-1].append((left, text))
        confs = [c * 100 for _b, _t, c in found]
        return ([" ".join(t for _l, t in sorted(line)) for line in lines],
                statistics.mean(confs) if confs else 0.0)


ENGINES = {"tesseract": TesseractEngine, "easyocr": EasyOCREngine}


# Per-process state: a worker builds these once, not once per page.
_OPS: ImageOps | None = None
_ENGINE: Any = None
_OPTS: dict[str, Any] | None = None


def worker_init(opts: dict[str, Any]) -> None:
    global _OPS, _OPTS, _ENGINE
    _OPTS = opts
    seg_device = "cpu"
    if opts["device"] == "gpu" and cuda_devices():
        try:
            cv2.cuda.setDevice(0)
            seg_device = "gpu"
        except Exception as exc:  # noqa: BLE001 - fall back rather than lose the run
            log.warning("cv2.cuda.setDevice failed (%s); segmenting on the CPU", exc)
    _OPS = ImageOps(seg_device)
    _ENGINE = ENGINES[opts.get("engine", "tesseract")](opts)


def ops() -> ImageOps:
    """Segmentation ops for this process, built on first use (covers workers=1,
    where main() runs the pages itself without a pool)."""
    if _OPS is None:
        worker_init(_OPTS or {"tesseract": pytesseract.pytesseract.tesseract_cmd,
                              "device": "cpu", "lang": "eng", "ocr_timeout": 0})
    assert _OPS is not None
    return _OPS


def engine() -> Any:
    if _ENGINE is None:
        ops()  # same lazy init path
    return _ENGINE


# ---------------------------------------------------------------- segmentation

def page_gray(pdf: Any, page_no: int, dpi: int) -> np.ndarray:
    """Render a 1-based page of an already-open PDF to a grayscale array."""
    img = pdf.pages[page_no - 1].to_image(resolution=dpi).original.convert("L")
    try:
        return np.array(img)
    finally:
        img.close()


def rect_components(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of the white regions enclosed by the printed rules."""
    h, w = gray.shape
    op = ops()
    binary = op.threshold_otsu_inv(gray)
    hor = op.open_rect(binary, (max(3, w // 40), 1))
    ver = op.open_rect(binary, (1, max(3, h // 40)))
    grid = op.dilate3(cv2.bitwise_or(hor, ver))
    count, _labels, stats, _c = cv2.connectedComponentsWithStats(cv2.bitwise_not(grid), 8)
    return [tuple(int(v) for v in stats[i][:4]) for i in range(1, count)]


def find_cards(comps: list[tuple[int, int, int, int]], shape: tuple[int, int]
               ) -> list[tuple[int, int, int, int]]:
    """Voter cards: ~1/3 page wide, ~1/10 page tall, sorted in reading order."""
    h, w = shape
    cards = [c for c in comps
             if 0.20 * w < c[2] < 0.40 * w and 0.04 * h < c[3] < 0.16 * h]
    cards.sort(key=lambda b: (round(b[1] / (0.02 * h)), b[0]))
    return cards


def card_regions(card: tuple[int, int, int, int], comps: list[tuple[int, int, int, int]]
                 ) -> tuple[tuple[int, int, int, int], int]:
    """(serial box, x of the photo box) for one card, with proportional fallbacks."""
    x, y, cw, ch = card
    inside = [c for c in comps
              if c[0] > x and c[1] > y and c[0] + c[2] < x + cw and c[1] + c[3] < y + ch]
    serials = [c for c in inside
               if 0.15 * cw < c[2] < 0.5 * cw and c[3] < 0.3 * ch and c[1] - y < 0.3 * ch]
    photos = [c for c in inside if 0.10 * cw < c[2] < 0.35 * cw and c[3] > 0.45 * ch]
    serial_box = max(serials, key=lambda c: c[2]) if serials else (
        x, y, int(0.35 * cw), int(0.18 * ch))
    photo_x = max(photos, key=lambda c: c[3])[0] if photos else x + int(0.74 * cw)
    return serial_box, photo_x


def ink_crop(img: np.ndarray, pad: int = 12) -> np.ndarray | None:
    """Tighten a crop onto its dark pixels; None when the region is blank."""
    ys, xs = np.where(img < 128)
    if not len(ys):
        return None
    return img[max(0, ys.min() - pad): ys.max() + pad,
               max(0, xs.min() - pad): xs.max() + pad]


# ---------------------------------------------------------------------- parsing

def _clean(value: Any) -> str:
    """Drop the separators and speckle-noise glyphs tesseract leaves around a
    value, but keep a lone "-", which rolls print to mean 'no house number'."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    stripped = text.strip(" :;.।|_=+*?¢~`'\"-")
    return stripped or ("-" if "-" in text else "")


def _digits(value: Any, repair: bool = True) -> int | str:
    text = str(value or "").translate(DEV_DIGITS)
    if repair and not text.strip().isdigit():
        text = text.translate(DIGIT_FIX)
    digits = re.sub(r"\D", "", text)
    return int(digits) if digits else ""


def fix_names(value: str) -> str:
    """`KARBHAR]` -> `KARBHARI`, `Shaf|` -> `Shafl`: tesseract substitutes bar-like
    glyphs for I/l. Case of the rest of the word decides which letter it was."""
    def repl(word: str) -> str:
        if not I_CONFUSIONS.search(word):
            return word
        letters = [c for c in word if c.isalpha()]
        upper = letters and all(c.isupper() for c in letters)
        return I_CONFUSIONS.sub("I" if upper else "l", word)
    return " ".join(repl(w) for w in value.split())


def canon_gender(value: str) -> str:
    """Gender is a closed set, so a misread can be snapped to the nearest member
    - "Fernale"/"Femaie" are the routine ones. The match must be unique: "male"
    is two edits from "female", so an ambiguous read is left alone rather than
    guessed at."""
    text = _clean(value).lower()
    if not text:
        return ""
    if text in GENDER_CANON:
        return GENDER_CANON[text]
    scored = sorted((_edit_distance(text, key), value_) for key, value_ in GENDER_CANON.items())
    if scored and scored[0][0] <= 2 and (len(scored) == 1 or scored[1][0] > scored[0][0]):
        return scored[0][1]
    return _clean(value)


def gender_from_lines(lines: list[str]) -> str:
    """Fallback when the Age/Gender line did not parse: look for a gender word
    anywhere on the card (it wraps onto its own line often enough to matter)."""
    for line in reversed(lines):
        for token in re.split(r"[\s:;.,|]+", line):
            if len(token) < 3:
                continue
            if (canon := canon_gender(token)) in {"Male", "Female", "Third Gender"}:
                return canon
    return ""


def arbitrate(primary: dict[str, str], secondary: dict[str, str],
              conf_primary: float, conf_secondary: float,
              fields: tuple[str, ...] = ("name", "relation_name"),
              simple_fields: tuple[str, ...] = ("age", "gender")) -> tuple[dict[str, str], list[str]]:
    """Merge two models' readings of the same card.

    The installed integer model and the tessdata_best float model make different
    character errors (best reads a terminal "i" as "l": Shaikh -> Shalkh; the
    standard one goes the other way, I -> L). Where they agree, the reading is
    almost certainly right (95-97% here). Where they disagree, the tie-break is
    the *other* field on the same card: a roll prints the father/husband's name
    as a suffix of the voter's own name, so the reading whose tokens are
    corroborated across fields is the better one. Confidence breaks a true tie.

    Measured on 240 cards: name 89.2 -> 91.2%, relation_name 93.8 -> 96.2%
    (a perfect arbiter would give 92.5 / 97.1)."""
    merged = dict(primary)
    flags: list[str] = []
    for field in fields:
        a, b = primary.get(field, ""), secondary.get(field, "")
        if a == b:
            continue
        other = "relation_name" if field == "name" else "name"
        pool = {t.upper() for t in (primary.get(other, "") + " "
                                    + secondary.get(other, "")).split()}
        score_a = sum(t.upper() in pool for t in a.split())
        score_b = sum(t.upper() in pool for t in b.split())
        if score_b > score_a or (score_b == score_a and conf_secondary > conf_primary):
            merged[field] = b
        flags.append(f"{field}_arbitrated")
    # age/gender have no corroborating field to vote with, so a disagreement
    # is just flagged (confidence tie-breaks). Previously age/gender were read
    # from primary alone with no cross-model check at all, so a wrong read
    # carried no signal unless it also failed the age_bad range check - the
    # in-range age misreads and both gender misreads found on AC 216 ground
    # truth were completely silent before this.
    for field in simple_fields:
        a, b = primary.get(field, ""), secondary.get(field, "")
        if a == b or a == "" or b == "":
            continue
        if conf_secondary > conf_primary:
            merged[field] = b
        flags.append(f"{field}_arbitrated")
    return merged, flags


def parse_card_lines(lines: list[str]) -> dict[str, str]:
    """Map the label lines of one card onto fields, appending wrapped values."""
    out = {"name": "", "relation_type": "", "relation_name": "", "house_number": "",
           "age": "", "gender": ""}
    last = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if m := RE_AGE_GENDER.search(line):
            out["age"] = _digits(m.group(1))
            out["gender"] = canon_gender(m.group(2))
            last = ""
            continue
        if m := RE_RELATION.match(line):
            label = (m.group(1) or m.group(2) or "").lower()
            out["relation_type"] = RELATION_CANON.get(label, "Other")
            out["relation_name"] = fix_names(_clean(m.group(3)))
            last = "relation_name"
            continue
        if m := RE_NAME.match(line):
            out["name"] = fix_names(_clean(m.group(1)))
            last = "name"
            continue
        if m := RE_HOUSE.match(line):
            out["house_number"] = _clean(m.group(1))
            last = "house_number"
            continue
        if last:  # wrapped continuation of the previous value
            out[last] = fix_names(f"{out[last]} {_clean(line)}".strip())
    if not out["gender"]:
        out["gender"] = gender_from_lines(lines)
    return out


def parse_header(lines: list[str]) -> dict[str, str]:
    """Section/part numbers from the strip above the first card."""
    text = " ".join(lines)
    out = {"section_no": "", "section_name": "", "part_no": ""}
    if m := re.search(r"(?:Section\s*No\s*and\s*Name|अनुभाग)\s*[:;.]?\s*([^|]*)", text,
                      re.IGNORECASE):
        section = _clean(m.group(1))
        if sm := re.match(r"(\d+)\s*[-–]\s*(.*)", section):
            out["section_no"], out["section_name"] = sm.group(1), _clean(sm.group(2))
        else:
            out["section_name"] = section
    if m := re.search(r"Part\s*No\.?\s*[:;.]?\s*(\d+)", text, re.IGNORECASE):
        out["part_no"] = m.group(1)
    return out


def repair_serials(serials: list[int | str]) -> tuple[list[int | str], list[str]]:
    """Serials on a page are consecutive, so position determines them once the
    page's first serial is known. Tesseract reads an isolated one-digit number
    badly, but the modal (serial - index) offset across the page is solid.

    A blank read is therefore not worth flagging; a read that disagrees with the
    fitted sequence is, since it means either the OCR or the fit is wrong."""
    offsets = [s - i for i, s in enumerate(serials) if isinstance(s, int) and s > 0]
    if len(offsets) < 3:
        return serials, ["serial_unfitted" if not isinstance(s, int) else ""
                         for s in serials]
    base = Counter(offsets).most_common(1)[0][0]
    fixed, flags = [], []
    for i, s in enumerate(serials):
        expected = base + i
        fixed.append(expected)
        flags.append("serial_mismatch" if isinstance(s, int) and s != expected else "")
    return fixed, flags


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# Glyphs tesseract routinely confuses between the letter and digit forms.
# Measured on AC 216 (910-card ground truth): I->1 was 557 of ~800 EPIC char
# errors and O->0 was 237 - both close to total confusion (nearly every I/O in
# the crop reads as 1/0). Z/S/B/G/Y are the same shape family and worth fixing
# on sight even though they were single-digit counts in that sample.
EPIC_LETTER_TO_DIGIT = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "Y": "7"}
EPIC_DIGIT_TO_LETTER = {v: k for k, v in EPIC_LETTER_TO_DIGIT.items()}


def fix_epic_glyphs(epic: str, head_len: int = 3) -> tuple[str, bool]:
    """Rewrite characters that violate the <letters><digits> shape: a digit-like
    glyph in the letter zone becomes its letter twin, and vice versa in the
    digit zone. head_len defaults to 3 (the nationwide EPIC prefix length)."""
    if len(epic) <= head_len:
        return epic, False
    head, tail = epic[:head_len], epic[head_len:]
    new_head = "".join(EPIC_DIGIT_TO_LETTER.get(c, c) if c.isdigit() else c for c in head)
    new_tail = "".join(EPIC_LETTER_TO_DIGIT.get(c, c) if c.isalpha() else c for c in tail)
    fixed = new_head + new_tail
    return fixed, fixed != epic


def repair_epics(epics: list[str]) -> tuple[list[str], list[str]]:
    """An EPIC is <letters><digits> and a booth uses only a handful of prefixes,
    so the prefix can be voted on: anything within 2 edits of a prefix that is
    common on this page is rewritten to it. The digits cannot be voted on, so an
    EPIC whose digit count is off the page's mode is flagged rather than fixed."""
    glyph_fixed = [False] * len(epics)
    fixed_epics = []
    for i, e in enumerate(epics):
        fixed, changed = fix_epic_glyphs(e)
        fixed_epics.append(fixed)
        glyph_fixed[i] = changed
    epics = fixed_epics
    well_formed = [e for e in epics if EPIC_RE.match(e)]
    common = {p for p, n in Counter(e[:3] for e in well_formed).items() if n >= 3}
    digit_counts = Counter(len(e) - 3 for e in well_formed).most_common(1)
    modal_digits = digit_counts[0][0] if digit_counts else 0

    out, flags = [], []
    for epic, was_glyph_fixed in zip(epics, glyph_fixed):
        glyph_flag = "epic_glyph_fixed" if was_glyph_fixed else ""
        if not epic:
            out.append("")
            flags.append(";".join(f for f in (glyph_flag, "epic_missing") if f))
            continue
        # Natural split first: a dropped/merged digit shifts the fixed-width
        # modal_digits boundary into the letters, so that split's tail stops
        # being pure digits and the vote below never fires (this is the case
        # that let "IOG963656" go unflagged as fixable - Y misread as G, plus
        # a lost digit - even though "IOG" is within 2 edits of "IYO").
        m = EPIC_NATURAL_SPLIT.match(epic)
        head, tail = (m.group(1), m.group(2)) if m else (
            epic[:-modal_digits], epic[-modal_digits:] if modal_digits else "")
        if common and tail.isdigit() and head not in common:
            near = sorted((_edit_distance(p, head), p) for p in common)
            if near and near[0][0] <= 2 and (len(near) == 1 or near[1][0] > near[0][0]):
                # The prefix vote only replaces the letters; a short tail (a
                # digit the crop lost entirely) is a separate problem it can't
                # fix, so that still needs to carry epic_length through to the
                # escalation rule rather than being marked clean.
                epic = near[0][1] + tail
                length_flag = ("epic_length" if modal_digits and len(tail) != modal_digits
                               else "")
                out.append(epic)
                flags.append(";".join(f for f in (glyph_flag, "epic_prefix_fixed", length_flag)
                                      if f))
                continue
        if not EPIC_RE.match(epic):
            out.append(epic)
            flags.append(";".join(f for f in (glyph_flag, "epic_bad") if f))
            continue
        out.append(epic)
        length_flag = "epic_length" if modal_digits and len(epic) - 3 != modal_digits else ""
        flags.append(";".join(f for f in (glyph_flag, length_flag) if f))
    return out, flags


# ------------------------------------------------------------------- page-level

def process_page(gray: np.ndarray, pdf_name: str, page_no: int, ac_no: str,
                 part_no: int, opts: dict[str, Any]
                 ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ocr = engine()
    comps = rect_components(gray)
    cards = find_cards(comps, gray.shape)

    header_bottom = min((c[1] for c in cards), default=int(0.06 * gray.shape[0]))
    header_lines, _hc = ocr.block(gray[: max(1, header_bottom - 4), :], psm=6)
    header = parse_header(header_lines)
    if header["part_no"] and int(header["part_no"]) != part_no:
        log.warning("%s p%d: printed part number %s != filename part number %d",
                    pdf_name, page_no, header["part_no"], part_no)

    rows: list[dict[str, Any]] = []
    confs: list[float] = []
    for card in cards:
        x, y, cw, ch = card
        serial_box, photo_x = card_regions(card, comps)
        # inside the serial box the status marker sits at the far left and the
        # number is right-aligned; OCR-ing them together invents letters
        sx, sy, sw, sh = serial_box
        inset = max(2, sh // 10)
        top, bottom = sy + inset, sy + sh - inset
        marker_img = ink_crop(gray[top:bottom, sx + inset: sx + int(0.45 * sw)], pad=8)
        serial_img = ink_crop(gray[top:bottom, sx + int(0.45 * sw): sx + sw - inset], pad=8)
        epic_img = ink_crop(gray[y + 2: sy + sh, photo_x - 8: x + cw - 2], pad=opts["epic_pad"])
        detail = gray[sy + sh: y + ch, x: max(x + 1, photo_x - 5)]

        serial_txt = ocr.text(serial_img, psm=10, whitelist="0123456789", scale=2,
                              lang="eng")
        marker_txt = ocr.text(marker_img, psm=10, whitelist="#ESRMQ", scale=3, lang="eng")
        epic_txt = ocr.text(epic_img, psm=opts["epic_psm"], whitelist=ALNUM_WL,
                            scale=opts["epic_scale"], lang="eng")
        lines, conf = ocr.block(detail, psm=6)
        confs.append(conf)

        fields = parse_card_lines(lines)
        card_flags: list[str] = []
        if opts.get("best_tessdata"):
            lines_b, conf_b = ocr.block(detail, psm=6, tessdata=opts["best_tessdata"])
            fields, card_flags = arbitrate(fields, parse_card_lines(lines_b), conf, conf_b)
        rows.append({
            "booth_pdf": pdf_name, "page_no": page_no,
            "ac_no": _digits(ac_no), "section_no": header["section_no"],
            "section_name": header["section_name"],
            # the printed part number is the true identifier - it does not
            # always match the sequential booth filename (see the warning
            # above), which is only a fallback when the header OCR misses it
            "part_no": header["part_no"] or part_no,
            "serial_no": _digits(serial_txt, repair=False),
            "epic_number": re.sub(r"[^A-Z0-9]", "", epic_txt.upper()),
            "name": fields["name"], "relation_type": fields["relation_type"],
            "relation_name": fields["relation_name"],
            "house_number": fields["house_number"], "age": fields["age"],
            "gender": fields["gender"],
            "marker": marker_txt.strip()[:1],
            "flags": ";".join(card_flags), "ocr_conf": f"{conf:.1f}",
            "card_box": f"{x},{y},{cw},{ch}", "dpi": opts["dpi"],
        })

    serials, serial_flags = repair_serials([r["serial_no"] for r in rows])
    epics, epic_flags = repair_epics([r["epic_number"] for r in rows])
    for row, serial, sflag, epic, eflag in zip(rows, serials, serial_flags, epics, epic_flags):
        row["serial_no"], row["epic_number"] = serial, epic
        age = row["age"]
        flags = [f for f in str(row["flags"]).split(";") if f] + [sflag, eflag]
        if not isinstance(age, int) or not 18 <= age <= 120:
            flags.append("age_bad")
        if not row["name"]:
            flags.append("name_missing")
        row["flags"] = ";".join(f for f in flags if f)

    rows = [r for r in rows if r["name"] or r["epic_number"]]
    page_row = {"booth_pdf": pdf_name, "page_no": page_no, "cards": len(rows),
                "mean_conf": f"{statistics.mean(confs):.1f}" if confs else "0.0",
                "section_no": header["section_no"], "section_name": header["section_name"]}
    if opts["debug"]:
        for r in rows:
            log.info("  %s", {k: r[k] for k in ("serial_no", "epic_number", "name",
                                                "relation_type", "relation_name",
                                                "house_number", "age", "gender", "marker")})
    return rows, page_row


def read_part_no_only(pdf_path: Path, first_page: int, opts: dict[str, Any]) -> str:
    """Cheaply read the printed part number off a booth's first elector page
    header only - one small OCR call, not the ~30-card full-page pass. Used
    to find PDFs that are duplicate scans of the same electoral part under a
    different filename before spending full OCR on them: the printed number
    is the true identifier, and it does not always match the sequential
    booth filename (see the "printed part number != filename part number"
    warning in process_page)."""
    if _OPS is None:
        worker_init(opts)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if first_page > len(pdf.pages):
                return ""
            gray = page_gray(pdf, first_page, opts["dpi"])
    except Exception:  # noqa: BLE001 - a truncated download should not stop the pre-scan
        return ""
    comps = rect_components(gray)
    cards = find_cards(comps, gray.shape)
    header_bottom = min((c[1] for c in cards), default=int(0.06 * gray.shape[0]))
    header_lines, _hc = engine().block(gray[: max(1, header_bottom - 4), :], psm=6)
    return parse_header(header_lines)["part_no"]


def process_chunk(pdf_path: Path, pages: list[int], ac_no: str, opts: dict[str, Any]
                  ) -> tuple[list[tuple[list[dict], dict]], list[tuple[int, str]]]:
    """OCR a run of pages from one PDF. Opening the PDF once per chunk (rather
    than per page) is most of the win from chunking; the rest is fewer futures.

    Returns (results, failures) - a page that fails all its attempts is reported
    rather than raised, so its siblings in the chunk still land."""
    if _OPS is None:            # spawn-based pools do not inherit the initializer
        worker_init(opts)
    part_no = int(re.sub(r"\D", "", pdf_path.stem).lstrip("0") or "0")
    results: list[tuple[list[dict], dict]] = []
    failures: list[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no in pages:
            for attempt in range(opts["retries"] + 1):
                try:
                    gray = page_gray(pdf, page_no, opts["dpi"])
                    results.append(process_page(gray, pdf_path.name, page_no, ac_no,
                                                part_no, opts))
                    break
                except Exception as exc:  # noqa: BLE001 - one page must not kill the chunk
                    if attempt < opts["retries"]:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    failures.append((page_no, f"{type(exc).__name__}: {exc}"))
    return results, failures


# ------------------------------------------------------------------------- I/O

def read_csv_rows(path: Path, columns: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as f:
        return [{c: r.get(c, "") for c in columns} for r in csv.DictReader(f)]


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def check_output_path(path: Path, force: bool) -> None:
    """Refuse to overwrite someone else's CSV - extract_voters_gemini.py writes
    voters_<AC>.csv, one character away from this script's default."""
    if force or not path.exists():
        return
    with path.open(encoding="utf-8-sig") as f:
        header = next(csv.reader(f), [])
    if not {"flags", "ocr_conf"} <= set(header):
        raise SystemExit(
            f"{path} exists and does not look like this script's output "
            f"(no flags/ocr_conf columns) - it is probably the Gemini extraction.\n"
            f"Pass --out <other path>, or --force to overwrite it anyway.")


def _sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    booth = int(re.sub(r"\D", "", str(row.get("booth_pdf", ""))) or 0)
    return booth, int(str(row.get("page_no") or 0)), int(str(row.get("serial_no") or 0) or 0)


def page_count(pdf_path: Path) -> int:
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


def chunked(pages: list[int], size: int) -> list[list[int]]:
    return [pages[i:i + size] for i in range(0, len(pages), size)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract voter rows from booth-roll PDFs with Tesseract OCR")
    ap.add_argument("--pdf-dir", type=Path, help="Directory of booth_*.pdf files")
    ap.add_argument("--ac", help="AC number; used for --out naming and the ac_no column "
                                 "(default: the --pdf-dir folder name)")
    ap.add_argument("--out", type=Path,
                    help="Output CSV (default booth_list_csv/voters_<AC>_ocr.csv)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite --out even if it is not this script's output")
    ap.add_argument("--tesseract", help="Path to the tesseract binary (default: auto-detect)")
    ap.add_argument("--lang", default="eng", help="Tesseract language for the label block, "
                                                  "e.g. eng, hin, mar, 'hin+eng' (default eng)")
    ap.add_argument("--first-page", type=int, default=3, help="First elector page, 1-based (default 3)")
    ap.add_argument("--last-page", type=int, default=0,
                    help="Last elector page, 1-based; 0 = total pages - --skip-last")
    ap.add_argument("--skip-last", type=int, default=2,
                    help="Trailing summary pages to skip when --last-page is 0 (default 2)")
    ap.add_argument("--max-pages", type=int, help="Process at most N elector pages per booth")
    ap.add_argument("--parts", help="Comma-separated booth part numbers, e.g. 1,2,7")
    ap.add_argument("--limit", type=int, help="Process only the first N booth PDFs")
    ap.add_argument("--dpi", type=int, default=200, help="Render DPI (200 reads best here)")
    # these three defaults were tuned on a page of AC 228; the padding matters most
    ap.add_argument("--epic-psm", type=int, default=8, help="Tesseract PSM for the EPIC number")
    ap.add_argument("--epic-scale", type=int, default=7,
                    help="Upscale factor for the EPIC crop only (page --dpi is separate - "
                         "raising that instead regressed other fields whose crop math is "
                         "tuned for it). Swept 3/5/7/10 on AC 216 ground truth: "
                         "epic_number exact match 85.6%% / 92.4%% / 93.3%% / 92.2%% - 7 is "
                         "the peak, 10 over-upscales (blur) and is slower for a worse result.")
    ap.add_argument("--epic-pad", type=int, default=35, help="Padding around the EPIC ink crop")
    ap.add_argument("--engine", default="tesseract", choices=sorted(ENGINES),
                    help="OCR engine: tesseract (CPU, default) or easyocr "
                         "(torch; uses CUDA/MPS when present)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="easyocr recognition batch size (raise it on a GPU)")
    ap.add_argument("--best-tessdata", default=str(HERE / "tessdata_best"),
                    help="Directory holding the tessdata_best eng.traineddata. When it "
                         "exists, every card is read by both models and the readings are "
                         "arbitrated (+2 pts name, +2.4 pts relation_name, ~2x slower)")
    ap.add_argument("--no-ensemble", action="store_true",
                    help="Read each card with the installed model only")
    ap.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"],
                    help="GPU for the OpenCV segmentation, and for the OCR engine "
                         "if it has a GPU path (easyocr does, tesseract does not). "
                         "Default auto.")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 4),
                    help="Worker processes (default: CPU count)")
    ap.add_argument("--chunk-pages", type=int, default=0,
                    help="Pages per work unit; one PDF open per chunk "
                         "(default 0 = size it so every worker gets ~3 chunks)")
    ap.add_argument("--retries", type=int, default=1, help="Retries per failed page")
    ap.add_argument("--ocr-timeout", type=int, default=120,
                    help="Seconds before one tesseract call is abandoned (0 = no limit)")
    ap.add_argument("--flush-every", type=int, default=25,
                    help="Write the CSVs every N pages (default 25)")
    ap.add_argument("--refresh", action="store_true", help="Ignore existing output and redo all pages")
    ap.add_argument("--no-dedup-scan", action="store_true",
                    help="Skip the pre-scan that finds booth PDFs printing the same part "
                         "number under a different filename and drops the full OCR pass "
                         "on the duplicates")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not args.pdf_dir and not args.ac:
        ap.error("provide --pdf-dir (or --ac for booth_list_pdf/<AC>)")
    pdf_dir = args.pdf_dir or (HERE / "booth_list_pdf" / str(args.ac))
    if not pdf_dir.is_dir():
        raise SystemExit(f"PDF dir not found: {pdf_dir}")
    ac_no = args.ac or pdf_dir.name
    out_csv = args.out or HERE / "booth_list_csv" / f"voters_{ac_no}_ocr.csv"
    pages_csv = out_csv.with_suffix(".pages.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    check_output_path(out_csv, args.force)

    tesseract = resolve_tesseract(args.tesseract)
    pytesseract.pytesseract.tesseract_cmd = tesseract
    device, note = resolve_device(args.device, args.engine)
    if args.engine == "tesseract":
        log.info("tesseract: %s (v%s), lang=%s", tesseract,
                 pytesseract.get_tesseract_version(), args.lang)
    log.info("engine=%s device=%s (%s)", args.engine, device, note)

    best_dir = Path(args.best_tessdata)
    best_tessdata = ""
    if args.engine == "tesseract" and not args.no_ensemble:
        if (best_dir / f"{args.lang.split('+')[0]}.traineddata").exists():
            best_tessdata = str(best_dir)
            log.info("ensemble: reading every card with the installed model and the "
                     "tessdata_best model in %s, arbitrating the two", best_dir)
        else:
            log.info("no tessdata_best model in %s - single-model read. Fetch it for "
                     "+2 pts on names:\n  curl -sL -o %s/%s.traineddata https://raw."
                     "githubusercontent.com/tesseract-ocr/tessdata_best/main/%s.traineddata",
                     best_dir, best_dir, args.lang.split('+')[0], args.lang.split('+')[0])

    opts = {"tesseract": tesseract, "lang": args.lang, "dpi": args.dpi,
            "epic_psm": args.epic_psm, "epic_scale": args.epic_scale,
            "epic_pad": args.epic_pad, "debug": args.debug, "device": device,
            "retries": args.retries, "ocr_timeout": args.ocr_timeout,
            "engine": args.engine, "batch_size": args.batch_size,
            "best_tessdata": best_tessdata}

    pdfs = sorted(pdf_dir.glob("booth_*.pdf"))
    if not pdfs:
        raise SystemExit(f"No booth_*.pdf files in {pdf_dir}")
    if args.parts:
        wanted = {int(x) for x in re.findall(r"\d+", args.parts)}
        pdfs = [p for p in pdfs if int(re.sub(r"\D", "", p.stem) or 0) in wanted]
        if missing := wanted - {int(re.sub(r"\D", "", p.stem) or 0) for p in pdfs}:
            log.warning("no PDF found for parts: %s", ",".join(map(str, sorted(missing))))
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        raise SystemExit("no booth PDFs selected")

    # Cheap pre-scan: read just the printed part number off each booth's
    # first elector page header (one small OCR call, not the ~30-card full
    # pass) to find PDFs that are duplicate scans of the same electoral part
    # under a different filename, and skip full OCR on the duplicates
    # entirely rather than pay for it and dedup after the fact.
    if not args.no_dedup_scan:
        log.info("pre-scanning %d booths for duplicate part numbers...", len(pdfs))
        prescan_workers = max(1, min(args.workers, len(pdfs)))
        seen_parts: dict[str, Path] = {}
        dup_rows: list[dict[str, str]] = []
        kept: list[Path] = []
        with ProcessPoolExecutor(max_workers=prescan_workers, initializer=worker_init,
                                 initargs=(opts,)) as ex:
            futs = {ex.submit(read_part_no_only, pdf, args.first_page, opts): pdf for pdf in pdfs}
            for fut in as_completed(futs):
                pdf = futs[fut]
                try:
                    part_no = fut.result()
                except Exception as exc:  # noqa: BLE001 - a bad header read should not block a booth
                    log.warning("%s: part-number pre-scan failed (%s) - processing anyway", pdf.name, exc)
                    part_no = ""
                if part_no and part_no in seen_parts:
                    log.warning("%s: printed part_no %s already covered by %s - skipping full OCR",
                                pdf.name, part_no, seen_parts[part_no].name)
                    dup_rows.append({"booth_pdf": pdf.name, "part_no": part_no,
                                     "kept_booth_pdf": seen_parts[part_no].name})
                    continue
                if part_no:
                    seen_parts[part_no] = pdf
                kept.append(pdf)
        if dup_rows:
            dup_path = out_csv.with_suffix(".duplicate_parts.csv")
            write_csv(dup_path, ["booth_pdf", "part_no", "kept_booth_pdf"], dup_rows)
            log.info("skipped %d duplicate-part booth(s), full list in %s", len(dup_rows), dup_path)
        pdfs = sorted(kept)

    rows = [] if args.refresh else read_csv_rows(out_csv, COLUMNS)
    page_rows = [] if args.refresh else read_csv_rows(pages_csv, PAGE_COLUMNS)
    done = {(p["booth_pdf"], str(p["page_no"])) for p in page_rows}
    if done:
        log.info("resuming; %d pages already in %s", len(done), out_csv.name)

    todo: list[tuple[Path, list[int]]] = []
    total_pages = 0
    for pdf in pdfs:
        try:
            total = page_count(pdf)
        except Exception as exc:  # noqa: BLE001 - a truncated download should not stop the run
            log.error("%s: cannot open (%s) - skipping", pdf.name, exc)
            continue
        last = min(args.last_page or (total - args.skip_last), total)
        pages = list(range(args.first_page, last + 1))
        if args.max_pages:
            pages = pages[: args.max_pages]
        if not pages:
            log.warning("%s: no elector pages in range (total_pages=%d)", pdf.name, total)
        pages = [p for p in pages if (pdf.name, str(p)) not in done]
        total_pages += len(pages)
        if pages:
            todo.append((pdf, pages))

    if not todo:
        log.info("nothing to do; %s already covers the selected pages", out_csv)
        return

    # size chunks so the pool actually fills: a fixed size would put a 4-page run
    # in one chunk and leave every other worker idle
    chunk_size = args.chunk_pages or max(
        1, min(8, -(-total_pages // max(1, (args.workers or 1) * 3))))
    tasks = [(pdf, chunk) for pdf, pages in todo for chunk in chunked(pages, chunk_size)]

    workers = max(1, min(args.workers, len(tasks)))
    if args.engine == "easyocr" and device == "gpu":
        # one torch model per process, and the GPU is the bottleneck, not the cores
        workers = 1 if args.workers == (os.cpu_count() or 4) else workers
    log.info("OCR-ing %d pages from %d booths: %d chunks x <=%d pages, %d workers",
             total_pages, len(pdfs), len(tasks), chunk_size, workers)

    done_pages = 0
    failures: list[str] = []
    started = time.time()

    def flush() -> None:
        rows.sort(key=_sort_key)
        page_rows.sort(key=_sort_key)
        write_csv(out_csv, COLUMNS, rows)
        write_csv(pages_csv, PAGE_COLUMNS, page_rows)

    def record(result: tuple[list[tuple[list[dict], dict]], list[tuple[int, str]]],
               pdf: Path) -> None:
        nonlocal done_pages
        chunk_results, chunk_failures = result
        for new_rows, page_row in chunk_results:
            rows.extend(new_rows)
            page_rows.append(page_row)
            done_pages += 1
            if page_row["cards"] != CARDS_PER_PAGE:
                log.warning("%s p%s: %d cards (expected %d) - check the page",
                            page_row["booth_pdf"], page_row["page_no"],
                            page_row["cards"], CARDS_PER_PAGE)
        for page_no, error in chunk_failures:
            failures.append(f"{pdf.name} p{page_no}")
            log.error("%s p%d failed: %s", pdf.name, page_no, error)
        rate = done_pages / max(1e-9, time.time() - started)
        log.info("[%d/%d pages] %s -> %d voters (%.1f pages/s, eta %s)",
                 done_pages, total_pages, pdf.name, len(rows), rate,
                 time.strftime("%M:%S", time.gmtime((total_pages - done_pages) / max(rate, 1e-9))))
        if done_pages % max(1, args.flush_every) < len(chunk_results):
            flush()

    try:
        if workers == 1:
            worker_init(opts)
            for pdf, pages in tasks:
                record(process_chunk(pdf, pages, ac_no, opts), pdf)
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=worker_init,
                                     initargs=(opts,)) as ex:
                futs = {ex.submit(process_chunk, pdf, pages, ac_no, opts): pdf
                        for pdf, pages in tasks}
                for fut in as_completed(futs):
                    pdf = futs[fut]
                    try:
                        record(fut.result(), pdf)
                    except Exception as exc:  # noqa: BLE001 - chunk-level crash
                        failures.append(f"{pdf.name} (chunk)")
                        log.error("%s chunk failed: %s", pdf.name, exc)
    except KeyboardInterrupt:
        flush()
        log.warning("interrupted; %d pages saved to %s (re-run to resume)",
                    done_pages, out_csv)
        raise SystemExit(130)
    finally:
        flush()

    flag_counts = Counter(f for r in rows for f in str(r["flags"]).split(";") if f)
    elapsed = time.time() - started
    log.info("Done. %d voters from %d pages in %.0fs (%.1f pages/s) -> %s",
             len(rows), len(page_rows), elapsed, done_pages / max(elapsed, 1e-9), out_csv)
    log.info("flagged rows: %d/%d | %s", sum(1 for r in rows if r["flags"]), len(rows),
             ", ".join(f"{k}={v}" for k, v in sorted(flag_counts.items())) or "none")
    if failures:
        log.warning("%d pages FAILED (re-run to retry): %s", len(failures), ", ".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
