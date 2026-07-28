"""
Extract page-1 booth details from ECI booth-roll PDFs using Gemini vision.

This is the cloud/Gemini alternative to extract_booth_info.py. It sends page 1
as an image to the Gemini API (Interactions API) and asks for one strict JSON
object matching the same booth_info_<AC>.csv columns.

Credentials
-----------
Two backends are supported; put one of these in survey/.env,
survey/booth_list/.env, or your shell env:

    # Gemini API (AI Studio key)
    GEMINI_API_KEY=...           # or GOOGLE_API_KEY=...

    # Vertex AI (GCP project credits) - service-account JSON key
    GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

With --backend auto (the default) Vertex is used when a service-account key is
found, otherwise the Gemini API key. The Vertex OAuth token is minted locally
from the key with `cryptography` (no google-auth dependency needed).

Usage
-----
    PY=../../jup_venv/bin/python
    $PY extract_booth_info_gemini.py --ac 324 --limit 3 --debug
    $PY extract_booth_info_gemini.py --ac 324 --parts 25,26,37,38,40,46,51 --debug
    $PY extract_booth_info_gemini.py --ac 324 --workers 4
    $PY extract_booth_info_gemini.py --ac 324 --backend vertex --sa-key key.json
    $PY extract_booth_info_gemini.py --pdf-dir booth_list_pdf/324 --out booth_list_csv/booth_info_gemini_324.csv

Notes
-----
Gemini is stronger at reading noisy Hindi than EasyOCR, but it is a generative
model. The script still derives exact part_no/ac_no from file/folder and validates
the voter-count identity in counts_ok.

Cost accounting: the Interactions API reports usage as snake_case fields
(usage.total_input_tokens / total_output_tokens / total_thought_tokens /
total_cached_tokens). Thinking tokens are billed at the output rate; cached
input tokens are billed at the (cheaper) cached rate.

panchayat_clean: once all requested booths are extracted, one extra text-only
Gemini call standardizes OCR/spelling variants of the AC's gram panchayat
names into the panchayat_clean column (cached in
booth_list_csv/panchayat_canon_<AC>.json, reused while the AC's distinct
panchayat spellings are unchanged). build_localities.py reads this column
directly instead of re-running standardization itself. Pass --no-panchayat-clean
to skip it, or --refresh-panchayat to force a re-ask.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import logging
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import pdfplumber

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extract-gemini")

HERE = Path(__file__).resolve().parent
API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
DEFAULT_MODEL = "gemini-3-flash-preview"
# gemini-3-flash-preview paid tier (ai.google.dev/gemini-api/docs/pricing)
DEFAULT_INPUT_PRICE_PER_M = 0.50
DEFAULT_OUTPUT_PRICE_PER_M = 3.00
DEFAULT_CACHED_PRICE_PER_M = 0.05
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

NUM_FIELDS = ["part_no", "ac_no", "pc_no", "revision_year", "qualifying_date",
              "publication_date", "pin_code", "start_serial", "end_serial",
              "male", "female", "third_gender", "total", "counts_ok"]
HIN_FIELDS = ["ac_name", "pc_name", "revision_type", "sections", "main_town_village",
              "ward", "post_office", "police_station", "panchayat", "block", "tehsil",
              "district", "ps_no_name", "ps_address", "ps_type"]
# derived after all booths in the AC are extracted, not read off any single image
DERIVED_FIELDS = ["panchayat_clean"]
USAGE_FIELDS = ["input_tokens", "output_tokens", "thought_tokens", "cached_tokens",
                "total_tokens", "approx_cost_usd"]
EXTRACT_COLUMNS = ["booth_pdf", "total_pages"] + NUM_FIELDS + HIN_FIELDS + DERIVED_FIELDS
COLUMNS = EXTRACT_COLUMNS + USAGE_FIELDS

# Fields the model must read off the image. part_no/ac_no/counts_ok/booth_pdf/
# total_pages are derived locally, so asking the model for them only spends
# output tokens — except part_no, kept as a cheap cross-check against the
# filename-derived part number.
MODEL_NUM_FIELDS = [f for f in NUM_FIELDS if f not in {"ac_no", "counts_ok"}]
MODEL_FIELDS = MODEL_NUM_FIELDS + HIN_FIELDS

PROMPT = """\
Extract fields from page 1 of this Hindi Election Commission booth-roll PDF.
Preserve Hindi text exactly as printed; do not translate, normalize to a more
common place name, or guess from outside knowledge. Unreadable/absent field ->
empty string.

- Convert Devanagari digits to ASCII in numeric/date fields; dates DD-MM-YYYY.
- ps_no_name: full printed polling-station number + name.
- sections: join the section/tola lines with " ; ".
- Do not include labels like "ग्राम", "ब्लॉक", "तहसील" unless part of the value.
- The voter-count row has start_serial, end_serial, male, female, third_gender,
  total. Read those carefully.
"""

FIELD_DESCRIPTIONS = {
    "part_no": "printed part number of the booth",
    "qualifying_date": "DD-MM-YYYY",
    "publication_date": "DD-MM-YYYY",
    "sections": "section/tola lines joined with ' ; '",
    "ps_no_name": "polling station number and name as printed",
}


def response_schema() -> dict[str, Any]:
    props: dict[str, Any] = {}
    for field in MODEL_FIELDS:
        if field in {"qualifying_date", "publication_date"} or field in HIN_FIELDS:
            typ = "string"
        else:
            typ = "integer"
        prop: dict[str, Any] = {"type": typ}
        if field in FIELD_DESCRIPTIONS:
            prop["description"] = FIELD_DESCRIPTIONS[field]
        props[field] = prop
    return {"type": "object", "properties": props, "required": MODEL_FIELDS}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_env_files(env_file: Path | None = None) -> None:
    for path in [HERE.parent / ".env", HERE / ".env", env_file]:
        if path:
            load_dotenv(path)


class VertexAuth:
    """Mints OAuth2 access tokens from a service-account JSON key (no google-auth)."""

    SCOPE = "https://www.googleapis.com/auth/cloud-platform"

    def __init__(self, sa_path: Path) -> None:
        info = json.loads(sa_path.read_text(encoding="utf-8"))
        missing = [k for k in ("client_email", "private_key", "token_uri", "project_id")
                   if not info.get(k)]
        if info.get("type") != "service_account" or missing:
            raise SystemExit(f"{sa_path} is not a service-account key "
                             f"(missing: {', '.join(missing) or 'type=service_account'})")
        from cryptography.hazmat.primitives import serialization
        self.project_id: str = info["project_id"]
        self._email: str = info["client_email"]
        self._token_uri: str = info["token_uri"]
        self._key = serialization.load_pem_private_key(info["private_key"].encode(), None)
        self._lock = threading.Lock()
        self._token = ""
        self._expiry = 0.0

    @staticmethod
    def _b64u(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode().rstrip("=")

    def token(self) -> str:
        with self._lock:
            if self._token and time.time() < self._expiry - 300:
                return self._token
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            now = int(time.time())
            header = self._b64u(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
            claims = self._b64u(json.dumps({
                "iss": self._email, "scope": self.SCOPE, "aud": self._token_uri,
                "iat": now - 10, "exp": now + 3600,
            }).encode())
            signing_input = f"{header}.{claims}".encode()
            sig = self._key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
            assertion = f"{header}.{claims}.{self._b64u(sig)}"
            body = urllib.parse.urlencode({
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": assertion,
            }).encode()
            req = urllib.request.Request(self._token_uri, data=body, method="POST",
                                         headers={"Content-Type": "application/x-www-form-urlencoded"})
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    tok = json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                raise SystemExit(f"Vertex token exchange failed (HTTP {exc.code}): {detail}") from exc
            self._token = tok["access_token"]
            self._expiry = time.time() + float(tok.get("expires_in", 3600))
            return self._token


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
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": response_schema(),
            },
            "generation_config": {"thinking_level": "minimal"},
        }


def _vertex_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Vertex responseSchema wants OpenAPI-style uppercase type names."""
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            out[key] = value.upper()
        elif isinstance(value, dict):
            out[key] = _vertex_schema(value)
        else:
            out[key] = value
    return out


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
                "thinkingConfig": {"thinkingLevel": "MINIMAL"},
            },
        }


def _vertex_preflight(backend: VertexBackend, model: str) -> str | None:
    """Free countTokens call to verify IAM/API access; returns the error or None."""
    url = backend.url(model).replace(":generateContent", ":countTokens")
    payload = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    try:
        api_request(url, backend.headers(), payload, timeout=30, retries=1)
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
            log.info("backend: Vertex AI (project %s, location %s)",
                     auth.project_id, args.location)
            return vertex
        hint = (f"Vertex AI not usable with this key yet: {err}\n"
                f"Fix in the Cloud Console (project {auth.project_id}):\n"
                f"  1. Enable the Vertex AI API: https://console.cloud.google.com/apis/"
                f"library/aiplatform.googleapis.com?project={auth.project_id}\n"
                f"  2. IAM -> Grant access -> principal {auth._email}\n"
                f"     role 'Vertex AI User' (roles/aiplatform.user)")
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


def render_page1_jpeg(pdf_path: Path, dpi: int, quality: int, max_dim: int) -> tuple[str, int]:
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        if not total_pages:
            raise RuntimeError("PDF has no pages")
        img = pdf.pages[0].to_image(resolution=dpi).original.convert("RGB")
    try:
        if max_dim and max(img.size) > max_dim:
            scale = max_dim / max(img.size)
            img = img.resize((round(img.width * scale), round(img.height * scale)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii"), total_pages
    finally:
        img.close()


def _retry_after_seconds(exc: urllib.error.HTTPError, fallback: float) -> float:
    value = exc.headers.get("Retry-After") if exc.headers else None
    try:
        return min(float(value), 60.0) if value else fallback
    except ValueError:
        return fallback


def api_request(url: str, headers: dict[str, str], payload: dict[str, Any],
                timeout: int, retries: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            if exc.code in {408, 429, 500, 502, 503, 504} and attempt < retries:
                delay = _retry_after_seconds(exc, 2 ** attempt + random.uniform(0, 1))
                log.warning("Gemini HTTP %d, retrying in %.1fs (attempt %d/%d)",
                            exc.code, delay, attempt + 1, retries)
                time.sleep(delay)
                continue
            raise RuntimeError(f"Gemini HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            if attempt < retries:
                delay = 2 ** attempt + random.uniform(0, 1)
                log.warning("Gemini request error (%s), retrying in %.1fs", exc, delay)
                time.sleep(delay)
                continue
            raise RuntimeError(f"Gemini request failed: {exc}") from exc
    raise RuntimeError("Gemini request failed after retries")


def _walk_strings(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, list):
        out: list[str] = []
        for item in obj:
            out.extend(_walk_strings(item))
        return out
    if isinstance(obj, dict):
        out = []
        for key in ("output_text", "text", "content", "steps", "outputs",
                    "candidates", "parts"):
            if key in obj:
                out.extend(_walk_strings(obj[key]))
        return out
    return []


def _extract_json_text(response: dict[str, Any]) -> str:
    status = response.get("status")
    if status in {"failed", "cancelled"}:
        err = response.get("error") or {}
        raise RuntimeError(f"Gemini interaction {status}: "
                           f"{err.get('code', '')} {err.get('message', json.dumps(response)[:500])}")
    for text in _walk_strings(response):
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            return match.group(0)
    raise RuntimeError(f"No JSON object found in Gemini response: {json.dumps(response)[:1000]}")


def parse_gemini_response(response: dict[str, Any]) -> dict[str, Any]:
    parsed = json.loads(_extract_json_text(response))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Gemini returned non-object JSON: {str(parsed)[:200]}")
    return parsed


def _first_int(obj: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return 0


def extract_usage(response: dict[str, Any]) -> dict[str, int]:
    """Read token usage; Interactions API uses snake_case `usage` fields,
    generateContent-style camelCase kept as a fallback."""
    usage = response.get("usage") or response.get("usageMetadata") or {}
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = _first_int(usage, ("total_input_tokens", "input_tokens", "promptTokenCount"))
    output_tokens = _first_int(usage, ("total_output_tokens", "output_tokens", "candidatesTokenCount"))
    thought_tokens = _first_int(usage, ("total_thought_tokens", "thought_tokens", "thoughtsTokenCount"))
    cached_tokens = _first_int(usage, ("total_cached_tokens", "cached_tokens", "cachedContentTokenCount"))
    total_tokens = _first_int(usage, ("total_tokens", "totalTokenCount"))
    if not total_tokens:
        total_tokens = input_tokens + output_tokens + thought_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thought_tokens": thought_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": total_tokens,
    }


def estimate_cost_from_usage(usage: dict[str, int], input_price_per_m: float,
                             output_price_per_m: float, cached_price_per_m: float) -> float:
    """Thinking tokens bill at the output rate; cached tokens (a subset of
    input tokens) bill at the cached rate instead of the input rate."""
    if usage["input_tokens"] or usage["output_tokens"] or usage["thought_tokens"]:
        cached = min(usage["cached_tokens"], usage["input_tokens"])
        billable_input = usage["input_tokens"] - cached
        billable_output = usage["output_tokens"] + usage["thought_tokens"]
        return (billable_input / 1e6 * input_price_per_m
                + cached / 1e6 * cached_price_per_m
                + billable_output / 1e6 * output_price_per_m)
    # only a total is known -> price everything at the dearer rate (upper bound)
    return usage["total_tokens"] / 1e6 * max(input_price_per_m, output_price_per_m)


def _row_int(row: dict[str, Any], field: str) -> int:
    value = str(row.get(field) or "").strip()
    return int(value) if value.isdigit() else 0


def log_usage_summary(rows: list[dict[str, Any]], input_price_per_m: float,
                      output_price_per_m: float) -> None:
    known = [r for r in rows if any(_row_int(r, f) for f in
                                    ("input_tokens", "output_tokens", "total_tokens"))]
    if not known:
        log.info("Gemini usage metadata was not present in saved rows; cost could not be estimated")
        return
    input_tokens = sum(_row_int(r, "input_tokens") for r in known)
    output_tokens = sum(_row_int(r, "output_tokens") for r in known)
    thought_tokens = sum(_row_int(r, "thought_tokens") for r in known)
    total_tokens = sum(_row_int(r, "total_tokens") for r in known)
    approx_cost = 0.0
    for r in known:
        try:
            approx_cost += float(r.get("approx_cost_usd") or 0)
        except ValueError:
            pass
    log.info("Gemini usage: %d/%d rows | input %s | output %s | thought %s | total %s tokens",
             len(known), len(rows), f"{input_tokens:,}", f"{output_tokens:,}",
             f"{thought_tokens:,}", f"{total_tokens:,}")
    if input_tokens or output_tokens:
        log.info("Approx paid-tier cost at $%.2f/M input + $%.2f/M output: $%.4f "
                 "(avg $%.5f/booth)", input_price_per_m, output_price_per_m,
                 approx_cost, approx_cost / len(known))
    else:
        log.info("Approx paid-tier cost upper bound at $%.2f/M tokens: $%.4f",
                 max(input_price_per_m, output_price_per_m), approx_cost)


GEMINI_PANCHAYAT_PROMPT = """\
These gram panchayat names were OCR-extracted from Hindi (Devanagari) ECI \
voter-roll booth pages of ONE assembly constituency. Many entries are OCR or \
spelling variants of the SAME panchayat. Group the variants and give each \
group one standardized Devanagari spelling.

Rules:
- Group two names ONLY when they are clearly the same panchayat (matra/anusvara
  confusions, missing or extra characters, digit style like नं0/नं०, spacing).
- Do NOT merge genuinely different panchayats, even if their names are similar.
- Every input name must appear in exactly one group's variants list, exactly as
  given (a group may hold a single name).
- "standard" is the corrected/most plausible spelling; prefer the most frequent
  input spelling unless it is an obvious OCR error.
- Do not translate, transliterate, or invent names.

Input panchayat names (with occurrence counts):
"""


def _panchayat_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "standard": {"type": "string"},
                        "variants": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["standard", "variants"],
                },
            },
        },
        "required": ["groups"],
    }


def standardize_panchayats(
    rows: list[dict[str, Any]],
    backend: GeminiApiBackend | VertexBackend,
    model: str,
    cache_path: Path,
    refresh: bool = False,
) -> dict[str, str]:
    """Map each distinct panchayat spelling in this AC to one Gemini-standardized
    spelling (one text call for the whole AC). Cached per AC in cache_path and
    reused as long as the set of input names is unchanged. Falls back to an
    identity mapping (each name maps to itself) if Gemini fails, so callers
    always get a complete mapping."""
    counts = Counter(r["panchayat"] for r in rows if r.get("panchayat"))
    if not counts:
        return {}

    if cache_path.exists() and not refresh:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if set(cached["names"]) == set(counts):
                log.info("using cached Gemini panchayat mapping: %s", cache_path.name)
                return {str(k): str(v) for k, v in cached["mapping"].items()}
            log.info("panchayat names changed since %s was written; re-asking Gemini",
                     cache_path.name)
        except (json.JSONDecodeError, KeyError, TypeError):
            log.warning("ignoring unreadable panchayat cache %s", cache_path.name)

    listing = "\n".join(f"- {name} (x{n})" for name, n in
                        sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    prompt = GEMINI_PANCHAYAT_PROMPT + listing
    schema = _panchayat_schema()
    if backend.name == "vertex":
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": _vertex_schema(schema),
                "thinkingConfig": {"thinkingLevel": "MINIMAL"},
            },
        }
    else:
        payload = {
            "model": model,
            "input": [{"type": "text", "text": prompt}],
            "response_format": {"type": "text", "mime_type": "application/json",
                                "schema": schema},
            "generation_config": {"thinking_level": "minimal"},
        }

    try:
        response = api_request(backend.url(model), backend.headers(), payload,
                               timeout=120, retries=3)
        parsed = parse_gemini_response(response)
    except RuntimeError as exc:
        log.warning("Gemini panchayat standardization failed (%s); keeping original "
                    "spellings", exc)
        return {name: name for name in counts}

    mapping: dict[str, str] = {}
    for group in parsed.get("groups") or []:
        std = _clean_text(group.get("standard", ""))
        variants = [_clean_text(v) for v in group.get("variants") or []]
        if not std:
            continue
        for v in variants:
            if v in counts and v not in mapping:
                mapping[v] = std
    missing = sorted(n for n in counts if n not in mapping)
    if missing:
        log.warning("Gemini mapping missed %d panchayat names; keeping them as-is: %s",
                    len(missing), " / ".join(missing))
        mapping.update({n: n for n in missing})

    changed: dict[str, list[str]] = {}
    for variant, std in mapping.items():
        if variant != std:
            changed.setdefault(std, []).append(variant)
    if changed:
        log.info("Gemini standardized %d panchayat spellings across %d names",
                 sum(len(v) for v in changed.values()), len(counts))
        for std, variants in sorted(changed.items()):
            log.info("  %s <= %s", std, " / ".join(sorted(variants)))
    usage = extract_usage(response)
    if usage["total_tokens"]:
        log.info("Gemini panchayat-standardization usage: %d input + %d output + "
                 "%d thought tokens", usage["input_tokens"], usage["output_tokens"],
                 usage["thought_tokens"])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"names": sorted(counts), "mapping": mapping},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("cached panchayat mapping: %s", cache_path.name)
    return mapping


def _to_int(value: Any) -> int | str:
    if value is None or value == "":
        return ""
    text = str(value).translate(DEV_DIGITS)
    digits = re.sub(r"\D", "", text)
    return int(digits) if digits else ""


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip(" :।")


def normalize_row(raw: dict[str, Any], pdf_path: Path, ac_no: str, total_pages: int) -> dict[str, Any]:
    part_no = int(re.sub(r"\D", "", pdf_path.stem).lstrip("0") or "0")
    row = {c: "" for c in COLUMNS}
    row["booth_pdf"] = pdf_path.name
    row["total_pages"] = total_pages
    row["part_no"] = part_no
    row["ac_no"] = int(re.sub(r"\D", "", str(ac_no)) or 0)

    model_part = _to_int(raw.get("part_no", ""))
    if isinstance(model_part, int) and model_part and model_part != part_no:
        log.warning("%s: printed part number %d != filename part number %d",
                    pdf_path.name, model_part, part_no)

    for field in NUM_FIELDS:
        if field in {"part_no", "ac_no", "counts_ok"}:
            continue
        if field in {"qualifying_date", "publication_date"}:
            row[field] = _clean_text(raw.get(field, ""))
        else:
            row[field] = _to_int(raw.get(field, ""))
    for field in HIN_FIELDS:
        row[field] = _clean_text(raw.get(field, ""))

    m, f, th, total = (row["male"], row["female"], row["third_gender"], row["total"])
    if th == "" and isinstance(m, int) and isinstance(f, int) and isinstance(total, int):
        th = total - m - f
        row["third_gender"] = th
    if all(isinstance(v, int) for v in (m, f, th, total)):
        row["counts_ok"] = "Y" if m + f + th == total else "N"
    else:
        row["counts_ok"] = "?"
    return row


def process_pdf(
    pdf_path: Path,
    ac_no: str,
    backend: GeminiApiBackend | VertexBackend,
    model: str,
    dpi: int,
    jpeg_quality: int,
    max_dim: int,
    timeout: int,
    retries: int,
    input_price_per_m: float,
    output_price_per_m: float,
    cached_price_per_m: float,
    debug: bool,
) -> dict[str, Any]:
    image_b64, total_pages = render_page1_jpeg(pdf_path, dpi, jpeg_quality, max_dim)
    response = api_request(backend.url(model), backend.headers(),
                           backend.payload(model, image_b64),
                           timeout=timeout, retries=retries)
    raw = parse_gemini_response(response)
    row = normalize_row(raw, pdf_path, ac_no, total_pages)
    usage = extract_usage(response)
    cost = estimate_cost_from_usage(usage, input_price_per_m, output_price_per_m,
                                    cached_price_per_m)
    row.update(usage)
    row["approx_cost_usd"] = f"{cost:.6f}"
    if debug:
        log.info("%s raw=%s", pdf_path.name, json.dumps(raw, ensure_ascii=False))
        log.info("%s usage=%s cost=$%.6f", pdf_path.name, json.dumps(usage), cost)
        log.info("%s row=%s", pdf_path.name, json.dumps(row, ensure_ascii=False))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract booth-roll first-page details using Gemini vision")
    ap.add_argument("--ac", help="AC number; reads booth_list_pdf/<AC>/*.pdf")
    ap.add_argument("--pdf-dir", type=Path, help="Directory of booth PDFs (overrides --ac)")
    ap.add_argument("--out", type=Path, help="Output CSV (default booth_list_csv/booth_info_gemini_<AC>.csv)")
    ap.add_argument("--env-file", type=Path, help="Optional extra .env path")
    ap.add_argument("--backend", choices=["auto", "gemini", "vertex"], default="auto",
                    help="auto = Vertex if a service-account key is found, else Gemini API key")
    ap.add_argument("--sa-key", type=Path,
                    help="Vertex service-account JSON key (default: GOOGLE_APPLICATION_CREDENTIALS)")
    ap.add_argument("--location", default="global", help="Vertex location (default global)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--jpeg-quality", type=int, default=82)
    ap.add_argument("--max-dim", type=int, default=0,
                    help="Downscale page image so its longest side is at most this many px (0 = off)")
    ap.add_argument("--limit", type=int, help="Process only first N booths")
    ap.add_argument("--parts", help="Comma-separated booth part numbers to process, e.g. 25,26,37")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--input-price-per-m", type=float, default=DEFAULT_INPUT_PRICE_PER_M,
                    help="Paid-tier input token price per 1M tokens in USD")
    ap.add_argument("--output-price-per-m", type=float, default=DEFAULT_OUTPUT_PRICE_PER_M,
                    help="Paid-tier output token price per 1M tokens in USD")
    ap.add_argument("--cached-price-per-m", type=float, default=DEFAULT_CACHED_PRICE_PER_M,
                    help="Paid-tier cached input token price per 1M tokens in USD")
    ap.add_argument("--no-panchayat-clean", action="store_true",
                    help="Skip Gemini panchayat standardization (panchayat_clean column)")
    ap.add_argument("--refresh-panchayat", action="store_true",
                    help="Ignore the cached Gemini panchayat mapping and re-ask")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not args.ac and not args.pdf_dir:
        ap.error("provide --ac or --pdf-dir")
    backend = make_backend(args)
    pdf_dir = args.pdf_dir or (HERE / "booth_list_pdf" / str(args.ac))
    if not pdf_dir.is_dir():
        raise SystemExit(f"PDF dir not found: {pdf_dir}")
    ac_no = args.ac or pdf_dir.name
    out_csv = args.out or HERE / "booth_list_csv" / f"booth_info_gemini_{ac_no}.csv"
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

    done, rows = set(), []
    if out_csv.exists():
        with out_csv.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                for field in USAGE_FIELDS:
                    r.setdefault(field, "")
                rows.append({c: r.get(c, "") for c in COLUMNS})
                done.add(r.get("booth_pdf", ""))
        log.info("resuming; %d booths already in %s", len(done), out_csv.name)

    tmp_csv = out_csv.with_suffix(out_csv.suffix + ".tmp")

    def flush() -> None:
        rows.sort(key=lambda r: int(re.sub(r"\D", "", str(r["booth_pdf"])) or 0))
        with tmp_csv.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp_csv, out_csv)

    def apply_panchayat_clean() -> None:
        if args.no_panchayat_clean or not rows:
            return
        cache_path = HERE / "booth_list_csv" / f"panchayat_canon_{ac_no}.json"
        pan_map = standardize_panchayats(rows, backend, args.model, cache_path,
                                         args.refresh_panchayat)
        if not pan_map:
            return
        for r in rows:
            r["panchayat_clean"] = pan_map.get(r.get("panchayat", ""), r.get("panchayat", ""))
        flush()

    todo = [p for p in pdfs if p.name not in done]
    if not todo:
        log.info("nothing to do; %s is already complete for selected PDFs", out_csv)
        apply_panchayat_clean()
        log_usage_summary(rows, args.input_price_per_m, args.output_price_per_m)
        return

    def report(n: int, pdf_name: str, row: dict[str, Any]) -> None:
        log.info("[%d/%d] %s | %s (M%s F%s T%s ok=%s)", n, len(todo), pdf_name,
                 row.get("main_town_village", ""), row["male"], row["female"],
                 row["total"], row["counts_ok"])

    def submit(ex: ThreadPoolExecutor, pdf: Path):
        return ex.submit(process_pdf, pdf, ac_no, backend, args.model, args.dpi,
                         args.jpeg_quality, args.max_dim, args.timeout, args.retries,
                         args.input_price_per_m, args.output_price_per_m,
                         args.cached_price_per_m, args.debug)

    failed: list[str] = []
    workers = max(1, min(args.workers, len(todo)))
    log.info("processing %d booths with Gemini model=%s workers=%d", len(todo), args.model, workers)
    if workers == 1:
        for i, pdf in enumerate(todo, 1):
            try:
                row = process_pdf(pdf, ac_no, backend, args.model, args.dpi,
                                  args.jpeg_quality, args.max_dim, args.timeout,
                                  args.retries, args.input_price_per_m,
                                  args.output_price_per_m, args.cached_price_per_m,
                                  args.debug)
                rows.append(row)
                report(i, pdf.name, row)
                flush()
            except Exception as exc:  # noqa: BLE001
                failed.append(pdf.name)
                log.error("%s failed: %s", pdf.name, exc)
    else:
        remaining = iter(todo)
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {}
            for _ in range(workers):
                try:
                    pdf = next(remaining)
                except StopIteration:
                    break
                futs[submit(ex, pdf)] = pdf

            while futs:
                done_futs, _pending = wait(futs, return_when=FIRST_COMPLETED)
                for fut in done_futs:
                    pdf = futs.pop(fut)
                    completed += 1
                    try:
                        row = fut.result()
                        rows.append(row)
                        report(completed, pdf.name, row)
                        flush()
                    except Exception as exc:  # noqa: BLE001
                        failed.append(pdf.name)
                        log.error("%s failed: %s", pdf.name, exc)

                    try:
                        next_pdf = next(remaining)
                    except StopIteration:
                        continue
                    futs[submit(ex, next_pdf)] = next_pdf

    apply_panchayat_clean()

    bad = sum(1 for r in rows if r.get("counts_ok") != "Y")
    log.info("Done. %d booths -> %s  (%d rows need a counts check)", len(rows), out_csv, bad)
    if failed:
        log.warning("%d booths FAILED (rerun to retry): %s", len(failed),
                    ", ".join(sorted(failed)))
    log_usage_summary(rows, args.input_price_per_m, args.output_price_per_m)


if __name__ == "__main__":
    main()
