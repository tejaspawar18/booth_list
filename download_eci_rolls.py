"""
Download booth-wise (part-wise) electoral-roll PDFs from the ECI voters portal.

For Uttar Pradesh SIR 2026 (and any state/year the portal exposes).

How it works
------------
The portal at https://voters.eci.gov.in/download-eroll is a React SPA talking to
https://gateway-voters.eci.gov.in.  Reverse-engineering that SPA showed the whole
flow is *public* - no login is required:

  1. Reference data (states / districts / ACs) is served as plain JSON.
  2. The published roll for a state/year is described by
     `printing-publish/get-publish-eroll-type` (FinalRoll, DraftRoll, ...).
     That call needs a small hybrid-encrypted (RSA-OAEP + AES-GCM) request
     envelope, reproduced in `EciCrypto` below.
  3. Each generated booth PDF is stored on a CDN under a *deterministic* path:

        https://voters.eci.gov.in/eroll/<year>/<state>/<rolltype>/<ac>/
            <year>-<pdfGenType>-<STATE>-<ac>-<rollTypeRefId>-Revision<n>-<lang>-<part>-WI.pdf

     The PDFs are pre-generated, so we can download them directly - no captcha.
  4. Booth (part) numbers per AC come from the plain `get-part-list` endpoint.

If a particular CDN file is ever missing (404), we fall back to the
`generate-published-pdfs` endpoint, which *does* need a captcha.  The captcha is
an AES-GCM-encrypted image whose key we recovered from the SPA; it is decrypted
and either solved automatically with Gemini vision (Vertex AI via
GOOGLE_APPLICATION_CREDENTIALS, or a Gemini API key, loaded from survey/.env the
same way the extractor does) or shown to you to type in.

Generation returns one item per requested booth plus a `refId`. When refId is
"CDN" the items are CDN paths to fetch; otherwise (archived rolls such as a past
General Election, which are NOT pre-generated on the CDN and 404 on every booth)
each item is a fileId, and the PDF is fetched base64-encoded from
gateway-vpd.eci.gov.in/api/v1/ext-printing-publish/get-published-file?fileId=...
So both current pre-generated rolls and archived generate-on-demand rolls work.

Caveat for generate-on-demand rolls (measured on Maharashtra 2024 and the 2026
Baramati bye-election roll, none of which are on the CDN): a fileId is single-use
and the gateway allows only ~15s to serve it.  Booths whose PDF can't be produced
in that window return `504 upstream request timeout`, which permanently burns the
fileId ("File Fetch Time Expire" on every later fetch).  Waiting before the first
fetch, smaller batches and the other language all fail to help — the same booths
fail across repeated fresh generates.  Expect a partial harvest on such rolls
(~20-35% of booths in testing); the rest are simply not retrievable this way.

Booth PDFs are written to <out>/<state>/<year>-<rolltype>/<ac>/ so different
states/rolls that share AC numbers don't collide (--flat-out keeps the legacy
<out>/<ac>/ layout).

The RSA public key, captcha key and `misKey` below are constants baked into the
SPA bundle (main.<hash>.js).  If the portal rotates them this script will start
failing on the encrypted calls - re-extract them from the current bundle.

Usage
-----
    PY=../../jup_venv/bin/python   # repo venv (see project memory)

    # All booths of two Gorakhpur ACs, Final Roll:
    $PY download_eci_rolls.py --state "Uttar Pradesh" --district Gorakhpur --ac 327 328

    # Every AC in a district:
    $PY download_eci_rolls.py --state "Uttar Pradesh" --district Gorakhpur --all-acs

    # Draft roll instead of the final roll, only booths 1-50:
    $PY download_eci_rolls.py --state "Uttar Pradesh" --ac 327 --roll draft --parts 1-50

    # Just list what's available (no download):
    $PY download_eci_rolls.py --state "Uttar Pradesh" --district Gorakhpur --list
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import random
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eci")

HERE = Path(__file__).resolve().parent

GATEWAY = "https://gateway-voters.eci.gov.in"
PORTAL = "https://voters.eci.gov.in"
CDN_ROOT = f"{PORTAL}/eroll"
# archived rolls aren't on the CDN; the portal serves them base64-encoded from
# this gateway via get-published-file?fileId=<uuid> (uuid comes from generate)
VPD_GATEWAY = "https://gateway-vpd.eci.gov.in"
# booths to try on the CDN before deciding a roll is generate-on-demand
CDN_PROBE_BOOTHS = 3

# --- constants extracted from the SPA bundle (voters.eci.gov.in/static/js/main.*.js) ---
# RSA public key (SPKI, base64) used to wrap the per-request AES key.
_RSA_SPKI_B64 = (
    "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEArb7++BxL/YN8OIln+6FL9Gnw5DNmQ/VF"
    "ZXss+J+TuQyJc891JbqbijxYQNEin2c2u+CnpXpoGQ/1gUSzDMJeNS3sNSlIUykp2dt7xIm/cmV4"
    "sZ/c769vCxVRosMfRaZJnBAah+m1X26lEhnOo0wpAB9Txr8RIyBe6h7PiQWykeJeh6UacOBBX28k"
    "gkq7+vJhW8HgB38lt32XRocznRYwS9LqR7ZweFmQhTr1+EGrqiEKCOCxMYgHR2SQckb96hZ9kWzf"
    "zeun4bUO5oXKJciLkiS1IgKieADEvYLgu129ZIpn1H+8H+8ikNNVETqEDDMtqcQcQmWppJvcWHaX"
    "As+f8QIDAQAB"
)
# AES-256-GCM key that decrypts the getCaptcha response (`oe.slice(15, 59)` in bundle).
_CAPTCHA_OE = "SFfIO0YsOlOKawZe855n97lc4tcPkj7WWsi38yNWpalLBLZzQdkqHWYbZ0=GhSJk2raUo"
_CAPTCHA_KEY = base64.b64decode(_CAPTCHA_OE[15:59])
# Opaque constant the printing-publish endpoints demand in every payload.
MIS_KEY = "EROLLA32DVI09AJH"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "applicationName": "VSP",
    "Accept": "application/json, text/plain, */*",
    "Origin": PORTAL,
    "Referer": f"{PORTAL}/",
}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_env_files(env_file: Path | None = None) -> None:
    """Populate os.environ from survey/.env, survey/booth_list/.env, and an
    optional extra file — so OPENAI_API_KEY (used to auto-solve captchas) is
    picked up the same way extract_booth_info_gemini.py does."""
    for path in [HERE.parent / ".env", HERE / ".env", env_file]:
        if path:
            load_dotenv(path)


def _resp_json(r: requests.Response, what: str) -> dict:
    """`.json()` with a readable error: the portal answers throttled/blocked
    requests with an empty body or an HTML page, which turns a bare .json()
    into an opaque "Expecting value: line 1 column 1 (char 0)"."""
    try:
        return r.json()
    except ValueError as exc:
        snippet = " ".join(r.text.split())[:200]
        raise RuntimeError(
            f"{what}: non-JSON HTTP {r.status_code} response "
            f"({r.headers.get('content-type', '?')}): {snippet!r}"
        ) from exc


class EciCrypto:
    """Reproduces the SPA's RSA-OAEP(SHA-256) + AES-256-GCM request envelope."""

    def __init__(self) -> None:
        self._pub = serialization.load_der_public_key(base64.b64decode(_RSA_SPKI_B64))

    def _wrap_key(self, aes_key: bytes) -> bytes:
        return self._pub.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None
            ),
        )

    def encrypt_body(self, payload: dict) -> dict:
        """POST-body envelope: {encryptedPayload, encryptedKey, iv} (standard base64)."""
        aes_key, iv = os.urandom(32), os.urandom(12)
        ct = AESGCM(aes_key).encrypt(
            iv, json.dumps(payload, separators=(",", ":")).encode(), None
        )
        return {
            "encryptedPayload": base64.b64encode(ct).decode(),
            "encryptedKey": base64.b64encode(self._wrap_key(aes_key)).decode(),
            "iv": base64.b64encode(iv).decode(),
        }

    def encrypt_params(self, *values: str) -> tuple[str, str, list[str]]:
        """GET-param envelope: returns (accept_yek, accept_rotcev, [enc(v) ...]) as base64url."""
        aes_key, iv = os.urandom(32), os.urandom(12)
        gcm = AESGCM(aes_key)

        def b64u(b: bytes) -> str:
            return base64.urlsafe_b64encode(b).decode().rstrip("=")

        enc = [
            b64u(gcm.encrypt(iv, json.dumps(v, separators=(",", ":")).encode(), None))
            for v in values
        ]
        return b64u(self._wrap_key(aes_key)), b64u(iv), enc


class EciClient:
    def __init__(self, pool_size: int = 8) -> None:
        self.s = requests.Session()
        # size the connection pool for parallel CDN downloads; retry dropped connections
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=2)
        self.s.mount("https://", adapter)
        self.s.headers.update(_HEADERS)
        self.crypto = EciCrypto()

    # ---- reference data (plain JSON) -------------------------------------
    def states(self) -> list[dict]:
        return self.s.get(f"{GATEWAY}/api/v1/common/states/", timeout=30).json()

    def districts(self, state_cd: str) -> list[dict]:
        return self.s.get(f"{GATEWAY}/api/v1/common/districts/{state_cd}", timeout=30).json()

    def acs(self, district_cd: str) -> list[dict]:
        return self.s.get(f"{GATEWAY}/api/v1/common/acs/{district_cd}", timeout=30).json()

    def ac_languages(self, state_cd: str, district_cd: str, ac_no: int) -> dict:
        r = self.s.post(
            f"{GATEWAY}/api/v1/printing-publish/get-ac-languages",
            json={"stateCd": state_cd, "districtCd": district_cd, "acNumber": ac_no},
            timeout=30,
        )
        return (r.json() or {}).get("payload") or {}

    def part_list(self, state_cd: str, district_cd: str, ac_no: int) -> list[dict]:
        """Booth (part) list for an AC - plain, unauthenticated."""
        body = self.crypto.encrypt_body(
            {"stateCd": state_cd, "districtCd": district_cd, "acNumber": ac_no, "misKey": MIS_KEY}
        )
        r = self.s.post(
            f"{GATEWAY}/api/v1/printing-publish/get-publish-part-list", json=body, timeout=40
        )
        return (r.json() or {}).get("payload") or []

    def eroll_types(self, state_cd: str, year: str) -> list[dict]:
        """Published roll variants (FinalRoll / DraftRoll ...) with their CDN metadata."""
        yek, rotcev, (st, yr, mk) = self.crypto.encrypt_params(state_cd, str(year), MIS_KEY)
        r = self.s.get(
            f"{GATEWAY}/api/v1/printing-publish/get-publish-eroll-type",
            params={"stateCd": st, "year": yr, "misKey": mk},
            headers={"accept_yek": yek, "accept_rotcev": rotcev},
            timeout=30,
        )
        return (r.json() or {}).get("payload") or []

    # ---- captcha (only needed for the generate fallback) -----------------
    def get_captcha(self) -> dict:
        r = self.s.get(f"{GATEWAY}/api/v1/captcha-service/getCaptcha/{uuid.uuid4()}", timeout=30)
        raw = base64.b64decode(_resp_json(r, "getCaptcha")["data"])
        obj = json.loads(AESGCM(_CAPTCHA_KEY).decrypt(raw[:12], raw[12:], None))
        return {"id": obj["id"], "image": base64.b64decode(obj["captcha"])}

    def generate_pdfs(self, roll: dict, ac_no: int, district_cd: str, parts: list[int],
                      lang: str, captcha_text: str, captcha_id: str) -> dict:
        """Trigger generation for a batch of parts. The response's `payload` is a
        list aligned with `parts`; `refId` picks the delivery mode: "CDN" means
        each payload item is a CDN-relative path, anything else means each item
        is a fileId to redeem via get_published_file (archived rolls). A null
        item means that part could not be generated."""
        body = self.crypto.encrypt_body({
            "stateCd": roll["stateCd"], "districtCd": district_cd, "acNumber": ac_no,
            "partNumberList": parts, "langCd": lang, "captcha": captcha_text,
            "captchaId": captcha_id, "misKey": MIS_KEY, "publishedRollId": roll["id"],
            "rollTypeRefId": roll["rollTypeRefId"], "pdfGenType": roll["pdfGenType"],
            "revisionNo": roll["revisionNo"],
        })
        r = self.s.post(
            f"{GATEWAY}/api/v1/printing-publish/generate-published-pdfs", json=body, timeout=120
        )
        j = _resp_json(r, "generate-published-pdfs")
        if j.get("status") != "Success":
            raise RuntimeError(f"generate failed: {j.get('message')}")
        return j

    def save_published_file(self, file_id: str, out: Path) -> bool:
        """Fetch a generated PDF the portal serves base64-encoded (archived rolls,
        not on the CDN) and write it out.

        A fileId is single-use and short-lived. The gateway allows roughly 15s to
        serve it; a booth whose PDF can't be produced in that window answers
        `504 upstream request timeout`, and from then on that fileId is dead
        forever ("File Fetch Time Expire"). So there is exactly one attempt here —
        re-polling a burnt fileId can never succeed, and the only real retry is a
        fresh generate call (see the round loop in main)."""
        url = f"{VPD_GATEWAY}/api/v1/ext-printing-publish/get-published-file"
        headers = {"CurrentRole": "citizen", "PLATFORM-TYPE": "web"}
        try:
            r = self.s.get(url, params={"fileId": file_id}, headers=headers, timeout=180)
        except requests.RequestException as exc:
            log.debug("get-published-file network error for %s (%s)", out.name, exc)
            return False
        if r.status_code == 504:
            log.debug("%s: gateway 504 (PDF too slow to serve; fileId burnt)", out.name)
            return False
        payload = ""
        if r.status_code == 200 and r.content:
            try:
                payload = _resp_json(r, "get-published-file").get("payload") or ""
            except RuntimeError:
                payload = ""
        if payload:
            raw = base64.b64decode(payload)
            if raw[:4] == b"%PDF":
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_name(out.name + ".part")
                tmp.write_bytes(raw)
                os.replace(tmp, out)
                log.info("saved %s (%.1f MB, generated)", out.name, len(raw) / 1e6)
                return True
        return False


def generate_with_captcha(cli: "EciClient", roll: dict, ac_no: int, district_cd: str,
                          parts: list[int], lang: str, out_dir: Path,
                          env_file: Path | None, tries: int) -> dict:
    """generate_pdfs, re-solving a fresh captcha when the portal rejects one.
    An auto-solved captcha is occasionally misread, and since a single request
    can cover a whole AC, one bad read must not sink the run."""
    for attempt in range(1, tries + 1):
        cap = cli.get_captcha()
        text = solve_captcha(cap["image"], out_dir, env_file)
        try:
            return cli.generate_pdfs(roll, ac_no, district_cd, parts, lang, text, cap["id"])
        except RuntimeError as exc:
            if "captcha" not in str(exc).lower() and "catpcha" not in str(exc).lower():
                raise
            log.warning("captcha %r rejected (attempt %d/%d); re-solving", text, attempt, tries)
    raise RuntimeError(f"captcha rejected {tries} times in a row")


def cdn_path(roll: dict, ac_no: int, part: int, lang: str) -> str:
    """Deterministic CDN object path for one booth PDF."""
    state = roll["stateCd"]
    return (
        f"{roll['year']}/{state.lower()}/{roll['rollTypeRefId'].lower()}/{ac_no}/"
        f"{roll['year']}-{roll['pdfGenType']}-{state}-{ac_no}-{roll['rollTypeRefId']}-"
        f"Revision{roll['revisionNo']}-{lang}-{part}-WI.pdf"
    )


_CAPTCHA_PROMPT = (
    "This image is a distorted alphanumeric CAPTCHA (letters a-z/A-Z and digits, "
    "usually 5-7 characters, mixed case, ignore the strike-through lines). "
    "Read it and reply with ONLY the characters, no spaces or explanation."
)
# built once per process; None = not yet tried, False = unavailable
_GEMINI_SOLVER: tuple | None | bool = None


def _gemini_solver(env_file: Path | None, model: str):
    """Lazily build a Gemini backend (Vertex via GOOGLE_APPLICATION_CREDENTIALS,
    else a Gemini API key) reusing extract_booth_info_gemini's auth. Returns
    (gx_module, backend) or None if no Google credentials are usable."""
    global _GEMINI_SOLVER
    if _GEMINI_SOLVER is None:
        try:
            import extract_booth_info_gemini as gx
            ns = argparse.Namespace(env_file=env_file, backend="auto", sa_key=None,
                                    location="global", model=model)
            _GEMINI_SOLVER = (gx, gx.make_backend(ns))
        except SystemExit as exc:
            log.warning("Gemini captcha solver unavailable (%s); manual entry only", exc)
            _GEMINI_SOLVER = False
    return _GEMINI_SOLVER or None


def _gemini_read_captcha(gx, backend, model: str, image: bytes) -> str:
    b64 = base64.b64encode(image).decode()
    mime = "image/png" if image[:8].startswith(b"\x89PNG") else "image/jpeg"
    if backend.name == "vertex":
        payload = {
            "contents": [{"role": "user", "parts": [
                {"text": _CAPTCHA_PROMPT},
                {"inlineData": {"mimeType": mime, "data": b64}}]}],
            "generationConfig": {"thinkingConfig": {"thinkingLevel": "MINIMAL"}},
        }
    else:
        payload = {
            "model": model,
            "input": [{"type": "text", "text": _CAPTCHA_PROMPT},
                      {"type": "image", "data": b64, "mime_type": mime}],
            "generation_config": {"thinking_level": "minimal"},
        }
    resp = gx.api_request(backend.url(model), backend.headers(), payload, timeout=40, retries=2)
    texts = gx._walk_strings(resp)
    # the model may wrap the answer in words; pick the alphanumeric token nearest
    # the expected captcha length rather than concatenating everything
    tokens = re.findall(r"[A-Za-z0-9]+", " ".join(texts))
    if not tokens:
        return ""
    return min(tokens, key=lambda t: abs(len(t) - 6))


def solve_captcha(image: bytes, out_dir: Path, env_file: Path | None = None,
                  model: str = "gemini-3-flash-preview") -> str:
    """Auto-solve with Gemini (Vertex AI / GOOGLE_APPLICATION_CREDENTIALS) if
    credentials are available, else save the image and ask interactively."""
    solver = _gemini_solver(env_file, model)
    if solver:
        gx, backend = solver
        try:
            text = _gemini_read_captcha(gx, backend, model, image)
            text = "".join(c for c in text if c.isalnum())
            if text:
                log.info("captcha auto-solved as %r", text)
                return text
        except Exception as exc:  # noqa: BLE001
            log.warning("Gemini captcha solve failed (%s); falling back to manual", exc)
    path = out_dir / "_captcha.jpg"
    path.write_bytes(image)
    log.info("Saved captcha image to %s", path)
    return input(f"Open {path} and type the captcha text: ").strip()


# The PDFs sit behind Akamai. Aggressive parallel access trips an edge
# rate-limit that answers with a *cached* 403 "Access Denied" (edgesuite.net,
# ~10-min TTL) rather than a 404 — so 403 here means "throttled, back off", not
# "forbidden". Retry it (and 429/5xx) with a longer backoff than transient 5xx.
_RETRY_STATUS = {403, 408, 429, 500, 502, 503, 504}


def download(session: requests.Session, path: str, out: Path, attempts: int = 4) -> bool:
    if out.exists() and out.stat().st_size > 10_000:
        log.info("skip existing %s", out.name)
        return True
    url = f"{CDN_ROOT}/{path}"
    for attempt in range(attempts):
        try:
            r = session.get(url, timeout=180)
        except requests.RequestException as exc:
            if attempt < attempts - 1:
                time.sleep(2 ** attempt + random.uniform(0, 1))
                continue
            log.warning("CDN error for %s (%s)", path, exc)
            return False
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".part")
            tmp.write_bytes(r.content)
            os.replace(tmp, out)
            log.info("saved %s (%.1f MB)", out.name, len(r.content) / 1e6)
            return True
        if r.status_code in _RETRY_STATUS and attempt < attempts - 1:
            # 403 is edge throttling; back off harder (and jitter) so parallel
            # workers don't re-trip the limit in lockstep
            base = 6 if r.status_code == 403 else 2
            time.sleep(min(45.0, base * 2 ** attempt) + random.uniform(0, 2))
            continue
        note = "throttled (edge 403)" if r.status_code == 403 else f"HTTP {r.status_code}"
        log.warning("CDN miss for %s (%s)", path, note)
        return False
    return False


def download_batch(session: requests.Session, jobs: dict[int, tuple[str, Path]],
                   workers: int) -> set[int]:
    """Download {part: (cdn_path, out_file)} in parallel; return failed part numbers."""
    failed: set[int] = set()
    if not jobs:
        return failed
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as ex:
        futs = {ex.submit(download, session, path, out): part
                for part, (path, out) in jobs.items()}
        for fut in as_completed(futs):
            part = futs[fut]
            try:
                if not fut.result():
                    failed.add(part)
            except Exception as exc:  # noqa: BLE001
                log.error("part %s download failed: %s", part, exc)
                failed.add(part)
    return failed


def _chunks(seq: list[int], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _match(items: list[dict], value: str, *keys: str):
    v = value.strip().lower()
    for it in items:
        for k in keys:
            if str(it.get(k, "")).strip().lower() == v:
                return it
    for it in items:  # loose contains-match fallback
        for k in keys:
            if v in str(it.get(k, "")).strip().lower():
                return it
    return None


def parse_parts(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif chunk:
            out.add(int(chunk))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Download booth-wise ECI electoral-roll PDFs")
    ap.add_argument("--state", default="Uttar Pradesh", help="State name or code (e.g. S24)")
    ap.add_argument("--district", help="District name or code (e.g. Gorakhpur / S2450)")
    ap.add_argument("--ac", nargs="+", type=int, help="AC number(s) to download")
    ap.add_argument("--all-acs", action="store_true", help="All ACs in --district")
    ap.add_argument("--year", default="2026")
    ap.add_argument("--roll", default="final", choices=["final", "draft"],
                    help="Which published roll (default: final)")
    ap.add_argument("--roll-id", help="Exact published-roll id from --list (e.g. "
                    "S13-2024-GEN-FIR). A state/year can publish several rolls of "
                    "the same type — Maharashtra 2024 has both a General Election "
                    "roll and an SSR final roll — and --roll alone picks the first.")
    ap.add_argument("--no-cdn", action="store_true",
                    help="Skip the CDN probe and generate every booth. Archived rolls "
                         "(e.g. any Maharashtra 2024 roll) are not pre-generated, so "
                         "the probe only collects 404s.")
    ap.add_argument("--lang", help="Language code (default: AC's first, usually HIN)")
    ap.add_argument("--parts", help="Booth subset, e.g. '1-50' or '1,5,9' (default: all)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "booth_list_pdf")
    ap.add_argument("--list", action="store_true", help="List rolls/ACs/booths, don't download")
    ap.add_argument("--no-generate", action="store_true",
                    help="Don't fall back to captcha-based generation on CDN misses")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel CDN downloads (default 4; the CDN edge throttles "
                         "aggressive parallelism with 403s, so raise with care)")
    ap.add_argument("--generate-batch", type=int, default=25,
                    help="Booths per captcha in the generation fallback (default 25). "
                         "The portal generates these server-side on demand; asking for "
                         "a whole AC at once overloads it and nothing becomes ready, "
                         "so keep this modest (0 = all booths in one request).")
    ap.add_argument("--captcha-tries", type=int, default=6,
                    help="Fresh captchas to try when the portal rejects a solve (default 6)")
    ap.add_argument("--generate-passes", type=int, default=3,
                    help="Rounds to try a batch of booths (default 3). On the fileId "
                         "path each extra round costs a fresh captcha+generate, "
                         "because a fileId that timed out is unusable.")
    ap.add_argument("--generate-wait", type=int, default=30,
                    help="Seconds between those rounds (default 30)")
    ap.add_argument("--env-file", type=Path,
                    help="Extra .env to load (for OPENAI_API_KEY captcha auto-solve)")
    ap.add_argument("--flat-out", action="store_true",
                    help="Legacy layout: write to <out>/<ac>/ instead of the "
                         "state/roll-namespaced <out>/<state>/<roll>/<ac>/")
    args = ap.parse_args()

    load_env_files(args.env_file)
    cli = EciClient(pool_size=max(1, args.workers))

    # ---- resolve state --------------------------------------------------
    state = _match(cli.states(), args.state, "stateName", "stateCd")
    if not state:
        sys.exit(f"State not found: {args.state!r}")
    state_cd = state["stateCd"]
    log.info("State: %s (%s)", state["stateName"], state_cd)

    # ---- resolve roll ---------------------------------------------------
    rolls = cli.eroll_types(state_cd, args.year)
    if not rolls:
        sys.exit(f"No published rolls for {state_cd} {args.year}")
    want = "FinalRoll" if args.roll == "final" else "DraftRoll"
    if args.roll_id:
        roll = _match(rolls, args.roll_id, "id")
        if not roll:
            sys.exit(f"Roll id not found: {args.roll_id!r} "
                     f"(available: {', '.join(r['id'] for r in rolls)})")
    else:
        roll = _match(rolls, want, "rollType") or rolls[0]
    log.info("Roll: %s (rev %s, published %s)",
             roll.get("displayName"), roll.get("revisionNo"), roll.get("publishDate"))

    # Booth PDFs are keyed by (state, roll, ac): different states/rolls share AC
    # numbers, so namespace the output to keep e.g. UP-228 and Maharashtra-228
    # apart. --flat-out keeps the legacy <out>/<ac>/ layout.
    roll_slug = f"{roll.get('year', args.year)}-{roll.get('rollTypeRefId') or want}".lower()
    # A state/year can publish two rolls of the same type (Maharashtra 2024 has a
    # General Election final roll and an SSR final roll), which would collide in
    # the same directory. Only then fall back to the unambiguous roll id, so
    # existing single-roll layouts (UP 2026) keep their paths.
    if sum(1 for r in rolls if r.get("rollTypeRefId") == roll.get("rollTypeRefId")
           and r.get("year") == roll.get("year")) > 1:
        roll_slug = str(roll["id"]).lower()
    base_out = args.out if args.flat_out else args.out / state_cd.lower() / roll_slug
    if not args.flat_out:
        log.info("output namespace: %s", base_out)

    if args.list and not (args.ac or args.all_acs):
        print(json.dumps(rolls, indent=2, ensure_ascii=False))
        return

    # ---- resolve district & ACs -----------------------------------------
    district = None
    if args.district:
        district = _match(cli.districts(state_cd), args.district, "districtValue", "districtCd", "districtNo")
        if not district:
            sys.exit(f"District not found: {args.district!r}")
        log.info("District: %s (%s)", district["districtValue"], district["districtCd"])

    ac_numbers: list[int] = list(args.ac or [])
    ac_district: dict[int, str] = {}
    if args.all_acs:
        if not district:
            sys.exit("--all-acs requires --district")
        for a in cli.acs(district["districtCd"]):
            ac_numbers.append(int(a["asmblyNo"]))
            ac_district[int(a["asmblyNo"])] = a["districtCd"]
    if not ac_numbers:
        sys.exit("Nothing to do: pass --ac N... or --all-acs (with --district).")

    # We need a districtCd per AC (for part-list / generate). Build a lookup.
    if district:
        for a in ac_numbers:
            ac_district.setdefault(a, district["districtCd"])
    else:
        # discover each AC's district by scanning the state's districts.
        for d in cli.districts(state_cd):
            for a in cli.acs(d["districtCd"]):
                n = int(a["asmblyNo"])
                if n in ac_numbers:
                    ac_district.setdefault(n, a["districtCd"])

    part_filter = parse_parts(args.parts)
    ok = miss = 0
    for ac in sorted(set(ac_numbers)):
        dcd = ac_district.get(ac)
        if not dcd:
            log.warning("AC %s: could not resolve district; skipping", ac)
            continue
        lang = args.lang or next(iter(cli.ac_languages(state_cd, dcd, ac) or {"HIN": ""}), "HIN")
        parts = [int(p["partNumber"]) for p in cli.part_list(state_cd, dcd, ac)]
        parts = sorted(set(parts) & part_filter) if part_filter else sorted(set(parts))
        log.info("AC %s: %d booths (lang %s)", ac, len(parts), lang)
        if args.list:
            continue

        ac_dir = base_out / str(ac)
        jobs = {part: (cdn_path(roll, ac, part, lang), ac_dir / f"booth_{part:04d}.pdf")
                for part in parts}
        t0 = time.time()
        # Archived rolls are not on the CDN at all, so sweeping every booth just
        # collects hundreds of 404s before the generate flow does the real work.
        # Probe a few booths first (a hit is a real download, so nothing is
        # wasted) and skip straight to generation when none are hosted.
        if args.no_cdn:
            missed = {p for p in jobs if not (jobs[p][1].exists()
                                              and jobs[p][1].stat().st_size > 10_000)}
            ok += len(jobs) - len(missed)
            log.info("AC %s: --no-cdn, generating %d booth(s)", ac, len(missed))
        else:
            probe = sorted(jobs)[:CDN_PROBE_BOOTHS]
            probe_missed = download_batch(cli.s, {p: jobs[p] for p in probe}, args.workers)
            ok += len(probe) - len(probe_missed)
            if len(probe_missed) == len(probe) and len(parts) > len(probe):
                log.info("AC %s: none of the first %d booths are on the CDN — this roll "
                         "is generated on demand; going straight to the captcha flow "
                         "for all %d booths", ac, len(probe), len(parts))
                missed = set(jobs) - (set(probe) - probe_missed)
            else:
                rest = {p: jobs[p] for p in jobs if p not in probe}
                missed = probe_missed | download_batch(cli.s, rest, args.workers)
                ok += len(rest) - len(missed - probe_missed)
                if missed:
                    log.info("AC %s: %d/%d booths from CDN in %.0fs; %d misses",
                             ac, len(jobs) - len(missed), len(jobs), time.time() - t0, len(missed))
                    # Many misses on a CDN-hosted roll usually means the edge
                    # rate-limit tripped (cached 403 ~10 min): re-run (saved files
                    # are skipped) and/or lower --workers.
                    if len(missed) > len(jobs) * 0.5:
                        log.warning("AC %s: %d/%d missed — likely CDN edge throttling; "
                                    "re-run or use fewer --workers.", ac, len(missed), len(jobs))
                else:
                    log.info("AC %s: all %d booths done in %.0fs", ac, len(jobs), time.time() - t0)
                    continue
        if not missed:
            log.info("AC %s: all %d booths done in %.0fs", ac, len(jobs), time.time() - t0)
            continue
        if args.no_generate:
            miss += len(missed)
            continue

        # Fallback: generate on demand (one captcha per batch). The response's
        # payload is aligned with the requested parts; refId=="CDN" means each
        # item is a CDN path to fetch, otherwise each item is a fileId the portal
        # serves base64-encoded (archived rolls that never hit the CDN).
        batch = args.generate_batch or len(missed)  # 0 => one captcha for the lot
        for chunk in _chunks(sorted(missed), max(1, batch)):
            try:
                gen = generate_with_captcha(cli, roll, ac, dcd, chunk, lang, ac_dir,
                                            args.env_file, args.captcha_tries)
            except Exception as exc:  # noqa: BLE001
                log.error("AC %s booths %s: generation fallback failed: %s",
                          ac, ",".join(map(str, chunk)), exc)
                miss += len(chunk)
                continue
            payload = gen.get("payload") or []
            via_cdn = gen.get("refId") == "CDN"
            if not payload or all(item is None for item in payload):
                log.error("AC %s booths %s: portal generated nothing (roll not "
                          "available for these booths)", ac, ",".join(map(str, chunk)))
                miss += len(chunk)
                continue
            pending = []
            for part, item in zip(chunk, payload):
                if item:
                    pending.append((part, item))
                else:
                    log.warning("AC %s booth %s: no file generated", ac, part)
                    miss += 1
            # Generation is asynchronous: a booth that isn't ready on the first
            # pass usually is a little later, so sweep the batch repeatedly
            # instead of giving up on the stragglers.
            for attempt in range(args.generate_passes):
                # One try per booth per round, in parallel. For fileIds that is
                # all we get (single-use, see save_published_file); for CDN paths
                # a round is a cheap re-check.
                def _fetch(job: tuple[int, str]) -> tuple[int, str, bool]:
                    part, item = job
                    out_file = jobs[part][1]
                    got = (download(cli.s, item, out_file, attempts=1) if via_cdn
                           else cli.save_published_file(item, out_file))
                    return part, item, got

                still: list[tuple[int, str]] = []
                with ThreadPoolExecutor(max_workers=max(1, min(args.workers, len(pending)))) as ex:
                    for part, item, got in ex.map(_fetch, pending):
                        if got:
                            ok += 1
                        else:
                            still.append((part, item))
                pending = still
                if not pending or attempt == args.generate_passes - 1:
                    break
                if via_cdn:
                    log.info("AC %s: %d booth(s) still generating; re-checking in %ds",
                             ac, len(pending), args.generate_wait)
                    time.sleep(args.generate_wait)
                    continue
                # fileId path: the ids we hold are spent, so the only way to try
                # again is to ask the portal to generate afresh (new captcha).
                log.info("AC %s: %d booth(s) timed out at the gateway; re-generating "
                         "fresh fileIds (round %d/%d)",
                         ac, len(pending), attempt + 2, args.generate_passes)
                time.sleep(args.generate_wait)
                retry_parts = [p for p, _ in pending]
                try:
                    regen = generate_with_captcha(cli, roll, ac, dcd, retry_parts, lang,
                                                  ac_dir, args.env_file, args.captcha_tries)
                except Exception as exc:  # noqa: BLE001
                    log.error("AC %s: re-generation failed: %s", ac, exc)
                    break
                new_payload = regen.get("payload") or []
                via_cdn = regen.get("refId") == "CDN"
                pending = [(p, it) for p, it in zip(retry_parts, new_payload) if it]
                if not pending:
                    break
            for part, _item in pending:
                log.warning("AC %s booth %s: gateway could not serve this booth's PDF "
                            "within its ~15s window after %d round(s)", ac, part,
                            args.generate_passes)
            miss += len(pending)
            # a batch shorter than its parts leaves the tail ungenerated
            miss += max(0, len(chunk) - len(payload))

    log.info("Done. %d downloaded, %d missing. Output under %s", ok, miss, base_out)


if __name__ == "__main__":
    main()
