#!/usr/bin/env bash
# End-to-end voter-roll pipeline for one or more ACs: download PDFs from
# S3, Tesseract OCR (all CPU cores), escalate the flagged residual to
# Gemini (cheap - see escalate_voters_gemini.py), convert to per-booth
# JSON and upload to the destination bucket. Pushes the code (not data -
# booth_list_pdf/, booth_list_csv/, validation/ are all git-ignored) once
# all requested ACs are done.
#
# Usage:
#   ./run_voter_pipeline.sh 216 217 218
#   ./run_voter_pipeline.sh $(seq 216 227)
#
# Env overrides: STATE_CD, ROLL_ID, SRC_BUCKET, DST_BUCKET, OCR_WORKERS,
# ESC_WORKERS, DL_WORKERS, ESC_RULE, ESC_BATCH, KEEP_LOCAL=1 (skip cleanup).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

if [ "$#" -eq 0 ]; then
  echo "usage: $0 AC [AC ...]" >&2
  exit 1
fi
ACS=("$@")

STATE_CD="${STATE_CD:-S13}"
ROLL_ID="${ROLL_ID:-s13-2024-fir}"
SRC_BUCKET="${SRC_BUCKET:-electoral-roll-pdfs}"
DST_BUCKET="${DST_BUCKET:-electoral-voter-details}"
OCR_WORKERS="${OCR_WORKERS:-$(nproc)}"
ESC_WORKERS="${ESC_WORKERS:-24}"
DL_WORKERS="${DL_WORKERS:-32}"
ESC_RULE="${ESC_RULE:-flags:epic_length,flags:epic_bad,flags:age_bad,flags:age_arbitrated}"
ESC_BATCH="${ESC_BATCH:-15}"
KEEP_LOCAL="${KEEP_LOCAL:-0}"

PY=".venv/bin/python"
ENV_FILE="/home/ubuntu/.env"
LOG="/home/ubuntu/voter_pipeline_$(date +%Y%m%d_%H%M%S).log"

# boto3 (download_ac_s3.py, voters_csv_to_json.py) reads AWS creds from the
# environment - without this they fail with NoCredentialsError even though
# ENV_FILE above gets AWS creds to escalate_voters_gemini.py's Gemini client.
if [ -f "$ENV_FILE" ]; then
  set -a; source "$ENV_FILE"; set +a
fi

# Tesseract has no internal use for more than one thread per worker process;
# without this, OpenCV/OMP each spawn up to nproc threads *per worker*,
# oversubscribing the box ~16x and roughly halving real throughput (measured
# this session: load average 60 vs 17 on a 16-core box for the same job).
export OPENCV_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_NUM_THREADS=1

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "=== voter pipeline started | ACs: ${ACS[*]} | ocr_workers=$OCR_WORKERS esc_workers=$ESC_WORKERS ==="

FAILED_ACS=()
for ac in "${ACS[@]}"; do
  log "----- AC $ac: downloading PDFs from s3://$SRC_BUCKET/$STATE_CD/$ROLL_ID/$ac/ -----"
  PDF_DIR="booth_list_pdf/${STATE_CD,,}/${ROLL_ID,,}/$ac"
  if ! $PY download_ac_s3.py --state-cd "$STATE_CD" --roll-id "$ROLL_ID" --ac "$ac" \
        --bucket "$SRC_BUCKET" --out-dir "$PDF_DIR" --workers "$DL_WORKERS" 2>&1 | tee -a "$LOG"; then
    log "AC $ac: download failed, skipping"; FAILED_ACS+=("$ac"); continue
  fi

  TESS_CSV="booth_list_csv/voters_${ac}_tesseract.csv"
  log "----- AC $ac: Tesseract OCR ($OCR_WORKERS workers) -----"
  if ! $PY extract_voters_tesseract.py --pdf-dir "$PDF_DIR" --ac "$ac" \
        --workers "$OCR_WORKERS" --out "$TESS_CSV" 2>&1 | tee -a "$LOG"; then
    log "AC $ac: OCR failed, skipping"; FAILED_ACS+=("$ac"); continue
  fi

  HYBRID_CSV="booth_list_csv/voters_${ac}_hybrid.csv"
  log "----- AC $ac: Gemini escalation (rule: $ESC_RULE) -----"
  if ! $PY escalate_voters_gemini.py --csv "$TESS_CSV" --pdf-dir "$PDF_DIR" \
        --rule "$ESC_RULE" --batch-size "$ESC_BATCH" --env-file "$ENV_FILE" \
        --workers "$ESC_WORKERS" --out "$HYBRID_CSV" 2>&1 | tee -a "$LOG"; then
    log "AC $ac: escalation failed, falling back to Tesseract-only CSV"
    HYBRID_CSV="$TESS_CSV"
  fi

  # A batch occasionally comes back short (see escalate_voters_gemini.py);
  # one smaller-batch retry on just the gaps recovers those for a few cents.
  MISSING=$($PY - "$HYBRID_CSV" <<'PYEOF'
import csv, sys
with open(sys.argv[1], encoding="utf-8-sig") as f:
    print(sum(1 for r in csv.DictReader(f) if "gemini_missing" in r.get("flags", "")))
PYEOF
)
  if [ "${MISSING:-0}" -gt 0 ]; then
    log "AC $ac: retrying $MISSING gemini_missing cards with a smaller batch"
    $PY escalate_voters_gemini.py --csv "$HYBRID_CSV" --pdf-dir "$PDF_DIR" \
        --rule "flags:gemini_missing" --batch-size 10 --env-file "$ENV_FILE" \
        --workers "$ESC_WORKERS" --out "$HYBRID_CSV" 2>&1 | tee -a "$LOG"
  fi

  log "----- AC $ac: JSON -> s3://$DST_BUCKET/$STATE_CD/$ROLL_ID/$ac/ -----"
  if ! $PY voters_csv_to_json.py --csv "$HYBRID_CSV" --state-cd "$STATE_CD" \
        --roll-id "$ROLL_ID" --ac "$ac" --bucket "$DST_BUCKET" \
        --workers "$DL_WORKERS" 2>&1 | tee -a "$LOG"; then
    log "AC $ac: JSON upload failed"; FAILED_ACS+=("$ac"); continue
  fi

  if [ "$KEEP_LOCAL" != "1" ]; then
    log "AC $ac: cleaning up local PDFs/CSVs (data lives in S3 now)"
    rm -rf "$PDF_DIR"
    rm -f "$TESS_CSV" "${TESS_CSV%.csv}.pages.csv" \
          "$HYBRID_CSV" "${HYBRID_CSV%.csv}.pages.csv" "${HYBRID_CSV%.csv}.usage.csv"
  fi
  log "----- AC $ac: done -----"
done

log "=== all ACs processed | failed: ${FAILED_ACS[*]:-none} ==="

log "----- pushing code -----"
# Only this pipeline's own files - never a blanket `git add -A`, which would
# sweep in whatever unrelated WIP happens to be sitting modified in the
# working tree (this bit us once: an earlier version committed unrelated
# in-progress edits to other scripts that this pipeline never touched).
PIPELINE_FILES=(run_voter_pipeline.sh extract_voters_tesseract.py
                 escalate_voters_gemini.py voters_csv_to_json.py
                 download_ac_s3.py .gitignore)
if [ -n "$(git status --porcelain -- "${PIPELINE_FILES[@]}")" ]; then
  git add -- "${PIPELINE_FILES[@]}" 2>&1 | tee -a "$LOG"
  git commit -m "Voter pipeline run: ACs ${ACS[*]} ($(date -Is))" 2>&1 | tee -a "$LOG"
else
  log "no pipeline code changes to commit"
fi
if [ "$(git rev-list --count '@{u}..HEAD' 2>/dev/null || echo 0)" -gt 0 ]; then
  # GIT_TERMINAL_PROMPT=0 + timeout: fail fast instead of hanging forever
  # waiting for interactive auth in a detached screen session (also bit us
  # once - a stuck `git push` sat there with no TTY to answer it).
  if GIT_TERMINAL_PROMPT=0 timeout 30 git push 2>&1 | tee -a "$LOG"; then
    log "push succeeded"
  else
    log "push failed or needs interactive auth - commit is local, push manually"
  fi
else
  log "nothing to push"
fi

log "=== pipeline complete. log: $LOG ==="
if [ "${#FAILED_ACS[@]}" -gt 0 ]; then
  exit 1
fi
