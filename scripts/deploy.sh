#!/usr/bin/env bash
# deploy.sh — Qwen3-Reranker self-hosted one-command deploy (with automatic "missing scoring head" check = this book's big pitfall, mechanized)
# Usage: ./deploy.sh [--size 0.6B|4B] [--port PORT] [--models-dir DIR] [--max-tokens 4096] [--base-url URL] [--dry-run]
set -euo pipefail
SIZE="0.6B"; PORT=8081; MODELS_DIR="./models"; MAX_TOKENS=4096; BASE_URL=""; DRY_RUN=0

say() { printf '\033[1m[deploy]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[deploy] FAIL:\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
case "$1" in
  --size|--port|--models-dir|--max-tokens|--base-url) [ $# -ge 2 ] || die "missing value for $1";;
esac
case "$1" in
  --size) SIZE="$2"; shift 2;;
  --port) PORT="$2"; shift 2;;
  --models-dir) MODELS_DIR="$2"; shift 2;;
  --max-tokens) MAX_TOKENS="$2"; shift 2;;
  --base-url) BASE_URL="$2"; shift 2;;
  --dry-run) DRY_RUN=1; shift;;
  *) echo "unknown arg: $1"; exit 2;;
esac; done
[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "--max-tokens must be a positive integer"
HF_REPO="Qwen/Qwen3-Reranker-$SIZE"
WORK="$MODELS_DIR/qwen3-reranker-$SIZE"
GGUF="$WORK/model-f16.gguf"
SERVER_COMMAND=(llama-server -m "$GGUF" --reranking --pooling rank
  --batch-size "$MAX_TOKENS" --ubatch-size "$MAX_TOKENS" --ctx-size "$MAX_TOKENS" --port "$PORT")

if [ "$DRY_RUN" = 1 ]; then
  printf '%q ' "${SERVER_COMMAND[@]}"; printf '\n'
  exit 0
fi
[ -n "$BASE_URL" ] || die "--base-url is required for health and scoring requests"
BASE_URL="${BASE_URL%/}"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

for c in llama-server curl find python3; do command -v "$c" >/dev/null || die "missing $c"; done
python3 -c "import gguf" 2>/dev/null || die "missing python package gguf. Install: pip install gguf (a dedicated env is recommended)"
if [ -f "$WORK/server.pid" ] && kill -0 "$(cat "$WORK/server.pid" 2>/dev/null)" 2>/dev/null; then
  SKIP_START=1
else
  SKIP_START=0
  lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 && die "port $PORT is occupied by another process"
fi

# ── 1. Convert official weights yourself (never use community GGUFs — this book's big pitfall) ──
if [ ! -f "$GGUF" ]; then
  say "Downloading official weights $HF_REPO"
  command -v hf >/dev/null || die "missing hf CLI. Install: pip install -U huggingface_hub (a dedicated env is recommended)"
  hf download "$HF_REPO" --local-dir "$WORK/hf"
  say "Converting with a recent convert_hf_to_gguf.py (keeps the scoring head)"
  CONVERT=$(find "$(dirname "$(command -v llama-server)")/.." -name "convert_hf_to_gguf.py" 2>/dev/null | head -1)
  # With multiple llama.cpp installs the above may pick the wrong one; set CONVERT_PY=<path> to override
  CONVERT="${CONVERT_PY:-$CONVERT}"
  [ -n "$CONVERT" ] || die "convert_hf_to_gguf.py not found; run from a llama.cpp source directory"
  python3 "$CONVERT" "$WORK/hf" --outfile "$GGUF" --outtype f16
fi

# ── 2. Scoring-head check (30 seconds that save you from the "masquerading as a weak model" abyss) ──
say "Checking cls.output.weight (missing = bad file with random scores)..."
python3 - "$GGUF" <<'PY'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
cls = [t.name for t in r.tensors if 'cls' in t.name]
assert cls, "FATAL: no cls tensors — scoring head missing; this GGUF will output random scores. Re-convert with a recent converter."
print(f"  scoring head present: {cls}")
PY

# ── 3. Start + real scoring assertion ──
if [ "${SKIP_START:-0}" = 1 ]; then
  say "Instance already running (pid $(cat "$WORK/server.pid")), skipping to assertions (idempotent)"
else
  say "Starting llama-server @ :$PORT"
  nohup "${SERVER_COMMAND[@]}" > "$WORK/server.log" 2>&1 &
  SRV_PID=$!; echo "$SRV_PID" > "$WORK/server.pid"
  ok=0
  for i in $(seq 1 30); do curl -fsS -m 2 "$BASE_URL/health" >/dev/null 2>&1 && { ok=1; break; }; sleep 2; done
  if [ "$ok" != 1 ]; then kill "$SRV_PID" 2>/dev/null || true; rm -f "$WORK/server.pid"; die "not ready after 60s (background process cleaned up), see $WORK/server.log"; fi
fi
say "Effective batch sizes from the server log:"
for key in n_batch n_ubatch; do
  grep -E "(^|[[:space:]])${key}[[:space:]]*=" "$WORK/server.log" || die "server log missing $key"
done
say "Real scoring assertions (short-document separation and long-document acceptance)..."
python3 "$SCRIPT_DIR/check_rerank.py" "$BASE_URL"
say "Deploy complete: $BASE_URL/v1/rerank"
