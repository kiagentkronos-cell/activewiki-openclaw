#!/usr/bin/env bash
set -u -o pipefail

# ── Configuration ────────────────────────────────────────────────────────────
# Locate activewiki.json: env var > script-relative > cwd
if [ -z "${ACTIVEWIKI_CONFIG:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    if [ -f "$SCRIPT_DIR/../activewiki.json" ]; then
        ACTIVEWIKI_CONFIG="$SCRIPT_DIR/../activewiki.json"
    elif [ -f "$SCRIPT_DIR/activewiki.json" ]; then
        ACTIVEWIKI_CONFIG="$SCRIPT_DIR/activewiki.json"
    elif [ -f "./activewiki.json" ]; then
        ACTIVEWIKI_CONFIG="./activewiki.json"
    else
        echo "[ERROR] activewiki.json not found. Set ACTIVEWIKI_CONFIG or place it next to scripts/." >&2
        exit 1
    fi
fi
export ACTIVEWIKI_CONFIG

# Parse config values safely using Python (avoids shell injection from config paths)
get_cfg() {
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    cfg = json.load(f)
keys = sys.argv[2].split('.')
val = cfg
for k in keys:
    val = val.get(k, '') if isinstance(val, dict) else ''
print(val if val else '')
" "$ACTIVEWIKI_CONFIG" "$1" 2>/dev/null
}

WIKIS_ROOT="$(get_cfg wikis_root)"
if [ -z "$WIKIS_ROOT" ]; then
    echo "[ERROR] wikis_root not set in activewiki.json" >&2
    exit 1
fi

DEADLINE="$(get_cfg ingest.deadline)"
TIMEZONE="$(get_cfg ingest.timezone)"
DOCLING_VENV="$(get_cfg ocr.venv_path)"

export WIKIS_DEADLINE="${DEADLINE:-03:00}"
export WIKIS_TIMEZONE="${TIMEZONE:-Europe/Berlin}"

cd "$WIKIS_ROOT"

# Determine Python binary (prefer docling venv if configured)
if [ -n "$DOCLING_VENV" ] && [ -f "$DOCLING_VENV/bin/python3" ]; then
    PYTHON="$DOCLING_VENV/bin/python3"
else
    PYTHON="$(command -v python3)"
fi

mkdir -p logs
LOG="logs/ingest-$(TZ="$WIKIS_TIMEZONE" date +%Y-%m-%d).log"
stamp() { TZ="$WIKIS_TIMEZONE" date +"%Y-%m-%dT%H:%M:%S %Z"; }

# Hard deadline in local time (HH:MM). Distill stops here,
# leaving buffer for vectordb + graph build.
past_deadline() {
    [ "$(date +%H:%M)" \> "$WIKIS_DEADLINE" ]
}

exec 9>logs/.ingest.lock
if ! flock -n 9; then
    echo "[$(stamp)] another run holds the lock, skipping" >>"$LOG"
    exit 0
fi

echo "[$(stamp)] run start (deadline=$WIKIS_DEADLINE local)" >>"$LOG"
ok=0
fail=0
shopt -s nullglob
stopped_at=""

# Read scopes from config
SCOPES_LIST="$(get_cfg scopes.enabled)"
# Convert JSON array to space-separated list
SCOPES_ARR=()
while IFS= read -r s; do
    [ -n "$s" ] && SCOPES_ARR+=("$s")
done < <(python3 -c "
import json
with open('$ACTIVEWIKI_CONFIG') as f:
    cfg = json.load(f)
for s in cfg.get('scopes', {}).get('enabled', ['private','family','public']):
    print(s)
" 2>/dev/null)

for scope in "${SCOPES_ARR[@]}"; do
    scope_dir="inbox/$scope"
    [ -d "$scope_dir" ] || continue
    # Recursive, skip dot-paths (.DS_Store, .git/, etc).
    while IFS= read -r -d '' f; do
        if past_deadline; then
            stopped_at="ingest"
            break
        fi
        rel="${f#"$scope_dir/"}"
        echo "[$(stamp)] ingest: $scope/$rel" >>"$LOG"
        if "$PYTHON" scripts/ingest.py "$f" >>"$LOG" 2>&1; then
            ok=$((ok + 1))
        else
            echo "[$(stamp)] ingest failed for $f (exit $?)" >>"$LOG"
            fail=$((fail + 1))
        fi
    done < <(find "$scope_dir" -type f -not -path '*/.*' -print0 | sort -z)
    [ -n "$stopped_at" ] && break
done
if [ -n "$stopped_at" ]; then
    remaining=$(find inbox/private inbox/family inbox/public -type f 2>/dev/null | wc -l)
    echo "[$(stamp)] ingest stopped at deadline, ok=$ok fail=$fail remaining_in_inbox=$remaining" >>"$LOG"
else
    echo "[$(stamp)] ingest end, ok=$ok fail=$fail" >>"$LOG"
fi

# Phase 1+2: Distill unprocessed sources into hierarchical wiki pages,
# then roll up the folder hierarchy (bottom-up synthesis).
if past_deadline; then
    echo "[$(stamp)] past deadline $WIKIS_DEADLINE, skipping distill+rollup" >>"$LOG"
else
    echo "[$(stamp)] distill+rollup start" >>"$LOG"
    if PYTHONUNBUFFERED=1 "$PYTHON" scripts/distill.py --all >>"$LOG" 2>&1; then
        echo "[$(stamp)] distill+rollup end ok" >>"$LOG"
    else
        echo "[$(stamp)] distill+rollup failed or truncated (exit $?)" >>"$LOG"
    fi
fi

# Auto-Commit ausstehender Build-Änderungen (hält das Repo sauber für QC-Commits)
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if [ -n "$(git status --porcelain)" ]; then
        echo "[$(stamp)] auto-commit: uncommitted build changes" >>"$LOG"
        if git add -A >>"$LOG" 2>&1 && git commit -m "build: auto-commit uncommitted build changes" >>"$LOG" 2>&1; then
            echo "[$(stamp)] auto-commit done" >>"$LOG"
        else
            echo "[$(stamp)] auto-commit FAILED (exit $?)" >>"$LOG"
        fi
    fi
fi

# QC-Phase (Wiki-Qualitäts-Check): nur wenn Inbox leer + Zeitfenster
QC_DEADLINE=02:30
# QC-Gate: Scopes aus der Config (wie Ingest-Phase); Fallback auf Default-Scopes
# bei leerem SCOPES_ARR (Config-Lesefehler), damit das Gate nicht still "leer" meldet.
QC_SCOPES=("${SCOPES_ARR[@]}")
if [ ${#QC_SCOPES[@]} -eq 0 ]; then
    QC_SCOPES=(private family public)
fi
if [ "$(find inbox/"${QC_SCOPES[@]}" -type f 2>/dev/null | wc -l)" -ne 0 ]; then
    echo "[$(stamp)] quality-check skipped: inbox not empty" >>"$LOG"
elif [ "$(date +%H:%M)" \> "$QC_DEADLINE" ]; then
    echo "[$(stamp)] quality-check skipped: time window" >>"$LOG"
else
    echo "[$(stamp)] quality-check start (oldest 2)" >>"$LOG"
    "$PYTHON" scripts/check.py --oldest 2 >>"$LOG" 2>&1
    echo "[$(stamp)] quality-check end (exit $?)" >>"$LOG"
fi

# vectordb is now incremental (content-hash based) — always run it,
# even past deadline.
if past_deadline; then
    echo "[$(stamp)] past deadline $WIKIS_DEADLINE, running vectordb anyway (incremental)" >>"$LOG"
fi

echo "[$(stamp)] vectordb start (build + graph + communities)" >>"$LOG"

GRAPH_FLAG="--graph-incremental"

if "$PYTHON" scripts/vectordb.py build $GRAPH_FLAG --communities --communities-incremental >>"$LOG" 2>&1; then
    echo "[$(stamp)] vectordb end ok" >>"$LOG"
else
    echo "[$(stamp)] vectordb failed (exit $?)" >>"$LOG"
fi

echo "[$(stamp)] run end" >>"$LOG"
