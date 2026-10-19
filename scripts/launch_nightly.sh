#!/usr/bin/env bash
# Nightly pipeline launcher — starts run_inbox.sh in a systemd USER unit.
#
# Why this exists: when run_inbox.sh is started with plain `nohup setsid`
# from an agent/exec session, the whole process tree can be reaped when the
# launching session's turn ends — the nightly run then dies mid-phase
# (observed with OpenClaw 2026.10.3; all runs of a week died in the QC
# phase). Fix: run the pipeline as its own systemd user oneshot unit, so it
# no longer hangs off the launching session's lifecycle.
#
# Usage: bash scripts/launch_nightly.sh   (typically from a cron/exec session)
#
# Repo root detection: script-relative (scripts/..), overridable via
# ACTIVEWIKI_ROOT. No hardcoded install paths.
#
# No --replace: a still-running unit for the same day should fail visibly;
# concurrent runs are guarded by the flock inside run_inbox.sh.
# Note: the unit name MUST be passed via --unit= — a bare positional
# argument is interpreted by systemd-run as the executable to run
# (live-verified: "Failed to find executable ...").
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${ACTIVEWIKI_ROOT:-$(dirname "$SCRIPT_DIR")}"
if [ ! -f "$REPO/scripts/run_inbox.sh" ]; then
    echo "ERROR: $REPO/scripts/run_inbox.sh not found — set ACTIVEWIKI_ROOT." >&2
    exit 1
fi

# In exec/cron environments neither variable is set, but the systemd user
# bus exists for the invoking user's uid.
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR}/bus"

# Per-day unit, e.g. activewiki-nightly-2026-10-19 (dashes are legal in
# systemd unit names). Timezone for the date stamp: ACTIVEWIKI_TZ if set,
# else the system local time (run_inbox.sh uses ingest.timezone from config).
if [ -n "${ACTIVEWIKI_TZ:-}" ]; then
    UNIT="activewiki-nightly-$(TZ="$ACTIVEWIKI_TZ" date +%F)"
else
    UNIT="activewiki-nightly-$(date +%F)"
fi

mkdir -p "$REPO/logs"

if command -v systemd-run >/dev/null 2>&1 \
   && systemd-run --user --service-type=oneshot --collect --unit="$UNIT" \
        /bin/bash -c "cd '$REPO' && exec bash scripts/run_inbox.sh >>logs/launcher.log 2>&1"; then
    echo "STARTED: $UNIT (systemd-run --user, oneshot)"
    exit 0
fi

# Fallback: legacy path if systemd-run is missing or the user bus is not
# reachable. Not ideal (the run may get reaped when the launching session
# ends), but better than no run at all.
echo "WARN: systemd-run unavailable, using legacy nohup setsid" >&2
if (cd "$REPO" && nohup setsid bash scripts/run_inbox.sh >>logs/launcher.log 2>&1 &); then
    # No $! here: the job runs in the subshell, the parent shell does not
    # know it (set -u would abort on "unbound variable").
    echo "STARTED: legacy nohup setsid"
    exit 0
else
    echo "ERROR: neither systemd-run nor legacy nohup start worked" >&2
    exit 1
fi
