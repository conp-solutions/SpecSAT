#!/usr/bin/env bash
#
# Copyright (C) 2021, Norbert Manthey <nmanthey@conp-solutions.com>
#
# Running SpecSAT.py with the specified arguments in a
# virtual environment, to keep python dependencies
# isolated from the system Python installation.

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" &>/dev/null && pwd)"
declare -r SCRIPT_DIR
FORCE_INSTALL="false"

# Install error handler, which will act like 'set -e', but print a message
error_handler() {
    local _PROG="$0"
    local LINE="$1"
    local ERR="$2"
    if [ "$ERR" != 0 ]; then
        echo "$_PROG: error_handler() invoked, line $LINE, exit status $ERR" 1>&2
    fi
    exit "$ERR"
}
trap 'error_handler ${LINENO} $?' ERR

# Use a log file to print output to in case of failure
trap '[ -f "$TMP_LOG_FILE" ] && rm -f "$TMP_LOG_FILE"' EXIT
TMP_LOG_FILE=$(mktemp SpecSAT-run-XXXXXX.log)

execute_silently() {
    local -i RET=0
    "$@" &>"$TMP_LOG_FILE" || RET=$?
    if [ "$RET" -ne 0 ]; then
        echo "Error: failed to run command '$*', returned with status $RET"
        cat "$TMP_LOG_FILE"
        exit $RET
    fi
}

# Hidden command line options
if [ "${1:-}" = "--trace" ]; then
    set -x
    shift
fi

RUN_DIR="$PWD"
# Use bash script dir as run-dir?
if [ "${1:-}" = "--sh-run-from-install" ]; then
    RUN_DIR="$SCRIPT_DIR"
    shift
fi

# Set specific run-dir?
if [ "${1:-}" = "--sh-run-dir" ] && [ -n "${2:-}" ]; then
    RUN_DIR="$2"
    shift 2
fi

# Re-install dependencies in virtual env?
if [ "${1:-}" = "--sh-force-install" ]; then
    FORCE_INSTALL="true"
    shift
fi

VENV_DIR="$RUN_DIR"/.SpecSATvenv
declare -r VENV_DIR

# Create virtual environment once
if [ ! -d "$VENV_DIR" ]; then
    execute_silently python3 -m venv "$VENV_DIR"
    # Activate virtual environment
    source "$VENV_DIR"/bin/activate

    # Install dependencies (for user)
    execute_silently python3 -m pip install --upgrade -U -r "$SCRIPT_DIR"/requirements.txt
else
    # Activate virtual environment
    source "$VENV_DIR"/bin/activate
fi

if [ "$FORCE_INSTALL" = "true" ]; then
    # Install dependencies (for user)
    execute_silently python3 -m pip install --upgrade -U -r "$SCRIPT_DIR"/requirements.txt
fi

# Actually execute SpecSAT
declare -i STATUS=0
"$SCRIPT_DIR"/SpecSAT.py "$@" || STATUS=$?

# Deactivate venv again
deactivate

# Leave this script, propagate the exit code
exit "$STATUS"
