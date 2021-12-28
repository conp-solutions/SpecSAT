#!/usr/bin/env bash
#
# Copyright (C) 2021, Norbert Manthey <nmanthey@conp-solutions.com>
#
# Running SpecSAT.py with the specified arguments in a
# virtual environment, to keep python dependencies
# isolated from the system Python installation.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
declare -r SCRIPT_DIR
VENV_DIR="$SCRIPT_DIR"/.SpecSATvenv
declare -r VENV_DIR

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

# Create virtual environment once
if [ ! -d "$VENV_DIR" ]; then
    execute_silently python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR"/bin/activate

# Install dependencies (for user)
execute_silently python3 -m pip install --upgrade -U -r requirements.txt

# Actually execute SpecSAT
declare -i STATUS=0
"$SCRIPT_DIR"/SpecSAT.py "$@" || STATUS=$?

# Deactivate venv again
deactivate

# Leave this script, propagate the exit code
exit "$STATUS"
