#!/usr/bin/env bash
# SIH26168 -- activate the project virtual environment (bash / Git Bash)
#
#   source scripts/setup/activate.sh
#
# Must be sourced, not executed: a subshell's exported variables die with the subshell.
#
# PYTHONUTF8=1 is not optional. Verified in Phase 1: torch 2.14's dynamo-based ONNX
# exporter prints a U+2705 check mark and dies with UnicodeEncodeError under the Windows
# cp1252 console codepage. UTF-8 mode fixes it. (Unrelated to the IO-VNBD CSVs also
# being cp1252 -- that is handled explicitly by the loader.)

_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ -f "$_repo/.venv/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source "$_repo/.venv/Scripts/activate"      # Windows layout
elif [ -f "$_repo/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$_repo/.venv/bin/activate"          # POSIX layout
else
    echo "No .venv found under $_repo" >&2
    echo "Create it with:" >&2
    echo "    python -m venv .venv" >&2
    echo "    .venv/Scripts/python.exe -m pip install -r requirements.txt" >&2
    echo "    .venv/Scripts/python.exe -m pip install -e ." >&2
    return 1 2>/dev/null || exit 1
fi

export PYTHONUTF8=1

echo "SIH26168 environment active"
echo "  python      : $(python -c 'import sys; print(sys.version.split()[0], "->", sys.prefix)')"
echo "  PYTHONUTF8  : $PYTHONUTF8"
echo "  validate    : python scripts/phase1_validate.py"
