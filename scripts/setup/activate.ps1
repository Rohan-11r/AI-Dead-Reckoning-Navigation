# SIH26168 -- activate the project virtual environment (PowerShell)
#
#   . .\scripts\setup\activate.ps1
#
# The leading dot matters: it dot-sources the script so the environment changes persist
# in your shell. Running it without the dot activates a venv in a child process that
# exits immediately, which does nothing.
#
# PYTHONUTF8=1 is not optional. Verified in Phase 1: torch 2.14's dynamo-based ONNX
# exporter prints a U+2705 check mark in its progress log and dies with
# UnicodeEncodeError under the Windows cp1252 console codepage. UTF-8 mode fixes it.
# The IO-VNBD CSVs are separately cp1252-encoded -- that is handled explicitly by the
# loader and is unrelated to this setting.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$activate = Join-Path $repo ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $activate)) {
    Write-Host "No .venv found at $activate" -ForegroundColor Red
    Write-Host "Create it with:" -ForegroundColor Yellow
    Write-Host "    python -m venv .venv"
    Write-Host "    .venv\Scripts\python.exe -m pip install -r requirements.txt"
    Write-Host "    .venv\Scripts\python.exe -m pip install -e ."
    return
}

. $activate
$env:PYTHONUTF8 = "1"

Write-Host "SIH26168 environment active" -ForegroundColor Green
Write-Host ("  python      : " + (& python -c "import sys; print(sys.version.split()[0], '->', sys.prefix)"))
Write-Host  "  PYTHONUTF8  : $env:PYTHONUTF8"
Write-Host  "  validate    : python scripts\phase1_validate.py"
