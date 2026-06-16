# Convenience runner for eval_phone_methods.py.
#
# 1. Pulls every measurement folder off the connected phone into
#    .\phone_measurements\ (the script's expected default).
# 2. Runs the evaluator, which writes phone_method_eval.csv and prints
#    the per-method aggregate to the console.
#
# Usage:  pwsh -File .\eval_phone_methods.ps1
#
# If a ground_truth.csv (header: ts,param,value) sits next to the script
# the evaluator picks it up automatically and adds per-method error
# columns to the aggregate.

param(
    [string]$Dest = 'phone_measurements',
    [string]$Pkg  = 'com.example.poolwatertester'
)

$ErrorActionPreference = 'Stop'

if (-not (Get-Command adb -ErrorAction SilentlyContinue)) {
    Write-Error "adb not on PATH. Install Android platform-tools or add D:\Android\AndroidSDK\platform-tools."
    exit 1
}

Write-Host "Pulling /sdcard/Android/data/$Pkg/files/measurements/ -> $Dest"
New-Item -ItemType Directory -Force $Dest | Out-Null
adb pull "/sdcard/Android/data/$Pkg/files/measurements/." $Dest

Write-Host ""
Write-Host "Running eval_phone_methods.py..."
python .\eval_phone_methods.py --frames $Dest
