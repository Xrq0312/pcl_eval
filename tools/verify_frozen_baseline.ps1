$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$output = Join-Path $root 'results\frozen\synth_00.json'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $output) | Out-Null

& (Join-Path $PSScriptRoot 'run_python.ps1') `
    (Join-Path $PSScriptRoot 'test_evaluate.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& (Join-Path $PSScriptRoot 'run_python.ps1') `
    (Join-Path $PSScriptRoot 'evaluate.py') `
    --sample-dir (Join-Path $root 'samples\synth\synth_00') `
    --pred-prefix (Join-Path $root 'samples\synth\synth_00\out') `
    --out $output `
    --dataset frozen-upstream-artifact
exit $LASTEXITCODE
