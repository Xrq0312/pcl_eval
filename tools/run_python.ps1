param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $PythonArgs
)

$ErrorActionPreference = 'Stop'

$candidates = @(
    $env:PCL_EVAL_PYTHON,
    'C:\Users\13437\anaconda3\python.exe',
    'python.exe'
) | Where-Object { $_ }

$python = $null
foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $python = $candidate
        break
    }
    $command = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($command -and $command.Source -notlike '*WindowsApps*') {
        $python = $command.Source
        break
    }
}

if (-not $python) {
    throw 'No usable Python installation found. Set PCL_EVAL_PYTHON.'
}

$pythonRoot = Split-Path -Parent $python
$dllDirs = @(
    $pythonRoot,
    (Join-Path $pythonRoot 'Library\mingw-w64\bin'),
    (Join-Path $pythonRoot 'Library\usr\bin'),
    (Join-Path $pythonRoot 'Library\bin'),
    (Join-Path $pythonRoot 'Scripts'),
    (Join-Path $pythonRoot 'bin')
) | Where-Object { Test-Path -LiteralPath $_ -PathType Container }

$env:Path = (($dllDirs + $env:Path) -join ';')
& $python @PythonArgs
exit $LASTEXITCODE

