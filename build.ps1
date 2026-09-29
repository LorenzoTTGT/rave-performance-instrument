$ErrorActionPreference = 'Stop'
$python = if ($env:RAVE_PYTHON) { $env:RAVE_PYTHON } else { 'python' }
& $python "$PSScriptRoot\scripts\build.py" @args
exit $LASTEXITCODE
