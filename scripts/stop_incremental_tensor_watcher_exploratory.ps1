$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$stopFile = (
    'results\collections\' +
    'v2_exploratory_diverse_v1_20260730_tensor_watcher.stop'
)
New-Item -ItemType Directory -Force -Path (
    Split-Path -Parent $stopFile
) | Out-Null
Set-Content -Encoding utf8 -LiteralPath $stopFile -Value (
    "stop requested at " + (Get-Date).ToString('o')
)
Write-Output "stop_requested=$stopFile"
