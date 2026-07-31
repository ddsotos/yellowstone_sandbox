$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$name = 'v2_exploratory_diverse_v1_20260730'
$stopFile = "results\collections\$name.stop"
$directory = Split-Path -Parent $stopFile
New-Item -ItemType Directory -Force -Path $directory | Out-Null
Set-Content -Encoding utf8 -LiteralPath $stopFile -Value (
    "stop requested at " + (Get-Date).ToString('o')
)
Write-Output "stop_requested=$stopFile"
