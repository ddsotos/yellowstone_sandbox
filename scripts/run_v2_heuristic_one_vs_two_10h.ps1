param(
    [double]$Hours = 10.0,
    [int]$Seed = 20260729,
    [int]$GameIdOffset = 954346
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'

$statusPath = 'results\collections\v2_heuristic_one_vs_two_v1_10h_20260729.status.json'
$output = 'data\v2_heuristic_one_vs_two_v1_10h_20260729'
$checkpoint = 'models\win_value_v1_original_generation0_197800_epoch002.pt'

function Write-Status {
    param([string]$State, [string]$Message = '')
    $directory = Split-Path -Parent $statusPath
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    $payload = [ordered]@{
        state = $State
        message = $Message
        updated_at = (Get-Date).ToString('o')
        pid = $PID
        output = $output
        checkpoint = $checkpoint
    }
    $temporary = "$statusPath.tmp"
    $payload | ConvertTo-Json | Set-Content -Encoding utf8 $temporary
    Move-Item -Force -LiteralPath $temporary -Destination $statusPath
}

try {
    Write-Status 'running'
    & python -m yellowstone.fast_value_npc collect `
        --checkpoint $checkpoint `
        --mode heuristic-one-vs-two `
        --duration-hours "$Hours" `
        --seed "$Seed" `
        --game-id-offset "$GameIdOffset" `
        --output $output `
        --shard-games 100
    if ($LASTEXITCODE -ne 0) {
        throw "collector failed with exit code $LASTEXITCODE"
    }
    $manifest = Get-Content -Raw -Encoding utf8 (
        Join-Path $output 'collection_manifest.json'
    ) | ConvertFrom-Json
    if ($manifest.status -ne 'complete') {
        throw "collector exited without a complete manifest"
    }
    Write-Status 'complete'
}
catch {
    Write-Status 'failed' $_.Exception.Message
    throw
}
