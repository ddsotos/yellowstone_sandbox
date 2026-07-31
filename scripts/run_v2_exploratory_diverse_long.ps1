param(
    [int]$Seed = 20260730,
    [int]$GameIdOffset = 1100912
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'

$name = 'v2_exploratory_diverse_v1_20260730'
$statusPath = "results\collections\$name.status.json"
$pidPath = "logs\$name.pid"
$stopFile = "results\collections\$name.stop"
$output = "data\$name"
$checkpoint = 'models\win_value_v1_original_generation0_197800_epoch002.pt'

if (Test-Path -LiteralPath $stopFile) {
    Remove-Item -LiteralPath $stopFile -Force
}
New-Item -ItemType Directory -Force -Path 'logs' | Out-Null
Set-Content -Encoding ascii -LiteralPath $pidPath -Value $PID

try {
    & python -m yellowstone.exploratory_collection `
        --checkpoint $checkpoint `
        --seed $Seed `
        --game-id-offset $GameIdOffset `
        --output $output `
        --stop-file $stopFile `
        --status-file $statusPath `
        --shard-games 100 `
        --lazy-single-pass
    if ($LASTEXITCODE -ne 0) {
        throw "collector failed with exit code $LASTEXITCODE"
    }
}
catch {
    $payload = [ordered]@{
        state = 'failed'
        step = 'collect'
        last_completed_step = ''
        message = $_.Exception.Message
        updated_at = (Get-Date).ToString('o')
        pid = $PID
        output = $output
        checkpoint = $checkpoint
        stop_file = $stopFile
    }
    $temporary = "$statusPath.tmp"
    $payload | ConvertTo-Json | Set-Content -Encoding utf8 $temporary
    Move-Item -Force -LiteralPath $temporary -Destination $statusPath
    throw
}
