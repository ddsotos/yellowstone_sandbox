param(
    [double]$Hours = 720.0,
    [int]$Seed = 20260729,
    [int]$GameIdOffset = 1043312
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'

$name = 'v2_heuristic_one_vs_two_v1_continuation_20260729'
$statusPath = "results\collections\$name.status.json"
$output = "data\$name"
$checkpoint = 'models\win_value_v1_original_generation0_197800_epoch002.pt'

function Write-AtomicStatus {
    param(
        [string]$State,
        [string]$Message = ''
    )
    $directory = Split-Path -Parent $statusPath
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    $payload = [ordered]@{
        state = $State
        step = 'collect'
        last_completed_step = ''
        message = $Message
        updated_at = (Get-Date).ToString('o')
        pid = $PID
        output = $output
        checkpoint = $checkpoint
        seed = $Seed
        game_id_offset = $GameIdOffset
        target_hours = $Hours
        preceding_collection = (
            'data\v2_heuristic_one_vs_two_v1_10h_20260729'
        )
        preceding_games = 88966
        preceding_game_id_max = 1043311
        stdout = "logs\$name.stdout.log"
        stderr = "logs\$name.stderr.log"
    }
    $temporary = "$statusPath.tmp"
    $payload | ConvertTo-Json | Set-Content -Encoding utf8 $temporary
    Move-Item -Force -LiteralPath $temporary -Destination $statusPath
}

try {
    Write-AtomicStatus 'running'
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
    Write-AtomicStatus 'complete'
}
catch {
    Write-AtomicStatus 'failed' $_.Exception.Message
    throw
}
