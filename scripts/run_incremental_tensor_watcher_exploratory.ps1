$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'

$name = 'v2_exploratory_diverse_v1_20260730_tensor_watcher'
$sourceName = 'v2_exploratory_diverse_v1_20260730'
$statusPath = "results\collections\$name.status.json"
$pidPath = "logs\$name.pid"
$stopFile = "results\collections\$name.stop"
$source = "data\$sourceName"
$sourceManifest = Join-Path $source 'collection_manifest.json'
$v1Output = 'data\v1_original_exploratory_diverse_v1_20260730_canonical'
$preplayOutput = (
    'data\privileged_state_exploratory_diverse_v1_20260730_tensors'
)

if (Test-Path -LiteralPath $stopFile) {
    Remove-Item -LiteralPath $stopFile -Force
}
New-Item -ItemType Directory -Force -Path 'logs' | Out-Null
Set-Content -Encoding ascii -LiteralPath $pidPath -Value $PID

try {
    & python -m yellowstone.incremental_tensor_watcher `
        --source $source `
        --v1-output $v1Output `
        --preplay-output $preplayOutput `
        --game-id-rebase 1100912 `
        --source-manifest $sourceManifest `
        --status-file $statusPath `
        --stop-file $stopFile `
        --poll-seconds 15
    if ($LASTEXITCODE -ne 0) {
        throw "tensor watcher failed with exit code $LASTEXITCODE"
    }
}
catch {
    $payload = [ordered]@{
        state = 'failed'
        step = 'tensorize_completed_shards'
        last_completed_step = ''
        message = $_.Exception.Message
        updated_at = (Get-Date).ToString('o')
        pid = $PID
        source = $source
        v1_output = $v1Output
        preplay_output = $preplayOutput
        stop_file = $stopFile
    }
    $temporary = "$statusPath.tmp"
    $payload | ConvertTo-Json | Set-Content -Encoding utf8 $temporary
    Move-Item -Force -LiteralPath $temporary -Destination $statusPath
    throw
}
