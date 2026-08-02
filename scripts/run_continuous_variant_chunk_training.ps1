param(
    [int]$ChunkGames = 200000,
    [int]$PollSeconds = 3600,
    [int]$MaxTrainingWorkers = 2
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path 'logs','results\evaluations','models','data' | Out-Null

$name = 'v2_variant_board5_hand6_oneoff_tiered_continuous_chunk_training'
$stdoutPath = Join-Path $root "logs\$name.stdout.log"
$stderrPath = Join-Path $root "logs\$name.stderr.log"
$pidPath = Join-Path $root "logs\$name.pid"
$statusPath = Join-Path $root "results\evaluations\$name.status.json"

function Write-AtomicJson {
    param([object]$Payload, [string]$Path)
    $temporary = "$Path.tmp"
    $Payload | ConvertTo-Json -Depth 12 | Set-Content -Encoding utf8 $temporary
    Move-Item -Force -LiteralPath $temporary -Destination $Path
}

function Write-LaunchStatus {
    param([string]$State, [string]$Message = '')
    Write-AtomicJson -Path $statusPath -Payload ([ordered]@{
        state = $State
        step = 'launch'
        last_completed_step = ''
        message = $Message
        updated_at = (Get-Date).ToString('o')
        source = 'data\v2_variant_board5_hand6_oneoff_tiered_heuristic4_continuous_20260802'
        summary = "results\evaluations\$name.json"
        stop_file = "results\evaluations\$name.stop"
        stdout = "logs\$name.stdout.log"
        stderr = "logs\$name.stderr.log"
    })
}

$arguments = @(
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-Command',
    @"
`$ErrorActionPreference='Stop'
Set-Location '$root'
`$p=[Environment]::GetEnvironmentVariable('Path','Process')
if([string]::IsNullOrEmpty(`$p)){`$p=[Environment]::GetEnvironmentVariable('PATH','Process')}
[Environment]::SetEnvironmentVariable('PATH',`$null,'Process')
[Environment]::SetEnvironmentVariable('Path',`$p,'Process')
`$env:PYTHONUNBUFFERED='1'
`$env:PYTHONDONTWRITEBYTECODE='1'
`$env:PYTHONPATH='src'
python -m yellowstone.run_continuous_variant_chunk_training --chunk-games $ChunkGames --poll-seconds $PollSeconds --max-training-workers $MaxTrainingWorkers
"@
)

try {
    Write-LaunchStatus -State 'launching'
    $process = Start-Process `
        -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
        -ArgumentList $arguments `
        -WorkingDirectory $root `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -WindowStyle Hidden `
        -PassThru
    Set-Content -Encoding ascii -LiteralPath $pidPath -Value $process.Id
    Write-LaunchStatus -State 'running' -Message "pid=$($process.Id)"
    Write-Output "started $name pid=$($process.Id)"
}
catch {
    Write-LaunchStatus -State 'failed' -Message $_.Exception.Message
    throw
}
